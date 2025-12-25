"""
Training functions for VAE, Diffusion, and Consistency models
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
import math
from tqdm import tqdm
from torch.cuda.amp import GradScaler

from data.dataset import DataPrefetcher

logger = logging.getLogger(__name__)


def check_gradient_health(model, clip_value=1.0, epoch=1, batch_idx=0, debug=False):
    """Check if gradients are healthy after clipping"""
    total_norm = 0
    param_count = 0
    has_nan = False
    has_inf = False
    problematic_params = []
    layer_norms = {}
    
    if epoch <= 5:
        warning_threshold = clip_value * 20
    elif epoch <= 10:
        warning_threshold = clip_value * 10
    else:
        warning_threshold = clip_value * 5
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            layer_name = name.split('.')[0]
            if layer_name not in layer_norms:
                layer_norms[layer_name] = []
            layer_norms[layer_name].append(param_norm.item())
            
            if torch.isnan(param_norm):
                has_nan = True
                problematic_params.append(f"{name} (NaN)")
            elif torch.isinf(param_norm):
                has_inf = True
                problematic_params.append(f"{name} (Inf)")
            elif param_norm.item() > warning_threshold:
                if epoch > 5:
                    problematic_params.append(f"{name} (norm={param_norm.item():.2f})")
    
    total_norm = total_norm ** 0.5
    
    if has_nan or has_inf:
        logger.warning(f"Critical gradient issues at epoch {epoch}, batch {batch_idx}: NaN={has_nan}, Inf={has_inf}")
        for param in problematic_params[:5]:
            if "(NaN)" in param or "(Inf)" in param:
                logger.warning(f"  - {param}")
    
    elif total_norm > warning_threshold and debug and epoch > 5:
        logger.debug(f"Large gradient norm at epoch {epoch}: {total_norm:.2f} (threshold: {warning_threshold:.2f})")
    
    if debug and layer_norms and (batch_idx % 1000 == 0):
        logger.debug(f"Layer-wise gradient norms at epoch {epoch}, batch {batch_idx}:")
        for layer, norms in layer_norms.items():
            avg_norm = np.mean(norms)
            max_norm = np.max(norms)
            if avg_norm > 1.0:
                logger.debug(f"  - {layer}: avg={avg_norm:.3f}, max={max_norm:.3f}")
    
    return {
        'total_norm': total_norm,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'is_critical': has_nan or has_inf,
        'is_large': total_norm > warning_threshold and epoch > 5,
        'problematic_count': len(problematic_params),
        'problematic_params': problematic_params[:5]
    }


def train_vae_optimized(vae, dataloader, optimizer, scheduler, device, epoch, config, metrics_tracker, 
              lpips_loss_fn=None, lpips_weight=0.1, scaler=None, accumulation_steps=1,
              gpu_augmenter=None):
    """Optimized VAE training with async prefetching and GPU augmentation"""
    vae.train()
    total_loss = 0
    recon_losses = []
    kl_losses = []
    lpips_losses = []
    
    from losses.losses import StableKLLoss
    kl_loss_fn = StableKLLoss(free_bits=config['vae'].get('free_bits', 0.0))
    grad_clip_value = config['training'].get('grad_clip', 0.8)
    
    optimizer.zero_grad()
    
    if epoch <= 20:
        beta = 0.0
    elif epoch <= 40:
        beta = 0.00001 * (epoch - 20) / 20
    elif epoch <= 60:
        beta = 0.00001 + 0.00009 * (epoch - 40) / 20
    else:
        beta = min(0.0001, config['vae']['beta_max'])
    
    gradient_overflow_count = 0
    skipped_steps = 0
    
    prefetcher = DataPrefetcher(iter(dataloader), device)
    
    pbar = tqdm(range(len(dataloader)), desc=f'VAE Epoch {epoch}/{config["training"]["vae_epochs"]}')
    
    for i in pbar:
        try:
            batch = next(prefetcher)
        except StopIteration:
            break
        
        ct_slice = batch['ct_slice']
        
        if gpu_augmenter is not None and dataloader.dataset.augment:
            ct_slice, _ = gpu_augmenter(ct_slice, None, epoch=epoch)
        
        if torch.isnan(ct_slice).any() or torch.isinf(ct_slice).any():
            logger.warning(f"NaN/Inf detected in input at batch {i}, folder: {batch.get('folder', ['unknown'])[0]}, skipping")
            continue
        
        grad_norm = torch.tensor(0.0)
        
        try:
            with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
                recon, mean, logvar = vae(ct_slice)
                
                if torch.isnan(recon).any() or torch.isinf(recon).any():
                    logger.warning(f"NaN/Inf in reconstruction at batch {i}")
                    continue
                
                if torch.isnan(mean).any() or torch.isinf(mean).any():
                    logger.warning(f"NaN/Inf in mean at batch {i}")
                    continue
                
                if torch.isnan(logvar).any() or torch.isinf(logvar).any():
                    logger.warning(f"NaN/Inf in logvar at batch {i}")
                    continue
                
                recon_loss = F.mse_loss(recon, ct_slice, reduction='mean')
                recon_loss = torch.maximum(recon_loss, torch.tensor(1e-6, device=device))
                
                kl_loss = kl_loss_fn(mean, logvar)
                
                if lpips_loss_fn is not None and lpips_weight > 0 and epoch > 20:
                    lpips_loss = lpips_loss_fn(recon, ct_slice)
                    loss = recon_loss + beta * kl_loss + lpips_weight * lpips_loss
                    lpips_losses.append(lpips_loss.item())
                else:
                    loss = recon_loss + beta * kl_loss
                    lpips_losses.append(0.0)
                
                loss = loss / accumulation_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss at batch {i}: loss={loss.item()}, recon={recon_loss.item()}, kl={kl_loss.item()}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in forward pass at batch {i}: {str(e)}")
            continue
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip_value)
                grad_info = check_gradient_health(vae, clip_value=grad_clip_value, epoch=epoch, batch_idx=i)
                
                if grad_norm > grad_clip_value * 2:
                    gradient_overflow_count += 1
                
                if grad_info['is_critical']:
                    logger.warning(f"Critical gradient issues at batch {i}: "
                                 f"nan={grad_info['has_nan']}, inf={grad_info['has_inf']}")
                    optimizer.zero_grad()
                    scaler.update()
                    skipped_steps += 1
                    continue
                
                scaler.step(optimizer)
                scaler.update()
                
                if hasattr(scheduler, 'step') and hasattr(scheduler, 'warmup_steps'):
                    scheduler.step()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip_value)
                grad_info = check_gradient_health(vae, clip_value=grad_clip_value, epoch=epoch, batch_idx=i)
                
                if grad_norm > grad_clip_value * 2:
                    gradient_overflow_count += 1
                
                if grad_info['is_critical']:
                    logger.warning(f"Critical gradient issues at batch {i}: "
                                 f"nan={grad_info['has_nan']}, inf={grad_info['has_inf']}")
                    optimizer.zero_grad()
                    skipped_steps += 1
                    continue
                
                optimizer.step()
                
                if hasattr(scheduler, 'step') and hasattr(scheduler, 'warmup_steps'):
                    scheduler.step()
            
            optimizer.zero_grad()
        else:
            if scaler is not None:
                with torch.no_grad():
                    total_norm = 0
                    for p in vae.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = torch.tensor((total_norm ** 0.5) / scaler.get_scale())
            else:
                with torch.no_grad():
                    grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), float('inf'))
        
        actual_loss = loss.item() * accumulation_steps
        total_loss += actual_loss
        recon_losses.append(recon_loss.item())
        kl_losses.append(kl_loss.item())
        
        pbar.set_postfix({
            'loss': actual_loss,
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'beta': beta,
            'lr': optimizer.param_groups[0]['lr'],
            'grad': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'mem': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
        })
    
    if scheduler is not None:
        from .schedulers import AdaptiveLearningRateScheduler
        if isinstance(scheduler, AdaptiveLearningRateScheduler):
            avg_loss = total_loss / max(len(recon_losses), 1)
            new_lr = scheduler.step(avg_loss)
            logger.info(f"Adaptive LR adjusted to: {new_lr:.6f}")
        elif hasattr(scheduler, 'step') and not hasattr(scheduler, 'warmup_steps'):
            scheduler.step(total_loss / len(dataloader))
    
    if gradient_overflow_count > 0:
        overflow_rate = gradient_overflow_count / len(dataloader) * 100
        logger.info(f"Gradient overflows this epoch: {gradient_overflow_count} ({overflow_rate:.2f}%)")
    
    if skipped_steps > 0:
        skip_rate = skipped_steps / len(dataloader) * 100
        logger.info(f"Skipped steps due to critical issues: {skipped_steps} ({skip_rate:.2f}%)")
    
    avg_loss = total_loss / max(len(recon_losses), 1)
    avg_recon = np.mean(recon_losses) if recon_losses else 0
    avg_kl = np.mean(kl_losses) if kl_losses else 0
    avg_lpips = np.mean(lpips_losses) if lpips_losses else 0
    
    metrics_tracker.update('vae', train_loss=avg_loss, recon_loss=avg_recon, 
                          kl_loss=avg_kl, lpips_loss=avg_lpips)
    metrics_tracker.log_epoch(epoch, 'vae_train', 
                            total_loss=avg_loss, 
                            recon_loss=avg_recon, 
                            kl_loss=avg_kl,
                            lpips_loss=avg_lpips,
                            beta=beta,
                            lr=optimizer.param_groups[0]['lr'],
                            overflow_count=gradient_overflow_count,
                            skipped_steps=skipped_steps)
    
    return avg_loss, avg_recon, avg_kl


def train_diffusion_optimized(diffusion_model, vae, diffusion_process, dataloader, optimizer, scheduler, 
                   device, epoch, config, metrics_tracker, gdl_loss_fn, lpips_loss_fn, l1_loss_fn,
                   l1_weight=0.5, gdl_weight=0.5, lpips_weight=0.1, scaler=None, ema_wrapper=None,
                   accumulation_steps=1, use_amp=True, autocast_dtype=torch.float16,
                   gpu_augmenter=None):
    """Optimized diffusion training with v-param, self-conditioning, and CFG"""
    diffusion_model.train()
    vae.eval()
    
    for param in vae.parameters():
        param.requires_grad = False
    
    total_loss = 0
    mse_losses = []
    l1_losses = []
    gdl_losses = []
    lpips_losses = []
    
    grad_clip_value = config['training'].get('grad_clip', 0.8)
    
    # MSE FIX: Adjusted thresholds
    high_mse_threshold = 1.2
    ignore_epochs = 3
    
    # CFG training parameters
    cfg_dropout_prob = config['diffusion'].get('cfg_dropout_prob', 0.1)
    use_self_conditioning = config['diffusion'].get('use_self_conditioning', True)
    self_cond_prob = config['diffusion'].get('self_cond_prob', 0.5)
    
    optimizer.zero_grad()
    
    grad_norm = torch.tensor(0.0)
    
    gradient_overflow_count = 0
    skipped_steps = 0
    nan_count = 0
    high_loss_count = 0
    
    prefetcher = DataPrefetcher(iter(dataloader), device)
    
    pbar = tqdm(range(len(dataloader)), desc=f'Diffusion Epoch {epoch}/{config["training"]["diffusion_epochs"]}')
    
    for i in pbar:
        try:
            batch = next(prefetcher)
        except StopIteration:
            break
        
        condition = batch['condition']
        ct_slice = batch['ct_slice']
        slice_position = batch.get('slice_position', None)
        
        if gpu_augmenter is not None and dataloader.dataset.augment:
            ct_slice, condition = gpu_augmenter(ct_slice, condition, epoch=epoch)
        
        if torch.isnan(condition).any() or torch.isinf(condition).any() or \
           torch.isnan(ct_slice).any() or torch.isinf(ct_slice).any():
            logger.warning(f"NaN/Inf detected in input at batch {i}, skipping")
            nan_count += 1
            continue
        
        with torch.no_grad():
            mean, logvar = vae.encoder(ct_slice)
            z = mean  # Deterministic target (no sampling)
            
            if torch.isnan(z).any() or torch.isinf(z).any():
                logger.warning(f"NaN/Inf in VAE encoding at batch {i}, skipping")
                nan_count += 1
                continue
        
        batch_size = z.shape[0]
        t = torch.randint(0, diffusion_process.num_timesteps, (batch_size,), device=device)
        
        noise = torch.randn_like(z)
        noisy_z = diffusion_process.q_sample(z, t, noise)
        
        # CFG: randomly drop condition
        if np.random.rand() < cfg_dropout_prob:
            condition_input = torch.zeros_like(condition)
        else:
            condition_input = condition
        
        # Self-conditioning
        x_self_cond = None
        if use_self_conditioning and np.random.rand() < self_cond_prob:
            with torch.no_grad():
                if np.random.rand() < 0.5:
                    x_self_cond = torch.zeros_like(z)
                else:
                    output_prev = diffusion_model(noisy_z, t, condition_input, slice_position, None)
                    if diffusion_process.prediction_type == 'v':
                        x_self_cond = diffusion_process.predict_x0_from_v(noisy_z, t, output_prev)
                    elif diffusion_process.prediction_type == 'epsilon':
                        x_self_cond = diffusion_process.predict_x0_from_eps(noisy_z, t, output_prev)
                    else:
                        x_self_cond = output_prev
                    x_self_cond = x_self_cond.detach()
        
        try:
            with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=autocast_dtype):
                model_output = diffusion_model(noisy_z, t, condition_input, slice_position, x_self_cond)
                
                if torch.isnan(model_output).any() or torch.isinf(model_output).any():
                    logger.warning(f"NaN/Inf in model output at batch {i}")
                    nan_count += 1
                    optimizer.zero_grad()
                    continue
                
                # Compute target based on prediction type
                if diffusion_process.prediction_type == 'v':
                    target = diffusion_process.get_v(z, noise, t)
                elif diffusion_process.prediction_type == 'epsilon':
                    target = noise
                elif diffusion_process.prediction_type == 'x0':
                    target = z
                else:
                    raise ValueError(f"Unknown prediction type: {diffusion_process.prediction_type}")
                
                # MSE loss on the prediction
                mse_loss = F.mse_loss(model_output, target)
                
                # MSE FIX: Only warn after ignore_epochs and with higher threshold
                if epoch > ignore_epochs and mse_loss.item() > high_mse_threshold:
                    high_loss_count += 1
                    logger.warning(f"Batch {i}: MSE {mse_loss.item():.4f} is above threshold "
                                 f"(>{high_mse_threshold}). This can indicate augmentation or "
                                 f"conditioning issues.")
                    
                    if high_loss_count == 1 and epoch > ignore_epochs:
                        debug_dir = os.path.join(config['save_dir'], 'debug_high_loss')
                        os.makedirs(debug_dir, exist_ok=True)
                        torch.save({
                            'condition': condition,
                            'ct_slice': ct_slice,
                            'model_output': model_output,
                            'target': target,
                            'epoch': epoch,
                            'batch': i,
                            'mse_loss': mse_loss.item()
                        }, os.path.join(debug_dir, f'high_loss_epoch{epoch}_batch{i}.pt'))
                
                # Predict x0 for perceptual losses
                if diffusion_process.prediction_type == 'v':
                    x0_pred = diffusion_process.predict_x0_from_v(noisy_z, t, model_output)
                elif diffusion_process.prediction_type == 'epsilon':
                    x0_pred = diffusion_process.predict_x0_from_eps(noisy_z, t, model_output)
                else:
                    x0_pred = model_output
                
                x0_pred = torch.clamp(x0_pred, -1, 1)
                
                decoded_pred = vae.decode(x0_pred)
                decoded_pred = torch.clamp(decoded_pred, -1, 1)
                
                if torch.isnan(decoded_pred).any() or torch.isinf(decoded_pred).any():
                    logger.warning(f"NaN/Inf in decoded prediction at batch {i}")
                    nan_count += 1
                    optimizer.zero_grad()
                    continue
                
                l1_loss = l1_loss_fn(decoded_pred, ct_slice) if l1_weight > 0 else torch.tensor(0.0, device=device)
                gdl_loss = gdl_loss_fn(decoded_pred, ct_slice) if gdl_weight > 0 else torch.tensor(0.0, device=device)
                lpips_loss = lpips_loss_fn(decoded_pred, ct_slice) if lpips_weight > 0 else torch.tensor(0.0, device=device)
                
                loss = mse_loss + l1_weight * l1_loss + gdl_weight * gdl_loss + lpips_weight * lpips_loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf in loss at batch {i}: mse={mse_loss.item()}, l1={l1_loss.item()}, gdl={gdl_loss.item()}, lpips={lpips_loss.item()}")
                    nan_count += 1
                    optimizer.zero_grad()
                    continue
                
                loss = loss / accumulation_steps
                
        except Exception as e:
            logger.error(f"Error in forward pass at batch {i}: {str(e)}")
            optimizer.zero_grad()
            continue
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), grad_clip_value)
                grad_info = check_gradient_health(diffusion_model, clip_value=grad_clip_value, 
                                                epoch=epoch, batch_idx=i)
                
                if grad_norm > grad_clip_value * 2:
                    gradient_overflow_count += 1
                
                if grad_info['is_critical']:
                    logger.warning(f"Critical gradient issues at batch {i}: "
                                 f"nan={grad_info['has_nan']}, inf={grad_info['has_inf']}")
                    optimizer.zero_grad()
                    scaler.update()
                    skipped_steps += 1
                    continue
                
                scaler.step(optimizer)
                scaler.update()
                
                if hasattr(scheduler, 'step') and hasattr(scheduler, 'warmup_steps'):
                    scheduler.step()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), grad_clip_value)
                grad_info = check_gradient_health(diffusion_model, clip_value=grad_clip_value, 
                                                epoch=epoch, batch_idx=i)
                
                if grad_norm > grad_clip_value * 2:
                    gradient_overflow_count += 1
                
                if grad_info['is_critical']:
                    logger.warning(f"Critical gradient issues at batch {i}: "
                                 f"nan={grad_info['has_nan']}, inf={grad_info['has_inf']}")
                    optimizer.zero_grad()
                    skipped_steps += 1
                    continue
                
                optimizer.step()
                
                if hasattr(scheduler, 'step') and hasattr(scheduler, 'warmup_steps'):
                    scheduler.step()
            
            optimizer.zero_grad()
            
            if ema_wrapper is not None:
                ema_wrapper.update()
        else:
            if scaler is not None:
                with torch.no_grad():
                    total_norm = 0
                    scale = scaler.get_scale()
                    if scale > 0:
                        for p in diffusion_model.parameters():
                            if p.grad is not None:
                                param_norm = (p.grad.data.float() / scale).norm(2)
                                total_norm += param_norm.item() ** 2
                        grad_norm = math.sqrt(total_norm)
                    else:
                        grad_norm = 0.0
            else:
                with torch.no_grad():
                    grad_norm = torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), float('inf'))
        
        actual_loss = loss.item() * accumulation_steps
        total_loss += actual_loss
        mse_losses.append(mse_loss.item())
        l1_losses.append(l1_loss.item() if l1_weight > 0 else 0.0)
        gdl_losses.append(gdl_loss.item() if gdl_weight > 0 else 0.0)
        lpips_losses.append(lpips_loss.item() if lpips_weight > 0 else 0.0)
        
        pbar.set_postfix({
            'loss': actual_loss,
            'mse': mse_loss.item(),
            'l1': l1_loss.item() if l1_weight > 0 else 0.0,
            'gdl': gdl_loss.item() if gdl_weight > 0 else 0.0,
            'lpips': lpips_loss.item() if lpips_weight > 0 else 0.0,
            'pred': diffusion_process.prediction_type[:3],
            'lr': optimizer.param_groups[0]['lr'],
            'grad': grad_norm if isinstance(grad_norm, (int, float)) else grad_norm.item(),
            'mem': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A",
            'nan': nan_count
        })
    
    if scheduler is not None and not hasattr(scheduler, 'warmup_steps'):
        scheduler.step()
    
    for param in vae.parameters():
        param.requires_grad = True
    
    if gradient_overflow_count > 0:
        overflow_rate = gradient_overflow_count / len(dataloader) * 100
        logger.info(f"Gradient overflows this epoch: {gradient_overflow_count} ({overflow_rate:.2f}%)")
    
    if skipped_steps > 0:
        skip_rate = skipped_steps / len(dataloader) * 100
        logger.info(f"Skipped steps due to critical issues: {skipped_steps} ({skip_rate:.2f}%)")
    
    if nan_count > 0:
        nan_rate = nan_count / len(dataloader) * 100
        logger.warning(f"NaN/Inf detected in {nan_count} batches ({nan_rate:.2f}%)")
        if nan_rate > 10:
            logger.error("High NaN rate detected! Consider reducing learning rate or disabling AMP")
    
    if high_loss_count > 0 and epoch > ignore_epochs:
        high_loss_rate = high_loss_count / len(dataloader) * 100
        logger.info(f"High MSE loss (>{high_mse_threshold}) detected in {high_loss_count} batches ({high_loss_rate:.2f}%)")
        if high_loss_rate > 50 and epoch > 10:
            logger.error("CRITICAL: Over 50% of batches have high MSE loss after warmup! Check augmentation alignment.")
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = np.mean(mse_losses)
    avg_l1 = np.mean(l1_losses)
    avg_gdl = np.mean(gdl_losses)
    avg_lpips = np.mean(lpips_losses)
    
    if avg_mse > 2.0 and epoch > 10:
        should_stop = metrics_tracker.check_early_stop(avg_mse)
        if should_stop:
            logger.error(f"Early stopping triggered due to very high MSE loss: {avg_mse:.4f}")
            return avg_loss
    
    metrics_tracker.update('diffusion', 
                         train_loss=avg_loss, 
                         mse_loss=avg_mse, 
                         l1_loss=avg_l1,
                         gdl_loss=avg_gdl,
                         lpips_loss=avg_lpips)
    metrics_tracker.log_epoch(epoch, 'diffusion_train', 
                            loss=avg_loss,
                            mse_loss=avg_mse,
                            l1_loss=avg_l1,
                            gdl_loss=avg_gdl,
                            lpips_loss=avg_lpips,
                            pred_type=diffusion_process.prediction_type,
                            lr=optimizer.param_groups[0]['lr'],
                            overflow_count=gradient_overflow_count,
                            nan_count=nan_count,
                            high_loss_count=high_loss_count)
    
    return avg_loss


def train_consistency_3d(consistency_net, vae, diffusion_model, diffusion_process, 
                        dataloader, optimizer, device, epoch, config, num_slices, metrics_tracker,
                        teacher_forcing_ratio=0.5, ema_wrapper=None):
    """Train 3D consistency network with teacher forcing"""
    consistency_net.train()
    vae.eval()
    
    if ema_wrapper is not None:
        eval_model = ema_wrapper.get_model()
        eval_model.eval()
    else:
        eval_model = diffusion_model
        eval_model.eval()
    
    total_loss = 0
    
    grad_clip_value = config['training'].get('grad_clip', 0.8)
    
    num_batches = min(10, len(dataloader))
    
    prefetcher = DataPrefetcher(iter(dataloader), device)
    
    pbar = tqdm(range(num_batches), desc=f'3D Consistency Epoch {epoch}/{config["training"]["consistency_epochs"]}')
    
    for batch_idx in pbar:
        try:
            batch = next(prefetcher)
        except StopIteration:
            break
            
        condition = batch['condition']
        volume_idx = batch['volume_idx'][0].item()
        folder = batch['folder'][0]
        
        generated_slices = []
        target_slices = []
        
        with torch.no_grad():
            latent_h, latent_w = vae.get_latent_shape((200, 200))
            
            start_idx, end_idx = dataloader.dataset.slice_range
            slice_indices = np.linspace(
                start_idx, 
                end_idx - 1,
                num_slices, 
                dtype=int
            )
            
            for j, slice_idx in enumerate(slice_indices):
                relative_slice_idx = slice_idx - start_idx
                absolute_idx = volume_idx * dataloader.dataset.slices_per_volume + relative_slice_idx
                
                if absolute_idx >= len(dataloader.dataset):
                    logger.warning(f"Slice index {absolute_idx} out of range, skipping")
                    continue
                
                if np.random.rand() < teacher_forcing_ratio:
                    try:
                        slice_data = dataloader.dataset[absolute_idx]
                        gt_slice = slice_data['ct_slice'].unsqueeze(0).to(device)
                        
                        if torch.isnan(gt_slice).any() or torch.isinf(gt_slice).any():
                            logger.warning(f"NaN/Inf in ground truth slice {absolute_idx}, skipping")
                            continue
                            
                        target_slices.append(gt_slice)
                        generated_slices.append(gt_slice)
                    except Exception as e:
                        logger.warning(f"Error getting slice {absolute_idx}: {e}")
                    continue
                
                try:
                    slice_data = dataloader.dataset[absolute_idx]
                    slice_condition = slice_data['condition'].unsqueeze(0).to(device)
                    
                    latent_shape = (1, vae.z_channels, latent_h, latent_w)
                    z = diffusion_process.p_sample_loop(eval_model, latent_shape, slice_condition, device)
                    slice_img = vae.decode(z)
                    generated_slices.append(slice_img)
                    
                    target_slices.append(slice_data['ct_slice'].unsqueeze(0).to(device))
                except Exception as e:
                    logger.warning(f"Error generating slice {absolute_idx}: {e}")
        
        if len(target_slices) < num_slices // 2:
            logger.warning(f"Not enough slices for volume {volume_idx}, skipping")
            continue
        
        generated_volume = torch.cat(generated_slices, dim=0).unsqueeze(0)
        generated_volume = generated_volume.permute(0, 2, 1, 3, 4)
        
        target_volume = torch.cat(target_slices, dim=0).unsqueeze(0)
        target_volume = target_volume.permute(0, 2, 1, 3, 4)
        
        optimizer.zero_grad()
        refined_volume = consistency_net(generated_volume)
        
        loss = F.mse_loss(refined_volume, target_volume)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(consistency_net.parameters(), grad_clip_value)
        optimizer.step()
        
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    metrics_tracker.update('consistency', train_loss=avg_loss)
    metrics_tracker.log_epoch(epoch, 'consistency_train', loss=avg_loss)
    
    return avg_loss
