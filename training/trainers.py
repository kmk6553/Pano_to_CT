"""
Training functions for 3D Slab-based VAE, Diffusion, and Consistency models
Processes 3-slice windows: [B, 1, D=3, H, W]

FIXES APPLIED:
1. Latent Scaling - scale_factor 적용 (기본값 0.18215, Stable Diffusion 표준)
2. Deterministic Encoding - z = mean 사용 (구조 정확도 향상) (v5.3)
3. Decode 전 Rescaling 적용
4. GPU Augmentation 조건문 수정 - dataset.augment 플래그 제거 (v5.1)
5. 모든 3개 슬라이스에 보조 손실 함수(LPIPS, L1, GDL) 적용 (v5.2)
6. compute_3d_loss 스케일 오류 수정 - 분모에 C 포함 (v5.4)
7. Image-space 보조 loss에 mid_weight 적용 (MSE는 균등 유지) (v5.4)
8. [신규] VAE 학습에 GDL 추가 및 균등 가중치(Equal Weight) 적용 (v5.5)
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
    
    return {
        'total_norm': total_norm,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'is_critical': has_nan or has_inf,
        'is_large': total_norm > warning_threshold and epoch > 5,
        'problematic_count': len(problematic_params),
        'problematic_params': problematic_params[:5]
    }


def compute_3d_loss(pred, target, mid_weight=None):
    """
    Compute loss for 3D volumes with optional mid-slice weighting
    
    Args:
        pred: [B, C, D=3, H, W] - predicted volume (C can be 1 for image or z_ch for latent)
        target: [B, C, D=3, H, W] - target volume
        mid_weight: Weight for middle slice (None=equal, 0.5=mid focused)
                    [수정] VAE 학습 시 None 사용하면 균등 가중치
    
    Returns:
        Weighted MSE loss (properly scaled)
    
    FIXED (v5.4): 분모에 C(채널) 포함하여 스케일 정확하게 계산
    FIXED (v5.5): mid_weight=None일 때 균등 가중치 (1/3) 사용
    """
    # Base MSE on all slices (element-wise)
    mse_all = F.mse_loss(pred, target, reduction='none')
    
    # [수정]: mid_weight가 None이면 균등 가중치 사용
    if mid_weight is None:
        mid_weight = 1/3  # 균등 가중치
    
    if mid_weight != 1/3:
        # Apply slice-wise weighting: [prev, mid, next]
        # mid_weight for middle, (1-mid_weight)/2 for sides
        side_weight = (1 - mid_weight) / 2
        weights = torch.tensor([side_weight, mid_weight, side_weight], 
                              device=pred.device, dtype=pred.dtype)
        # Reshape for broadcasting: [1, 1, 3, 1, 1]
        weights = weights.view(1, 1, 3, 1, 1)
        
        # Apply weights
        mse_weighted = mse_all * weights
        
        # FIXED (v5.4): 분모에 모든 차원 포함 (B, C, D, H, W)
        B, C, D, H, W = pred.shape
        denominator = weights.sum() * B * C * H * W
        return mse_weighted.sum() / denominator
    else:
        # Equal weighting: just use standard mean
        return mse_all.mean()


def flatten_3d_to_4d(volume):
    """
    Flatten 5D volume [B, C, D=3, H, W] to 4D [B*D, C, H, W] for 2D loss functions
    
    Args:
        volume: [B, C, D=3, H, W] tensor
    
    Returns:
        [B*D, C, H, W] tensor
    """
    B, C, D, H, W = volume.shape
    # Permute to [B, D, C, H, W] then reshape to [B*D, C, H, W]
    return volume.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)


def compute_slicewise_weighted_loss(pred, target, loss_fn, mid_weight=None):
    """
    Compute loss with slice-wise weighting for 3D volumes
    
    Args:
        pred: [B, C, D=3, H, W] - predicted volume
        target: [B, C, D=3, H, W] - target volume
        loss_fn: Loss function that takes (pred, target) and returns scalar
        mid_weight: Weight for middle slice (None=equal, 0.5=mid focused)
                    [수정] VAE 학습 시 None 사용하면 균등 가중치
    
    Returns:
        Weighted loss scalar
    """
    # [수정]: mid_weight가 None이면 균등 가중치 사용
    if mid_weight is None:
        mid_weight = 1/3  # 균등 가중치
    
    # Slice weights: [prev, mid, next]
    side_weight = (1 - mid_weight) / 2
    slice_weights = [side_weight, mid_weight, side_weight]
    
    weighted_loss = 0.0
    total_weight = sum(slice_weights)
    
    for i, w in enumerate(slice_weights):
        pred_slice = pred[:, :, i, :, :]    # [B, C, H, W]
        target_slice = target[:, :, i, :, :]
        slice_loss = loss_fn(pred_slice, target_slice)
        weighted_loss += w * slice_loss
    
    return weighted_loss / total_weight


def train_vae_optimized(vae, dataloader, optimizer, scheduler, device, epoch, config, metrics_tracker, 
              lpips_loss_fn=None, lpips_weight=0.1, scaler=None, accumulation_steps=1,
              gpu_augmenter=None, gdl_loss_fn=None, gdl_weight=0.0):
    """
    Optimized 3D VAE training with async prefetching and GPU augmentation
    
    Processes 3-slice windows: [B, 1, D=3, H, W]
    
    FIXES APPLIED:
    1. [수정 v5.2]: LPIPS Loss is now applied to all 3 slices
    2. [수정 v5.5]: VAE 학습 시 균등 가중치(Equal Weight) 사용 - mid_weight=None
    3. [수정 v5.5]: GDL(Gradient Difference Loss) 추가
    
    Args:
        gdl_loss_fn: GradientDifferenceLoss 인스턴스 (None이면 비활성화)
        gdl_weight: GDL loss 가중치 (0이면 비활성화)
    """
    vae.train()
    total_loss = 0
    recon_losses = []
    kl_losses = []
    lpips_losses = []
    gdl_losses = []  # [추가] GDL loss 추적
    
    from losses.losses import StableKLLoss
    kl_loss_fn = StableKLLoss(free_bits=config['vae'].get('free_bits', 0.0))
    grad_clip_value = config['training'].get('grad_clip', 0.8)
    
    # [수정 v5.5]: VAE 학습 시 균등 가중치 사용
    # mid_weight를 None으로 설정하여 모든 슬라이스에 동일한 가중치 부여
    # Z축 전체 복원이 중요하므로 mid_slice_weight 무시
    current_mid_weight = None  # 균등 가중치 (1/3)
    
    # [기록용] 설정된 mid_weight 로깅
    config_mid_weight = config['vae'].get('mid_slice_weight', 0.5)
    if epoch == 1:
        logger.info(f"VAE training config:")
        logger.info(f"  - Config mid_slice_weight: {config_mid_weight} (IGNORED for VAE)")
        logger.info(f"  - Actual mid_weight: Equal (1/3 for all slices)")
        logger.info(f"  - GDL weight: {gdl_weight}")
        logger.info(f"  - LPIPS weight: {lpips_weight}")
    
    optimizer.zero_grad()
    
    # KL annealing schedule
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
        
        # Get 3D CT volume: [B, 1, D=3, H, W]
        ct_volume = batch['ct_volume']
        
        # Ensure correct shape for 3D VAE
        if ct_volume.dim() == 4:
            # [B, 3, H, W] -> [B, 1, 3, H, W]
            ct_volume = ct_volume.unsqueeze(1)
        
        # GPU Augmentation
        if gpu_augmenter is not None:
            ct_volume_2d = ct_volume.squeeze(1)  # [B, 3, H, W] treat depth as channels
            ct_volume_2d, _ = gpu_augmenter(ct_volume_2d, None, epoch=epoch)
            ct_volume = ct_volume_2d.unsqueeze(1)  # [B, 1, 3, H, W]
        
        if torch.isnan(ct_volume).any() or torch.isinf(ct_volume).any():
            logger.warning(f"NaN/Inf detected in input at batch {i}, skipping")
            continue
        
        grad_norm = torch.tensor(0.0)
        
        try:
            with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
                # Forward pass through 3D VAE
                recon, mean, logvar = vae(ct_volume)
                
                if torch.isnan(recon).any() or torch.isinf(recon).any():
                    logger.warning(f"NaN/Inf in reconstruction at batch {i}")
                    continue
                
                if torch.isnan(mean).any() or torch.isinf(mean).any():
                    logger.warning(f"NaN/Inf in mean at batch {i}")
                    continue
                
                if torch.isnan(logvar).any() or torch.isinf(logvar).any():
                    logger.warning(f"NaN/Inf in logvar at batch {i}")
                    continue
                
                # [수정 v5.5]: Reconstruction loss with 균등 가중치
                recon_loss = compute_3d_loss(recon, ct_volume, mid_weight=current_mid_weight)
                recon_loss = torch.maximum(recon_loss, torch.tensor(1e-6, device=device))
                
                # KL loss
                kl_loss = kl_loss_fn(mean, logvar)
                
                # Initialize total loss
                loss = recon_loss + beta * kl_loss
                
                # [수정 v5.5]: GDL Loss 추가 (균등 가중치)
                if gdl_loss_fn is not None and gdl_weight > 0:
                    gdl_loss = compute_slicewise_weighted_loss(
                        recon, ct_volume, gdl_loss_fn, mid_weight=current_mid_weight
                    )
                    loss = loss + gdl_weight * gdl_loss
                    gdl_losses.append(gdl_loss.item())
                else:
                    gdl_losses.append(0.0)
                
                # LPIPS loss (균등 가중치)
                if lpips_loss_fn is not None and lpips_weight > 0 and epoch > 20:
                    lpips_loss = compute_slicewise_weighted_loss(
                        recon, ct_volume, lpips_loss_fn, mid_weight=current_mid_weight
                    )
                    loss = loss + lpips_weight * lpips_loss
                    lpips_losses.append(lpips_loss.item())
                else:
                    lpips_losses.append(0.0)
                
                loss = loss / accumulation_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss at batch {i}")
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
                    logger.warning(f"Critical gradient issues at batch {i}")
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
                    optimizer.zero_grad()
                    skipped_steps += 1
                    continue
                
                optimizer.step()
                
                if hasattr(scheduler, 'step') and hasattr(scheduler, 'warmup_steps'):
                    scheduler.step()
            
            optimizer.zero_grad()
        
        actual_loss = loss.item() * accumulation_steps
        total_loss += actual_loss
        recon_losses.append(recon_loss.item())
        kl_losses.append(kl_loss.item())
        
        pbar.set_postfix({
            'loss': actual_loss,
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'gdl': gdl_losses[-1] if gdl_losses else 0.0,
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
    avg_gdl = np.mean(gdl_losses) if gdl_losses else 0  # [추가]
    
    metrics_tracker.update('vae', train_loss=avg_loss, recon_loss=avg_recon, 
                          kl_loss=avg_kl, lpips_loss=avg_lpips)
    metrics_tracker.log_epoch(epoch, 'vae_train', 
                            total_loss=avg_loss, 
                            recon_loss=avg_recon, 
                            kl_loss=avg_kl,
                            lpips_loss=avg_lpips,
                            gdl_loss=avg_gdl,  # [추가]
                            beta=beta,
                            lr=optimizer.param_groups[0]['lr'],
                            overflow_count=gradient_overflow_count,
                            skipped_steps=skipped_steps,
                            mid_weight='equal')  # [추가] 균등 가중치 사용 로깅
    
    return avg_loss, avg_recon, avg_kl


def train_diffusion_optimized(diffusion_model, vae, diffusion_process, dataloader, optimizer, scheduler, 
                   device, epoch, config, metrics_tracker, gdl_loss_fn, lpips_loss_fn, l1_loss_fn,
                   l1_weight=0.5, gdl_weight=0.5, lpips_weight=0.1, scaler=None, ema_wrapper=None,
                   accumulation_steps=1, use_amp=True, autocast_dtype=torch.float16,
                   gpu_augmenter=None):
    """
    Optimized 3D diffusion training with v-param, self-conditioning, and CFG
    
    Processes 3D latent volumes: [B, C, D=3, h, w]
    
    FIXES APPLIED:
    1. Scale factor 적용 (기본값 0.18215)
    2. Deterministic Encoding - z = mean 사용 (v5.3, 구조 정확도 향상)
    3. Decode 전 rescaling 적용
    4. GPU Augmentation 조건문 수정 (v5.1)
    5. 모든 3개 슬라이스에 보조 손실 함수(L1, GDL, LPIPS) 적용 (v5.2)
    6. Image-space 보조 loss에 mid_weight 적용 (v5.4)
       - MSE(노이즈 예측)는 균등 유지: z-연속성 학습에 기여
       - Image-space loss에만 mid_weight 적용: 최종 출력 품질 최적화
    """
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
    
    # Mid-slice weight for image-space loss (not for MSE)
    mid_weight = config['diffusion'].get('mid_slice_weight', 0.5)
    
    # Scale Factor
    scale_factor = config['model'].get('scale_factor', 0.18215)
    
    # Deterministic vs Stochastic Encoding
    use_stochastic_encoding = config['diffusion'].get('use_stochastic_encoding', False)
    encoding_type = "stochastic (sample)" if use_stochastic_encoding else "deterministic (mean)"
    
    # Log only on first epoch
    if epoch == 1:
        logger.info(f"Diffusion training config:")
        logger.info(f"  - Latent scale factor: {scale_factor}")
        logger.info(f"  - Encoding type: {encoding_type}")
        logger.info(f"  - Mid-slice weight (image-space only): {mid_weight}")
        logger.info(f"  - MSE loss: uniform weighting (for z-continuity)")
        if not use_stochastic_encoding:
            logger.info(f"    (Using z=mean for better structural accuracy)")
    
    # MSE threshold for warnings
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
        
        # Get data
        condition = batch['condition']  # [B, 3, H, W] - 3-channel panorama
        ct_volume = batch['ct_volume']  # [B, 1, D=3, H, W]
        slice_position = batch.get('slice_position', None)
        
        # Ensure correct shape
        if ct_volume.dim() == 4:
            ct_volume = ct_volume.unsqueeze(1)
        
        # GPU Augmentation
        if gpu_augmenter is not None:
            ct_volume_2d = ct_volume.squeeze(1)  # [B, 3, H, W]
            ct_volume_2d, condition = gpu_augmenter(ct_volume_2d, condition, epoch=epoch)
            ct_volume = ct_volume_2d.unsqueeze(1)  # [B, 1, 3, H, W]
        
        if torch.isnan(condition).any() or torch.isinf(condition).any() or \
           torch.isnan(ct_volume).any() or torch.isinf(ct_volume).any():
            logger.warning(f"NaN/Inf detected in input at batch {i}, skipping")
            nan_count += 1
            continue
        
        # Encoding
        with torch.no_grad():
            mean, logvar = vae.encoder(ct_volume)
            
            if use_stochastic_encoding:
                z = vae.sample(mean, logvar)
            else:
                z = mean  # Deterministic for better structural accuracy
            
            z = z * scale_factor
            
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
                
                # MSE loss: UNIFORM weighting (not mid_weight)
                mse_loss = F.mse_loss(model_output, target)
                
                # Warning for high MSE
                if epoch > ignore_epochs and mse_loss.item() > high_mse_threshold:
                    high_loss_count += 1
                    if high_loss_count == 1:
                        debug_dir = os.path.join(config['save_dir'], 'debug_high_loss')
                        os.makedirs(debug_dir, exist_ok=True)
                        torch.save({
                            'condition': condition.cpu(),
                            'ct_volume': ct_volume.cpu(),
                            'model_output': model_output.cpu(),
                            'target': target.cpu(),
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
                
                # Decode
                x0_pred_rescaled = x0_pred / scale_factor
                decoded_pred = vae.decode(x0_pred_rescaled)
                decoded_pred = torch.clamp(decoded_pred, -1, 1)
                
                if torch.isnan(decoded_pred).any() or torch.isinf(decoded_pred).any():
                    logger.warning(f"NaN/Inf in decoded prediction at batch {i}")
                    nan_count += 1
                    optimizer.zero_grad()
                    continue
                
                # Image-space 보조 loss에 mid_weight 적용
                
                # L1 Loss with mid_weight
                if l1_weight > 0:
                    l1_loss = compute_slicewise_weighted_loss(
                        decoded_pred, ct_volume, l1_loss_fn, mid_weight=mid_weight
                    )
                else:
                    l1_loss = torch.tensor(0.0, device=device)
                
                # GDL Loss with mid_weight
                if gdl_weight > 0:
                    gdl_loss = compute_slicewise_weighted_loss(
                        decoded_pred, ct_volume, gdl_loss_fn, mid_weight=mid_weight
                    )
                else:
                    gdl_loss = torch.tensor(0.0, device=device)
                
                # LPIPS Loss with mid_weight
                if lpips_weight > 0:
                    lpips_loss = compute_slicewise_weighted_loss(
                        decoded_pred, ct_volume, lpips_loss_fn, mid_weight=mid_weight
                    )
                else:
                    lpips_loss = torch.tensor(0.0, device=device)
                
                loss = mse_loss + l1_weight * l1_loss + gdl_weight * gdl_loss + lpips_weight * lpips_loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf in loss at batch {i}")
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
                    optimizer.zero_grad()
                    skipped_steps += 1
                    continue
                
                optimizer.step()
                
                if hasattr(scheduler, 'step') and hasattr(scheduler, 'warmup_steps'):
                    scheduler.step()
            
            optimizer.zero_grad()
            
            if ema_wrapper is not None:
                ema_wrapper.update()
        
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
    
    if high_loss_count > 0 and epoch > ignore_epochs:
        high_loss_rate = high_loss_count / len(dataloader) * 100
        logger.info(f"High MSE loss (>{high_mse_threshold}) in {high_loss_count} batches ({high_loss_rate:.2f}%)")
    
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
                            encoding_type=encoding_type,
                            mid_weight=mid_weight,
                            lr=optimizer.param_groups[0]['lr'],
                            overflow_count=gradient_overflow_count,
                            nan_count=nan_count,
                            high_loss_count=high_loss_count,
                            scale_factor=scale_factor)
    
    return avg_loss


def train_consistency_3d(consistency_net, vae, diffusion_model, diffusion_process, 
                        dataloader, optimizer, device, epoch, config, num_slices, metrics_tracker,
                        teacher_forcing_ratio=0.5, ema_wrapper=None):
    """
    Train 3D consistency network with teacher forcing
    
    For 3D slab approach, this refines the generated 3-slice windows
    to ensure z-axis consistency
    """
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
    
    # Scale factor for consistency with diffusion training
    scale_factor = config['model'].get('scale_factor', 0.18215)
    
    num_batches = min(10, len(dataloader))
    
    prefetcher = DataPrefetcher(iter(dataloader), device)
    
    pbar = tqdm(range(num_batches), desc=f'3D Consistency Epoch {epoch}/{config["training"]["consistency_epochs"]}')
    
    for batch_idx in pbar:
        try:
            batch = next(prefetcher)
        except StopIteration:
            break
            
        condition = batch['condition']  # [B, 3, H, W]
        ct_volume = batch['ct_volume']  # [B, 1, D=3, H, W]
        slice_position = batch.get('slice_position', None)
        
        if ct_volume.dim() == 4:
            ct_volume = ct_volume.unsqueeze(1)
        
        # For consistency training, we generate the 3-slice window and refine it
        with torch.no_grad():
            # Use teacher forcing or generate from diffusion
            if np.random.rand() < teacher_forcing_ratio:
                # Use ground truth
                generated_volume = ct_volume
            else:
                # Generate from diffusion
                latent_shape = vae.get_latent_shape((3, 200, 200))
                z_shape = (ct_volume.shape[0], vae.z_channels, *latent_shape)
                
                z_gen = diffusion_process.p_sample_loop(
                    eval_model, z_shape, condition, device,
                    slice_pos=slice_position, show_progress=False
                )
                
                # Rescale before decoding
                z_gen_rescaled = z_gen / scale_factor
                
                generated_volume = vae.decode(z_gen_rescaled)
                generated_volume = torch.clamp(generated_volume, -1, 1)
        
        optimizer.zero_grad()
        
        # Refine the generated volume
        # consistency_net expects [B, 1, D, H, W]
        refined_volume = consistency_net(generated_volume)
        
        # Loss against ground truth
        loss = F.mse_loss(refined_volume, ct_volume)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(consistency_net.parameters(), grad_clip_value)
        optimizer.step()
        
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    metrics_tracker.update('consistency', train_loss=avg_loss)
    metrics_tracker.log_epoch(epoch, 'consistency_train', loss=avg_loss)
    
    return avg_loss