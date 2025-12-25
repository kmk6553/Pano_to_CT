"""
Model evaluation and visualization functions
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging

logger = logging.getLogger(__name__)


def compute_metrics(pred, target):
    """Compute evaluation metrics"""
    pred_np = ((pred + 1) / 2).cpu().numpy()
    target_np = ((target + 1) / 2).cpu().numpy()
    
    pred_np = pred_np.squeeze()
    target_np = target_np.squeeze()
    
    if pred_np.ndim == 4:
        pred_np = pred_np[0]
        target_np = target_np[0]
    if pred_np.ndim == 3:
        pred_np = pred_np[0]
        target_np = target_np[0]
    
    psnr_val = psnr(target_np, pred_np, data_range=1.0)
    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    
    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'mae': F.l1_loss(pred, target).item(),
        'mse': F.mse_loss(pred, target).item()
    }


def evaluate_and_visualize(vae, diffusion_model, diffusion_process, val_loader, 
                          device, epoch, save_dir, phase_name, metrics_tracker, ema_wrapper=None,
                          guidance_scale=1.5, fixed_slice_indices=[20,40,60,80,100]):
    """Comprehensive evaluation and visualization with CFG and fixed slice indices"""
    vae.eval()
    
    if ema_wrapper is not None and diffusion_model is not None:
        eval_model = ema_wrapper.get_model()
        eval_model.eval()
    elif diffusion_model is not None:
        eval_model = diffusion_model
        eval_model.eval()
    else:
        eval_model = None
    
    epoch_save_dir = os.path.join(save_dir, f'{phase_name}_epoch_{epoch}')
    os.makedirs(epoch_save_dir, exist_ok=True)
    
    metrics_list = []
    
    # Use fixed slice indices if provided, otherwise use loader sequentially
    if fixed_slice_indices is not None:
        # Get dataset from loader
        dataset = val_loader.dataset
        
        # Validate indices
        valid_indices = []
        for idx in fixed_slice_indices:
            if 0 <= idx < len(dataset):
                valid_indices.append(idx)
            else:
                logger.warning(f"Slice index {idx} is out of range (dataset size: {len(dataset)})")
        
        if not valid_indices:
            logger.error("No valid slice indices provided, using loader default")
            use_loader = True
            num_samples = min(5, len(val_loader))
        else:
            use_loader = False
            logger.info(f"Evaluating on fixed slice indices: {valid_indices}")
    else:
        use_loader = True
        num_samples = min(5, len(val_loader))
    
    with torch.no_grad():
        if use_loader:
            # Original behavior: use loader
            for i, batch in enumerate(val_loader):
                if i >= num_samples:
                    break
                
                condition = batch['condition'].to(device)
                ct_slice = batch['ct_slice'].to(device)
                slice_position = batch.get('slice_position', None)
                if slice_position is not None:
                    slice_position = slice_position.to(device)
                
                if phase_name == 'vae':
                    recon, _, _ = vae(ct_slice)
                    metrics = compute_metrics(recon, ct_slice)
                    
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    
                    axes[0].imshow(ct_slice[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
                    axes[0].set_title('Original')
                    axes[0].axis('off')
                    
                    axes[1].imshow(recon[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
                    axes[1].set_title(f'Reconstructed\nPSNR: {metrics["psnr"]:.2f}dB')
                    axes[1].axis('off')
                    
                    diff = torch.abs(ct_slice - recon)
                    axes[2].imshow(diff[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                    axes[2].set_title('Absolute Difference')
                    axes[2].axis('off')
                    
                else:
                    z, _, _ = vae.encode(ct_slice)
                    z_gen = diffusion_process.p_sample_loop(
                        eval_model, 
                        z.shape, 
                        condition, 
                        device,
                        slice_pos=slice_position,
                        guidance_scale=guidance_scale,
                        use_self_conditioning=True
                    )
                    gen_slice = vae.decode(z_gen)
                    
                    metrics = compute_metrics(gen_slice, ct_slice)
                    
                    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                    
                    if condition.shape[1] == 3:
                        cond_to_show = condition[0, 1].cpu().numpy()
                    else:
                        cond_to_show = condition[0, 0].cpu().numpy()
                    
                    axes[0, 0].imshow(cond_to_show, cmap='gray', vmin=-1, vmax=1)
                    axes[0, 0].set_title('Panorama (Input)')
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(gen_slice[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
                    title_text = f'Generated\nPSNR: {metrics["psnr"]:.2f}dB'
                    if slice_position is not None:
                        title_text += f'\nPos: {slice_position[0].item():.3f}'
                    axes[0, 1].set_title(title_text)
                    axes[0, 1].axis('off')
                    
                    axes[0, 2].imshow(ct_slice[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
                    axes[0, 2].set_title('Ground Truth')
                    axes[0, 2].axis('off')
                    
                    diff = torch.abs(ct_slice - gen_slice)
                    axes[1, 0].imshow(diff[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                    axes[1, 0].set_title('Absolute Difference')
                    axes[1, 0].axis('off')
                    
                    axes[1, 1].hist(gen_slice[0, 0].cpu().numpy().flatten(), bins=50, alpha=0.5, label='Generated')
                    axes[1, 1].hist(ct_slice[0, 0].cpu().numpy().flatten(), bins=50, alpha=0.5, label='GT')
                    axes[1, 1].set_title('Intensity Distribution')
                    axes[1, 1].legend()
                    
                    metrics_text = f"PSNR: {metrics['psnr']:.2f} dB\n"
                    metrics_text += f"SSIM: {metrics['ssim']:.3f}\n"
                    metrics_text += f"MAE: {metrics['mae']:.4f}\n"
                    metrics_text += f"MSE: {metrics['mse']:.4f}\n"
                    if guidance_scale != 1.0:
                        metrics_text += f"\nCFG Scale: {guidance_scale}"
                    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, transform=axes[1, 2].transAxes)
                    axes[1, 2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(epoch_save_dir, f'val_sample_{i}.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                metrics_list.append(metrics)
        else:
            # Fixed slice indices: directly access dataset
            for i, idx in enumerate(valid_indices):
                try:
                    sample = dataset[idx]
                    
                    # Add batch dimension and move to device
                    condition = sample['condition'].unsqueeze(0).to(device)
                    ct_slice = sample['ct_slice'].unsqueeze(0).to(device)
                    slice_position = sample.get('slice_position', None)
                    if slice_position is not None:
                        slice_position = slice_position.unsqueeze(0).to(device)
                    
                    folder = sample.get('folder', 'unknown')
                    slice_idx = sample.get('slice_idx', 0)
                    
                    if phase_name == 'vae':
                        recon, _, _ = vae(ct_slice)
                        metrics = compute_metrics(recon, ct_slice)
                        
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        
                        axes[0].imshow(ct_slice[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
                        axes[0].set_title('Original')
                        axes[0].axis('off')
                        
                        axes[1].imshow(recon[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
                        axes[1].set_title(f'Reconstructed\nPSNR: {metrics["psnr"]:.2f}dB')
                        axes[1].axis('off')
                        
                        diff = torch.abs(ct_slice - recon)
                        axes[2].imshow(diff[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                        axes[2].set_title('Absolute Difference')
                        axes[2].axis('off')
                        
                    else:
                        z, _, _ = vae.encode(ct_slice)
                        z_gen = diffusion_process.p_sample_loop(
                            eval_model, 
                            z.shape, 
                            condition, 
                            device,
                            slice_pos=slice_position,
                            guidance_scale=guidance_scale,
                            use_self_conditioning=True
                        )
                        gen_slice = vae.decode(z_gen)
                        
                        metrics = compute_metrics(gen_slice, ct_slice)
                        
                        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                        
                        if condition.shape[1] == 3:
                            cond_to_show = condition[0, 1].cpu().numpy()
                        else:
                            cond_to_show = condition[0, 0].cpu().numpy()
                        
                        axes[0, 0].imshow(cond_to_show, cmap='gray', vmin=-1, vmax=1)
                        axes[0, 0].set_title('Panorama (Input)')
                        axes[0, 0].axis('off')
                        
                        axes[0, 1].imshow(gen_slice[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
                        title_text = f'Generated\nPSNR: {metrics["psnr"]:.2f}dB'
                        if slice_position is not None:
                            title_text += f'\nPos: {slice_position[0].item():.3f}'
                        title_text += f'\nSlice: {slice_idx}'
                        axes[0, 1].set_title(title_text)
                        axes[0, 1].axis('off')
                        
                        axes[0, 2].imshow(ct_slice[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
                        axes[0, 2].set_title('Ground Truth')
                        axes[0, 2].axis('off')
                        
                        diff = torch.abs(ct_slice - gen_slice)
                        axes[1, 0].imshow(diff[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                        axes[1, 0].set_title('Absolute Difference')
                        axes[1, 0].axis('off')
                        
                        axes[1, 1].hist(gen_slice[0, 0].cpu().numpy().flatten(), bins=50, alpha=0.5, label='Generated')
                        axes[1, 1].hist(ct_slice[0, 0].cpu().numpy().flatten(), bins=50, alpha=0.5, label='GT')
                        axes[1, 1].set_title('Intensity Distribution')
                        axes[1, 1].legend()
                        
                        metrics_text = f"Folder: {folder}\n"
                        metrics_text += f"Slice: {slice_idx}\n"
                        metrics_text += f"PSNR: {metrics['psnr']:.2f} dB\n"
                        metrics_text += f"SSIM: {metrics['ssim']:.3f}\n"
                        metrics_text += f"MAE: {metrics['mae']:.4f}\n"
                        metrics_text += f"MSE: {metrics['mse']:.4f}\n"
                        if guidance_scale != 1.0:
                            metrics_text += f"\nCFG Scale: {guidance_scale}"
                        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, transform=axes[1, 2].transAxes)
                        axes[1, 2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(epoch_save_dir, f'val_sample_idx{idx}_slice{slice_idx}.png'), 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    metrics_list.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Error evaluating slice index {idx}: {e}")
                    continue
    
    torch.cuda.empty_cache()
    
    if metrics_list:
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
        
        metrics_tracker.update('val', **avg_metrics)
        
        logger.info(f"\n{phase_name.upper()} - VALIDATION Set Metrics:")
        logger.info(f"PSNR: {avg_metrics['psnr']:.2f} dB")
        logger.info(f"SSIM: {avg_metrics['ssim']:.3f}")
        logger.info(f"MAE: {avg_metrics['mae']:.4f}")
        logger.info(f"MSE: {avg_metrics['mse']:.4f}")
        if phase_name != 'vae' and guidance_scale != 1.0:
            logger.info(f"CFG Scale: {guidance_scale}")
