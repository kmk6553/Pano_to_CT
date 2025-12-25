"""
Model evaluation and visualization functions for 3D Slab-based generation
Evaluates 3-slice windows: [B, 1, D=3, H, W]
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
    """
    Compute evaluation metrics for single 2D slice
    
    Args:
        pred: [B, 1, H, W] or [H, W] - predicted slice
        target: [B, 1, H, W] or [H, W] - target slice
    Returns:
        dict with psnr, ssim, mae, mse
    """
    # Convert to numpy and normalize to [0, 1]
    if isinstance(pred, torch.Tensor):
        pred_np = ((pred + 1) / 2).cpu().numpy()
        target_np = ((target + 1) / 2).cpu().numpy()
    else:
        pred_np = (pred + 1) / 2
        target_np = (target + 1) / 2
    
    # Squeeze to 2D
    pred_np = pred_np.squeeze()
    target_np = target_np.squeeze()
    
    # Handle batch dimension
    if pred_np.ndim == 3:
        pred_np = pred_np[0]
        target_np = target_np[0]
    
    # Clip to valid range
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    psnr_val = psnr(target_np, pred_np, data_range=1.0)
    ssim_val = ssim(target_np, pred_np, data_range=1.0)
    
    # Compute MAE and MSE on original tensors
    if isinstance(pred, torch.Tensor):
        mae = F.l1_loss(pred, target).item()
        mse = F.mse_loss(pred, target).item()
    else:
        mae = np.mean(np.abs(pred_np - target_np))
        mse = np.mean((pred_np - target_np) ** 2)
    
    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'mae': mae,
        'mse': mse
    }


def compute_metrics_3d(pred_volume, target_volume):
    """
    Compute metrics for 3D volume (all 3 slices + middle slice separately)
    
    Args:
        pred_volume: [B, 1, D=3, H, W] - predicted volume
        target_volume: [B, 1, D=3, H, W] - target volume
    Returns:
        dict with metrics for each slice and average
    """
    metrics = {}
    
    # Metrics for each slice
    slice_names = ['prev', 'mid', 'next']
    all_psnr = []
    all_ssim = []
    all_mae = []
    all_mse = []
    
    for i, name in enumerate(slice_names):
        pred_slice = pred_volume[:, :, i, :, :]  # [B, 1, H, W]
        target_slice = target_volume[:, :, i, :, :]
        
        slice_metrics = compute_metrics(pred_slice, target_slice)
        metrics[f'{name}_psnr'] = slice_metrics['psnr']
        metrics[f'{name}_ssim'] = slice_metrics['ssim']
        metrics[f'{name}_mae'] = slice_metrics['mae']
        metrics[f'{name}_mse'] = slice_metrics['mse']
        
        all_psnr.append(slice_metrics['psnr'])
        all_ssim.append(slice_metrics['ssim'])
        all_mae.append(slice_metrics['mae'])
        all_mse.append(slice_metrics['mse'])
    
    # Average metrics
    metrics['psnr'] = np.mean(all_psnr)
    metrics['ssim'] = np.mean(all_ssim)
    metrics['mae'] = np.mean(all_mae)
    metrics['mse'] = np.mean(all_mse)
    
    # Z-consistency metric: measure smoothness across slices
    # Lower is better - measures difference between adjacent slices
    pred_np = pred_volume.cpu().numpy()
    z_diff_01 = np.mean(np.abs(pred_np[:, :, 0, :, :] - pred_np[:, :, 1, :, :]))
    z_diff_12 = np.mean(np.abs(pred_np[:, :, 1, :, :] - pred_np[:, :, 2, :, :]))
    metrics['z_consistency'] = (z_diff_01 + z_diff_12) / 2
    
    return metrics


def visualize_3slice_comparison(condition, pred_volume, target_volume, save_path, 
                                metrics=None, slice_position=None, folder=None, slice_idx=None):
    """
    Visualize comparison of 3-slice window
    
    Args:
        condition: [3, H, W] - 3-channel panorama condition
        pred_volume: [1, D=3, H, W] - predicted volume (no batch dim)
        target_volume: [1, D=3, H, W] - target volume (no batch dim)
        save_path: Path to save figure
        metrics: Optional dict of metrics
        slice_position: Optional slice position value
        folder: Optional folder name
        slice_idx: Optional slice index
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    slice_names = ['Prev', 'Mid', 'Next']
    
    for i, name in enumerate(slice_names):
        # Column 0: Condition (panorama)
        if condition.shape[0] == 3:
            cond_slice = condition[i].cpu().numpy() if isinstance(condition, torch.Tensor) else condition[i]
        else:
            cond_slice = condition[0].cpu().numpy() if isinstance(condition, torch.Tensor) else condition[0]
        axes[i, 0].imshow(cond_slice, cmap='gray', vmin=-1, vmax=1)
        axes[i, 0].set_title(f'Panorama ({name})')
        axes[i, 0].axis('off')
        
        # Column 1: Predicted slice
        pred_slice = pred_volume[0, i].cpu().numpy() if isinstance(pred_volume, torch.Tensor) else pred_volume[0, i]
        axes[i, 1].imshow(pred_slice, cmap='gray', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Generated ({name})')
        axes[i, 1].axis('off')
        
        # Column 2: Target slice
        target_slice = target_volume[0, i].cpu().numpy() if isinstance(target_volume, torch.Tensor) else target_volume[0, i]
        axes[i, 2].imshow(target_slice, cmap='gray', vmin=-1, vmax=1)
        axes[i, 2].set_title(f'Ground Truth ({name})')
        axes[i, 2].axis('off')
        
        # Column 3: Difference map
        diff = np.abs(pred_slice - target_slice)
        axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[i, 3].set_title(f'|Difference| ({name})')
        axes[i, 3].axis('off')
        
        # Column 4: Histogram comparison
        axes[i, 4].hist(pred_slice.flatten(), bins=50, alpha=0.5, label='Generated', density=True)
        axes[i, 4].hist(target_slice.flatten(), bins=50, alpha=0.5, label='GT', density=True)
        axes[i, 4].set_title(f'Intensity Dist ({name})')
        axes[i, 4].legend(fontsize=8)
    
    # Add metrics as text
    if metrics is not None:
        metrics_text = "=== Metrics ===\n"
        if folder is not None:
            metrics_text += f"Folder: {folder}\n"
        if slice_idx is not None:
            metrics_text += f"Slice: {slice_idx}\n"
        if slice_position is not None:
            pos_val = slice_position.item() if isinstance(slice_position, torch.Tensor) else slice_position
            metrics_text += f"Position: {pos_val:.3f}\n"
        metrics_text += "\n"
        metrics_text += f"Avg PSNR: {metrics.get('psnr', 0):.2f} dB\n"
        metrics_text += f"Avg SSIM: {metrics.get('ssim', 0):.3f}\n"
        metrics_text += f"Avg MAE: {metrics.get('mae', 0):.4f}\n"
        metrics_text += f"Z-Consistency: {metrics.get('z_consistency', 0):.4f}\n"
        metrics_text += "\n--- Per Slice ---\n"
        metrics_text += f"Mid PSNR: {metrics.get('mid_psnr', 0):.2f} dB\n"
        metrics_text += f"Mid SSIM: {metrics.get('mid_ssim', 0):.3f}\n"
        
        fig.text(0.02, 0.02, metrics_text, fontsize=10, fontfamily='monospace',
                verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_vae_3d(recon_volume, target_volume, save_path, metrics=None):
    """
    Visualize VAE reconstruction for 3-slice window
    
    Args:
        recon_volume: [1, D=3, H, W] - reconstructed volume
        target_volume: [1, D=3, H, W] - target volume
        save_path: Path to save figure
        metrics: Optional dict of metrics
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    slice_names = ['Prev', 'Mid', 'Next']
    
    for i, name in enumerate(slice_names):
        # Original
        target_slice = target_volume[0, i].cpu().numpy() if isinstance(target_volume, torch.Tensor) else target_volume[0, i]
        axes[i, 0].imshow(target_slice, cmap='gray', vmin=-1, vmax=1)
        axes[i, 0].set_title(f'Original ({name})')
        axes[i, 0].axis('off')
        
        # Reconstructed
        recon_slice = recon_volume[0, i].cpu().numpy() if isinstance(recon_volume, torch.Tensor) else recon_volume[0, i]
        axes[i, 1].imshow(recon_slice, cmap='gray', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Reconstructed ({name})')
        axes[i, 1].axis('off')
        
        # Difference
        diff = np.abs(target_slice - recon_slice)
        axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].set_title(f'|Difference| ({name})')
        axes[i, 2].axis('off')
        
        # Per-slice metrics
        slice_psnr = metrics.get(f'{name.lower()}_psnr', 0) if metrics else 0
        slice_ssim = metrics.get(f'{name.lower()}_ssim', 0) if metrics else 0
        info_text = f"PSNR: {slice_psnr:.2f} dB\nSSIM: {slice_ssim:.3f}"
        axes[i, 3].text(0.5, 0.5, info_text, fontsize=14, transform=axes[i, 3].transAxes,
                       verticalalignment='center', horizontalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        axes[i, 3].axis('off')
    
    # Overall metrics
    if metrics is not None:
        fig.suptitle(f"VAE Reconstruction - Avg PSNR: {metrics.get('psnr', 0):.2f} dB, "
                    f"Avg SSIM: {metrics.get('ssim', 0):.3f}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_and_visualize(vae, diffusion_model, diffusion_process, val_loader, 
                          device, epoch, save_dir, phase_name, metrics_tracker, ema_wrapper=None,
                          guidance_scale=1.5, fixed_slice_indices=[20, 40, 60, 80, 100]):
    """
    Comprehensive evaluation and visualization for 3D Slab models
    
    Args:
        vae: 3D VAE model
        diffusion_model: 3D Diffusion UNet model
        diffusion_process: Diffusion process
        val_loader: Validation dataloader
        device: Device
        epoch: Current epoch
        save_dir: Directory to save results
        phase_name: 'vae' or 'diffusion'
        metrics_tracker: Metrics tracker object
        ema_wrapper: Optional EMA wrapper for diffusion model
        guidance_scale: CFG scale for diffusion
        fixed_slice_indices: Specific indices to evaluate
    """
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
    
    # Determine which indices to use
    if fixed_slice_indices is not None:
        dataset = val_loader.dataset
        valid_indices = [idx for idx in fixed_slice_indices if 0 <= idx < len(dataset)]
        
        if not valid_indices:
            logger.warning("No valid slice indices provided, using loader default")
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
            # Use dataloader
            for i, batch in enumerate(val_loader):
                if i >= num_samples:
                    break
                
                condition = batch['condition'].to(device)  # [B, 3, H, W]
                ct_volume = batch['ct_volume'].to(device)  # [B, 1, D=3, H, W]
                slice_position = batch.get('slice_position', None)
                if slice_position is not None:
                    slice_position = slice_position.to(device)
                
                # Ensure correct shape
                if ct_volume.dim() == 4:
                    ct_volume = ct_volume.unsqueeze(1)
                
                if phase_name == 'vae':
                    # VAE evaluation
                    recon, _, _ = vae(ct_volume)
                    metrics = compute_metrics_3d(recon, ct_volume)
                    
                    # Visualize
                    save_path = os.path.join(epoch_save_dir, f'val_sample_{i}.png')
                    visualize_vae_3d(recon[0], ct_volume[0], save_path, metrics)
                    
                else:
                    # Diffusion evaluation
                    # Get latent shape
                    z, _, _ = vae.encode(ct_volume)
                    latent_shape = z.shape
                    
                    # Generate from diffusion
                    z_gen = diffusion_process.p_sample_loop(
                        eval_model,
                        latent_shape,
                        condition,
                        device,
                        slice_pos=slice_position,
                        guidance_scale=guidance_scale,
                        use_self_conditioning=True,
                        show_progress=False
                    )
                    
                    # Decode
                    gen_volume = vae.decode(z_gen)
                    gen_volume = torch.clamp(gen_volume, -1, 1)
                    
                    metrics = compute_metrics_3d(gen_volume, ct_volume)
                    
                    # Visualize
                    save_path = os.path.join(epoch_save_dir, f'val_sample_{i}.png')
                    visualize_3slice_comparison(
                        condition[0], gen_volume[0], ct_volume[0], save_path,
                        metrics=metrics, slice_position=slice_position[0] if slice_position is not None else None
                    )
                
                metrics_list.append(metrics)
                
        else:
            # Use fixed indices
            dataset = val_loader.dataset
            
            for i, idx in enumerate(valid_indices):
                try:
                    sample = dataset[idx]
                    
                    # Add batch dimension and move to device
                    condition = sample['condition'].unsqueeze(0).to(device)  # [1, 3, H, W]
                    ct_volume = sample['ct_volume'].unsqueeze(0).to(device)  # [1, 1, D=3, H, W]
                    slice_position = sample.get('slice_position', None)
                    if slice_position is not None:
                        slice_position = slice_position.unsqueeze(0).to(device)
                    
                    folder = sample.get('folder', 'unknown')
                    slice_idx = sample.get('slice_idx', 0)
                    
                    # Ensure correct shape
                    if ct_volume.dim() == 4:
                        ct_volume = ct_volume.unsqueeze(1)
                    
                    if phase_name == 'vae':
                        # VAE evaluation
                        recon, _, _ = vae(ct_volume)
                        metrics = compute_metrics_3d(recon, ct_volume)
                        
                        # Visualize
                        save_path = os.path.join(epoch_save_dir, f'val_idx{idx}_slice{slice_idx}.png')
                        visualize_vae_3d(recon[0], ct_volume[0], save_path, metrics)
                        
                    else:
                        # Diffusion evaluation
                        z, _, _ = vae.encode(ct_volume)
                        latent_shape = z.shape
                        
                        z_gen = diffusion_process.p_sample_loop(
                            eval_model,
                            latent_shape,
                            condition,
                            device,
                            slice_pos=slice_position,
                            guidance_scale=guidance_scale,
                            use_self_conditioning=True,
                            show_progress=False
                        )
                        
                        gen_volume = vae.decode(z_gen)
                        gen_volume = torch.clamp(gen_volume, -1, 1)
                        
                        metrics = compute_metrics_3d(gen_volume, ct_volume)
                        
                        # Visualize
                        save_path = os.path.join(epoch_save_dir, f'val_idx{idx}_slice{slice_idx}.png')
                        visualize_3slice_comparison(
                            condition[0], gen_volume[0], ct_volume[0], save_path,
                            metrics=metrics, 
                            slice_position=slice_position[0] if slice_position is not None else None,
                            folder=folder, slice_idx=slice_idx
                        )
                    
                    metrics_list.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Error evaluating slice index {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    torch.cuda.empty_cache()
    
    # Aggregate metrics
    if metrics_list:
        avg_metrics = {}
        
        # Get all keys from first metrics dict
        all_keys = metrics_list[0].keys()
        
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                avg_metrics[key] = np.mean(values)
        
        # Update metrics tracker with main metrics
        metrics_tracker.update('val', 
                              psnr=avg_metrics.get('psnr', 0),
                              ssim=avg_metrics.get('ssim', 0),
                              mae=avg_metrics.get('mae', 0),
                              mse=avg_metrics.get('mse', 0))
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"{phase_name.upper()} - VALIDATION Set Metrics (3D Slab)")
        logger.info(f"{'='*60}")
        logger.info(f"Average PSNR: {avg_metrics.get('psnr', 0):.2f} dB")
        logger.info(f"Average SSIM: {avg_metrics.get('ssim', 0):.3f}")
        logger.info(f"Average MAE: {avg_metrics.get('mae', 0):.4f}")
        logger.info(f"Average MSE: {avg_metrics.get('mse', 0):.4f}")
        logger.info(f"Z-Consistency: {avg_metrics.get('z_consistency', 0):.4f}")
        logger.info(f"\n--- Per-Slice Metrics ---")
        logger.info(f"Prev PSNR: {avg_metrics.get('prev_psnr', 0):.2f} dB, SSIM: {avg_metrics.get('prev_ssim', 0):.3f}")
        logger.info(f"Mid  PSNR: {avg_metrics.get('mid_psnr', 0):.2f} dB, SSIM: {avg_metrics.get('mid_ssim', 0):.3f}")
        logger.info(f"Next PSNR: {avg_metrics.get('next_psnr', 0):.2f} dB, SSIM: {avg_metrics.get('next_ssim', 0):.3f}")
        if phase_name != 'vae' and guidance_scale != 1.0:
            logger.info(f"CFG Scale: {guidance_scale}")
        logger.info(f"{'='*60}\n")
        
        # Save metrics to file
        metrics_file = os.path.join(epoch_save_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"{phase_name.upper()} Validation Metrics - Epoch {epoch}\n")
            f.write("="*50 + "\n\n")
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value:.4f}\n")


def evaluate_sliding_window(vae, diffusion_model, diffusion_process, pano_volume, 
                           device, slice_range=(0, 120), guidance_scale=1.5,
                           ema_wrapper=None, save_dir=None):
    """
    Evaluate full volume generation using sliding window approach
    
    This generates each slice using a 3-slice window and extracts the middle slice,
    demonstrating the inference pipeline for full volume reconstruction.
    
    Args:
        vae: 3D VAE model
        diffusion_model: 3D Diffusion UNet
        diffusion_process: Diffusion process
        pano_volume: [D, H, W] panorama volume
        device: Device
        slice_range: (start, end) slice range
        guidance_scale: CFG scale
        ema_wrapper: Optional EMA wrapper
        save_dir: Optional directory to save results
    
    Returns:
        generated_volume: [D, H, W] generated CT volume (middle slices only)
    """
    vae.eval()
    
    if ema_wrapper is not None:
        eval_model = ema_wrapper.get_model()
        eval_model.eval()
    else:
        eval_model = diffusion_model
        eval_model.eval()
    
    start_idx, end_idx = slice_range
    num_slices = end_idx - start_idx
    
    generated_slices = []
    
    # Get latent shape
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, 3, 200, 200, device=device)
        latent_shape = vae.encoder(dummy_input)[0].shape
    
    logger.info(f"Generating {num_slices} slices using sliding window approach...")
    
    from tqdm import tqdm
    
    for center_idx in tqdm(range(start_idx, end_idx), desc="Generating slices"):
        # Get 3-slice panorama window
        prev_idx = max(start_idx, center_idx - 1)
        next_idx = min(end_idx - 1, center_idx + 1)
        
        pano_window = np.stack([
            pano_volume[prev_idx],
            pano_volume[center_idx],
            pano_volume[next_idx]
        ], axis=0)  # [3, H, W]
        
        # Normalize and convert to tensor
        pano_tensor = torch.from_numpy(pano_window).unsqueeze(0).float().to(device)  # [1, 3, H, W]
        pano_tensor = torch.clamp(pano_tensor, -1, 1)
        
        # Slice position
        slice_position = torch.tensor(
            [(center_idx - start_idx) / max(1, num_slices - 1)],
            dtype=torch.float32, device=device
        )
        
        with torch.no_grad():
            # Generate 3-slice window
            z_gen = diffusion_process.p_sample_loop(
                eval_model,
                (1, latent_shape[1], latent_shape[2], latent_shape[3], latent_shape[4]),
                pano_tensor,
                device,
                slice_pos=slice_position,
                guidance_scale=guidance_scale,
                use_self_conditioning=True,
                show_progress=False
            )
            
            # Decode
            gen_volume = vae.decode(z_gen)
            gen_volume = torch.clamp(gen_volume, -1, 1)
            
            # Extract middle slice
            mid_slice = gen_volume[0, 0, 1, :, :].cpu().numpy()  # [H, W]
            generated_slices.append(mid_slice)
    
    # Stack all middle slices
    generated_volume = np.stack(generated_slices, axis=0)  # [D, H, W]
    
    logger.info(f"Generated volume shape: {generated_volume.shape}")
    logger.info(f"Generated volume range: [{generated_volume.min():.3f}, {generated_volume.max():.3f}]")
    
    # Optionally save
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as raw
        raw_path = os.path.join(save_dir, 'generated_volume.raw')
        generated_volume.astype(np.float32).tofile(raw_path)
        logger.info(f"Saved generated volume to {raw_path}")
        
        # Save visualization
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        # Show sample slices
        sample_indices = np.linspace(0, num_slices - 1, 10, dtype=int)
        
        for i, idx in enumerate(sample_indices):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(generated_volume[idx], cmap='gray', vmin=-1, vmax=1)
            axes[row, col].set_title(f'Slice {idx + start_idx}')
            axes[row, col].axis('off')
        
        plt.suptitle('Generated CT Volume - Sample Slices', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'generated_samples.png'), dpi=150)
        plt.close()
    
    return generated_volume