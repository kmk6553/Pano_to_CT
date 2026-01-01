"""
3D Slab-based Latent Diffusion Model - Inference Script

Generates CT volumes from panoramic X-ray images using sliding window approach.
Each 3-slice window is generated jointly, then middle slices are extracted.

FIXES APPLIED:
1. Scale factor 적용 - decode 전 rescaling
2. [v5.7] 정규화 로직 버그 수정 - dataset.py와 일관성 유지
   - normalize_input 플래그 추가
   - [0, 1] 데이터가 변환 없이 통과되는 버그 수정
3. [v5.7] 데이터 범위 검증 및 경고 메시지 추가

Usage examples:
    # Generate single slice (data is in [0, 1] range)
    python inference_latent.py --checkpoint model.pth --pano_path pano.raw --slice_idx 60 --normalize_input
    
    # Generate full volume (data is already in [-1, 1] range)
    python inference_latent.py --checkpoint model.pth --pano_dir folder/ --output_dir results/
    
    # Generate with DDIM (faster)
    python inference_latent.py --checkpoint model.pth --pano_dir folder/ --use_ddim --ddim_steps 50
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our modules
from models import VAE3D, ConditionalUNet3D, EMAWrapper
from diffusion_process import DiffusionProcess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def validate_data_range(data, name, expected_range=(-1, 1), tolerance=0.1):
    """
    Validate data range and log warnings if unexpected
    
    Args:
        data: numpy array
        name: name for logging
        expected_range: (min, max) expected range
        tolerance: tolerance for range check
    
    Returns:
        True if data is in expected range, False otherwise
    """
    data_min, data_max = data.min(), data.max()
    exp_min, exp_max = expected_range
    
    in_range = (data_min >= exp_min - tolerance) and (data_max <= exp_max + tolerance)
    
    if not in_range:
        logger.warning(f"{name} data range [{data_min:.4f}, {data_max:.4f}] "
                      f"differs from expected [{exp_min}, {exp_max}]")
    
    return in_range


def normalize_to_model_range(data, input_range='auto', target_range=(-1, 1)):
    """
    Normalize data to target range for model input
    
    [v5.7 FIX] dataset.py와 동일한 정규화 정책 적용
    
    Args:
        data: numpy array
        input_range: 'auto', '0_1', '-1_1', or tuple (min, max)
        target_range: output range (default: (-1, 1))
    
    Returns:
        Normalized data in target_range
    """
    data_min, data_max = data.min(), data.max()
    
    # Auto-detect input range
    if input_range == 'auto':
        if data_min >= -0.1 and data_max <= 1.1:
            if data_min >= -0.1 and data_min <= 0.1 and data_max >= 0.9:
                # Likely [0, 1] range
                input_range = '0_1'
                logger.info(f"Auto-detected input range: [0, 1] (actual: [{data_min:.4f}, {data_max:.4f}])")
            elif data_min >= -1.1 and data_min <= -0.5:
                # Likely [-1, 1] range
                input_range = '-1_1'
                logger.info(f"Auto-detected input range: [-1, 1] (actual: [{data_min:.4f}, {data_max:.4f}])")
            else:
                # Ambiguous, assume [0, 1]
                input_range = '0_1'
                logger.warning(f"Ambiguous input range [{data_min:.4f}, {data_max:.4f}], assuming [0, 1]")
        else:
            # Values outside [0, 1] or [-1, 1], need full normalization
            input_range = 'full'
            logger.info(f"Input range [{data_min:.4f}, {data_max:.4f}] requires full normalization")
    
    # Apply normalization based on detected/specified input range
    if input_range == '0_1':
        # [0, 1] -> [-1, 1]: x' = x * 2 - 1
        normalized = data * 2.0 - 1.0
        logger.info("Applied transformation: [0, 1] -> [-1, 1]")
    elif input_range == '-1_1':
        # Already in [-1, 1], no transformation needed
        normalized = data
        logger.info("No transformation needed (already [-1, 1])")
    elif input_range == 'full' or isinstance(input_range, tuple):
        # Full min-max normalization
        if isinstance(input_range, tuple):
            src_min, src_max = input_range
        else:
            src_min, src_max = data_min, data_max
        
        if src_max - src_min < 1e-8:
            logger.warning("Data has near-zero range, returning zeros")
            return np.zeros_like(data)
        
        # Normalize to [0, 1] first, then to target range
        normalized = (data - src_min) / (src_max - src_min)
        normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
        logger.info(f"Applied full normalization: [{src_min:.4f}, {src_max:.4f}] -> {target_range}")
    else:
        raise ValueError(f"Unknown input_range: {input_range}")
    
    # Final clip to ensure we're in target range
    normalized = np.clip(normalized, target_range[0], target_range[1])
    
    return normalized


def load_panorama_volume(pano_path, slice_range=(0, 120), shape=(200, 200), 
                        normalize_input=True, input_range='auto'):
    """
    Load panorama volume from raw file or directory
    
    [v5.7 FIX] 정규화 로직 개선
    
    Args:
        pano_path: Path to .raw file or directory containing slice files
        slice_range: (start, end) slice indices
        shape: (H, W) shape of each slice
        normalize_input: If True, apply normalization to [-1, 1]
        input_range: 'auto', '0_1', '-1_1', or tuple for manual range
    
    Returns:
        pano_volume: [D, H, W] numpy array normalized to [-1, 1]
    """
    start_idx, end_idx = slice_range
    num_slices = end_idx - start_idx
    
    if os.path.isfile(pano_path):
        # Load from raw file
        if pano_path.endswith('.raw'):
            data = np.fromfile(pano_path, dtype=np.float32)
            total_elements = data.size
            
            # Try to infer dimensions
            if total_elements == num_slices * shape[0] * shape[1]:
                pano_volume = data.reshape(num_slices, shape[0], shape[1])
            else:
                # Assume it's stored as (D, H, W) with full slices
                inferred_slices = total_elements // (shape[0] * shape[1])
                pano_volume = data.reshape(inferred_slices, shape[0], shape[1])
                pano_volume = pano_volume[start_idx:end_idx]
        elif pano_path.endswith('.npy'):
            pano_volume = np.load(pano_path)
            if pano_volume.ndim == 3:
                pano_volume = pano_volume[start_idx:end_idx]
        else:
            raise ValueError(f"Unsupported file format: {pano_path}")
    
    elif os.path.isdir(pano_path):
        # Load from directory of slice files
        slices = []
        for i in range(start_idx, end_idx):
            slice_path = os.path.join(pano_path, f'pano_{i:04d}.raw')
            if os.path.exists(slice_path):
                slice_data = np.fromfile(slice_path, dtype=np.float32).reshape(shape)
                slices.append(slice_data)
            else:
                # Try alternative naming
                slice_path = os.path.join(pano_path, f'slice_{i:04d}.raw')
                if os.path.exists(slice_path):
                    slice_data = np.fromfile(slice_path, dtype=np.float32).reshape(shape)
                    slices.append(slice_data)
                else:
                    logger.warning(f"Slice {i} not found, using zeros")
                    slices.append(np.zeros(shape, dtype=np.float32))
        
        pano_volume = np.stack(slices, axis=0)
    else:
        raise FileNotFoundError(f"Panorama path not found: {pano_path}")
    
    # Log raw data range before normalization
    logger.info(f"Raw panorama data range: [{pano_volume.min():.4f}, {pano_volume.max():.4f}]")
    
    # ============================================================
    # [v5.7 FIX] 정규화 로직 개선
    # 기존 버그: [0, 1] 데이터가 변환 없이 통과됨
    # 수정: normalize_input 플래그와 input_range로 명시적 제어
    # ============================================================
    if normalize_input:
        pano_volume = normalize_to_model_range(pano_volume, input_range=input_range)
    else:
        logger.info("Normalization disabled, using data as-is")
        # Still validate the range
        if not validate_data_range(pano_volume, "Panorama"):
            logger.warning("Data is not in [-1, 1] range but normalization is disabled!")
            logger.warning("This may cause poor generation quality.")
    
    logger.info(f"Loaded panorama volume: {pano_volume.shape}")
    logger.info(f"Final value range: [{pano_volume.min():.4f}, {pano_volume.max():.4f}]")
    
    return pano_volume


def load_from_dataset_folder(folder_path, panorama_type='axial', slice_range=(0, 120),
                            normalize_input=True, input_range='auto'):
    """
    Load panorama from standard dataset folder structure
    
    [v5.7 FIX] 정규화 로직 개선
    
    Expected structure:
        folder_path/
            Pano/Pano_Normalized_float32_200x200x120.raw
            CT/CT_Normalized_float32_200x200x120.raw (optional, for comparison)
    
    Args:
        folder_path: Path to dataset folder
        panorama_type: Type of panorama extraction
        slice_range: (start, end) slice indices
        normalize_input: If True, apply normalization to [-1, 1]
        input_range: 'auto', '0_1', '-1_1', or tuple for manual range
    
    Returns:
        pano_volume: [D, H, W] panorama volume
        ct_volume: [D, H, W] CT volume or None
    """
    # Try different path patterns
    pano_paths = [
        os.path.join(folder_path, 'Pano', 'Pano_Normalized_float32_200x200x120.raw'),
        os.path.join(folder_path, 'pano', 'pano_normalized.raw'),
        os.path.join(folder_path, 'panorama', f'{panorama_type}_pano.raw'),
        os.path.join(folder_path, f'{panorama_type}_pano.raw'),
    ]
    
    pano_path = None
    for p in pano_paths:
        if os.path.exists(p):
            pano_path = p
            break
    
    if pano_path is None:
        raise FileNotFoundError(f"Panorama file not found in: {folder_path}")
    
    logger.info(f"Loading panorama from: {pano_path}")
    
    # Load panorama
    pano_data = np.fromfile(pano_path, dtype=np.float32)
    
    # Infer shape (assuming 200x200 slices)
    H, W = 200, 200
    num_total_slices = pano_data.size // (H * W)
    pano_volume = pano_data.reshape(num_total_slices, H, W)
    
    start_idx, end_idx = slice_range
    pano_volume = pano_volume[start_idx:end_idx]
    
    # Log raw data range
    logger.info(f"Raw panorama data range: [{pano_volume.min():.4f}, {pano_volume.max():.4f}]")
    
    # ============================================================
    # [v5.7 FIX] 정규화 로직 개선
    # 기존 버그: np.clip만 적용하여 [0, 1] 데이터가 그대로 유지됨
    # 수정: normalize_input 플래그와 input_range로 명시적 제어
    # ============================================================
    if normalize_input:
        pano_volume = normalize_to_model_range(pano_volume, input_range=input_range)
    else:
        logger.info("Normalization disabled, using data as-is")
        if not validate_data_range(pano_volume, "Panorama"):
            logger.warning("Data is not in [-1, 1] range but normalization is disabled!")
    
    logger.info(f"Final panorama range: [{pano_volume.min():.4f}, {pano_volume.max():.4f}]")
    
    # Try to load CT for comparison
    ct_volume = None
    ct_paths = [
        os.path.join(folder_path, 'CT', 'CT_Normalized_float32_200x200x120.raw'),
        os.path.join(folder_path, 'ct', 'ct_normalized.raw'),
        os.path.join(folder_path, 'ct_volume.raw'),
    ]
    
    for ct_path in ct_paths:
        if os.path.exists(ct_path):
            logger.info(f"Loading CT from: {ct_path}")
            ct_data = np.fromfile(ct_path, dtype=np.float32)
            ct_total = ct_data.size // (H * W)
            ct_volume = ct_data.reshape(ct_total, H, W)[start_idx:end_idx]
            
            logger.info(f"Raw CT data range: [{ct_volume.min():.4f}, {ct_volume.max():.4f}]")
            
            # Apply same normalization to CT
            if normalize_input:
                ct_volume = normalize_to_model_range(ct_volume, input_range=input_range)
            
            logger.info(f"Final CT range: [{ct_volume.min():.4f}, {ct_volume.max():.4f}]")
            break
    
    return pano_volume, ct_volume


def generate_single_window(vae, diffusion_model, diffusion_process, 
                          pano_window, slice_position, device,
                          guidance_scale=1.5, use_ddim=False, ddim_steps=50,
                          use_self_conditioning=True, scale_factor=0.18215):
    """
    Generate a single 3-slice window
    
    Args:
        vae: 3D VAE model
        diffusion_model: 3D Diffusion UNet
        diffusion_process: Diffusion process
        pano_window: [3, H, W] panorama condition (must be in [-1, 1])
        slice_position: Normalized position [0, 1]
        device: Device
        guidance_scale: CFG scale
        use_ddim: Use DDIM sampling
        ddim_steps: Number of DDIM steps
        use_self_conditioning: Enable self-conditioning
        scale_factor: Latent scaling factor (default: 0.18215, Stable Diffusion standard)
    
    Returns:
        generated: [1, D=3, H, W] generated volume
    """
    vae.eval()
    diffusion_model.eval()
    
    # Validate input range
    if pano_window.min() < -1.1 or pano_window.max() > 1.1:
        logger.warning(f"pano_window range [{pano_window.min():.4f}, {pano_window.max():.4f}] "
                      f"is outside expected [-1, 1]. This may cause poor results.")
    
    # Prepare inputs
    pano_tensor = torch.from_numpy(pano_window).unsqueeze(0).float().to(device)  # [1, 3, H, W]
    slice_pos = torch.tensor([slice_position], dtype=torch.float32, device=device)
    
    # Get latent shape from VAE
    with torch.no_grad():
        # Create dummy input to get latent shape
        dummy = torch.zeros(1, 1, 3, pano_window.shape[1], pano_window.shape[2], device=device)
        z_dummy, _, _ = vae.encode(dummy)
        latent_shape = z_dummy.shape
    
    with torch.no_grad():
        if use_ddim:
            # DDIM sampling (faster)
            z_gen = diffusion_process.ddim_sample(
                diffusion_model,
                latent_shape,
                pano_tensor,
                device,
                slice_pos=slice_pos,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                eta=0.0,  # Deterministic
                use_self_conditioning=use_self_conditioning
            )
        else:
            # Full DDPM sampling
            z_gen = diffusion_process.p_sample_loop(
                diffusion_model,
                latent_shape,
                pano_tensor,
                device,
                slice_pos=slice_pos,
                guidance_scale=guidance_scale,
                use_self_conditioning=use_self_conditioning,
                show_progress=False
            )
        
        # ============================================================
        # FIX: Decode 전 Rescaling 적용
        # Diffusion이 생성한 scaled latent를 원래 스케일로 복원
        # ============================================================
        z_gen_rescaled = z_gen / scale_factor
        
        # Decode
        generated = vae.decode(z_gen_rescaled)
        generated = torch.clamp(generated, -1, 1)
    
    return generated


def generate_full_volume(vae, diffusion_model, diffusion_process,
                        pano_volume, device, slice_range=(0, 120),
                        guidance_scale=1.5, use_ddim=False, ddim_steps=50,
                        use_self_conditioning=True, overlap_blend=True,
                        scale_factor=0.18215):
    """
    Generate full CT volume using sliding window approach
    
    Args:
        vae: 3D VAE model
        diffusion_model: 3D Diffusion UNet
        diffusion_process: Diffusion process
        pano_volume: [D, H, W] panorama volume (must be in [-1, 1])
        device: Device
        slice_range: (start, end) indices
        guidance_scale: CFG scale
        use_ddim: Use DDIM sampling
        ddim_steps: DDIM steps
        use_self_conditioning: Enable self-conditioning
        overlap_blend: Blend overlapping regions
        scale_factor: Latent scaling factor
    
    Returns:
        generated_volume: [D, H, W] generated CT volume
    """
    # Validate input range
    pano_min, pano_max = pano_volume.min(), pano_volume.max()
    if pano_min < -1.1 or pano_max > 1.1:
        logger.warning(f"pano_volume range [{pano_min:.4f}, {pano_max:.4f}] "
                      f"is outside expected [-1, 1]. This may cause poor results.")
    elif pano_min > -0.5 and pano_max < 0.5:
        logger.warning(f"pano_volume has narrow range [{pano_min:.4f}, {pano_max:.4f}]. "
                      f"Check if normalization was applied correctly.")
    
    start_idx, end_idx = slice_range
    num_slices = end_idx - start_idx
    H, W = pano_volume.shape[1], pano_volume.shape[2]
    
    # Storage for generated slices
    generated_slices = []
    
    # For overlap blending
    if overlap_blend:
        slice_counts = np.zeros(num_slices)
        accumulated = np.zeros((num_slices, H, W))
    
    logger.info(f"Generating {num_slices} slices using sliding window...")
    logger.info(f"Input panorama range: [{pano_min:.4f}, {pano_max:.4f}]")
    logger.info(f"Guidance scale: {guidance_scale}")
    logger.info(f"Sampling: {'DDIM' if use_ddim else 'DDPM'} ({ddim_steps if use_ddim else 1000} steps)")
    logger.info(f"Scale factor: {scale_factor}")
    
    for center_idx in tqdm(range(num_slices), desc="Generating"):
        # Get 3-slice panorama window with boundary handling
        prev_idx = max(0, center_idx - 1)
        next_idx = min(num_slices - 1, center_idx + 1)
        
        pano_window = np.stack([
            pano_volume[prev_idx],
            pano_volume[center_idx],
            pano_volume[next_idx]
        ], axis=0)  # [3, H, W]
        
        # Slice position (normalized)
        slice_position = center_idx / max(1, num_slices - 1)
        
        # Generate 3-slice window
        generated = generate_single_window(
            vae, diffusion_model, diffusion_process,
            pano_window, slice_position, device,
            guidance_scale=guidance_scale,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            use_self_conditioning=use_self_conditioning,
            scale_factor=scale_factor  # Pass scale_factor
        )
        
        # Extract slices [1, 1, 3, H, W] -> [3, H, W]
        gen_np = generated[0, 0].cpu().numpy()
        
        if overlap_blend:
            # Accumulate all 3 slices with weights
            # Middle slice gets higher weight
            weights = [0.25, 0.5, 0.25]
            
            for i, (offset, weight) in enumerate(zip([-1, 0, 1], weights)):
                target_idx = center_idx + offset
                if 0 <= target_idx < num_slices:
                    accumulated[target_idx] += gen_np[i + 1 if i == -1 else i] * weight  # Adjust index
                    slice_counts[target_idx] += weight
        else:
            # Just use middle slice
            generated_slices.append(gen_np[1])  # Middle slice
    
    if overlap_blend:
        # Average overlapping regions
        slice_counts = np.maximum(slice_counts, 1e-8)[:, None, None]
        generated_volume = accumulated / slice_counts
    else:
        generated_volume = np.stack(generated_slices, axis=0)
    
    # Final clipping
    generated_volume = np.clip(generated_volume, -1, 1)
    
    logger.info(f"Generated volume shape: {generated_volume.shape}")
    logger.info(f"Value range: [{generated_volume.min():.4f}, {generated_volume.max():.4f}]")
    
    return generated_volume


def save_volume(volume, output_path, format='raw'):
    """
    Save generated volume
    
    Args:
        volume: [D, H, W] numpy array
        output_path: Output path (without extension for multi-format)
        format: 'raw', 'npy', 'png', or 'all'
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if format in ['raw', 'all']:
        raw_path = output_path if output_path.endswith('.raw') else output_path + '.raw'
        volume.astype(np.float32).tofile(raw_path)
        logger.info(f"Saved raw volume: {raw_path}")
    
    if format in ['npy', 'all']:
        npy_path = output_path if output_path.endswith('.npy') else output_path + '.npy'
        np.save(npy_path, volume)
        logger.info(f"Saved npy volume: {npy_path}")
    
    if format in ['png', 'all']:
        # Save sample slices as PNG
        png_dir = output_path + '_slices'
        os.makedirs(png_dir, exist_ok=True)
        
        num_slices = volume.shape[0]
        sample_indices = np.linspace(0, num_slices - 1, min(20, num_slices), dtype=int)
        
        for idx in sample_indices:
            slice_img = ((volume[idx] + 1) / 2 * 255).astype(np.uint8)
            plt.imsave(os.path.join(png_dir, f'slice_{idx:04d}.png'), slice_img, cmap='gray')
        
        logger.info(f"Saved {len(sample_indices)} sample slices to: {png_dir}")


def visualize_comparison(generated, ground_truth, pano_volume, output_path, 
                        sample_indices=None):
    """
    Create comparison visualization
    
    Args:
        generated: [D, H, W] generated volume
        ground_truth: [D, H, W] ground truth volume (can be None)
        pano_volume: [D, H, W] panorama volume
        output_path: Path to save visualization
        sample_indices: Specific indices to visualize
    """
    num_slices = generated.shape[0]
    
    if sample_indices is None:
        sample_indices = np.linspace(0, num_slices - 1, 5, dtype=int)
    
    num_samples = len(sample_indices)
    has_gt = ground_truth is not None
    
    num_rows = num_samples
    num_cols = 4 if has_gt else 3
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sample_indices):
        # Panorama
        axes[i, 0].imshow(pano_volume[idx], cmap='gray', vmin=-1, vmax=1)
        axes[i, 0].set_title(f'Panorama (Slice {idx})')
        axes[i, 0].axis('off')
        
        # Generated
        axes[i, 1].imshow(generated[idx], cmap='gray', vmin=-1, vmax=1)
        axes[i, 1].set_title(f'Generated (Slice {idx})')
        axes[i, 1].axis('off')
        
        if has_gt:
            # Ground truth
            axes[i, 2].imshow(ground_truth[idx], cmap='gray', vmin=-1, vmax=1)
            axes[i, 2].set_title(f'Ground Truth (Slice {idx})')
            axes[i, 2].axis('off')
            
            # Difference
            diff = np.abs(generated[idx] - ground_truth[idx])
            axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[i, 3].set_title(f'|Difference| (Slice {idx})')
            axes[i, 3].axis('off')
        else:
            # Histogram
            axes[i, 2].hist(generated[idx].flatten(), bins=50, alpha=0.7, density=True)
            axes[i, 2].set_title(f'Intensity Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison visualization: {output_path}")


def compute_metrics(generated, ground_truth):
    """Compute evaluation metrics"""
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    # Normalize to [0, 1] for metrics
    gen_norm = (generated + 1) / 2
    gt_norm = (ground_truth + 1) / 2
    
    metrics = {
        'psnr': [],
        'ssim': [],
        'mae': [],
        'mse': []
    }
    
    for i in range(generated.shape[0]):
        metrics['psnr'].append(psnr(gt_norm[i], gen_norm[i], data_range=1.0))
        metrics['ssim'].append(ssim(gt_norm[i], gen_norm[i], data_range=1.0))
        metrics['mae'].append(np.mean(np.abs(generated[i] - ground_truth[i])))
        metrics['mse'].append(np.mean((generated[i] - ground_truth[i]) ** 2))
    
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    # Z-consistency
    z_diff = np.mean([np.mean(np.abs(generated[i] - generated[i+1])) 
                     for i in range(generated.shape[0] - 1)])
    avg_metrics['z_consistency'] = z_diff
    
    return avg_metrics


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint.get('config', {})
    
    # Get model configuration
    z_channels = config.get('model', {}).get('z_channels', args.z_channels)
    vae_channels = config.get('model', {}).get('vae_channels', args.vae_channels)
    diffusion_channels = config.get('model', {}).get('diffusion_channels', args.diffusion_channels)
    cond_channels = config.get('model', {}).get('cond_channels', args.cond_channels)
    panorama_type = config.get('data', {}).get('panorama_type', args.panorama_type)
    
    # ============================================================
    # Scale factor 로드 (checkpoint config에서 가져오거나 기본값 사용)
    # ============================================================
    scale_factor = config.get('model', {}).get('scale_factor', args.scale_factor)
    logger.info(f"Using scale factor: {scale_factor}")
    
    # ============================================================
    # [v5.7] Normalization 설정 확인
    # ============================================================
    # 학습 시 사용된 normalize_volumes 설정 확인
    train_normalize = config.get('data', {}).get('normalize_volumes', None)
    if train_normalize is not None:
        logger.info(f"Training used normalize_volumes={train_normalize}")
        if train_normalize and not args.normalize_input:
            logger.warning("Training used normalize_volumes=True but --normalize_input not specified!")
            logger.warning("Consider adding --normalize_input if your data is in [0, 1] range.")
        elif not train_normalize and args.normalize_input:
            logger.warning("Training used normalize_volumes=False but --normalize_input specified!")
            logger.warning("This may cause a mismatch. Check your data range.")
    
    # Diffusion config
    num_timesteps = config.get('diffusion', {}).get('num_timesteps', args.num_timesteps)
    beta_start = config.get('diffusion', {}).get('beta_start', 0.0001)
    beta_end = config.get('diffusion', {}).get('beta_end', 0.02)
    prediction_type = config.get('diffusion', {}).get('prediction_type', args.prediction_type)
    use_self_conditioning = config.get('diffusion', {}).get('use_self_conditioning', args.use_self_conditioning)
    
    logger.info("\n" + "="*60)
    logger.info("Model Configuration")
    logger.info("="*60)
    logger.info(f"VAE channels: {vae_channels}")
    logger.info(f"Latent channels: {z_channels}")
    logger.info(f"Diffusion channels: {diffusion_channels}")
    logger.info(f"Prediction type: {prediction_type}")
    logger.info(f"Self-conditioning: {use_self_conditioning}")
    logger.info(f"Scale factor: {scale_factor}")
    logger.info("="*60)
    logger.info("Normalization Settings")
    logger.info("="*60)
    logger.info(f"normalize_input: {args.normalize_input}")
    logger.info(f"input_range: {args.input_range}")
    logger.info("="*60 + "\n")
    
    # Create models
    logger.info("Initializing models...")
    
    vae = VAE3D(
        in_channels=1,
        z_channels=z_channels,
        channels=vae_channels
    ).to(device)
    
    diffusion_model = ConditionalUNet3D(
        in_channels=z_channels,
        out_channels=z_channels,
        channels=diffusion_channels,
        cond_channels=cond_channels,
        panorama_type=panorama_type,
        pano_triplet=True,
        use_self_conditioning=use_self_conditioning
    ).to(device)
    
    # Load weights
    vae.load_state_dict(checkpoint['vae_state_dict'])
    logger.info("VAE weights loaded")
    
    # Try to load EMA weights first, fall back to regular weights
    if 'ema_state_dict' in checkpoint and args.use_ema:
        diffusion_model.load_state_dict(checkpoint['ema_state_dict'])
        logger.info("Diffusion model weights loaded (EMA)")
    elif 'diffusion_state_dict' in checkpoint:
        diffusion_model.load_state_dict(checkpoint['diffusion_state_dict'])
        logger.info("Diffusion model weights loaded")
    else:
        raise KeyError("No diffusion model weights found in checkpoint")
    
    # Set to eval mode
    vae.eval()
    diffusion_model.eval()
    
    # Create diffusion process
    diffusion_process = DiffusionProcess(
        num_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule='cosine',
        prediction_type=prediction_type
    ).to(device)
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'./inference_results/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input data
    slice_range = tuple(args.slice_range)
    
    if args.data_folder:
        # Load from dataset folder structure
        logger.info(f"Loading from dataset folder: {args.data_folder}")
        pano_volume, ct_volume = load_from_dataset_folder(
            args.data_folder, 
            panorama_type=panorama_type,
            slice_range=slice_range,
            normalize_input=args.normalize_input,
            input_range=args.input_range
        )
    elif args.pano_path:
        # Load from panorama file
        logger.info(f"Loading panorama from: {args.pano_path}")
        pano_volume = load_panorama_volume(
            args.pano_path,
            slice_range=slice_range,
            shape=(args.image_size, args.image_size),
            normalize_input=args.normalize_input,
            input_range=args.input_range
        )
        ct_volume = None
        
        # Optionally load CT for comparison
        if args.ct_path:
            logger.info(f"Loading CT from: {args.ct_path}")
            ct_data = np.fromfile(args.ct_path, dtype=np.float32)
            num_total = ct_data.size // (args.image_size * args.image_size)
            ct_volume = ct_data.reshape(num_total, args.image_size, args.image_size)
            ct_volume = ct_volume[slice_range[0]:slice_range[1]]
            
            # Apply same normalization to CT
            if args.normalize_input:
                ct_volume = normalize_to_model_range(ct_volume, input_range=args.input_range)
            
            logger.info(f"CT volume range: [{ct_volume.min():.4f}, {ct_volume.max():.4f}]")
    else:
        raise ValueError("Must specify either --data_folder or --pano_path")
    
    logger.info(f"Panorama volume shape: {pano_volume.shape}")
    if ct_volume is not None:
        logger.info(f"CT volume shape: {ct_volume.shape}")
    
    # Generate
    logger.info("\n" + "="*60)
    logger.info("Starting Generation")
    logger.info("="*60)
    
    if args.single_slice is not None:
        # Generate single slice
        center_idx = args.single_slice
        num_slices = pano_volume.shape[0]
        
        prev_idx = max(0, center_idx - 1)
        next_idx = min(num_slices - 1, center_idx + 1)
        
        pano_window = np.stack([
            pano_volume[prev_idx],
            pano_volume[center_idx],
            pano_volume[next_idx]
        ], axis=0)
        
        slice_position = center_idx / max(1, num_slices - 1)
        
        generated = generate_single_window(
            vae, diffusion_model, diffusion_process,
            pano_window, slice_position, device,
            guidance_scale=args.guidance_scale,
            use_ddim=args.use_ddim,
            ddim_steps=args.ddim_steps,
            use_self_conditioning=use_self_conditioning,
            scale_factor=scale_factor  # Pass scale_factor
        )
        
        # Extract middle slice
        gen_slice = generated[0, 0, 1].cpu().numpy()
        
        # Save
        save_path = os.path.join(output_dir, f'generated_slice_{center_idx}')
        np.save(save_path + '.npy', gen_slice)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(pano_volume[center_idx], cmap='gray', vmin=-1, vmax=1)
        plt.title('Panorama')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(gen_slice, cmap='gray', vmin=-1, vmax=1)
        plt.title('Generated')
        plt.axis('off')
        
        if ct_volume is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(ct_volume[center_idx], cmap='gray', vmin=-1, vmax=1)
            plt.title('Ground Truth')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path + '.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved generated slice to: {save_path}")
        
    else:
        # Generate full volume
        generated_volume = generate_full_volume(
            vae, diffusion_model, diffusion_process,
            pano_volume, device,
            slice_range=(0, pano_volume.shape[0]),
            guidance_scale=args.guidance_scale,
            use_ddim=args.use_ddim,
            ddim_steps=args.ddim_steps,
            use_self_conditioning=use_self_conditioning,
            overlap_blend=args.overlap_blend,
            scale_factor=scale_factor  # Pass scale_factor
        )
        
        # Save volume
        volume_path = os.path.join(output_dir, 'generated_volume')
        save_volume(generated_volume, volume_path, format=args.output_format)
        
        # Visualize comparison
        vis_path = os.path.join(output_dir, 'comparison.png')
        visualize_comparison(generated_volume, ct_volume, pano_volume, vis_path)
        
        # Compute metrics if ground truth available
        if ct_volume is not None:
            metrics = compute_metrics(generated_volume, ct_volume)
            
            logger.info("\n" + "="*60)
            logger.info("Evaluation Metrics")
            logger.info("="*60)
            logger.info(f"PSNR: {metrics['psnr']:.2f} dB")
            logger.info(f"SSIM: {metrics['ssim']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            logger.info(f"MSE: {metrics['mse']:.4f}")
            logger.info(f"Z-Consistency: {metrics['z_consistency']:.4f}")
            logger.info("="*60 + "\n")
            
            # Save metrics
            metrics_path = os.path.join(output_dir, 'metrics.txt')
            with open(metrics_path, 'w') as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v:.6f}\n")
            logger.info(f"Saved metrics to: {metrics_path}")
    
    logger.info("\n" + "="*60)
    logger.info(f"Inference completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60 + "\n")
    
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak GPU memory: {max_memory:.2f}GB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Slab-based LDM Inference')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Input options (one required)
    parser.add_argument('--data_folder', type=str, default=None,
                       help='Path to dataset folder (standard structure)')
    parser.add_argument('--pano_path', type=str, default=None,
                       help='Path to panorama volume (.raw or .npy)')
    parser.add_argument('--ct_path', type=str, default=None,
                       help='Path to ground truth CT (optional)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--output_format', type=str, default='all',
                       choices=['raw', 'npy', 'png', 'all'],
                       help='Output format')
    
    # Generation options
    parser.add_argument('--single_slice', type=int, default=None,
                       help='Generate single slice at this index')
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 120],
                       help='Slice range [start, end)')
    parser.add_argument('--guidance_scale', type=float, default=1.5,
                       help='CFG guidance scale')
    parser.add_argument('--use_ddim', action='store_true',
                       help='Use DDIM sampling (faster)')
    parser.add_argument('--ddim_steps', type=int, default=50,
                       help='Number of DDIM steps')
    parser.add_argument('--overlap_blend', action='store_true', default=True,
                       help='Blend overlapping regions in sliding window')
    parser.add_argument('--no_overlap_blend', action='store_true',
                       help='Disable overlap blending')
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Use EMA weights if available')
    parser.add_argument('--use_self_conditioning', action='store_true', default=True,
                       help='Enable self-conditioning')
    
    # Scale factor
    parser.add_argument('--scale_factor', type=float, default=0.18215,
                       help='Latent scaling factor (default: 0.18215, Stable Diffusion standard)')
    
    # ============================================================
    # [v5.7] Normalization options - 학습 시 설정과 일치시켜야 함
    # ============================================================
    parser.add_argument('--normalize_input', action='store_true', default=False,
                       help='Apply [0,1] -> [-1,1] normalization to input data. '
                            'Use this if your data is in [0,1] range. '
                            'Should match --normalize_volumes setting used during training.')
    parser.add_argument('--input_range', type=str, default='auto',
                       choices=['auto', '0_1', '-1_1'],
                       help='Input data range: auto (detect), 0_1, or -1_1. '
                            'Use with --normalize_input for explicit control.')
    
    # Model config (overrides checkpoint config)
    parser.add_argument('--z_channels', type=int, default=8)
    parser.add_argument('--vae_channels', type=int, default=64)
    parser.add_argument('--diffusion_channels', type=int, default=128)
    parser.add_argument('--cond_channels', type=int, default=512)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--prediction_type', type=str, default='v')
    parser.add_argument('--panorama_type', type=str, default='axial')
    parser.add_argument('--image_size', type=int, default=200)
    
    args = parser.parse_args()
    
    # Handle overlap blend flag
    if args.no_overlap_blend:
        args.overlap_blend = False
    
    main(args)