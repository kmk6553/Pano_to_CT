"""
Enhanced Pseudo-3D Latent Diffusion Model - Inference Script
Generate CT slices from panorama images
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Import model components
from models import VAE, ConditionalUNet, ConsistencyNet3D
from diffusion_process import DiffusionProcess
from utils import set_seed, load_checkpoint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_panorama_raw(file_path, shape=(120, 200, 200)):
    """Load panorama from raw file"""
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        volume = data.reshape(shape, order='C')
        logger.info(f"Loaded panorama: {volume.shape}, range [{volume.min():.3f}, {volume.max():.3f}]")
        return volume
    except Exception as e:
        logger.error(f"Error loading panorama: {e}")
        raise


def normalize_volume(volume):
    """Normalize volume to [-1, 1] range"""
    vmin = volume.min()
    vmax = volume.max()
    
    if vmax - vmin < 1e-6:
        logger.warning("Volume has constant values, returning zeros")
        return np.zeros_like(volume)
    
    normalized = 2.0 * (volume - vmin) / (vmax - vmin) - 1.0
    normalized = np.clip(normalized, -1.0, 1.0)
    
    logger.info(f"Normalized volume: [{normalized.min():.3f}, {normalized.max():.3f}]")
    return normalized


def extract_panorama_slice(pano_volume, slice_idx, panorama_type='axial'):
    """Extract 2D panorama slice from volume"""
    if panorama_type == 'axial':
        return pano_volume[slice_idx, :, :]
    elif panorama_type == 'coronal':
        return pano_volume[:, 200, :]
    elif panorama_type == 'mip':
        return np.max(pano_volume[:, 150:250, :], axis=1)
    elif panorama_type == 'curved':
        h, _, w = pano_volume.shape
        panorama = np.zeros((h, w))
        for i in range(w):
            curve_idx = int(200 + 50 * np.sin(2 * np.pi * i / w))
            curve_idx = np.clip(curve_idx, 0, pano_volume.shape[1] - 1)
            panorama[:, i] = pano_volume[:, curve_idx, i]
        return panorama
    else:
        raise ValueError(f"Unknown panorama type: {panorama_type}")


def prepare_condition(pano_volume, slice_idx, panorama_type, pano_triplet=False, 
                     slice_range=(0, 240), device='cuda'):
    """Prepare condition tensor (single or triplet)"""
    if pano_triplet:
        # Get previous, current, next slices
        prev_idx = max(slice_range[0], slice_idx - 1)
        next_idx = min(slice_range[1] - 1, slice_idx + 1)
        
        pano_prev = extract_panorama_slice(pano_volume, prev_idx, panorama_type)
        pano_curr = extract_panorama_slice(pano_volume, slice_idx, panorama_type)
        pano_next = extract_panorama_slice(pano_volume, next_idx, panorama_type)
        
        # Stack as 3 channels
        condition = np.stack([pano_prev, pano_curr, pano_next], axis=0)  # [3, H, W]
    else:
        # Single channel
        pano_2d = extract_panorama_slice(pano_volume, slice_idx, panorama_type)
        condition = pano_2d[np.newaxis, ...]  # [1, H, W]
    
    # Convert to tensor
    condition_tensor = torch.from_numpy(condition).unsqueeze(0).float().to(device)
    condition_tensor = torch.clamp(condition_tensor, -1.0, 1.0)
    
    return condition_tensor


def save_slice_png(slice_data, save_path, title=""):
    """Save single slice as PNG"""
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_data, cmap='gray', vmin=-1, vmax=1)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison_grid(conditions, generated_slices, save_path, num_samples=5):
    """Save comparison grid of conditions and generated slices"""
    num_samples = min(num_samples, len(generated_slices))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i in range(num_samples):
        idx = i * (len(generated_slices) // num_samples)
        
        # Show condition (middle channel if triplet)
        cond = conditions[idx]
        if cond.shape[0] == 3:
            cond_show = cond[1]  # Middle channel
        else:
            cond_show = cond[0]
        
        axes[0, i].imshow(cond_show, cmap='gray', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Panorama Slice {idx}')
        axes[0, i].axis('off')
        
        # Show generated
        axes[1, i].imshow(generated_slices[idx], cmap='gray', vmin=-1, vmax=1)
        axes[1, i].set_title(f'Generated CT Slice {idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison grid: {save_path}")


def denormalize_to_uint16(volume):
    """Denormalize from [-1, 1] to uint16 range"""
    # Convert from [-1, 1] to [0, 1]
    volume_01 = (volume + 1.0) * 0.5
    volume_01 = np.clip(volume_01, 0, 1)
    
    # Scale to uint16
    volume_uint16 = (volume_01 * 65535).astype(np.uint16)
    
    return volume_uint16


@torch.no_grad()
def inference(args):
    """Main inference function"""
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slices_dir = output_dir / 'slices'
    slices_dir.mkdir(exist_ok=True)
    
    if args.visualize_samples > 0:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    
    # Override config with command line arguments
    if args.slice_range:
        slice_range = tuple(args.slice_range)
    else:
        slice_range = config.get('data', {}).get('slice_range', (0, 240))
    
    panorama_type = args.panorama_type or config.get('data', {}).get('panorama_type', 'axial')
    pano_triplet = args.pano_triplet or config.get('data', {}).get('pano_triplet', False)
    
    # Model hyperparameters
    z_channels = config.get('model', {}).get('z_channels', 8)
    vae_channels = config.get('model', {}).get('vae_channels', 64)
    diffusion_channels = config.get('model', {}).get('diffusion_channels', 128)
    cond_channels = config.get('model', {}).get('cond_channels', 512)
    
    # Diffusion parameters
    num_timesteps = config.get('diffusion', {}).get('num_timesteps', 1000)
    beta_start = config.get('diffusion', {}).get('beta_start', 0.0001)
    beta_end = config.get('diffusion', {}).get('beta_end', 0.02)
    prediction_type = config.get('diffusion', {}).get('prediction_type', 'v')
    use_self_conditioning = config.get('diffusion', {}).get('use_self_conditioning', True)
    
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else config.get('diffusion', {}).get('guidance_scale', 1.0)
    
    logger.info("\n" + "="*60)
    logger.info("Inference Configuration:")
    logger.info("="*60)
    logger.info(f"Slice range: {slice_range}")
    logger.info(f"Panorama type: {panorama_type}")
    logger.info(f"Panorama triplet: {pano_triplet}")
    logger.info(f"Prediction type: {prediction_type}")
    logger.info(f"Self-conditioning: {use_self_conditioning}")
    logger.info(f"Guidance scale: {guidance_scale}")
    logger.info(f"Use consistency: {not args.no_consistency}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*60 + "\n")
    
    # Initialize models
    logger.info("Initializing models...")
    
    vae = VAE(
        in_channels=1,
        z_channels=z_channels,
        channels=vae_channels
    ).to(device)
    vae.eval()
    
    diffusion_model = ConditionalUNet(
        in_channels=z_channels,
        out_channels=z_channels,
        channels=diffusion_channels,
        cond_channels=cond_channels,
        panorama_type=panorama_type,
        pano_triplet=pano_triplet,
        use_self_conditioning=use_self_conditioning
    ).to(device)
    diffusion_model.eval()
    
    consistency_net = None
    if not args.no_consistency:
        consistency_net = ConsistencyNet3D(
            in_channels=1,
            features=32,
            use_axial_attention=True
        ).to(device)
        consistency_net.eval()
    
    # Load model weights
    if 'vae_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['vae_state_dict'])
        logger.info("✓ VAE weights loaded")
    else:
        logger.error("VAE weights not found in checkpoint!")
        return
    
    # Load diffusion weights (prefer EMA if available)
    if 'ema_state_dict' in checkpoint:
        diffusion_model.load_state_dict(checkpoint['ema_state_dict'])
        logger.info("✓ Diffusion model weights loaded (EMA)")
    elif 'diffusion_state_dict' in checkpoint:
        diffusion_model.load_state_dict(checkpoint['diffusion_state_dict'])
        logger.info("✓ Diffusion model weights loaded")
    else:
        logger.error("Diffusion model weights not found in checkpoint!")
        return
    
    if consistency_net is not None and 'consistency_state_dict' in checkpoint:
        consistency_net.load_state_dict(checkpoint['consistency_state_dict'])
        logger.info("✓ Consistency network weights loaded")
    
    # Initialize diffusion process
    diffusion_process = DiffusionProcess(
        num_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule='cosine',
        prediction_type=prediction_type
    ).to(device)
    
    logger.info("✓ Models initialized successfully\n")
    
    # Load panorama
    logger.info(f"Loading panorama from: {args.input}")
    pano_volume = load_panorama_raw(args.input, shape=(120, 200, 200))
    
    # Normalize panorama
    pano_volume = normalize_volume(pano_volume)
    
    # Generate slices
    num_slices = slice_range[1] - slice_range[0]
    logger.info(f"\nGenerating {num_slices} CT slices...")
    
    generated_slices = []
    conditions_list = []
    
    # Get latent shape
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, 200, 200, device=device)
        latent_shape = vae.encoder(dummy_input)[0].shape[2:]
        logger.info(f"Latent shape: {latent_shape}")
    
    # Process in batches
    slice_indices = list(range(slice_range[0], slice_range[1]))
    
    for batch_start in tqdm(range(0, len(slice_indices), args.batch_size), desc="Generating slices"):
        batch_end = min(batch_start + args.batch_size, len(slice_indices))
        batch_indices = slice_indices[batch_start:batch_end]
        
        # Prepare batch conditions
        batch_conditions = []
        batch_positions = []
        
        for slice_idx in batch_indices:
            condition = prepare_condition(
                pano_volume, slice_idx, panorama_type, 
                pano_triplet, slice_range, device
            )
            batch_conditions.append(condition)
            
            # Calculate normalized slice position
            slice_position = (slice_idx - slice_range[0]) / max(1, num_slices - 1)
            batch_positions.append(slice_position)
        
        batch_conditions = torch.cat(batch_conditions, dim=0)
        batch_positions = torch.tensor(batch_positions, dtype=torch.float32, device=device)
        
        # Sample latents
        latent_shape_batch = (len(batch_indices), z_channels, latent_shape[0], latent_shape[1])
        
        z_gen = diffusion_process.p_sample_loop(
            diffusion_model,
            latent_shape_batch,
            batch_conditions,
            device,
            slice_pos=batch_positions,
            guidance_scale=guidance_scale,
            use_self_conditioning=use_self_conditioning
        )
        
        # Decode latents
        decoded_slices = vae.decode(z_gen)
        decoded_slices = torch.clamp(decoded_slices, -1.0, 1.0)
        
        # Store results
        for i, slice_idx in enumerate(batch_indices):
            slice_np = decoded_slices[i, 0].cpu().numpy()
            generated_slices.append(slice_np)
            conditions_list.append(batch_conditions[i].cpu().numpy())
        
        # Free memory
        del z_gen, decoded_slices, batch_conditions
        torch.cuda.empty_cache()
    
    logger.info("✓ Slice generation completed")
    
    # Apply 3D consistency
    if consistency_net is not None and not args.no_consistency:
        logger.info("\nApplying 3D consistency refinement...")
        
        # Stack slices into volume [1, C, D, H, W]
        volume_tensor = torch.from_numpy(np.stack(generated_slices, axis=0)).unsqueeze(0).unsqueeze(0).float().to(device)
        volume_tensor = volume_tensor.permute(0, 1, 2, 3, 4)  # Ensure correct dimension order
        
        # Apply consistency
        with torch.no_grad():
            refined_volume = consistency_net(volume_tensor)
            refined_volume = torch.clamp(refined_volume, -1.0, 1.0)
        
        # Extract refined slices
        refined_slices = refined_volume[0, 0].cpu().numpy()
        generated_slices = [refined_slices[i] for i in range(refined_slices.shape[0])]
        
        logger.info("✓ 3D consistency applied")
        
        del volume_tensor, refined_volume
        torch.cuda.empty_cache()
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save individual slices as PNG
    if args.save_individual:
        logger.info("Saving individual slice PNGs...")
        for i, (slice_idx, slice_data) in enumerate(zip(slice_indices, generated_slices)):
            save_path = slices_dir / f'slice_{slice_idx:03d}.png'
            save_slice_png(slice_data, save_path, f'CT Slice {slice_idx}')
        logger.info(f"✓ Saved {len(generated_slices)} individual slices")
    
    # Save as raw volume
    if args.save_raw:
        # Stack slices
        volume_np = np.stack(generated_slices, axis=0)  # [D, H, W]
        
        # Save float32 version (normalized [-1, 1])
        raw_path_float = output_dir / 'generated_volume_float32.raw'
        volume_np.astype(np.float32).tofile(raw_path_float)
        logger.info(f"✓ Saved float32 raw volume: {raw_path_float}")
        logger.info(f"  Shape: {volume_np.shape}, Range: [{volume_np.min():.3f}, {volume_np.max():.3f}]")
        
        # Save uint16 version (denormalized)
        raw_path_uint16 = output_dir / 'generated_volume_uint16.raw'
        volume_uint16 = denormalize_to_uint16(volume_np)
        volume_uint16.tofile(raw_path_uint16)
        logger.info(f"✓ Saved uint16 raw volume: {raw_path_uint16}")
        logger.info(f"  Shape: {volume_uint16.shape}, Range: [{volume_uint16.min()}, {volume_uint16.max()}]")
    
    # Save visualizations
    if args.visualize_samples > 0:
        logger.info(f"Creating visualization with {args.visualize_samples} samples...")
        vis_path = vis_dir / 'comparison_grid.png'
        save_comparison_grid(conditions_list, generated_slices, vis_path, args.visualize_samples)
    
    # Save metadata
    metadata_path = output_dir / 'metadata.txt'
    with open(metadata_path, 'w') as f:
        f.write(f"Inference Metadata\n")
        f.write(f"="*60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Slice range: {slice_range}\n")
        f.write(f"Number of slices: {num_slices}\n")
        f.write(f"Panorama type: {panorama_type}\n")
        f.write(f"Panorama triplet: {pano_triplet}\n")
        f.write(f"Prediction type: {prediction_type}\n")
        f.write(f"Guidance scale: {guidance_scale}\n")
        f.write(f"Self-conditioning: {use_self_conditioning}\n")
        f.write(f"3D consistency: {not args.no_consistency}\n")
        f.write(f"Output shape: {volume_np.shape if args.save_raw else 'N/A'}\n")
        f.write(f"Output range (float32): [{volume_np.min():.3f}, {volume_np.max():.3f}]\n" if args.save_raw else "")
        f.write(f"Output range (uint16): [{volume_uint16.min()}, {volume_uint16.max()}]\n" if args.save_raw else "")
    
    logger.info(f"✓ Saved metadata: {metadata_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Inference completed successfully!")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Generated {len(generated_slices)} CT slices")
    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Pseudo-3D Latent Diffusion - Inference')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input panorama raw file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for generated slices')
    
    # Model arguments
    parser.add_argument('--slice_range', type=int, nargs=2, default=None,
                       help='Slice range [start, end) to generate (default: from config)')
    parser.add_argument('--panorama_type', type=str, default=None,
                       choices=['coronal', 'axial', 'mip', 'curved'],
                       help='Panorama extraction type (default: from config)')
    parser.add_argument('--pano_triplet', action='store_true', default=False,
                       help='Use 3-channel panorama condition')
    
    # Inference arguments
    parser.add_argument('--guidance_scale', type=float, default=None,
                       help='Classifier-free guidance scale (default: from config or 1.0)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--no_consistency', action='store_true',
                       help='Skip 3D consistency refinement')
    
    # Output arguments
    parser.add_argument('--save_raw', action='store_true',
                       help='Save output as raw volume file')
    parser.add_argument('--save_individual', action='store_true',
                       help='Save individual slice PNGs')
    parser.add_argument('--visualize_samples', type=int, default=0,
                       help='Number of samples to visualize in comparison grid')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Run inference
    inference(args)


if __name__ == '__main__':
    main()