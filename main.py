"""
3D Slab-based Latent Diffusion Model
Main training script with 3D VAE, 3D UNet, v-param, self-conditioning, and CFG

Processes 3-slice windows: [B, 1, D=3, H, W]
Generates 3 slices simultaneously for improved z-axis consistency

FIXES APPLIED:
1. GPU Augmentation 사용 시 CPU Augmentation 비활성화
2. Scale factor 설정 추가 (config['model']['scale_factor'])
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

import numpy as np
import os
import random
import argparse
import logging
from datetime import datetime

# Import our modules
from models import VAE3D, ConditionalUNet3D, ConsistencyNet3D, EMAWrapper
from data import OptimizedDentalSliceDataset, GPUAugmentation
from losses import CharbonnierLoss, GradientDifferenceLoss, LPIPSLoss
from training import (
    WarmupCosineScheduler, 
    AdaptiveLearningRateScheduler,
    train_vae_optimized,
    train_diffusion_optimized,
    train_consistency_3d,
    evaluate_and_visualize
)
from utils import (
    set_seed,
    enable_flash_attention,
    check_gpu_support,
    load_checkpoint,
    load_config,
    save_config,
    MetricsTracker
)
from diffusion_process import DiffusionProcess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_latent_scale_factor(vae, dataloader, device, num_samples=100):
    """
    VAE 학습 후 latent의 표준편차를 계산하여 scale_factor 결정
    
    Stable Diffusion 방식: scale_factor = 1 / std(latent)
    이렇게 하면 scaled latent가 대략 N(0, 1)에 가까워짐
    
    Args:
        vae: 학습된 VAE 모델
        dataloader: 데이터 로더
        device: 장치
        num_samples: 샘플링할 배치 수
    
    Returns:
        scale_factor: 계산된 스케일 팩터
    """
    vae.eval()
    latents = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            ct_volume = batch['ct_volume'].to(device)
            if ct_volume.dim() == 4:
                ct_volume = ct_volume.unsqueeze(1)
            
            mean, logvar = vae.encoder(ct_volume)
            z = vae.sample(mean, logvar)
            latents.append(z.cpu())
    
    all_latents = torch.cat(latents, dim=0)
    latent_std = all_latents.std().item()
    
    # 1/std로 scale_factor 계산 (Stable Diffusion 방식)
    scale_factor = 1.0 / latent_std
    
    logger.info(f"Computed latent statistics:")
    logger.info(f"  Mean: {all_latents.mean().item():.4f}")
    logger.info(f"  Std: {latent_std:.4f}")
    logger.info(f"  Scale factor (1/std): {scale_factor:.4f}")
    
    return scale_factor


def main(args):
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
        logger.warning("CLI arguments will override config file values if specified")
    else:
        # Build config from command line arguments
        config = {
            'data': {
                'data_dir': args.data_dir,
                'val_ratio': args.val_ratio,
                'slice_range': args.slice_range,
                'panorama_type': args.panorama_type,
                'normalize_volumes': args.normalize_volumes,
                'pano_triplet': True,  # Always True for 3D Slab
                'cache_volumes': args.cache_volumes,
                'use_memmap': not args.no_memmap,
                'use_gpu_augmentation': args.use_gpu_augmentation,
                'debug_augmentation': args.debug_augmentation,
                'augment_from_epoch': args.augment_from_epoch,
                'augmentation': {
                    'elastic_alpha': 10 * args.elastic_strength,
                    'elastic_sigma': 4,
                    'elastic_alpha_affine': 10 * args.elastic_strength,
                    'use_elastic': not args.no_elastic,
                    'random_flip': True,
                    'random_rotation': True,
                    'gamma_correction': True,
                    'random_noise': True,
                    'augment_from_epoch': args.augment_from_epoch,
                    'intensity_to_condition': True
                }
            },
            'model': {
                'z_channels': args.z_channels,
                'vae_channels': args.vae_channels,
                'diffusion_channels': args.diffusion_channels,
                'cond_channels': args.cond_channels,
                # ============================================================
                # FIX 1: Scale factor 설정 추가
                # 기본값 0.18215 (Stable Diffusion 표준)
                # --auto_scale_factor 사용 시 VAE 학습 후 자동 계산
                # ============================================================
                'scale_factor': args.scale_factor,
                'auto_scale_factor': args.auto_scale_factor
            },
            'diffusion': {
                'num_timesteps': args.num_timesteps,
                'beta_start': args.beta_start,
                'beta_end': args.beta_end,
                'schedule': 'cosine',
                'prediction_type': args.prediction_type,
                'use_self_conditioning': args.use_self_conditioning,
                'self_cond_prob': args.self_cond_prob,
                'cfg_dropout_prob': args.cfg_dropout_prob,
                'guidance_scale': args.guidance_scale,
                'l1_weight': args.l1_weight,
                'gdl_weight': args.gdl_weight,
                'lpips_weight': args.lpips_weight,
                'diffusion_lr': args.diffusion_lr,
                'mid_slice_weight': args.mid_slice_weight  # Weight for middle slice in loss
            },
            'vae': {
                'beta_max': 0.0001,
                'beta_schedule': 'linear',
                'beta_warmup': 20,
                'beta_cycle_length': 10,
                'free_bits': 0.1,
                'lpips_weight': args.vae_lpips_weight,
                'initial_lr': args.vae_lr,
                'target_lr': max(args.vae_lr * 10, 0.0001),
                'warmup_epochs': 5,
                'mid_slice_weight': args.mid_slice_weight  # Weight for middle slice in loss
            },
            'training': {
                'batch_size': args.batch_size,
                'vae_epochs': args.vae_epochs,
                'diffusion_epochs': args.diffusion_epochs,
                'consistency_epochs': args.consistency_epochs,
                'consistency_num_slices': args.consistency_num_slices,
                'vae_lr': args.vae_lr,
                'diffusion_lr': args.diffusion_lr,
                'consistency_lr': args.consistency_lr,
                'num_workers': args.num_workers,
                'grad_clip': args.grad_clip if args.grad_clip else 0.8,
                'use_scheduler': True,
                'skip_vae_training': args.skip_vae_training,
                'skip_consistency_training': args.skip_consistency_training,
                'use_amp': args.use_amp,
                'no_amp_vae': args.no_amp_vae,
                'use_ema': args.use_ema,
                'ema_decay': args.ema_decay,
                'warmup_steps': args.warmup_steps,
                'teacher_forcing_ratio': args.teacher_forcing_ratio,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'use_bfloat16': args.use_bfloat16,
                'detect_anomaly': args.detect_anomaly
            },
            'logging': {
                'sample_interval': args.sample_interval,
                'save_interval': args.save_interval,
                'eval_interval': 5,
                'plot_interval': 5  
            },
            'seed': args.seed,
            'vae_checkpoint': args.vae_checkpoint,
            'resume_from_checkpoint': args.resume_from_checkpoint,
            'resume_diffusion_from': args.resume_diffusion_from,
            'max_data_folders': args.max_data_folders,
            'save_dir': args.save_dir
        }
    
    # Set seed and device
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Enable Flash Attention if available
    flash_enabled = enable_flash_attention()
    
    # Check GPU support
    has_gpu, supports_bfloat16, gpu_name = check_gpu_support()
    
    if has_gpu:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        
        if config['training']['use_bfloat16']:
            if supports_bfloat16:
                logger.info("BFloat16 support detected! Using BFloat16 for improved numerical stability.")
            else:
                logger.warning(f"BFloat16 requested but not supported on {gpu_name}. Falling back to Float16.")
                config['training']['use_bfloat16'] = False
        
        # Memory recommendations for 3D models (higher memory usage)
        if gpu_memory < 12:
            logger.warning(f"Low GPU memory detected ({gpu_memory:.1f}GB). 3D models require more memory.")
            logger.warning("Consider reducing batch size to 1-2 or using gradient accumulation.")
        elif gpu_memory < 24:
            logger.info(f"Medium GPU memory detected ({gpu_memory:.1f}GB). Batch size 2-4 recommended.")
        else:
            logger.info(f"High GPU memory detected ({gpu_memory:.1f}GB). Batch size 4-8 should work well.")
    
    if config['training']['detect_anomaly']:
        torch.autograd.set_detect_anomaly(True)
        logger.warning("Anomaly detection enabled. This will slow down training but help identify NaN sources.")
    
    # Log 3D Slab architecture info
    logger.info("\n" + "="*60)
    logger.info("3D SLAB-BASED ARCHITECTURE")
    logger.info("="*60)
    logger.info("This model processes 3-slice windows simultaneously:")
    logger.info("  - Input:  [B, 1, D=3, H, W] CT volume")
    logger.info("  - Cond:   [B, 3, H, W] 3-channel panorama")
    logger.info("  - Latent: [B, z_ch, D=3, h, w] 3D latent")
    logger.info("  - Output: [B, 1, D=3, H, W] generated volume")
    logger.info("")
    logger.info("Benefits over 2D slice-by-slice:")
    logger.info("  - Direct z-axis consistency learning")
    logger.info("  - Joint generation of adjacent slices")
    logger.info("  - Reduced slice-to-slice artifacts")
    logger.info("="*60 + "\n")
    
    # ============================================================
    # FIX 2: GPU Augmentation 사용 시 CPU Augmentation 비활성화 결정
    # ============================================================
    from data.augmentation import KORNIA_AVAILABLE
    
    use_gpu_aug = config['data']['use_gpu_augmentation'] and KORNIA_AVAILABLE
    
    # CPU augmentation 플래그 결정
    if use_gpu_aug:
        logger.info("="*60)
        logger.info("AUGMENTATION CONFLICT PREVENTION")
        logger.info("="*60)
        logger.info("GPU Augmentation enabled -> Disabling heavy CPU augmentation")
        logger.info("to avoid double distortion (데이터 분포 망가짐 방지)")
        logger.info("="*60 + "\n")
        train_augment_cpu = False
    else:
        train_augment_cpu = True
    
    # Log optimization settings
    logger.info("\n" + "="*60)
    logger.info("Performance Optimizations Enabled:")
    logger.info("="*60)
    logger.info(f"- Volume caching: {config['data']['cache_volumes']}")
    logger.info(f"- GPU augmentation: {use_gpu_aug}")
    logger.info(f"- CPU augmentation: {train_augment_cpu}")
    logger.info(f"- Async data prefetching: Enabled")
    logger.info(f"- Memory-mapped I/O: {config['data']['use_memmap']}")
    logger.info(f"- Flash Attention: {flash_enabled}")
    logger.info(f"- Gradient accumulation: {config['training']['gradient_accumulation_steps']} steps")
    logger.info(f"- Augmentation starts from epoch: {config['data']['augment_from_epoch']}")
    logger.info(f"- Latent scale factor: {config['model']['scale_factor']}")
    logger.info(f"- Auto compute scale factor: {config['model'].get('auto_scale_factor', False)}")
    logger.info("="*60 + "\n")
    
    # Log model configuration
    logger.info("\n" + "="*60)
    logger.info("MODEL CONFIGURATION:")
    logger.info("="*60)
    logger.info(f"VAE channels: {config['model']['vae_channels']}")
    logger.info(f"Latent channels: {config['model']['z_channels']}")
    logger.info(f"Diffusion channels: {config['model']['diffusion_channels']}")
    logger.info(f"Condition channels: {config['model']['cond_channels']}")
    logger.info(f"Prediction type: {config['diffusion']['prediction_type']}")
    logger.info(f"Self-conditioning: {config['diffusion']['use_self_conditioning']} (prob={config['diffusion']['self_cond_prob']})")
    logger.info(f"CFG dropout: {config['diffusion']['cfg_dropout_prob']}")
    logger.info(f"CFG guidance scale: {config['diffusion']['guidance_scale']}")
    logger.info(f"Mid-slice weight: {config['diffusion']['mid_slice_weight']}")
    logger.info("="*60 + "\n")
    
    # Setup AMP
    use_amp = config['training']['use_amp']
    use_bf16 = config['training']['use_bfloat16']
    use_amp_fp16 = use_amp and not use_bf16
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    if use_amp:
        logger.info("\nAutomatic Mixed Precision (AMP) Settings:")
        logger.info(f"- Precision: {'BFloat16' if use_bf16 else 'Float16'}")
        logger.info(f"- GradScaler: {'Disabled (BFloat16)' if use_bf16 else 'Enabled (Float16)'}")
        logger.info(f"- Gradient clipping: {config['training']['grad_clip']}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'3d_slab_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'interim_plots'), exist_ok=True)
    
    # Update save_dir in config
    config['save_dir'] = save_dir
    
    # Save configuration
    save_config(config, save_dir)
    
    # Prepare data
    max_folders = config.get('max_data_folders', 500)
    all_folders = [f'{i:04d}' for i in range(0, min(max_folders, 500))]
    
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    random.shuffle(all_folders)
    
    val_size = int(len(all_folders) * config['data']['val_ratio'])
    
    val_folders = all_folders[:val_size]
    train_folders = all_folders[val_size:]
    
    logger.info(f"Total folders available: {len(all_folders)}")
    logger.info(f"Training folders: {len(train_folders)}")
    logger.info(f"Validation folders: {len(val_folders)}")
    logger.info(f"Slice range: {config['data']['slice_range']}")
    logger.info(f"Panorama type: {config['data']['panorama_type']}")
    logger.info(f"Volume normalization: {config['data']['normalize_volumes']}")
    
    accumulation_steps = config['training']['gradient_accumulation_steps']
    logger.info(f"Gradient accumulation steps: {accumulation_steps}")
    logger.info(f"Effective batch size: {config['training']['batch_size']} × {accumulation_steps} = {config['training']['batch_size'] * accumulation_steps}")
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Create 3D models
    logger.info("\nInitializing 3D models...")
    
    vae = VAE3D(
        in_channels=1, 
        z_channels=config['model']['z_channels'], 
        channels=config['model']['vae_channels']
    ).to(device)
    
    # Count parameters
    vae_params = sum(p.numel() for p in vae.parameters())
    logger.info(f"3D VAE parameters: {vae_params:,}")
    
    diffusion_model = ConditionalUNet3D(
        in_channels=config['model']['z_channels'], 
        out_channels=config['model']['z_channels'],
        channels=config['model']['diffusion_channels'],
        cond_channels=config['model']['cond_channels'],
        panorama_type=config['data']['panorama_type'],
        pano_triplet=True,  # Always True for 3D Slab
        use_self_conditioning=config['diffusion']['use_self_conditioning']
    ).to(device)
    
    diffusion_params = sum(p.numel() for p in diffusion_model.parameters())
    logger.info(f"3D Diffusion UNet parameters: {diffusion_params:,}")
    
    consistency_net = ConsistencyNet3D(
        in_channels=1, 
        features=32, 
        use_axial_attention=args.use_axial_attention
    ).to(device)
    
    consistency_params = sum(p.numel() for p in consistency_net.parameters())
    logger.info(f"Consistency Network parameters: {consistency_params:,}")
    logger.info(f"Total parameters: {vae_params + diffusion_params + consistency_params:,}")
    
    # Setup EMA
    ema_wrapper = None
    if config['training']['use_ema']:
        effective_batch_size = config['training']['batch_size'] * accumulation_steps
        if effective_batch_size <= 4:
            ema_decay = min(config['training']['ema_decay'], 0.995)
            if ema_decay != config['training']['ema_decay']:
                logger.info(f"Adjusting EMA decay from {config['training']['ema_decay']} to {ema_decay} for small batch size")
        else:
            ema_decay = config['training']['ema_decay']
        
        ema_wrapper = EMAWrapper(diffusion_model, decay=ema_decay)
        logger.info(f"EMA enabled with decay={ema_decay}")
    
    # Create diffusion process
    diffusion_process = DiffusionProcess(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule=config['diffusion']['schedule'],
        prediction_type=config['diffusion']['prediction_type']
    ).to(device)
    
    logger.info(f"Diffusion process: {config['diffusion']['prediction_type']} parametrization, {config['diffusion']['num_timesteps']} steps")
    
    # Create loss functions
    l1_loss_fn = CharbonnierLoss(epsilon=1e-2).to(device)
    gdl_loss_fn = GradientDifferenceLoss().to(device)
    lpips_loss_fn = LPIPSLoss(net='vgg', device=device) if config['vae']['lpips_weight'] > 0 or config['diffusion']['lpips_weight'] > 0 else None
    
    # GPU augmentation
    gpu_augmenter = None
    if use_gpu_aug:
        gpu_augmenter = GPUAugmentation(config['data']['augmentation'], device).to(device)
        logger.info("GPU augmentation module initialized")
    
    # ============================================================
    # FIX 2 적용: GPU augmentation 사용 시 CPU augmentation 비활성화
    # train_augment_cpu 플래그 사용
    # ============================================================
    train_dataset = OptimizedDentalSliceDataset(
        config['data']['data_dir'], 
        train_folders, 
        slice_range=config['data']['slice_range'],
        augment=train_augment_cpu,  # GPU aug 사용 시 False
        panorama_type=config['data']['panorama_type'],
        normalize_volumes=config['data']['normalize_volumes'],
        augment_config=config['data']['augmentation'] if train_augment_cpu else None,
        pano_triplet=True,  # Always True for 3D Slab
        cache_volumes=config['data']['cache_volumes'],
        use_memmap=config['data'].get('use_memmap', True)
    )
    
    val_dataset = OptimizedDentalSliceDataset(
        config['data']['data_dir'], 
        val_folders,
        slice_range=config['data']['slice_range'],
        augment=False,
        panorama_type=config['data']['panorama_type'],
        normalize_volumes=config['data']['normalize_volumes'],
        pano_triplet=True,  # Always True for 3D Slab
        cache_volumes=config['data']['cache_volumes'],
        use_memmap=config['data'].get('use_memmap', True)
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)} samples (3-slice windows)")
    logger.info(f"Val dataset size: {len(val_dataset)} samples (3-slice windows)")
    logger.info(f"Train CPU augmentation: {train_augment_cpu}")
    logger.info(f"Train GPU augmentation: {use_gpu_aug}")
    
    # Windows multiprocessing check
    if os.name == 'nt' and config['training']['num_workers'] > 0:
        logger.warning(f"Windows detected with num_workers={config['training']['num_workers']}.")
        logger.warning("If you encounter errors, consider setting --num_workers 0")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True, 
        num_workers=config['training']['num_workers'], 
        pin_memory=True, 
        persistent_workers=True if config['training']['num_workers'] > 0 else False,
        prefetch_factor=2 if config['training']['num_workers'] > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False, 
        num_workers=config['training']['num_workers'], 
        pin_memory=True, 
        persistent_workers=True if config['training']['num_workers'] > 0 else False,
        prefetch_factor=2 if config['training']['num_workers'] > 0 else None
    )
    
    logger.info(f"Batches per epoch: {len(train_loader)}")
    logger.info(f"Optimizer steps per epoch: {len(train_loader) // accumulation_steps}")
    
    # Create optimizers
    vae_optimizer = optim.AdamW(vae.parameters(), lr=config['training']['vae_lr'], weight_decay=1e-4)
    diffusion_optimizer = optim.AdamW(diffusion_model.parameters(), 
                                     lr=config['diffusion']['diffusion_lr'],
                                     weight_decay=1e-4)
    consistency_optimizer = optim.AdamW(consistency_net.parameters(), 
                                        lr=config['training']['consistency_lr'], 
                                        weight_decay=1e-4)
    
    # Create schedulers
    vae_scheduler = AdaptiveLearningRateScheduler(
        vae_optimizer,
        initial_lr=config['vae']['initial_lr'],
        target_lr=config['vae']['target_lr'],
        warmup_epochs=config['vae'].get('warmup_epochs', 5),
        stability_threshold=0.1
    ) if config['training']['use_scheduler'] else None
    
    # Load checkpoints if provided
    start_epoch = 1
    if config.get('vae_checkpoint'):
        logger.info(f"Loading VAE checkpoint from: {config['vae_checkpoint']}")
        checkpoint = load_checkpoint(config['vae_checkpoint'], device)
        vae.load_state_dict(checkpoint['vae_state_dict'])
        logger.info("VAE checkpoint loaded successfully!")
    
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = load_checkpoint(args.resume_from_checkpoint, device)
        
        vae.load_state_dict(checkpoint['vae_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            vae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Restored optimizer state")
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        
        if start_epoch > config['training']['vae_epochs']:
            logger.info(f"VAE training already completed")
            config['training']['skip_vae_training'] = True
    
    # Initialize diffusion training variables
    diffusion_start_epoch = 1
    nan_epochs = []
    
    # Load diffusion checkpoint if provided
    if config.get('resume_diffusion_from'):
        logger.info(f"Loading diffusion checkpoint from: {config['resume_diffusion_from']}")
        diffusion_checkpoint = load_checkpoint(config['resume_diffusion_from'], device)
        
        if 'diffusion_state_dict' in diffusion_checkpoint:
            diffusion_model.load_state_dict(diffusion_checkpoint['diffusion_state_dict'])
            logger.info("Diffusion model state loaded")
        
        if 'ema_state_dict' in diffusion_checkpoint and ema_wrapper is not None:
            ema_wrapper.get_model().load_state_dict(diffusion_checkpoint['ema_state_dict'])
            logger.info("EMA model state loaded")
        
        if 'diffusion_optimizer' in diffusion_checkpoint:
            diffusion_optimizer.load_state_dict(diffusion_checkpoint['diffusion_optimizer'])
            logger.info("Diffusion optimizer state loaded")
        
        if 'epoch' in diffusion_checkpoint:
            diffusion_start_epoch = diffusion_checkpoint['epoch'] + 1
            logger.info(f"Resuming diffusion training from epoch {diffusion_start_epoch}")
    
    # Setup diffusion scheduler with warmup
    steps_per_epoch = len(train_loader) // accumulation_steps
    total_steps = config['training']['diffusion_epochs'] * steps_per_epoch
    
    if diffusion_start_epoch > 1:
        warmup_steps = 0
        completed_steps = (diffusion_start_epoch - 1) * steps_per_epoch
    else:
        warmup_steps = min(5000, total_steps // 10)
        completed_steps = 0
    
    diffusion_scheduler = WarmupCosineScheduler(
        diffusion_optimizer, 
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=1e-6,
        base_lr=config['diffusion']['diffusion_lr']
    )
    
    if diffusion_start_epoch > 1:
        diffusion_scheduler.current_step = completed_steps
    
    # Setup GradScaler for AMP
    scaler = None
    vae_scaler = None
    
    use_amp_vae = config['training']['use_amp'] and not config['training'].get('no_amp_vae', False)
    
    if use_amp_fp16:
        scaler = GradScaler(
            init_scale=2.**6,
            growth_interval=5000,
            growth_factor=1.5,
            backoff_factor=0.5,
            enabled=True
        )
        logger.info("GradScaler enabled for Float16 AMP")
        
        if use_amp_vae:
            vae_scaler = GradScaler(
                init_scale=2.**4,
                growth_interval=10000,
                growth_factor=1.2,
                backoff_factor=0.5,
                enabled=True
            )
            logger.info("VAE GradScaler enabled")
        else:
            logger.info("VAE AMP disabled for stability")
    elif use_bf16:
        logger.info("Using BFloat16 - GradScaler not needed")
    
    # =========================================================================
    # Training Phase 1: 3D VAE
    # =========================================================================
    if not config['training']['skip_vae_training']:
        logger.info("\n" + "="*60)
        logger.info("=== Phase 1: Training 3D VAE ===")
        logger.info("="*60)
        logger.info(f"Processing 3-slice windows: [B, 1, D=3, H, W]")
        logger.info(f"Mid-slice weight: {config['vae']['mid_slice_weight']}")
        logger.info(f"Total epochs: {config['training']['vae_epochs']}")
        logger.info(f"Batch size: {config['training']['batch_size']}")
        logger.info(f"Effective batch size: {config['training']['batch_size'] * accumulation_steps}")
        logger.info("="*60 + "\n")
        
        for epoch in range(start_epoch, config['training']['vae_epochs'] + 1):
            if hasattr(train_dataset, 'set_epoch'):
                train_dataset.set_epoch(epoch)
            
            vae_loss, recon_loss, kl_loss = train_vae_optimized(
                vae, train_loader, vae_optimizer, vae_scheduler, device, epoch, config, metrics_tracker,
                lpips_loss_fn=lpips_loss_fn if config['vae']['lpips_weight'] > 0 else None,
                lpips_weight=config['vae']['lpips_weight'],
                scaler=vae_scaler,
                accumulation_steps=accumulation_steps,
                gpu_augmenter=gpu_augmenter
            )
            
            if np.isnan(vae_loss) or np.isinf(vae_loss):
                logger.error(f"Training instability at epoch {epoch}")
                for param_group in vae_optimizer.param_groups:
                    param_group['lr'] *= 0.5
                logger.warning(f"Reduced learning rate to {vae_optimizer.param_groups[0]['lr']:.6f}")
                continue
            
            if epoch % config['logging']['eval_interval'] == 0:
                evaluate_and_visualize(vae, None, None, val_loader, 
                                     device, epoch, os.path.join(save_dir, 'samples'), 
                                     'vae', metrics_tracker)
            
            if epoch % config['logging']['plot_interval'] == 0:
                metrics_tracker.plot_interim_metrics(save_dir, 'vae', epoch)
            
            if epoch % config['logging']['save_interval'] == 0:
                checkpoint_dict = {
                    'epoch': epoch,
                    'vae_state_dict': vae.state_dict(),
                    'optimizer_state_dict': vae_optimizer.state_dict(),
                    'metrics': metrics_tracker.metrics,
                    'config': config
                }
                
                if vae_scheduler is not None:
                    if isinstance(vae_scheduler, AdaptiveLearningRateScheduler):
                        checkpoint_dict['scheduler_state_dict'] = vae_scheduler.state_dict()
                
                torch.save(checkpoint_dict, os.path.join(save_dir, 'checkpoints', f'vae_epoch_{epoch}.pth'))
            
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
        
        # ============================================================
        # FIX 1: VAE 학습 완료 후 Scale Factor 자동 계산 (옵션)
        # ============================================================
        if config['model'].get('auto_scale_factor', False):
            logger.info("\n" + "="*60)
            logger.info("Computing Latent Scale Factor from trained VAE...")
            logger.info("="*60)
            
            computed_scale_factor = compute_latent_scale_factor(
                vae, train_loader, device, num_samples=100
            )
            
            # Config 업데이트
            config['model']['scale_factor'] = computed_scale_factor
            
            # Config 파일 다시 저장
            save_config(config, save_dir)
            
            logger.info(f"Updated scale_factor in config: {computed_scale_factor:.4f}")
            logger.info("="*60 + "\n")
    else:
        logger.info("\n=== Skipping VAE Training (Using loaded checkpoint) ===")
        
        # VAE가 이미 학습된 경우에도 scale factor 계산 옵션
        if config['model'].get('auto_scale_factor', False):
            logger.info("Computing scale factor from loaded VAE...")
            computed_scale_factor = compute_latent_scale_factor(
                vae, train_loader, device, num_samples=100
            )
            config['model']['scale_factor'] = computed_scale_factor
            save_config(config, save_dir)
            logger.info(f"Computed scale_factor: {computed_scale_factor:.4f}")
    
    # =========================================================================
    # Training Phase 2: 3D Diffusion
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("=== Phase 2: Training 3D Diffusion Model ===")
    logger.info("="*60)
    logger.info(f"Processing 3D latents: [B, z_ch, D=3, h, w]")
    logger.info(f"Prediction type: {config['diffusion']['prediction_type']}")
    logger.info(f"Self-conditioning: {config['diffusion']['use_self_conditioning']}")
    logger.info(f"CFG dropout: {config['diffusion']['cfg_dropout_prob']}")
    logger.info(f"Guidance scale: {config['diffusion']['guidance_scale']}")
    logger.info(f"Mid-slice weight: {config['diffusion']['mid_slice_weight']}")
    logger.info(f"Latent scale factor: {config['model']['scale_factor']}")
    logger.info(f"Total epochs: {config['training']['diffusion_epochs']}")
    if diffusion_start_epoch > 1:
        logger.info(f"Resuming from epoch: {diffusion_start_epoch}")
    logger.info("="*60 + "\n")
    
    early_stop = False
    
    for epoch in range(diffusion_start_epoch, config['training']['diffusion_epochs'] + 1):
        if early_stop:
            logger.warning("Early stopping triggered. Ending diffusion training.")
            break
            
        if hasattr(train_dataset, 'set_epoch'):
            train_dataset.set_epoch(epoch)
            
        diffusion_loss = train_diffusion_optimized(
            diffusion_model, vae, diffusion_process, 
            train_loader, diffusion_optimizer, diffusion_scheduler, 
            device, epoch, config, metrics_tracker, gdl_loss_fn, lpips_loss_fn, l1_loss_fn,
            l1_weight=config['diffusion']['l1_weight'],
            gdl_weight=config['diffusion']['gdl_weight'],
            lpips_weight=config['diffusion']['lpips_weight'],
            scaler=scaler,
            ema_wrapper=ema_wrapper,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
            autocast_dtype=autocast_dtype,
            gpu_augmenter=gpu_augmenter
        )
        
        if hasattr(metrics_tracker, 'patience_counter') and metrics_tracker.patience_counter >= metrics_tracker.early_stop_patience:
            early_stop = True
        
        if epoch % config['logging']['eval_interval'] == 0:
            evaluate_and_visualize(vae, diffusion_model, diffusion_process, 
                                 val_loader, device, epoch, 
                                 os.path.join(save_dir, 'samples'), 
                                 'diffusion', metrics_tracker, ema_wrapper=ema_wrapper,
                                 guidance_scale=config['diffusion']['guidance_scale'])
        
        if epoch % config['logging']['plot_interval'] == 0:
            metrics_tracker.plot_interim_metrics(save_dir, 'diffusion', epoch)
        
        if epoch % config['logging']['save_interval'] == 0:
            checkpoint_dict = {
                'epoch': epoch,
                'vae_state_dict': vae.state_dict(),
                'diffusion_state_dict': diffusion_model.state_dict(),
                'consistency_state_dict': consistency_net.state_dict(),
                'diffusion_optimizer': diffusion_optimizer.state_dict(),
                'scheduler_state_dict': diffusion_scheduler.state_dict() if hasattr(diffusion_scheduler, 'state_dict') else None,
                'metrics': metrics_tracker.metrics,
                'config': config,
                'nan_epochs': nan_epochs
            }
            
            if ema_wrapper is not None:
                checkpoint_dict['ema_state_dict'] = ema_wrapper.get_model().state_dict()
            
            torch.save(checkpoint_dict, os.path.join(save_dir, 'checkpoints', f'model_epoch_{epoch}.pth'))
            
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
    
    # =========================================================================
    # Training Phase 3: 3D Consistency (Optional)
    # =========================================================================
    if not config['training']['skip_consistency_training']:
        logger.info("\n" + "="*60)
        logger.info("=== Phase 3: Training 3D Consistency Network ===")
        logger.info("="*60)
        logger.info(f"Refining 3-slice windows for z-axis consistency")
        logger.info(f"Total epochs: {config['training']['consistency_epochs']}")
        logger.info(f"Teacher forcing ratio: {config['training']['teacher_forcing_ratio']}")
        logger.info("="*60 + "\n")
        
        for epoch in range(1, config['training']['consistency_epochs'] + 1):
            if hasattr(train_dataset, 'set_epoch'):
                train_dataset.set_epoch(epoch)
                
            consistency_loss = train_consistency_3d(
                consistency_net, vae, diffusion_model, diffusion_process,
                train_loader, consistency_optimizer, device, epoch, config,
                config['training']['consistency_num_slices'], metrics_tracker,
                teacher_forcing_ratio=config['training']['teacher_forcing_ratio'],
                ema_wrapper=ema_wrapper
            )
            
            if epoch % config['logging']['save_interval'] == 0:
                checkpoint_dict = {
                    'epoch': epoch,
                    'vae_state_dict': vae.state_dict(),
                    'diffusion_state_dict': diffusion_model.state_dict(),
                    'consistency_state_dict': consistency_net.state_dict(),
                    'consistency_optimizer': consistency_optimizer.state_dict(),
                    'metrics': metrics_tracker.metrics,
                    'config': config
                }
                
                if ema_wrapper is not None:
                    checkpoint_dict['ema_state_dict'] = ema_wrapper.get_model().state_dict()
                
                torch.save(checkpoint_dict, os.path.join(save_dir, 'checkpoints', f'consistency_epoch_{epoch}.pth'))
    else:
        logger.info("\n=== Skipping 3D Consistency Training ===")
    
    # =========================================================================
    # Training completed
    # =========================================================================
    
    # Final cleanup
    if hasattr(train_dataset, 'cleanup_cache'):
        train_dataset.cleanup_cache()
    if hasattr(val_dataset, 'cleanup_cache'):
        val_dataset.cleanup_cache()
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info("="*60)
    logger.info(f"Output directory: {save_dir}")
    logger.info(f"Final model saved at: {os.path.join(save_dir, 'checkpoints', 'final_model.pth')}")
    logger.info(f"Scale factor used: {config['model']['scale_factor']}")
    
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak GPU memory usage: {max_memory:.1f}GB")
    
    logger.info("="*60 + "\n")
    
    # Save final model
    final_checkpoint = {
        'vae_state_dict': vae.state_dict(),
        'diffusion_state_dict': diffusion_model.state_dict(),
        'consistency_state_dict': consistency_net.state_dict(),
        'metrics': metrics_tracker.metrics,
        'config': config
    }
    
    if ema_wrapper is not None:
        final_checkpoint['ema_state_dict'] = ema_wrapper.get_model().state_dict()
    
    torch.save(final_checkpoint, os.path.join(save_dir, 'checkpoints', 'final_model.pth'))
    
    # Save final metrics
    metrics_tracker.save_metrics(save_dir)
    metrics_tracker.plot_metrics(save_dir)


if __name__ == '__main__':
    if os.name == 'nt':
        import multiprocessing
        multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='3D Slab-based Latent Diffusion Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='D:\\ProjectS\\Dataset')
    parser.add_argument('--save_dir', type=str, default='./experiments/3d_slab_ldm')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 120],
                       help='Slice range [start, end)')
    parser.add_argument('--panorama_type', type=str, default='axial', 
                       choices=['coronal', 'axial', 'mip', 'curved'],
                       help='Panorama extraction type')
    parser.add_argument('--normalize_volumes', action='store_true', default=False,
                       help='Use per-volume normalization')
    parser.add_argument('--max_data_folders', type=int, default=500,
                       help='Maximum number of data folders to use')
    parser.add_argument('--cache_volumes', action='store_true', default=False,
                       help='Enable volume caching')
    parser.add_argument('--no_memmap', action='store_true', default=False,
                       help='Disable memory mapping')
    parser.add_argument('--use_gpu_augmentation', action='store_true', default=False,
                       help='Use GPU-based augmentation with Kornia')
    parser.add_argument('--debug_augmentation', action='store_true', default=False,
                       help='Enable debug logging for augmentation')
    
    # Model arguments
    parser.add_argument('--z_channels', type=int, default=8)
    parser.add_argument('--vae_channels', type=int, default=64)
    parser.add_argument('--diffusion_channels', type=int, default=128)
    parser.add_argument('--cond_channels', type=int, default=512)
    
    # ============================================================
    # FIX: Scale factor 인자 추가
    # ============================================================
    parser.add_argument('--scale_factor', type=float, default=0.18215,
                       help='Latent scaling factor (default: 0.18215, Stable Diffusion standard)')
    parser.add_argument('--auto_scale_factor', action='store_true', default=False,
                       help='Automatically compute scale factor from VAE after training')
    
    # Diffusion arguments
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--prediction_type', type=str, default='v', 
                       choices=['v', 'epsilon', 'x0'],
                       help='Diffusion prediction type')
    parser.add_argument('--use_self_conditioning', action='store_true', default=True,
                       help='Enable self-conditioning')
    parser.add_argument('--self_cond_prob', type=float, default=0.5,
                       help='Self-conditioning probability')
    parser.add_argument('--cfg_dropout_prob', type=float, default=0.1,
                       help='CFG dropout probability')
    parser.add_argument('--guidance_scale', type=float, default=1.5,
                       help='CFG guidance scale for inference')
    parser.add_argument('--l1_weight', type=float, default=0.5)
    parser.add_argument('--gdl_weight', type=float, default=0.5)
    parser.add_argument('--lpips_weight', type=float, default=0.1)
    parser.add_argument('--mid_slice_weight', type=float, default=0.5,
                       help='Weight for middle slice in 3D loss (0.33=equal, 0.5=mid focused)')
    
    # VAE specific
    parser.add_argument('--vae_lpips_weight', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (reduced default for 3D models)')
    parser.add_argument('--vae_epochs', type=int, default=50)
    parser.add_argument('--diffusion_epochs', type=int, default=200)
    parser.add_argument('--consistency_epochs', type=int, default=20)
    parser.add_argument('--consistency_num_slices', type=int, default=8)
    parser.add_argument('--vae_lr', type=float, default=0.00005)
    parser.add_argument('--diffusion_lr', type=float, default=0.00005)
    parser.add_argument('--consistency_lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation (increased default for 3D)')
    
    # Checkpoint arguments
    parser.add_argument('--vae_checkpoint', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--resume_diffusion_from', type=str, default=None,
                       help='Resume diffusion training from checkpoint')
    parser.add_argument('--skip_vae_training', action='store_true')
    parser.add_argument('--skip_consistency_training', action='store_true')
    
    # Advanced training
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--no_amp_vae', action='store_true', default=False)
    parser.add_argument('--use_bfloat16', action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--use_axial_attention', action='store_true', default=False)
    parser.add_argument('--no_elastic', action='store_true', default=False)
    parser.add_argument('--elastic_strength', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=0.8)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    
    # Logging arguments
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    # Config file
    parser.add_argument('--config', type=str, default=None)
    
    # Augmentation
    parser.add_argument('--augment_from_epoch', type=int, default=10,
                       help='Start augmentation from this epoch')
    
    args = parser.parse_args()
    main(args)