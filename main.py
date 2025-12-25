"""
Enhanced Pseudo-3D Latent Diffusion Model with Phase 1 Improvements
Main training script with v-param, self-conditioning, CFG, multi-scale conditioning, and position embedding
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
from models import VAE, ConditionalUNet, ConsistencyNet3D, EMAWrapper
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
                'pano_triplet': args.pano_triplet,
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
                'cond_channels': args.cond_channels
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
                'diffusion_lr': args.diffusion_lr
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
                'warmup_epochs': 5
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
        
        if gpu_memory < 8:
            logger.warning(f"Low GPU memory detected ({gpu_memory:.1f}GB). Consider reducing batch size or using gradient accumulation.")
        elif gpu_memory < 16:
            logger.info(f"Medium GPU memory detected ({gpu_memory:.1f}GB). Current settings should work well.")
        else:
            logger.info(f"High GPU memory detected ({gpu_memory:.1f}GB). You can increase batch size for faster training.")
    
    if config['training']['detect_anomaly']:
        torch.autograd.set_detect_anomaly(True)
        logger.warning("Anomaly detection enabled. This will slow down training but help identify NaN sources.")
    
    # Log optimization settings
    logger.info("\n" + "="*60)
    logger.info("Performance Optimizations Enabled:")
    logger.info("="*60)
    logger.info(f"- Volume caching: {config['data']['cache_volumes']}")
    from data.augmentation import KORNIA_AVAILABLE
    logger.info(f"- GPU augmentation: {config['data']['use_gpu_augmentation'] and KORNIA_AVAILABLE}")
    logger.info(f"- Safe gamma correction: Enabled")
    logger.info(f"- Async data prefetching: Enabled")
    logger.info(f"- Memory-mapped I/O: {config['data']['use_memmap']}")
    logger.info(f"- Flash Attention: {flash_enabled}")
    logger.info(f"- Gradient accumulation: {config['training']['gradient_accumulation_steps']} steps")
    logger.info(f"- Augmentation starts from epoch: {config['data']['augment_from_epoch']}")
    logger.info("="*60 + "\n")
    
    # Log Phase 1 improvements
    logger.info("\n" + "="*60)
    logger.info("PHASE 1 IMPROVEMENTS APPLIED:")
    logger.info("="*60)
    logger.info("1. Multi-scale Spatial Condition Injection")
    logger.info("   - 4 resolution levels with spatial features")
    logger.info("   - Direct spatial information preservation")
    logger.info("2. Slice Position Embedding")
    logger.info("   - Z-axis awareness for depth consistency")
    logger.info(f"3. v-Parametrization (prediction_type={config['diffusion']['prediction_type']})")
    logger.info("   - More stable training dynamics")
    logger.info(f"4. Self-Conditioning (enabled={config['diffusion']['use_self_conditioning']})")
    logger.info(f"   - Probability: {config['diffusion']['self_cond_prob']}")
    logger.info(f"5. Classifier-Free Guidance")
    logger.info(f"   - Dropout prob: {config['diffusion']['cfg_dropout_prob']}")
    logger.info(f"   - Guidance scale: {config['diffusion']['guidance_scale']}")
    logger.info("="*60 + "\n")
    
    if config['data']['pano_triplet']:
        logger.info("3-Channel Panorama Condition ENABLED")
        logger.info("Using previous, current, and next panorama slices as condition")
    else:
        logger.info("Single-Channel Panorama Condition")
    
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
        if use_amp_fp16:
            logger.info(f"- Initial scale: {2.**6}")
            logger.info(f"- Growth interval: 5000 steps")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
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
    logger.info(f"Effective batch size: {config['training']['batch_size']} Ã— {accumulation_steps} = {config['training']['batch_size'] * accumulation_steps}")
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Create models
    vae = VAE(
        in_channels=1, 
        z_channels=config['model']['z_channels'], 
        channels=config['model']['vae_channels']
    ).to(device)
    
    diffusion_model = ConditionalUNet(
        in_channels=config['model']['z_channels'], 
        out_channels=config['model']['z_channels'],
        channels=config['model']['diffusion_channels'],
        cond_channels=config['model']['cond_channels'],
        panorama_type=config['data']['panorama_type'],
        pano_triplet=config['data']['pano_triplet'],
        use_self_conditioning=config['diffusion']['use_self_conditioning']
    ).to(device)
    
    consistency_net = ConsistencyNet3D(
        in_channels=1, 
        features=32, 
        use_axial_attention=args.use_axial_attention
    ).to(device)
    
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
        logger.info(f"EMA enabled with decay={ema_decay} (effective batch size: {effective_batch_size})")
    
    # Create diffusion process with v-parametrization
    diffusion_process = DiffusionProcess(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule=config['diffusion']['schedule'],
        prediction_type=config['diffusion']['prediction_type']
    ).to(device)
    
    logger.info(f"Diffusion process initialized with {config['diffusion']['prediction_type']} parametrization")
    
    # Create loss functions
    l1_loss_fn = CharbonnierLoss(epsilon=1e-2).to(device)
    gdl_loss_fn = GradientDifferenceLoss().to(device)
    lpips_loss_fn = LPIPSLoss(net='vgg', device=device) if config['vae']['lpips_weight'] > 0 or config['diffusion']['lpips_weight'] > 0 else None
    
    # GPU augmentation
    gpu_augmenter = None
    if config['data']['use_gpu_augmentation'] and KORNIA_AVAILABLE:
        gpu_augmenter = GPUAugmentation(config['data']['augmentation'], device).to(device)
        logger.info("GPU augmentation module initialized with FIXED parameter sharing")
    
    # Create datasets
    train_dataset = OptimizedDentalSliceDataset(
        config['data']['data_dir'], 
        train_folders, 
        slice_range=config['data']['slice_range'],
        augment=True,
        panorama_type=config['data']['panorama_type'],
        normalize_volumes=config['data']['normalize_volumes'],
        augment_config=config['data']['augmentation'],
        pano_triplet=config['data']['pano_triplet'],
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
        pano_triplet=config['data']['pano_triplet'],
        cache_volumes=config['data']['cache_volumes'],
        use_memmap=config['data'].get('use_memmap', True)
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)} slices")
    logger.info(f"Val dataset size: {len(val_dataset)} slices")
    
    # Windows multiprocessing check
    if os.name == 'nt' and config['training']['num_workers'] > 0:
        logger.warning(f"Windows detected with num_workers={config['training']['num_workers']}.")
        logger.warning("This may cause multiprocessing issues. If you encounter errors, consider setting --num_workers 0")
        logger.warning("Make sure this script is run with if __name__ == '__main__': guard.")
    
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
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory > 16 and accumulation_steps > 1:
            logger.info(f"GPU memory {gpu_memory:.1f}GB detected. Consider increasing batch_size and reducing gradient_accumulation_steps for faster training.")
    
    # Create optimizers
    vae_optimizer = optim.AdamW(vae.parameters(), lr=config['training']['vae_lr'], weight_decay=1e-4)
    diffusion_optimizer = optim.AdamW(diffusion_model.parameters(), 
                                     lr=config['diffusion']['diffusion_lr'],
                                     weight_decay=1e-4)
    consistency_optimizer = optim.AdamW(consistency_net.parameters(), lr=config['training']['consistency_lr'], weight_decay=1e-4)
    
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
        
        if 'scheduler_state_dict' in checkpoint and config['training']['use_scheduler'] and vae_scheduler is not None:
            try:
                if isinstance(checkpoint['scheduler_state_dict'], dict) and 'current_epoch' in checkpoint['scheduler_state_dict']:
                    vae_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info(f"Restored scheduler state: epoch {vae_scheduler.current_epoch}")
            except Exception as e:
                logger.warning(f"Could not restore scheduler state: {e}")
    
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = load_checkpoint(args.resume_from_checkpoint, device)
        
        vae.load_state_dict(checkpoint['vae_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            vae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Restored optimizer state")
        
        if 'scheduler_state_dict' in checkpoint and vae_scheduler is not None:
            try:
                if isinstance(checkpoint['scheduler_state_dict'], dict):
                    vae_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info(f"Restored scheduler state")
            except Exception as e:
                logger.warning(f"Could not restore scheduler state: {e}")
        
        if 'metrics' in checkpoint:
            metrics_tracker.metrics = checkpoint['metrics']
            logger.info("Restored metrics history")
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.warning("No epoch information in checkpoint, starting from epoch 1")
        
        if start_epoch > config['training']['vae_epochs']:
            logger.info(f"VAE training already completed (epoch {start_epoch-1}/{config['training']['vae_epochs']})")
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
        
        if 'metrics' in diffusion_checkpoint:
            metrics_tracker.metrics = diffusion_checkpoint['metrics']
            logger.info("Restored metrics history")
        
        if 'nan_epochs' in diffusion_checkpoint:
            nan_epochs = diffusion_checkpoint.get('nan_epochs', [])
            logger.info(f"Restored NaN tracking history: {len(nan_epochs)} epochs with NaN issues")
    
    # Setup diffusion scheduler with warmup
    if config['training']['warmup_steps'] > 0:
        steps_per_epoch = len(train_loader) // accumulation_steps
        total_steps = config['training']['diffusion_epochs'] * steps_per_epoch
        
        remaining_epochs = config['training']['diffusion_epochs'] - diffusion_start_epoch + 1
        remaining_steps = remaining_epochs * steps_per_epoch
        
        completed_steps = (diffusion_start_epoch - 1) * steps_per_epoch
        
        if diffusion_start_epoch > 1:
            warmup_steps = 0
            logger.info(f"Resuming diffusion training - warmup disabled")
        else:
            warmup_steps = min(5000, total_steps // 10)
            logger.info(f"Diffusion warmup: {warmup_steps} optimizer steps")
        
        logger.info(f"Steps per epoch (with accumulation): {steps_per_epoch}")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Completed steps: {completed_steps}")
        logger.info(f"Remaining steps: {remaining_steps}")
        
        diffusion_scheduler = WarmupCosineScheduler(
            diffusion_optimizer, 
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-6,
            base_lr=config['diffusion']['diffusion_lr']
        )

        if diffusion_start_epoch > 1:
            diffusion_scheduler.current_step = completed_steps
            diffusion_scheduler.last_epoch = completed_steps
            logger.info(f"Scheduler step adjusted to {completed_steps}")
        
        if config.get('resume_diffusion_from'):
            if 'scheduler_state_dict' in diffusion_checkpoint and diffusion_checkpoint['scheduler_state_dict'] is not None:
                try:
                    diffusion_scheduler.load_state_dict(diffusion_checkpoint['scheduler_state_dict'])
                    logger.info("Diffusion scheduler state loaded")
                except Exception as e:
                    logger.warning(f"Could not restore diffusion scheduler state: {e}")
    else:
        diffusion_scheduler = CosineAnnealingLR(diffusion_optimizer, T_max=config['training']['diffusion_epochs']) if config['training']['use_scheduler'] else None
    
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
        logger.info(f"Diffusion Scaler: init_scale={2.**6}, growth_interval=5000")
        
        if use_amp_vae:
            vae_scaler = GradScaler(
                init_scale=2.**4,
                growth_interval=10000,
                growth_factor=1.2,
                backoff_factor=0.5,
                enabled=True
            )
            logger.info("VAE GradScaler enabled with conservative settings")
            logger.info(f"VAE Scaler: init_scale={2.**4}, growth_interval=10000")
        else:
            logger.info("VAE AMP disabled for stability (--no_amp_vae flag is set)")
            vae_scaler = None
    elif use_bf16:
        logger.info("Using BFloat16 - GradScaler not needed")
        scaler = None
        vae_scaler = None
    
    # Training Phase 1: VAE
    if not config['training']['skip_vae_training']:
        logger.info("\n" + "="*60)
        if start_epoch > 1:
            logger.info(f"=== Phase 1: Resuming VAE Training from Epoch {start_epoch} ===")
        else:
            logger.info("=== Phase 1: Training VAE ===")
        logger.info("="*60)
        logger.info(f"Initial learning rate: {config['vae']['initial_lr']}")
        logger.info(f"Target learning rate: {config['vae']['target_lr']}")
        logger.info(f"Total epochs: {config['training']['vae_epochs']}")
        if start_epoch > 1:
            logger.info(f"Starting from epoch: {start_epoch}")
            logger.info(f"Remaining epochs: {config['training']['vae_epochs'] - start_epoch + 1}")
        logger.info(f"Batch size: {config['training']['batch_size']}")
        logger.info(f"Effective batch size: {config['training']['batch_size'] * accumulation_steps}")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Batches per epoch: {len(train_loader)}")
        logger.info(f"Optimizer steps per epoch: {len(train_loader) // accumulation_steps}")
        logger.info(f"AMP enabled: {use_amp_vae}")
        if use_amp_vae:
            logger.info(f"VAE Scaler initial scale: {2.**4}")
            logger.info(f"VAE Scaler growth interval: 10000")
        logger.info(f"Gradient clipping: {config['training']['grad_clip']}")
        logger.info(f"KL warmup epochs: 60")
        logger.info("="*60 + "\n")
        
        gradient_issue_count = 0
        max_gradient_issues = 10
        epoch_overflow_stats = []
        
        if args.resume_from_checkpoint and 'epoch_overflow_stats' in checkpoint:
            epoch_overflow_stats = checkpoint.get('epoch_overflow_stats', [])
            logger.info(f"Restored {len(epoch_overflow_stats)} epochs of overflow statistics")
        
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
            
            if hasattr(metrics_tracker, 'epoch_logs') and metrics_tracker.epoch_logs:
                last_log = metrics_tracker.epoch_logs[-1]
                overflow_count = last_log.get('overflow_count', 0)
                epoch_overflow_stats.append((epoch, overflow_count))
                
                if len(epoch_overflow_stats) >= 3:
                    recent_overflows = [stat[1] for stat in epoch_overflow_stats[-3:]]
                    overflow_rate = overflow_count / len(train_loader) * 100 if len(train_loader) > 0 else 0
                    
                    if all(o > 0 for o in recent_overflows) and recent_overflows[-1] > recent_overflows[-2]:
                        logger.warning(f"Overflow increasing trend detected: {recent_overflows} (current rate: {overflow_rate:.2f}%)")
                        
                        for param_group in vae_optimizer.param_groups:
                            param_group['lr'] *= 0.8
                        logger.warning(f"Reduced learning rate to {vae_optimizer.param_groups[0]['lr']:.6f}")
                        
                        if vae_scaler is not None and hasattr(vae_scaler, '_scale'):
                            current_scale = vae_scaler._scale.item()
                            vae_scaler._scale = torch.tensor(current_scale * 0.5, device=device)
                            logger.warning(f"Reduced VAE scaler scale from {current_scale:.2f} to {vae_scaler._scale.item():.2f}")
                    
                    elif overflow_rate >= 1.0:
                        logger.warning(f"High overflow rate detected: {overflow_rate:.2f}% at epoch {epoch}")
                        if vae_scaler is not None:
                            logger.warning("Consider using --no_amp_vae flag for more stable training")
            
            if np.isnan(vae_loss) or np.isinf(vae_loss):
                gradient_issue_count += 1
                logger.error(f"Training instability detected at epoch {epoch}. Issue count: {gradient_issue_count}/{max_gradient_issues}")
                
                if gradient_issue_count >= max_gradient_issues:
                    logger.error("Too many gradient issues. Stopping VAE training.")
                    break
                
                for param_group in vae_optimizer.param_groups:
                    param_group['lr'] *= 0.5
                logger.warning(f"Reduced learning rate to {vae_optimizer.param_groups[0]['lr']:.6f}")
                
                continue
            else:
                gradient_issue_count = 0
            
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
                    'epoch_overflow_stats': epoch_overflow_stats
                }
                
                if vae_scheduler is not None:
                    if isinstance(vae_scheduler, AdaptiveLearningRateScheduler):
                        checkpoint_dict['scheduler_state_dict'] = vae_scheduler.state_dict()
                    else:
                        checkpoint_dict['scheduler_state_dict'] = vae_scheduler.state_dict() if hasattr(vae_scheduler, 'state_dict') else None
                
                torch.save(checkpoint_dict, os.path.join(save_dir, 'checkpoints', f'vae_epoch_{epoch}.pth'))
            
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                if hasattr(train_dataset, 'cleanup_cache') and epoch % 20 == 0:
                    train_dataset.cleanup_cache()
                    logger.info("Cleaned up dataset cache")
        
        if epoch_overflow_stats:
            logger.info("\n" + "="*60)
            logger.info("VAE Training Overflow Statistics Summary:")
            logger.info("="*60)
            total_overflows = sum(stat[1] for stat in epoch_overflow_stats)
            total_batches = len(train_loader) * len(epoch_overflow_stats)
            logger.info(f"Total gradient overflows: {total_overflows} / {total_batches} batches ({total_overflows/total_batches*100:.3f}%)")
            logger.info("Per-epoch breakdown:")
            for ep, count in epoch_overflow_stats[-10:]:
                if count > 0:
                    logger.info(f"  Epoch {ep}: {count} overflows ({count/len(train_loader)*100:.2f}%)")
            
            avg_overflow_rate = total_overflows / total_batches * 100
            if avg_overflow_rate > 0.1:
                logger.info("\nRecommendations:")
                if use_amp_vae:
                    logger.info("  - Consider using --no_amp_vae flag for more stable training")
                logger.info("  - Reduce learning rate further")
                logger.info("  - Increase gradient clipping strength")
            
            logger.info("="*60 + "\n")
    else:
        logger.info("\n=== Skipping VAE Training (Using loaded checkpoint) ===")
    
    # Training Phase 2: Diffusion
    logger.info("\n" + "="*60)
    if diffusion_start_epoch > 1:
        logger.info(f"=== Phase 2: Resuming Diffusion Training from Epoch {diffusion_start_epoch} ===")
    else:
        logger.info("=== Phase 2: Training Diffusion Model with Phase 1 Improvements ===")
    logger.info("="*60)
    logger.info(f"Total epochs: {config['training']['diffusion_epochs']}")
    if diffusion_start_epoch > 1:
        logger.info(f"Starting from epoch: {diffusion_start_epoch}")
        logger.info(f"Remaining epochs: {config['training']['diffusion_epochs'] - diffusion_start_epoch + 1}")
    logger.info(f"Learning rate: {config['diffusion']['diffusion_lr']}")
    logger.info(f"Warmup steps: {warmup_steps if 'warmup_steps' in locals() else config['training']['warmup_steps']}")
    logger.info(f"Prediction type: {config['diffusion']['prediction_type']}")
    logger.info(f"Self-conditioning: {config['diffusion']['use_self_conditioning']} (prob={config['diffusion']['self_cond_prob']})")
    logger.info(f"CFG dropout: {config['diffusion']['cfg_dropout_prob']}")
    logger.info(f"CFG guidance scale: {config['diffusion']['guidance_scale']}")
    logger.info(f"Loss weights - L1: {config['diffusion']['l1_weight']}, GDL: {config['diffusion']['gdl_weight']}, LPIPS: {config['diffusion']['lpips_weight']}")
    logger.info(f"EMA enabled: {config['training']['use_ema']}")
    logger.info(f"3-Channel Panorama Condition: {config['data']['pano_triplet']}")
    logger.info(f"AMP: {'BFloat16' if use_bf16 else 'Float16' if use_amp else 'Disabled'}")
    logger.info(f"Gradient clipping: {config['training']['grad_clip']}")
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
        
        if hasattr(metrics_tracker, 'epoch_logs') and metrics_tracker.epoch_logs:
            last_log = metrics_tracker.epoch_logs[-1]
            nan_count = last_log.get('nan_count', 0)
            if nan_count > 0:
                nan_epochs.append((epoch, nan_count))
                
                if len(nan_epochs) >= 3 and all(e[1] > 0 for e in nan_epochs[-3:]):
                    logger.error(f"Persistent NaN issues detected in epochs {[e[0] for e in nan_epochs[-3:]]}")
                    
                    for param_group in diffusion_optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    logger.warning(f"Reduced learning rate to {diffusion_optimizer.param_groups[0]['lr']:.6f}")
                    
                    if scaler is not None and not use_bf16:
                        logger.error("Consider using --use_bfloat16 flag or disabling AMP for better stability")
        
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
                if hasattr(train_dataset, 'cleanup_cache') and epoch % 20 == 0:
                    train_dataset.cleanup_cache()
                    logger.info("Cleaned up dataset cache")
    
    # Training Phase 3: 3D Consistency
    if not config['training']['skip_consistency_training']:
        logger.info("\n" + "="*60)
        logger.info("=== Phase 3: Training 3D Consistency ===")
        logger.info("="*60)
        logger.info(f"Total epochs: {config['training']['consistency_epochs']}")
        logger.info(f"Learning rate: {config['training']['consistency_lr']}")
        logger.info(f"Number of slices: {config['training']['consistency_num_slices']}")
        logger.info(f"Teacher forcing ratio: {config['training']['teacher_forcing_ratio']}")
        logger.info(f"Axial attention enabled: {args.use_axial_attention}")
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
            
            if epoch % config['logging']['plot_interval'] == 0:
                metrics_tracker.plot_interim_metrics(save_dir, 'consistency', epoch)
            
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
                
                if epoch % 5 == 0:
                    torch.cuda.empty_cache()
                    if hasattr(train_dataset, 'cleanup_cache'):
                        train_dataset.cleanup_cache()
                        logger.info("Cleaned up dataset cache")
    else:
        logger.info("\n=== Skipping 3D Consistency Training ===")
    
    # Final cleanup
    if hasattr(train_dataset, 'cleanup_cache'):
        train_dataset.cleanup_cache()
        logger.info("Final dataset cache cleanup")
    if hasattr(val_dataset, 'cleanup_cache'):
        val_dataset.cleanup_cache()
    
    # Training completed
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info("="*60)
    logger.info(f"Output directory: {save_dir}")
    logger.info(f"Final model saved at: {os.path.join(save_dir, 'checkpoints', 'final_model.pth')}")
    logger.info(f"Training logs saved at: {os.path.join(save_dir, 'training_logs.csv')}")
    logger.info(f"Metrics saved at: {os.path.join(save_dir, 'metrics.json')}")
    logger.info(f"Plots saved at: {os.path.join(save_dir, 'training_metrics_final.png')}")
    
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
    
    parser = argparse.ArgumentParser(description='Enhanced Pseudo-3D Latent Diffusion Model - Phase 1 Improvements')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='D:\\ProjectS\\Dataset')
    parser.add_argument('--save_dir', type=str, default='./experiments/enhanced_ldm_phase1')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--slice_range', type=int, nargs=2, default=[0, 240],
                       help='Slice range [start, end) - end is exclusive')
    parser.add_argument('--panorama_type', type=str, default='axial', 
                       choices=['coronal', 'axial', 'mip', 'curved'],
                       help='Type of panorama extraction')
    parser.add_argument('--normalize_volumes', action='store_true', default=False,
                       help='Use per-volume normalization')
    parser.add_argument('--max_data_folders', type=int, default=500,
                       help='Maximum number of data folders to use')
    parser.add_argument('--pano_triplet', action='store_true', default=False,
                       help='Use 3-channel panorama condition')
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
    
    # Diffusion arguments
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--prediction_type', type=str, default='v', 
                       choices=['v', 'epsilon', 'x0'],
                       help='Diffusion prediction type: v-param (recommended), epsilon, or x0')
    parser.add_argument('--use_self_conditioning', action='store_true', default=True,
                       help='Enable self-conditioning in diffusion model')
    parser.add_argument('--self_cond_prob', type=float, default=0.5,
                       help='Probability of using self-conditioning during training (0.0-1.0)')
    parser.add_argument('--cfg_dropout_prob', type=float, default=0.1,
                       help='Probability of dropping condition for CFG training (0.0-1.0)')
    parser.add_argument('--guidance_scale', type=float, default=1.5,
                       help='Classifier-free guidance scale for inference (1.0=no guidance)')
    parser.add_argument('--l1_weight', type=float, default=0.5)
    parser.add_argument('--gdl_weight', type=float, default=0.5)
    parser.add_argument('--lpips_weight', type=float, default=0.1)
    
    # VAE specific
    parser.add_argument('--vae_lpips_weight', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--vae_epochs', type=int, default=50)
    parser.add_argument('--diffusion_epochs', type=int, default=200)
    parser.add_argument('--consistency_epochs', type=int, default=20)
    parser.add_argument('--consistency_num_slices', type=int, default=16)
    parser.add_argument('--vae_lr', type=float, default=0.00005)
    parser.add_argument('--diffusion_lr', type=float, default=0.00005)
    parser.add_argument('--consistency_lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    # Checkpoint arguments
    parser.add_argument('--vae_checkpoint', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--resume_diffusion_from', type=str, default=None,
                       help='Resume diffusion training from specific checkpoint')
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
                       help='Start augmentation from this epoch (default: 10)')
    
    args = parser.parse_args()
    main(args)