"""
General utility functions
"""

import torch
import numpy as np
import random
import os
import logging

logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    # CUBLAS environment variable for deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch 2.0+ reproducibility settings
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except:
            # May not be supported in some environments
            pass


def enable_flash_attention():
    """Enable Flash Attention for faster training if available"""
    try:
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info("Flash Attention enabled for faster training")
            return True
    except:
        pass
    return False


def check_gpu_support():
    """Check GPU capabilities and provide recommendations"""
    if not torch.cuda.is_available():
        return False, None, None
    
    device_capability = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0)
    
    supports_bfloat16 = device_capability[0] >= 8
    
    return True, supports_bfloat16, gpu_name


def load_checkpoint(path, device):
    """Load checkpoint with backward compatibility"""
    return torch.load(path, map_location=device, weights_only=False)
