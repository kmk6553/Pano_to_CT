"""
General utility functions

FIXES APPLIED:
1. [v5.12] load_checkpoint - CPU로 먼저 로드 후 필요한 것만 GPU로 전송
2. [v5.12] 명시적 메모리 해제 함수 추가
"""

import torch
import numpy as np
import random
import os
import logging
import gc

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


def load_checkpoint(path, device=None):
    """
    Load checkpoint with memory-efficient loading
    
    [FIX v5.12] CPU로 먼저 로드하여 GPU 메모리 폭증 방지
    
    Args:
        path: checkpoint file path
        device: target device (deprecated, always loads to CPU first)
    
    Returns:
        checkpoint dict (on CPU)
    """
    logger.info(f"Loading checkpoint from: {path}")
    
    # 항상 CPU로 먼저 로드 (GPU 메모리 절약)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    logger.info(f"Checkpoint loaded to CPU successfully")
    
    return checkpoint


def load_state_dict_to_model(model, state_dict, strict=True):
    """
    [v5.12] state_dict를 모델에 로드하고 원본 state_dict 메모리 해제
    
    Args:
        model: target model (already on GPU)
        state_dict: state dict from checkpoint (on CPU)
        strict: whether to strictly enforce matching keys
    
    Returns:
        None (모델에 직접 로드)
    """
    model.load_state_dict(state_dict, strict=strict)
    
    # state_dict 메모리 해제
    del state_dict
    gc.collect()


def cleanup_checkpoint(checkpoint):
    """
    [v5.12] 체크포인트 딕셔너리 메모리 해제
    
    Args:
        checkpoint: checkpoint dictionary to clean up
    """
    if checkpoint is None:
        return
    
    # 모든 텐서 참조 해제
    for key in list(checkpoint.keys()):
        if isinstance(checkpoint[key], dict):
            for sub_key in list(checkpoint[key].keys()):
                del checkpoint[key][sub_key]
        del checkpoint[key]
    
    del checkpoint
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Checkpoint memory cleaned up")


def log_gpu_memory(tag=""):
    """
    [v5.12] GPU 메모리 사용량 로깅
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"[GPU Memory {tag}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")


def force_cuda_cleanup():
    """
    [v5.12] 강제 CUDA 메모리 정리
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()