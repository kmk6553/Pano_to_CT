"""
Data augmentation utilities for both CPU and GPU

FIXES APPLIED:
1. GT CT(Target)에는 Intensity Augmentation 적용 금지
2. Geometric 변환은 유지하되, 밝기/노이즈 변환은 Condition(Pano)에만 적용
3. [FIX v5.8] Kornia 미설치 시 Import 에러 수정 - 클래스 정의를 조건문 안으로 이동
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
import logging

logger = logging.getLogger(__name__)

# ============================================================
# [FIX v5.8] Kornia Import 방어 코드 개선
# 기존: K가 정의되지 않은 상태에서 상속 시도하여 NameError 발생
# 수정: 클래스 정의 전체를 if KORNIA_AVAILABLE 블록 안으로 이동
# ============================================================
try:
    import kornia
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    K = None  # Placeholder
    logger.warning("Kornia not installed. GPU augmentation disabled. Install with: pip install kornia")


class ElasticTransform:
    """Elastic deformation of images with optional affine transformation"""
    def __init__(self, alpha, sigma, alpha_affine, random_state=None):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.random_state = random_state
    
    def __call__(self, image):
        if self.random_state is None:
            random_state = np.random.RandomState(None)
        else:
            random_state = self.random_state
        
        shape = image.shape
        shape_size = shape[:2]
        
        # 1. Affine transformation (rotation, scale, shear)
        if self.alpha_affine > 0:
            center = np.array(shape_size) / 2.
            
            # Generate random affine parameters
            angle = random_state.uniform(-self.alpha_affine, self.alpha_affine) * np.pi / 180
            scale = random_state.uniform(1 - self.alpha_affine/100, 1 + self.alpha_affine/100)
            shear = random_state.uniform(-self.alpha_affine/100, self.alpha_affine/100)
            
            # Construct affine transformation matrix
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            # Rotation + scale + shear matrix
            affine_matrix = np.array([
                [scale * cos_angle - shear * sin_angle, -scale * sin_angle - shear * cos_angle],
                [scale * sin_angle + shear * cos_angle, scale * cos_angle - shear * sin_angle]
            ])
            
            # Transform around center
            y, x = np.mgrid[0:shape_size[0], 0:shape_size[1]]
            indices = np.stack([y - center[0], x - center[1]], axis=0).reshape(2, -1)
            transformed = affine_matrix @ indices
            transformed[0] += center[0]
            transformed[1] += center[1]
            
            # Apply affine transformation to image
            image = map_coordinates(image, transformed.reshape(2, *shape_size), order=1, mode='reflect')
        
        # 2. Elastic deformation (non-linear distortion)
        dx = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((random_state.rand(*shape_size) * 2 - 1), self.sigma) * self.alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


class DataAugmentation:
    """
    Advanced data augmentation for medical images (CPU-based)
    
    [FIX v5.8] 3-slice window에 동일한 augmentation 파라미터 적용을 위한
               apply_consistent() 메서드 추가
    """
    def __init__(self, config):
        self.config = config
        self.use_elastic = config.get('use_elastic', True)
        self.augment_from_epoch = config.get('augment_from_epoch', 10)
        
        if self.use_elastic:
            self.elastic = ElasticTransform(
                alpha=config.get('elastic_alpha', 10),
                sigma=config.get('elastic_sigma', 4),
                alpha_affine=config.get('elastic_alpha_affine', 10)
            )
    
    def random_flip(self, image, pano, do_flip=None):
        """Random horizontal flip with optional fixed parameter"""
        if do_flip is None:
            do_flip = np.random.rand() > 0.5
        
        if do_flip:
            image = np.flip(image, axis=1).copy()
            pano = np.flip(pano, axis=1).copy()
        return image, pano
    
    def random_rotation(self, image, pano, angle=None, do_rotate=None):
        """Random rotation with optional fixed parameters"""
        if do_rotate is None:
            do_rotate = np.random.rand() > 0.8
        
        if do_rotate:
            if angle is None:
                angle = np.random.uniform(-10, 10)
            image = rotate(image, angle, reshape=False, order=1, mode='constant', cval=image.min())
            pano = rotate(pano, angle, reshape=False, order=1, mode='constant', cval=pano.min())
        return image, pano
    
    def elastic_deformation(self, image, pano, epoch=None, do_elastic=None, random_state=None):
        """Apply elastic deformation with reduced intensity"""
        # Disable elastic deformation in early epochs
        if epoch is not None and epoch < self.augment_from_epoch:
            return image, pano
        
        if do_elastic is None:
            do_elastic = np.random.rand() > 0.9
            
        if self.use_elastic and do_elastic:
            # Use consistent random state if provided
            if random_state is not None:
                elastic = ElasticTransform(
                    alpha=self.config.get('elastic_alpha', 10),
                    sigma=self.config.get('elastic_sigma', 4),
                    alpha_affine=self.config.get('elastic_alpha_affine', 10),
                    random_state=random_state
                )
                image = elastic(image)
                pano = elastic(pano)
            else:
                image = self.elastic(image)
                pano = self.elastic(pano)
        return image, pano
    
    def gamma_correction(self, image, gamma=None):
        """Random gamma correction - [수정]: Pano에만 적용"""
        if gamma is None:
            if np.random.rand() > 0.8:
                gamma = np.random.uniform(0.9, 1.1)
            else:
                return image
        
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image_norm = (image - img_min) / (img_max - img_min)
            image_gamma = np.power(image_norm, gamma)
            image = image_gamma * (img_max - img_min) + img_min
        return image
    
    def random_noise(self, image, noise_std=None):
        """Add random Gaussian noise - [수정]: Pano에만 적용"""
        if noise_std is None:
            if np.random.rand() > 0.9:
                noise_std = np.random.uniform(0.005, 0.02)
            else:
                return image
        
        noise = np.random.normal(0, noise_std, image.shape)
        image = image + noise
        return image
    
    def apply(self, image, pano, epoch=None):
        """
        Apply all augmentations (legacy method for backward compatibility)
        
        WARNING: 이 메서드는 각 호출마다 다른 랜덤 파라미터를 사용합니다.
                 3-slice window에 일관된 augmentation을 적용하려면
                 apply_consistent() 또는 sample_params() + apply_with_params()를 사용하세요.
        
        [수정]: Intensity augmentation (gamma, noise)는 Pano에만 적용
        Target CT는 기하학적 변환만 유지
        """
        # Geometric augmentations (apply to both)
        image, pano = self.random_flip(image, pano)
        image, pano = self.random_rotation(image, pano)
        image, pano = self.elastic_deformation(image, pano, epoch)
        
        # [수정]: Intensity augmentations - Pano에만 적용, Target CT는 제외
        pano = self.gamma_correction(pano)
        pano = self.random_noise(pano)
        
        return image, pano
    
    def sample_params(self, epoch=None):
        """
        [FIX v5.8] 3-slice window에 일관된 augmentation을 위한 파라미터 샘플링
        
        Returns:
            dict: augmentation 파라미터들
        """
        params = {
            'do_flip': np.random.rand() > 0.5,
            'do_rotate': np.random.rand() > 0.8,
            'rotation_angle': np.random.uniform(-10, 10),
            'do_elastic': np.random.rand() > 0.9 if (epoch is None or epoch >= self.augment_from_epoch) else False,
            'elastic_seed': np.random.randint(0, 2**31),
            'do_gamma': np.random.rand() > 0.8,
            'gamma_value': np.random.uniform(0.9, 1.1),
            'do_noise': np.random.rand() > 0.9,
            'noise_std': np.random.uniform(0.005, 0.02)
        }
        return params
    
    def apply_with_params(self, image, pano, params, epoch=None):
        """
        [FIX v5.8] 미리 샘플링된 파라미터로 augmentation 적용
        
        Args:
            image: CT slice
            pano: Panorama slice
            params: sample_params()에서 반환된 파라미터 dict
            epoch: 현재 epoch (elastic deformation 조건용)
        
        Returns:
            augmented (image, pano) tuple
        """
        # Geometric augmentations (apply to both)
        image, pano = self.random_flip(image, pano, do_flip=params['do_flip'])
        image, pano = self.random_rotation(image, pano, 
                                           angle=params['rotation_angle'],
                                           do_rotate=params['do_rotate'])
        
        if params['do_elastic']:
            random_state = np.random.RandomState(params['elastic_seed'])
            image, pano = self.elastic_deformation(image, pano, epoch=epoch,
                                                   do_elastic=True,
                                                   random_state=random_state)
        
        # Intensity augmentations - Pano에만 적용
        if params['do_gamma']:
            pano = self.gamma_correction(pano, gamma=params['gamma_value'])
        
        if params['do_noise']:
            pano = self.random_noise(pano, noise_std=params['noise_std'])
        
        return image, pano


# ============================================================
# [FIX v5.8] GPU Augmentation Components - Kornia 조건부 정의
# 기존: try-except 후 클래스 정의가 전역에 있어 NameError 발생
# 수정: KORNIA_AVAILABLE 조건 안에서만 클래스 정의
# ============================================================

if KORNIA_AVAILABLE:
    class SafeRandomGamma(K.RandomGamma):
        """Safe RandomGamma that handles negative values in [-1, 1] range"""
        def apply_transform(self, input: torch.Tensor, params, flags, transform=None):
            # Convert from [-1, 1] to [0, 1] for gamma correction
            input_shifted = (input + 1.0) * 0.5
            
            # Apply gamma correction
            output_shifted = super().apply_transform(input_shifted, params, flags, transform)
            
            # Convert back to [-1, 1]
            output = output_shifted * 2.0 - 1.0
            
            return output


    class StableElasticTransform(K.RandomElasticTransform):
        """Stable elastic transform with safety constraints"""
        def __init__(self, *args, **kwargs):
            # Override default parameters for stability
            kwargs['align_corners'] = kwargs.get('align_corners', True)
            kwargs['padding_mode'] = kwargs.get('padding_mode', 'reflection')
            super().__init__(*args, **kwargs)
        
        def apply_transform(self, input: torch.Tensor, params, flags, transform=None):
            # Apply transform
            output = super().apply_transform(input, params, flags, transform)
            
            # Ensure output is in valid range
            output = torch.clamp(output, -1.0, 1.0)
            
            return output

else:
    # Placeholder classes when Kornia is not available
    class SafeRandomGamma:
        """Placeholder when Kornia is not available"""
        def __init__(self, *args, **kwargs):
            raise ImportError("Kornia is required for SafeRandomGamma. Install with: pip install kornia")
    
    class StableElasticTransform:
        """Placeholder when Kornia is not available"""
        def __init__(self, *args, **kwargs):
            raise ImportError("Kornia is required for StableElasticTransform. Install with: pip install kornia")


class GPUAugmentation(nn.Module):
    """
    GPU-based augmentation using Kornia
    
    FIXES APPLIED:
    1. Target(image)에는 Intensity Augmentation 적용 금지
    2. Geometric 변환만 Target에 적용
    3. Intensity 변환은 Condition(Pano)에만 적용
    4. [FIX v5.8] Kornia 미설치 시 graceful fallback
    """
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.device = device
        self.config = config
        self.debug = config.get('debug_augmentation', False)
        self.augment_from_epoch = config.get('augment_from_epoch', 10)
        
        if not KORNIA_AVAILABLE:
            self.use_gpu_aug = False
            logger.warning("Kornia not available, GPU augmentation disabled")
            return
        
        self.use_gpu_aug = True
        
        # Build augmentation pipeline
        geometric_transforms = []
        
        # Geometric augmentations
        if config.get('random_flip', True):
            geometric_transforms.append(K.RandomHorizontalFlip(p=0.5, same_on_batch=True))
        
        if config.get('random_rotation', True):
            geometric_transforms.append(K.RandomAffine(
                degrees=10, 
                p=0.2,
                align_corners=True,
                padding_mode='reflection',
                same_on_batch=True
            ))
        
        # Elastic deformation with safety constraints
        if config.get('use_elastic', True) and config.get('elastic_alpha', 10) > 0:
            # Conservative parameters for stability
            elastic_alpha = min(config.get('elastic_alpha', 10) * 0.3, 5.0)
            elastic_sigma = config.get('elastic_sigma', 4)
            
            geometric_transforms.append(StableElasticTransform(
                alpha=(elastic_alpha, elastic_alpha),
                sigma=(elastic_sigma, elastic_sigma),
                p=0.03,  # Low probability
                same_on_batch=True
            ))
        
        # Geometric augmentation pipeline
        self.geometric_augmentation = K.AugmentationSequential(
            *geometric_transforms,
            data_keys=['input'],
            same_on_batch=True,
            keepdim=True
        ).to(device) if geometric_transforms else None
        
        # Intensity augmentations (only for Condition/Pano, NOT for Target/CT)
        intensity_transforms = []
        
        # Safe gamma correction
        if config.get('gamma_correction', True):
            intensity_transforms.append(SafeRandomGamma(gamma=(0.9, 1.1), p=0.2))
        
        # Random noise with conservative parameters
        if config.get('random_noise', True):
            intensity_transforms.append(K.RandomGaussianNoise(mean=0., std=0.01, p=0.05))
        
        self.intensity_augmentation = K.AugmentationSequential(
            *intensity_transforms,
            data_keys=['input'],
            same_on_batch=True,
            keepdim=True
        ).to(device) if intensity_transforms else None        

        # For progressive augmentation
        self.base_elastic_alpha = config.get('elastic_alpha', 10)
        self.base_noise_std = 0.01
        
        logger.info(f"GPU Augmentation initialized with {len(geometric_transforms)} geometric transforms")
        logger.info(f"Intensity transforms ({len(intensity_transforms)} total) will be applied to CONDITION ONLY")
    
    @torch.no_grad()
    def forward(self, image, condition=None, epoch=None):
        """
        Apply GPU augmentation to image and optionally condition
        
        [수정]: Target(image)에는 Geometric 변환만 적용
                Intensity Augmentation은 Condition(Pano)에만 적용
        
        Args:
            image: [B, C, H, W] or [B, D, H, W] - Target CT (D=3 for 3-slice)
            condition: [B, 3, H, W] - Condition Pano (3-channel)
            epoch: Current epoch for progressive augmentation
        
        Returns:
            image_aug: Augmented target (geometric only)
            condition_aug: Augmented condition (geometric + intensity)
        """
        if not self.use_gpu_aug:
            return image, condition
        
        # Skip augmentation in very early epochs
        if epoch is not None and epoch < self.augment_from_epoch:
            return image, condition
        
        # Ensure inputs are on GPU and in valid range
        image = image.to(self.device)
        image = torch.clamp(image, -1.0, 1.0)
        
        if condition is not None:
            condition = condition.to(self.device)
            condition = torch.clamp(condition, -1.0, 1.0)
        
        try:
            # ============================================================
            # [수정]: Geometric augmentation - 동일한 파라미터로 둘 다 적용
            # ============================================================
            if self.geometric_augmentation is not None and condition is not None:
                # Sample parameters once
                params = self.geometric_augmentation.forward_parameters(image.shape)
                
                # Apply geometric transform to target (CT)
                image_geom = self.geometric_augmentation(image, params=params)
                
                # Apply same geometric transform to condition (Pano)
                condition_geom = self.geometric_augmentation(condition, params=params)
                
            elif self.geometric_augmentation is not None:
                image_geom = self.geometric_augmentation(image)
                condition_geom = condition
            else:
                image_geom = image
                condition_geom = condition
            
            # ============================================================
            # [수정]: Intensity augmentation - Condition(Pano)에만 적용
            # Target(CT)에는 Intensity Augmentation 적용하지 않음!
            # ============================================================
            image_aug = image_geom  # Target은 geometric만 적용
            
            if self.intensity_augmentation is not None and condition_geom is not None:
                # Intensity는 Condition에만 적용
                params_intensity = self.intensity_augmentation.forward_parameters(condition_geom.shape)
                condition_aug = self.intensity_augmentation(condition_geom, params=params_intensity)
            else:
                condition_aug = condition_geom
            
            # Safety checks and recovery
            image_aug = torch.nan_to_num(image_aug, nan=0.0, posinf=1.0, neginf=-1.0)
            if condition_aug is not None:
                condition_aug = torch.nan_to_num(condition_aug, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Final clamping
            image_aug = torch.clamp(image_aug, -1.0, 1.0)
            if condition_aug is not None:
                condition_aug = torch.clamp(condition_aug, -1.0, 1.0)
            
            # Debug logging
            if self.debug and epoch is not None and epoch % 10 == 0:
                logger.info(f"Augmentation stats at epoch {epoch}:")
                logger.info(f"  Image (Target) range: [{image_aug.min():.4f}, {image_aug.max():.4f}] (Geometric only)")
                if condition_aug is not None:
                    logger.info(f"  Condition (Pano) range: [{condition_aug.min():.4f}, {condition_aug.max():.4f}] (Geometric + Intensity)")
            
            return image_aug, condition_aug
            
        except Exception as e:
            logger.error(f"Error in GPU augmentation: {str(e)}")
            logger.error(f"Image shape: {image.shape}, Condition shape: {condition.shape if condition is not None else 'None'}")
            logger.error("Returning original inputs")
            return image, condition