"""
Data augmentation utilities for both CPU and GPU
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
import logging

# Try to import Kornia for GPU augmentation
try:
    import kornia
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: Kornia not installed. GPU augmentation disabled. Install with: pip install kornia")

logger = logging.getLogger(__name__)


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
    """Advanced data augmentation for medical images"""
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
    
    def random_flip(self, image, pano):
        """Random horizontal flip"""
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            pano = np.flip(pano, axis=1).copy()
        return image, pano
    
    def random_rotation(self, image, pano):
        """Random rotation"""
        if np.random.rand() > 0.8:
            angle = np.random.uniform(-10, 10)
            image = rotate(image, angle, reshape=False, order=1, mode='constant', cval=image.min())
            pano = rotate(pano, angle, reshape=False, order=1, mode='constant', cval=pano.min())
        return image, pano
    
    def elastic_deformation(self, image, pano, epoch=None):
        """Apply elastic deformation with reduced intensity"""
        # Disable elastic deformation in early epochs
        if epoch is not None and epoch < self.augment_from_epoch:
            return image, pano
            
        if self.use_elastic and np.random.rand() > 0.9:
            image = self.elastic(image)
            pano = self.elastic(pano)
        return image, pano
    
    def gamma_correction(self, image):
        """Random gamma correction"""
        if np.random.rand() > 0.8:
            gamma = np.random.uniform(0.9, 1.1)
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image_norm = (image - img_min) / (img_max - img_min)
                image_gamma = np.power(image_norm, gamma)
                image = image_gamma * (img_max - img_min) + img_min
        return image
    
    def random_noise(self, image):
        """Add random Gaussian noise"""
        if np.random.rand() > 0.9:
            noise_std = np.random.uniform(0.005, 0.02)
            noise = np.random.normal(0, noise_std, image.shape)
            image = image + noise
        return image
    
    def apply(self, image, pano, epoch=None):
        """Apply all augmentations"""
        # Geometric augmentations (apply to both)
        image, pano = self.random_flip(image, pano)
        image, pano = self.random_rotation(image, pano)
        image, pano = self.elastic_deformation(image, pano, epoch)
        
        # Intensity augmentations (apply only to image)
        image = self.gamma_correction(image)
        image = self.random_noise(image)
        
        return image, pano


# ==================== Safe GPU Augmentation Components ====================

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


# ==================== GPU Augmentation Module ====================

class GPUAugmentation(nn.Module):
    """GPU-based augmentation using Kornia - Fixed version with same parameters for all inputs"""
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.device = device
        self.config = config
        self.debug = config.get('debug_augmentation', False)
        self.augment_from_epoch = config.get('augment_from_epoch', 10)
        self.intensity_to_condition = config.get('intensity_to_condition', True)
        
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
        
        # Intensity augmentations (only for CT slices)
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
        
        logger.info(f"GPU Augmentation initialized with {len(geometric_transforms)} geometric and {len(intensity_transforms)} intensity transforms")
    
    @torch.no_grad()
    def forward(self, image, condition=None, epoch=None):
        """Apply GPU augmentation to image and optionally condition - FIXED VERSION"""
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
        
        # Calculate augmentation strength based on epoch (progressive)
        if epoch is not None:
            strength = min(1.0, (epoch - self.augment_from_epoch + 1) / 10.0)  # Ramp up over 10 epochs
        else:
            strength = 1.0
        
        try:
            # Apply geometric augmentations with SAME parameters
            if self.geometric_augmentation is not None and condition is not None:
                # CRITICAL FIX: Always use same parameters for both image and condition
                params = self.geometric_augmentation.forward_parameters(image.shape)
                
                image_geom = self.geometric_augmentation(image, params=params)
                
                # Handle different channel counts by expanding/contracting as needed
                if condition.shape[1] != image.shape[1]:
                    # If condition has different channels, we need to handle it carefully
                    if condition.shape[1] == 3 and image.shape[1] == 1:
                        # For 3-channel condition, apply same transform to each channel
                        condition_geom = self.geometric_augmentation(condition, params=params)
                    else:
                        # General case - apply same spatial transform
                        condition_geom = self.geometric_augmentation(condition, params=params)
                else:
                    condition_geom = self.geometric_augmentation(condition, params=params)
            elif self.geometric_augmentation is not None:
                image_geom = self.geometric_augmentation(image)
                condition_geom = condition
            else:
                image_geom = image
                condition_geom = condition
            
            # Apply intensity augmentations only to the image
            if self.intensity_augmentation is not None:
                # Check for negative values before intensity augmentation
                if self.debug and (image_geom < 0).any():
                    neg_ratio = (image_geom < 0).float().mean().item()
                    logger.debug(f"Negative values before intensity aug: {neg_ratio*100:.2f}%")
                
                # ✅ 파라미터를 한 번만 샘플링
                params_intensity = self.intensity_augmentation.forward_parameters(image_geom.shape)
                
                # ✅ Image에 적용 (같은 params)
                image_aug = self.intensity_augmentation(image_geom, params=params_intensity)

                # ✅ Condition에도 동일한 params 적용
                if condition_geom is not None and self.intensity_to_condition:
                    condition_geom = self.intensity_augmentation(condition_geom, params=params_intensity)
             
            else:
                image_aug = image_geom
            
            # Safety checks and recovery
            # Use nan_to_num for robust handling
            image_aug = torch.nan_to_num(image_aug, nan=0.0, posinf=1.0, neginf=-1.0)
            if condition_geom is not None:
                condition_geom = torch.nan_to_num(condition_geom, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Final clamping
            image_aug = torch.clamp(image_aug, -1.0, 1.0)
            if condition_geom is not None:
                condition_geom = torch.clamp(condition_geom, -1.0, 1.0)
            
            # Debug logging
            if self.debug and epoch is not None and epoch % 10 == 0:
                logger.info(f"Augmentation stats at epoch {epoch}:")
                logger.info(f"  Image range: [{image_aug.min():.4f}, {image_aug.max():.4f}]")
                if condition_geom is not None:
                    logger.info(f"  Condition range: [{condition_geom.min():.4f}, {condition_geom.max():.4f}]")
                logger.info(f"  Augmentation strength: {strength:.2f}")
            
            return image_aug, condition_geom
            
        except Exception as e:
            logger.error(f"Error in GPU augmentation: {str(e)}")
            logger.error(f"Image shape: {image.shape}, Condition shape: {condition.shape if condition is not None else 'None'}")
            logger.error("Returning original inputs")
            return image, condition