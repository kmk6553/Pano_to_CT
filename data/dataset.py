"""
Dataset and data loading utilities for 3D Slab-based generation
Loads 3-slice CT windows: [B, 1, D=3, H, W]
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import logging
import gc
from .augmentation import DataAugmentation

logger = logging.getLogger(__name__)


class OptimizedDentalSliceDataset(Dataset):
    """
    Dataset for 3D Slab-based CT generation
    
    Returns:
        ct_volume: [1, D=3, H, W] - 3-slice CT window (prev, mid, next)
        condition: [3, H, W] or [1, 3, H, W] - 3-channel panorama condition
        slice_position: float in [0, 1] - normalized position of middle slice
    """
    def __init__(self, root_dir, folders, slice_range=(0, 120), augment=True, 
                 panorama_type='axial', normalize_volumes=True, augment_config=None,
                 pano_triplet=True, cache_volumes=True, use_memmap=True):
        self.root_dir = root_dir
        self.folders = folders
        self.slice_range = slice_range
        self.augment = augment
        self.panorama_type = panorama_type
        self.normalize_volumes = normalize_volumes
        self.volume_shape = (120, 200, 200)  # (D, H, W) for raw volume
        self.max_retry_count = 5
        self.current_epoch = 1
        self.pano_triplet = pano_triplet  # Always True for 3D slab
        self.cache_volumes = cache_volumes
        self.use_memmap = use_memmap
        
        # Volume cache using memory mapping
        self._volume_cache = {}
        self._memmap_cache = {}
        
        # Data augmentation (CPU only)
        if augment and augment_config:
            self.data_augmentation = DataAugmentation(augment_config)
        else:
            self.data_augmentation = None
        
        # Calculate total slices (we generate windows, so effective range is smaller)
        # Each sample is centered at a middle slice, so we need at least 1 slice on each side
        self.slices_per_volume = slice_range[1] - slice_range[0]
        self.total_slices = len(folders) * self.slices_per_volume
        
        # Cache for volume statistics (for normalization)
        self.volume_stats = {}
        
        # Windows multiprocessing compatibility
        if os.name == 'nt':
            self.persistent_cache = False
        else:
            self.persistent_cache = True
    
    def _get_volume_memmap(self, path):
        """Get memory-mapped volume for efficient access"""
        if not self.cache_volumes and path in self._memmap_cache:
            return self._memmap_cache[path]
        
        if path not in self._memmap_cache:
            if os.path.exists(path):
                try:
                    memmap = np.memmap(
                        path, 
                        dtype=np.float32, 
                        mode='r',
                        shape=self.volume_shape,
                        order='C'
                    )
                    if self.cache_volumes:
                        self._memmap_cache[path] = memmap
                    return memmap
                except Exception as e:
                    logger.warning(f"Failed to create memmap for {path}: {e}")
                    data = np.fromfile(path, dtype=np.float32)
                    return data.reshape(self.volume_shape, order='C')
            else:
                logger.error(f"Volume file not found: {path}")
                return None
        
        return self._memmap_cache[path]
    
    def _get_volume_cached(self, path, folder):
        """Get volume with caching"""
        cache_key = f"{folder}_{path}"
        
        if self.cache_volumes and cache_key in self._volume_cache:
            return self._volume_cache[cache_key]
        
        volume = self._get_volume_memmap(path)
        
        if volume is None:
            return None
        
        if self.cache_volumes:
            volume = np.array(volume, copy=True)
            if self.persistent_cache:
                self._volume_cache[cache_key] = volume
        
        return volume
    
    def _read_slice_directly(self, path, slice_idx):
        """Read a single slice directly from file without loading entire volume"""
        if not os.path.exists(path):
            logger.error(f"Volume file not found: {path}")
            return None
        
        try:
            slice_size = self.volume_shape[1] * self.volume_shape[2]
            offset = slice_idx * slice_size * 4
            
            with open(path, 'rb') as f:
                f.seek(offset)
                slice_data = np.fromfile(f, dtype=np.float32, count=slice_size)
                slice_data = slice_data.reshape(self.volume_shape[1], self.volume_shape[2])
            
            return slice_data
        except Exception as e:
            logger.error(f"Error reading slice from {path}: {e}")
            return None
    
    def _get_volume_or_slice(self, path, folder, slice_idx=None):
        """Get volume or specific slice based on caching settings"""
        if slice_idx is None:
            return self._get_volume_cached(path, folder)
        
        if self.cache_volumes:
            volume = self._get_volume_cached(path, folder)
            if volume is not None:
                if isinstance(volume, np.memmap):
                    return np.array(volume[slice_idx, :, :])
                else:
                    return volume[slice_idx, :, :]
            return None
        
        if not self.use_memmap:
            return self._read_slice_directly(path, slice_idx)
        else:
            volume = self._get_volume_memmap(path)
            if volume is not None:
                return np.array(volume[slice_idx, :, :])
            return None
    
    def _get_3slice_window(self, path, folder, center_idx):
        """
        Get 3-slice window centered at center_idx
        
        Args:
            path: path to volume file
            folder: folder name for caching
            center_idx: center slice index
        
        Returns:
            [3, H, W] numpy array - (prev, mid, next) slices
        """
        # Calculate indices with boundary clamping
        prev_idx = max(self.slice_range[0], center_idx - 1)
        next_idx = min(self.slice_range[1] - 1, center_idx + 1)
        
        # Get slices
        prev_slice = self._get_volume_or_slice(path, folder, prev_idx)
        mid_slice = self._get_volume_or_slice(path, folder, center_idx)
        next_slice = self._get_volume_or_slice(path, folder, next_idx)
        
        if prev_slice is None or mid_slice is None or next_slice is None:
            return None
        
        # Stack to [3, H, W]
        window = np.stack([prev_slice, mid_slice, next_slice], axis=0)
        return window
    
    def set_epoch(self, epoch):
        """Set current epoch for conditional augmentation"""
        self.current_epoch = epoch
    
    def __len__(self):
        return self.total_slices
    
    def get_volume_stats(self, volume, folder):
        """Get or compute volume statistics for normalization"""
        if folder not in self.volume_stats:
            mean = np.mean(volume)
            std = np.std(volume)
            std = max(std, 0.01)
            self.volume_stats[folder] = {'mean': mean, 'std': std}
        return self.volume_stats[folder]
    
    def normalize_slice(self, slice_data, folder):
        """Normalize slice using per-volume statistics with outlier handling"""
        if self.normalize_volumes and folder in self.volume_stats:
            stats = self.volume_stats[folder]
            slice_data = (slice_data - stats['mean']) / stats['std']
            slice_data = np.clip(slice_data, -5.0, 5.0)
        return slice_data
    
    def normalize_volume(self, volume, folder):
        """Normalize volume using per-volume statistics with outlier handling"""
        if self.normalize_volumes:
            stats = self.get_volume_stats(volume, folder)
            volume = (volume - stats['mean']) / stats['std']
            volume = np.clip(volume, -5.0, 5.0)
        return volume
    
    def safe_normalize_tensor(self, tensor):
        """Safe tensor normalization to [-1, 1] range"""
        if torch.isnan(tensor).any():
            logger.warning("NaN detected in tensor, replacing with zeros")
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        
        if torch.isinf(tensor).any():
            logger.warning("Inf detected in tensor, clamping")
            tensor = torch.clamp(tensor, -1e6, 1e6)
        
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        if tensor_max - tensor_min < 1e-6:
            return torch.zeros_like(tensor)
        else:
            tensor = 2.0 * (tensor - tensor_min) / (tensor_max - tensor_min) - 1.0
            return torch.clamp(tensor, -1.0, 1.0)
    
    def extract_panorama(self, pano_volume, ct_volume, slice_idx=None):
        """Extract panorama from panorama volume or CT volume"""
        is_memmap = isinstance(pano_volume, np.memmap)
        
        if self.panorama_type == 'axial':
            if slice_idx is not None:
                if is_memmap:
                    return np.array(pano_volume[slice_idx, :, :])
                else:
                    return pano_volume[slice_idx, :, :]
            else:
                if is_memmap:
                    return np.array(pano_volume[60, :, :])
                else:
                    return pano_volume[60, :, :]
        elif self.panorama_type == 'coronal':
            if is_memmap:
                return np.array(pano_volume[:, 100, :])
            else:
                return pano_volume[:, 100, :]
        elif self.panorama_type == 'mip':
            if is_memmap:
                return np.max(np.array(pano_volume[:, 75:125, :]), axis=1)
            else:
                return np.max(pano_volume[:, 75:125, :], axis=1)
        elif self.panorama_type == 'curved':
            h, _, w = pano_volume.shape
            panorama = np.zeros((h, w))
            for i in range(w):
                curve_idx = int(100 + 25 * np.sin(2 * np.pi * i / w))
                curve_idx = np.clip(curve_idx, 0, pano_volume.shape[1] - 1)
                if is_memmap:
                    panorama[:, i] = np.array(pano_volume[:, curve_idx, i])
                else:
                    panorama[:, i] = pano_volume[:, curve_idx, i]
            return panorama
        else:
            raise ValueError(f"Unknown panorama type: {self.panorama_type}")
    
    def _get_pano_3window(self, volume_idx, center_idx):
        """
        Get 3-slice panorama window
        
        Returns:
            [3, H, W] numpy array - panorama at (prev, mid, next) positions
        """
        folder = self.folders[volume_idx]
        pano_path = os.path.join(self.root_dir, folder, 'Pano', 'Pano_Normalized_float32_200x200x120.raw')
        ct_path = os.path.join(self.root_dir, folder, 'CT', 'CT_Normalized_float32_200x200x120.raw')
        
        # Calculate indices with boundary clamping
        prev_idx = max(self.slice_range[0], center_idx - 1)
        next_idx = min(self.slice_range[1] - 1, center_idx + 1)
        
        pano_slices = []
        
        for idx in [prev_idx, center_idx, next_idx]:
            if os.path.exists(pano_path):
                try:
                    pano_volume = self._get_volume_cached(pano_path, folder + '_pano')
                    if pano_volume is not None:
                        if self.normalize_volumes:
                            if folder + '_pano' not in self.volume_stats:
                                self.get_volume_stats(pano_volume, folder + '_pano')
                        pano_2d = self.extract_panorama(pano_volume, pano_volume, idx)
                        if self.normalize_volumes:
                            pano_2d = self.normalize_slice(pano_2d, folder + '_pano')
                    else:
                        pano_2d = np.zeros((200, 200), dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Error loading panorama for {folder}: {e}, using zeros")
                    pano_2d = np.zeros((200, 200), dtype=np.float32)
            else:
                # Fall back to CT volume for panorama extraction
                try:
                    ct_volume = self._get_volume_cached(ct_path, folder)
                    if ct_volume is not None:
                        if self.normalize_volumes:
                            if folder not in self.volume_stats:
                                self.get_volume_stats(ct_volume, folder)
                        pano_2d = self.extract_panorama(ct_volume, ct_volume, idx)
                        if self.normalize_volumes:
                            pano_2d = self.normalize_slice(pano_2d, folder)
                    else:
                        pano_2d = np.zeros((200, 200), dtype=np.float32)
                except:
                    pano_2d = np.zeros((200, 200), dtype=np.float32)
            
            pano_slices.append(pano_2d)
        
        # Stack to [3, H, W]
        return np.stack(pano_slices, axis=0)
    
    def _get_item_with_retry(self, idx, retry_count=0):
        """Internal function with retry logic - returns 3D slab data"""
        if retry_count >= self.max_retry_count:
            logger.error(f"Max retry count exceeded for index {idx}")
            # Return dummy 3D data
            dummy_ct = np.zeros((3, 200, 200), dtype=np.float32)
            dummy_pano = np.zeros((3, 200, 200), dtype=np.float32)
            
            return {
                'condition': torch.from_numpy(dummy_pano).float(),  # [3, H, W]
                'ct_volume': torch.from_numpy(dummy_ct).unsqueeze(0).float(),  # [1, 3, H, W]
                'slice_idx': 0,
                'volume_idx': 0,
                'folder': 'dummy',
                'slice_position': torch.tensor(0.0)
            }
        
        volume_idx = idx // self.slices_per_volume
        slice_idx = idx % self.slices_per_volume + self.slice_range[0]
        
        # Calculate normalized slice position [0, 1] for the CENTER slice
        slice_position = (slice_idx - self.slice_range[0]) / max(1, self.slices_per_volume - 1)
        slice_position = torch.tensor(slice_position, dtype=torch.float32)
        
        folder = self.folders[volume_idx]
        ct_path = os.path.join(self.root_dir, folder, 'CT', 'CT_Normalized_float32_200x200x120.raw')
        
        # Initialize volume stats if needed
        if self.normalize_volumes and folder not in self.volume_stats:
            ct_volume = self._get_volume_cached(ct_path, folder)
            if ct_volume is not None:
                self.get_volume_stats(ct_volume, folder)
        
        # Get 3-slice CT window
        ct_window = self._get_3slice_window(ct_path, folder, slice_idx)
        if ct_window is None:
            logger.error(f"Failed to load CT window: {ct_path}, slice {slice_idx}")
            return self._get_item_with_retry((idx + 1) % len(self), retry_count + 1)
        
        # Normalize if needed
        if self.normalize_volumes and folder in self.volume_stats:
            stats = self.volume_stats[folder]
            ct_window = (ct_window - stats['mean']) / stats['std']
            ct_window = np.clip(ct_window, -5.0, 5.0)
        
        # Get 3-slice panorama condition
        pano_window = self._get_pano_3window(volume_idx, slice_idx)
        
        # Apply augmentation (same transform to all 3 slices)
        if self.augment and self.data_augmentation is not None:
            # Augment each slice pair consistently
            for i in range(3):
                ct_window[i], pano_window[i] = self.data_augmentation.apply(
                    ct_window[i], pano_window[i], epoch=self.current_epoch
                )
        
        # Convert to tensors
        # CT: [3, H, W] -> [1, 3, H, W] (add channel dim)
        ct_tensor = torch.from_numpy(ct_window.copy()).unsqueeze(0).float()  # [1, 3, H, W]
        # Panorama: [3, H, W] stays as is for condition
        pano_tensor = torch.from_numpy(pano_window.copy()).float()  # [3, H, W]
        
        # Scale to [-1, 1] range
        if not self.normalize_volumes:
            pano_tensor = (pano_tensor - 0.5) * 2
            ct_tensor = (ct_tensor - 0.5) * 2
        else:
            pano_tensor = pano_tensor / 5.0
            ct_tensor = ct_tensor / 5.0
        
        pano_tensor = torch.clamp(pano_tensor, -1.0, 1.0)
        ct_tensor = torch.clamp(ct_tensor, -1.0, 1.0)
        
        # Check for NaN
        if torch.isnan(pano_tensor).any() or torch.isnan(ct_tensor).any():
            logger.error(f"NaN in final tensor for {folder}, returning dummy data")
            pano_tensor = torch.zeros_like(pano_tensor)
            ct_tensor = torch.zeros_like(ct_tensor)
        
        return {
            'condition': pano_tensor,      # [3, H, W] - 3-channel panorama
            'ct_volume': ct_tensor,        # [1, 3, H, W] - 3-slice CT volume
            'ct_slice': ct_tensor[:, 1, :, :],  # [1, H, W] - middle slice only (for compatibility)
            'slice_idx': slice_idx,
            'volume_idx': volume_idx,
            'folder': folder,
            'slice_position': slice_position
        }
    
    def __getitem__(self, idx):
        """Public interface with retry logic"""
        return self._get_item_with_retry(idx)
    
    def cleanup_cache(self):
        """Clean up memory mapped files and caches"""
        self._volume_cache.clear()
        
        for path, memmap in list(self._memmap_cache.items()):
            try:
                del self._memmap_cache[path]
            except:
                pass
        
        self._memmap_cache.clear()
        gc.collect()


class DataPrefetcher:
    """Asynchronous data prefetcher for GPU"""
    def __init__(self, loader, device='cuda'):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()
    
    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        
        with torch.cuda.stream(self.stream):
            self.batch = {k: v.pin_memory().to(self.device, non_blocking=True) 
                         if isinstance(v, torch.Tensor) else v 
                         for k, v in self.batch.items()}
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        
        if batch is None:
            raise StopIteration
        
        self.preload()
        return batch
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()