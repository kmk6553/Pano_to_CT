"""
Dataset and data loading utilities
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
    def __init__(self, root_dir, folders, slice_range=(0, 120), augment=True, 
                 panorama_type='coronal', normalize_volumes=True, augment_config=None,
                 pano_triplet=False, cache_volumes=True, use_memmap=True):
        self.root_dir = root_dir
        self.folders = folders
        self.slice_range = slice_range
        self.augment = augment
        self.panorama_type = panorama_type
        self.normalize_volumes = normalize_volumes
        self.volume_shape = (120, 200, 200)  # Updated for downsampled data size
        self.max_retry_count = 5
        self.current_epoch = 1
        self.pano_triplet = pano_triplet
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
        
        # Calculate total slices
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
                    return np.array(pano_volume[60, :, :])  # Middle slice for 120 depth
                else:
                    return pano_volume[60, :, :]
        elif self.panorama_type == 'coronal':
            if is_memmap:
                return np.array(pano_volume[:, 100, :])  # Middle slice for 200 height
            else:
                return pano_volume[:, 100, :]
        elif self.panorama_type == 'mip':
            if is_memmap:
                return np.max(np.array(pano_volume[:, 75:125, :]), axis=1)  # Center 50 slices
            else:
                return np.max(pano_volume[:, 75:125, :], axis=1)
        elif self.panorama_type == 'curved':
            h, _, w = pano_volume.shape
            panorama = np.zeros((h, w))
            for i in range(w):
                curve_idx = int(100 + 25 * np.sin(2 * np.pi * i / w))  # Adjusted for 200 height
                curve_idx = np.clip(curve_idx, 0, pano_volume.shape[1] - 1)
                if is_memmap:
                    panorama[:, i] = np.array(pano_volume[:, curve_idx, i])
                else:
                    panorama[:, i] = pano_volume[:, curve_idx, i]
            return panorama
        else:
            raise ValueError(f"Unknown panorama type: {self.panorama_type}")
    
    def _get_pano_slice(self, volume_idx, slice_idx):
        """Helper method to get a panorama slice tensor"""
        folder = self.folders[volume_idx]
        pano_path = os.path.join(self.root_dir, folder, 'Pano', 'Pano_Normalized_float32_200x200x120.raw')
        
        if os.path.exists(pano_path):
            try:
                pano_volume = self._get_volume_cached(pano_path, folder + '_pano')
                if pano_volume is not None:
                    if self.normalize_volumes:
                        pano_volume = self.normalize_volume(pano_volume, folder + '_pano')
                    pano_2d = self.extract_panorama(pano_volume, pano_volume, slice_idx)
                else:
                    pano_2d = np.zeros((200, 200), dtype=np.float32)
            except Exception as e:
                logger.warning(f"Error loading panorama for {folder}: {e}, using zeros")
                pano_2d = np.zeros((200, 200), dtype=np.float32)
        else:
            ct_path = os.path.join(self.root_dir, folder, 'CT', 'CT_Normalized_float32_200x200x120.raw')
            try:
                ct_volume = self._get_volume_cached(ct_path, folder)
                if ct_volume is not None:
                    if self.normalize_volumes:
                        ct_volume = self.normalize_volume(ct_volume, folder)
                    pano_2d = self.extract_panorama(ct_volume, ct_volume, slice_idx)
                else:
                    pano_2d = np.zeros((200, 200), dtype=np.float32)
            except:
                pano_2d = np.zeros((200, 200), dtype=np.float32)
        
        pano_tensor = torch.from_numpy(pano_2d).unsqueeze(0).float()
        
        if not self.normalize_volumes:
            pano_tensor = (pano_tensor - 0.5) * 2
        else:
            pano_tensor = pano_tensor / 5.0
        
        return torch.clamp(pano_tensor, -1.0, 1.0)
    
    def _get_item_with_retry(self, idx, retry_count=0):
        """Internal function with retry logic - with slice position"""
        if retry_count >= self.max_retry_count:
            logger.error(f"Max retry count exceeded for index {idx}")
            dummy_slice = np.zeros((200, 200), dtype=np.float32)
            dummy_pano = np.zeros((200, 200), dtype=np.float32)
            
            return {
                'panorama': torch.from_numpy(dummy_pano).unsqueeze(0).float(),
                'condition': torch.stack([torch.from_numpy(dummy_pano).unsqueeze(0).float()] * 3, dim=0).squeeze(1) if self.pano_triplet else torch.from_numpy(dummy_pano).unsqueeze(0).float(),
                'ct_slice': torch.from_numpy(dummy_slice).unsqueeze(0).float(),
                'slice_idx': 0,
                'volume_idx': 0,
                'folder': 'dummy',
                'slice_position': torch.tensor(0.0)
            }
        
        volume_idx = idx // self.slices_per_volume
        slice_idx = idx % self.slices_per_volume + self.slice_range[0]
        
        # Calculate normalized slice position [0, 1]
        slice_position = (slice_idx - self.slice_range[0]) / max(1, self.slices_per_volume - 1)
        slice_position = torch.tensor(slice_position, dtype=torch.float32)
        
        folder = self.folders[volume_idx]
        
        ct_path = os.path.join(self.root_dir, folder, 'CT', 'CT_Normalized_float32_200x200x120.raw')
        
        if not self.cache_volumes:
            ct_slice = self._get_volume_or_slice(ct_path, folder, slice_idx)
            if ct_slice is None:
                logger.error(f"Failed to load CT slice: {ct_path}")
                return self._get_item_with_retry((idx + 1) % len(self), retry_count + 1)
            
            if self.normalize_volumes:
                if folder not in self.volume_stats:
                    sample_slices = []
                    for i in range(0, 120, 15):  # Sample every 15 slices for 120 depth
                        sample = self._read_slice_directly(ct_path, i)
                        if sample is not None:
                            sample_slices.append(sample)
                    if sample_slices:
                        sample_data = np.stack(sample_slices)
                        mean = np.mean(sample_data)
                        std = np.std(sample_data)
                        std = max(std, 0.01)
                        self.volume_stats[folder] = {'mean': mean, 'std': std}
                
                if folder in self.volume_stats:
                    stats = self.volume_stats[folder]
                    ct_slice = (ct_slice - stats['mean']) / stats['std']
                    ct_slice = np.clip(ct_slice, -5.0, 5.0)
        else:
            ct_volume = self._get_volume_cached(ct_path, folder)
            if ct_volume is None:
                logger.error(f"Failed to load CT volume: {ct_path}")
                return self._get_item_with_retry((idx + 1) % len(self), retry_count + 1)
            
            if self.normalize_volumes:
                ct_volume = self.normalize_volume(ct_volume, folder)
            
            if isinstance(ct_volume, np.memmap):
                ct_slice = np.array(ct_volume[slice_idx, :, :])
            else:
                ct_slice = ct_volume[slice_idx, :, :].copy()
        
        pano_path = os.path.join(self.root_dir, folder, 'Pano', 'Pano_Normalized_float32_200x200x120.raw')
        if os.path.exists(pano_path):
            pano_volume = self._get_volume_cached(pano_path, folder + '_pano')
            if pano_volume is not None:
                if self.normalize_volumes:
                    pano_volume = self.normalize_volume(pano_volume, folder + '_pano')
                pano_2d = self.extract_panorama(pano_volume, None, slice_idx)
            else:
                if not self.cache_volumes:
                    ct_volume_temp = self._get_volume_cached(ct_path, folder)
                    if ct_volume_temp is not None:
                        if self.normalize_volumes:
                            ct_volume_temp = self.normalize_volume(ct_volume_temp, folder)
                        pano_2d = self.extract_panorama(ct_volume_temp, ct_volume_temp, slice_idx)
                    else:
                        pano_2d = np.zeros((200, 200), dtype=np.float32)
                else:
                    pano_2d = self.extract_panorama(ct_volume, ct_volume, slice_idx)
        else:
            logger.warning(f"Panorama file not found for {folder}, extracting from CT volume")
            if not self.cache_volumes:
                ct_volume_temp = self._get_volume_cached(ct_path, folder)
                if ct_volume_temp is not None:
                    if self.normalize_volumes:
                        ct_volume_temp = self.normalize_volume(ct_volume_temp, folder)
                    pano_2d = self.extract_panorama(ct_volume_temp, ct_volume_temp, slice_idx)
                else:
                    pano_2d = np.zeros((200, 200), dtype=np.float32)
            else:
                pano_2d = self.extract_panorama(ct_volume, ct_volume, slice_idx)
        
        if self.augment and self.data_augmentation is not None:
            ct_slice, pano_2d = self.data_augmentation.apply(ct_slice, pano_2d, epoch=self.current_epoch)
        
        pano_tensor = torch.from_numpy(pano_2d.copy()).unsqueeze(0).float()
        ct_tensor = torch.from_numpy(ct_slice.copy()).unsqueeze(0).float()
        
        if not self.normalize_volumes:
            pano_tensor = (pano_tensor - 0.5) * 2
            ct_tensor = (ct_tensor - 0.5) * 2
        else:
            pano_tensor = pano_tensor / 5.0
            ct_tensor = ct_tensor / 5.0
        
        pano_tensor = torch.clamp(pano_tensor, -1.0, 1.0)
        ct_tensor = torch.clamp(ct_tensor, -1.0, 1.0)
        
        if torch.isnan(pano_tensor).any() or torch.isnan(ct_tensor).any():
            logger.error(f"NaN in final tensor for {folder}, returning dummy data")
            pano_tensor = torch.zeros_like(pano_tensor)
            ct_tensor = torch.zeros_like(ct_tensor)
        
        if self.pano_triplet:
            prev_idx = max(self.slice_range[0], slice_idx - 1)
            next_idx = min(self.slice_range[1] - 1, slice_idx + 1)
            
            pano_prev = self._get_pano_slice(volume_idx, prev_idx)
            pano_curr = pano_tensor
            pano_next = self._get_pano_slice(volume_idx, next_idx)
            
            condition = torch.cat([pano_prev, pano_curr, pano_next], dim=0)
        else:
            condition = pano_tensor
        
        return {
            'panorama': pano_tensor,
            'condition': condition,
            'ct_slice': ct_tensor,
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