"""
Dataset and data loading utilities for 3D Slab-based generation
Loads 3-slice CT windows: [B, 1, D=3, H, W]

FIXES APPLIED:
1. 이중 정규화 문제 완전 수정 (v5.7)
   - normalize_volumes=True: [0,1] -> [-1,1] 변환 수행
   - normalize_volumes=False: 변환 없음 (데이터가 이미 [-1,1]이라고 가정)
   - 하단의 중복 정규화 블록 완전 제거
2. 디버그용 min/max 로깅 추가 (선택적)
3. [FIX v5.8] Slice별 독립 Augmentation 문제 수정
   - 기존: 3장의 슬라이스에 각각 별도로 Augmentation 적용 (Z축 일관성 파괴)
   - 수정: 동일한 Augmentation 파라미터를 3장 모두에 적용
4. [FIX v5.10] Memmap 일관성 수정
   - 기존: CT는 no_memmap 적용되지만 Pano는 여전히 memmap 사용 가능
   - 수정: use_memmap 플래그가 CT와 Pano 모두에 일관되게 적용
   - _get_pano_slice_directly() 메서드 추가
   - memmap 핸들 관리 개선 (cache_volumes와 무관하게 핸들 유지)
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
    
    Normalization Policy (v5.7):
        - normalize_volumes=True: 데이터가 [0, 1] 범위라고 가정하고 [-1, 1]로 변환
        - normalize_volumes=False: 데이터가 이미 [-1, 1] 범위라고 가정하고 변환 없음
    
    Augmentation Policy (v5.8):
        - 3장의 슬라이스에 동일한 Augmentation 파라미터 적용
        - Z축 일관성 보존
    
    I/O Policy (v5.10):
        - use_memmap=True: Memory-mapped I/O 사용 (빠르지만 파일 핸들 소비)
        - use_memmap=False: 직접 파일 읽기 (느리지만 안정적)
        - CT와 Pano 모두에 동일하게 적용
    """
    def __init__(self, root_dir, folders, slice_range=(0, 120), augment=True, 
                 panorama_type='axial', normalize_volumes=True, augment_config=None,
                 pano_triplet=True, cache_volumes=True, use_memmap=True,
                 debug_normalization=False):
        """
        Args:
            root_dir: Root directory containing data folders
            folders: List of folder names to use
            slice_range: (start, end) slice indices
            augment: Whether to apply CPU augmentation
            panorama_type: Type of panorama extraction ('axial', 'coronal', 'mip', 'curved')
            normalize_volumes: If True, apply [0,1] -> [-1,1] scaling
                              If False, assume data is already in [-1,1] range
            augment_config: Configuration for data augmentation
            pano_triplet: Always True for 3D slab
            cache_volumes: Whether to cache volumes in memory
            use_memmap: Whether to use memory-mapped files (applies to both CT and Pano)
            debug_normalization: If True, log min/max values for debugging
        """
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
        self.debug_normalization = debug_normalization
        
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
        
        # Cache for volume statistics (DEPRECATED - kept for compatibility only)
        self.volume_stats = {}
        
        # Windows multiprocessing compatibility
        if os.name == 'nt':
            self.persistent_cache = False
        else:
            self.persistent_cache = True
        
        # Log normalization policy clearly
        if normalize_volumes:
            logger.info("="*60)
            logger.info("NORMALIZATION POLICY: normalize_volumes=True")
            logger.info("  - Input data expected range: [0, 1]")
            logger.info("  - Output data range: [-1, 1]")
            logger.info("  - Transformation: x' = x * 2 - 1")
            logger.info("="*60)
        else:
            logger.info("="*60)
            logger.info("NORMALIZATION POLICY: normalize_volumes=False")
            logger.info("  - Input data expected range: [-1, 1]")
            logger.info("  - Output data range: [-1, 1] (no transformation)")
            logger.info("  - Transformation: None (passthrough)")
            logger.info("="*60)
        
        # [FIX v5.8] Log augmentation policy
        if augment and augment_config:
            logger.info("="*60)
            logger.info("AUGMENTATION POLICY (v5.8):")
            logger.info("  - 3-slice window에 동일한 파라미터 적용")
            logger.info("  - Z축 일관성 보존")
            logger.info("  - [v5.10] CT/Pano 정합성이 보장된 elastic deformation")
            logger.info("="*60)
        
        # [FIX v5.10] Log I/O policy
        logger.info("="*60)
        logger.info(f"I/O POLICY (v5.10): use_memmap={use_memmap}")
        if use_memmap:
            logger.info("  - Memory-mapped I/O 사용 (CT and Pano)")
            logger.info("  - 빠르지만 파일 핸들 소비")
        else:
            logger.info("  - 직접 파일 읽기 사용 (CT and Pano)")
            logger.info("  - 느리지만 안정적 (Windows 호환성 향상)")
        logger.info(f"  - cache_volumes={cache_volumes}")
        logger.info("="*60)
    
    def _get_volume_memmap(self, path):
        """
        Get memory-mapped volume for efficient access
        
        [FIX v5.10] memmap 핸들을 항상 캐시에 유지 (cache_volumes와 무관)
        이렇게 하면 매 샘플마다 memmap 객체를 새로 만드는 것을 방지
        """
        if path in self._memmap_cache:
            return self._memmap_cache[path]
        
        if os.path.exists(path):
            try:
                memmap = np.memmap(
                    path, 
                    dtype=np.float32, 
                    mode='r',
                    shape=self.volume_shape,
                    order='C'
                )
                # [FIX v5.10] 항상 캐시에 저장 (파일 핸들 재사용)
                self._memmap_cache[path] = memmap
                return memmap
            except Exception as e:
                logger.warning(f"Failed to create memmap for {path}: {e}")
                # Fallback to direct read
                data = np.fromfile(path, dtype=np.float32)
                return data.reshape(self.volume_shape, order='C')
        else:
            logger.error(f"Volume file not found: {path}")
            return None
    
    def _get_volume_cached(self, path, folder):
        """Get volume with caching"""
        cache_key = f"{folder}_{path}"
        
        if self.cache_volumes and cache_key in self._volume_cache:
            return self._volume_cache[cache_key]
        
        # [FIX v5.10] use_memmap 플래그에 따라 분기
        if self.use_memmap:
            volume = self._get_volume_memmap(path)
        else:
            # 직접 파일 읽기
            volume = self._read_volume_directly(path)
        
        if volume is None:
            return None
        
        if self.cache_volumes:
            # memmap이면 복사해서 캐시 (numpy array로 변환)
            if isinstance(volume, np.memmap):
                volume = np.array(volume, copy=True)
            if self.persistent_cache:
                self._volume_cache[cache_key] = volume
        
        return volume
    
    def _read_volume_directly(self, path):
        """
        [FIX v5.10] 전체 볼륨을 직접 읽기 (memmap 사용 안 함)
        """
        if not os.path.exists(path):
            logger.error(f"Volume file not found: {path}")
            return None
        
        try:
            data = np.fromfile(path, dtype=np.float32)
            expected_size = self.volume_shape[0] * self.volume_shape[1] * self.volume_shape[2]
            
            if data.size != expected_size:
                logger.warning(f"Volume size mismatch: expected {expected_size}, got {data.size}")
                # Try to reshape anyway
                inferred_slices = data.size // (self.volume_shape[1] * self.volume_shape[2])
                return data.reshape(inferred_slices, self.volume_shape[1], self.volume_shape[2])
            
            return data.reshape(self.volume_shape, order='C')
        except Exception as e:
            logger.error(f"Error reading volume from {path}: {e}")
            return None
    
    def _read_slice_directly(self, path, slice_idx):
        """Read a single slice directly from file without loading entire volume"""
        if not os.path.exists(path):
            logger.error(f"Volume file not found: {path}")
            return None
        
        try:
            slice_size = self.volume_shape[1] * self.volume_shape[2]
            offset = slice_idx * slice_size * 4  # 4 bytes per float32
            
            with open(path, 'rb') as f:
                f.seek(offset)
                slice_data = np.fromfile(f, dtype=np.float32, count=slice_size)
                slice_data = slice_data.reshape(self.volume_shape[1], self.volume_shape[2])
            
            return slice_data
        except Exception as e:
            logger.error(f"Error reading slice from {path}: {e}")
            return None
    
    def _get_volume_or_slice(self, path, folder, slice_idx=None):
        """
        Get volume or specific slice based on caching settings
        
        [FIX v5.10] use_memmap 플래그를 일관되게 적용
        """
        if slice_idx is None:
            return self._get_volume_cached(path, folder)
        
        # 캐시된 볼륨이 있으면 사용
        if self.cache_volumes:
            volume = self._get_volume_cached(path, folder)
            if volume is not None:
                if isinstance(volume, np.memmap):
                    return np.array(volume[slice_idx, :, :])
                else:
                    return volume[slice_idx, :, :]
            return None
        
        # 캐시 없이 직접 읽기
        if self.use_memmap:
            volume = self._get_volume_memmap(path)
            if volume is not None:
                return np.array(volume[slice_idx, :, :])
            return None
        else:
            # [FIX v5.10] memmap 사용 안 함 - 직접 슬라이스 읽기
            return self._read_slice_directly(path, slice_idx)
    
    def _get_pano_slice_directly(self, path, slice_idx):
        """
        [FIX v5.10] Pano 슬라이스를 직접 읽기 (memmap 사용 안 함)
        CT의 _read_slice_directly와 동일한 로직
        """
        return self._read_slice_directly(path, slice_idx)
    
    def _get_pano_volume_or_slice(self, path, folder, slice_idx=None):
        """
        [FIX v5.10] Pano 볼륨 또는 슬라이스 읽기
        CT와 동일한 use_memmap 정책 적용
        """
        cache_key = folder + '_pano'
        
        if slice_idx is None:
            return self._get_volume_cached(path, cache_key)
        
        # 캐시된 볼륨이 있으면 사용
        if self.cache_volumes:
            volume = self._get_volume_cached(path, cache_key)
            if volume is not None:
                if isinstance(volume, np.memmap):
                    return np.array(volume[slice_idx, :, :])
                else:
                    return volume[slice_idx, :, :]
            return None
        
        # 캐시 없이 직접 읽기
        if self.use_memmap:
            volume = self._get_volume_memmap(path)
            if volume is not None:
                return np.array(volume[slice_idx, :, :])
            return None
        else:
            # [FIX v5.10] memmap 사용 안 함 - 직접 슬라이스 읽기
            return self._get_pano_slice_directly(path, slice_idx)
    
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
        """
        Get or compute volume statistics for normalization
        DEPRECATED: Kept for backward compatibility but not used for actual normalization
        """
        if folder not in self.volume_stats:
            mean = np.mean(volume)
            std = np.std(volume)
            std = max(std, 0.01)
            self.volume_stats[folder] = {'mean': mean, 'std': std}
        return self.volume_stats[folder]
    
    def normalize_slice(self, slice_data, folder):
        """
        Normalize slice from [0, 1] to [-1, 1]
        
        This function is ONLY called when normalize_volumes=True.
        Assumes input data is in [0, 1] range.
        
        Args:
            slice_data: numpy array in [0, 1] range
            folder: folder name (unused, kept for compatibility)
        
        Returns:
            numpy array in [-1, 1] range
        """
        # Transform: [0, 1] -> [-1, 1]
        slice_data = (slice_data * 2.0) - 1.0
        slice_data = np.clip(slice_data, -1.0, 1.0)
        return slice_data
    
    def normalize_volume(self, volume, folder):
        """
        Normalize volume from [0, 1] to [-1, 1]
        
        This function is ONLY called when normalize_volumes=True.
        Assumes input data is in [0, 1] range.
        
        Args:
            volume: numpy array in [0, 1] range
            folder: folder name (unused, kept for compatibility)
        
        Returns:
            numpy array in [-1, 1] range
        """
        # Transform: [0, 1] -> [-1, 1]
        volume = (volume * 2.0) - 1.0
        volume = np.clip(volume, -1.0, 1.0)
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
        
        [FIX v5.10] use_memmap 플래그를 Pano에도 일관되게 적용
        
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
            pano_2d = None
            
            if os.path.exists(pano_path):
                try:
                    # [FIX v5.10] use_memmap 플래그에 따라 Pano도 동일하게 처리
                    if self.panorama_type == 'axial':
                        # axial일 때는 슬라이스 단위 읽기 가능
                        pano_2d = self._get_pano_volume_or_slice(pano_path, folder, idx)
                    else:
                        # 다른 타입은 전체 볼륨 필요
                        pano_volume = self._get_volume_cached(pano_path, folder + '_pano')
                        if pano_volume is not None:
                            pano_2d = self.extract_panorama(pano_volume, pano_volume, idx)
                    
                    if pano_2d is not None:
                        # Apply normalization only if normalize_volumes=True
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
                    if self.panorama_type == 'axial':
                        # [FIX v5.10] CT도 동일한 로직으로 처리
                        pano_2d = self._get_volume_or_slice(ct_path, folder, idx)
                    else:
                        ct_volume = self._get_volume_cached(ct_path, folder)
                        if ct_volume is not None:
                            pano_2d = self.extract_panorama(ct_volume, ct_volume, idx)
                    
                    if pano_2d is not None:
                        # Apply normalization only if normalize_volumes=True
                        if self.normalize_volumes:
                            pano_2d = self.normalize_slice(pano_2d, folder)
                    else:
                        pano_2d = np.zeros((200, 200), dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Error loading CT for pano fallback: {e}")
                    pano_2d = np.zeros((200, 200), dtype=np.float32)
            
            if pano_2d is None:
                pano_2d = np.zeros((200, 200), dtype=np.float32)
            
            pano_slices.append(pano_2d)
        
        # Stack to [3, H, W]
        return np.stack(pano_slices, axis=0)
    
    def _get_item_with_retry(self, idx, retry_count=0):
        """Internal function with retry logic - returns 3D slab data"""
        if retry_count >= self.max_retry_count:
            logger.error(f"Max retry count exceeded for index {idx}")
            # Return dummy 3D data (already in [-1, 1] range)
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
        
        # Get 3-slice CT window
        ct_window = self._get_3slice_window(ct_path, folder, slice_idx)
        if ct_window is None:
            logger.error(f"Failed to load CT window: {ct_path}, slice {slice_idx}")
            return self._get_item_with_retry((idx + 1) % len(self), retry_count + 1)
        
        # Debug: log raw data range before normalization
        if self.debug_normalization and idx % 1000 == 0:
            logger.info(f"[DEBUG] Raw CT data range (before norm): [{ct_window.min():.4f}, {ct_window.max():.4f}]")
        
        # Apply normalization ONLY if normalize_volumes=True
        # This is the ONLY place where normalization happens
        if self.normalize_volumes:
            ct_window = self.normalize_volume(ct_window, folder)
        
        # Get 3-slice panorama condition (normalization is handled inside _get_pano_3window)
        pano_window = self._get_pano_3window(volume_idx, slice_idx)
        
        # ============================================================
        # [FIX v5.8] Slice별 독립 Augmentation 문제 수정
        # 기존: 각 슬라이스에 별도로 Augmentation 적용 (Z축 일관성 파괴)
        # 수정: 동일한 파라미터를 3장 모두에 적용
        # [FIX v5.10] CT/Pano 정합성이 보장된 elastic deformation 사용
        # ============================================================
        if self.augment and self.data_augmentation is not None:
            # 파라미터를 한 번만 샘플링
            aug_params = self.data_augmentation.sample_params(epoch=self.current_epoch)
            
            # 동일한 파라미터로 모든 슬라이스에 적용
            for i in range(3):
                ct_window[i], pano_window[i] = self.data_augmentation.apply_with_params(
                    ct_window[i], pano_window[i], aug_params, epoch=self.current_epoch
                )
        
        # Convert to tensors
        # CT: [3, H, W] -> [1, 3, H, W] (add channel dim)
        ct_tensor = torch.from_numpy(ct_window.copy()).unsqueeze(0).float()  # [1, 3, H, W]
        # Panorama: [3, H, W] stays as is for condition
        pano_tensor = torch.from_numpy(pano_window.copy()).float()  # [3, H, W]
        
        # ============================================================
        # [FIX v5.7]: 이중 정규화 블록 완전 제거
        # ============================================================
        # 기존 코드 (제거됨):
        # if not self.normalize_volumes:
        #     pano_tensor = (pano_tensor * 2.0) - 1.0
        #     ct_tensor = (ct_tensor * 2.0) - 1.0
        #
        # 이제 정규화는 오직 normalize_volume() / normalize_slice() 에서만 수행됨
        # normalize_volumes=False 이면 데이터가 이미 [-1, 1]이라고 가정하고 변환 없음
        # ============================================================
        
        # Safety clamp (always apply as safeguard)
        pano_tensor = torch.clamp(pano_tensor, -1.0, 1.0)
        ct_tensor = torch.clamp(ct_tensor, -1.0, 1.0)
        
        # Debug: log final data range after all processing
        if self.debug_normalization and idx % 1000 == 0:
            logger.info(f"[DEBUG] Final CT tensor range: [{ct_tensor.min():.4f}, {ct_tensor.max():.4f}]")
            logger.info(f"[DEBUG] Final Pano tensor range: [{pano_tensor.min():.4f}, {pano_tensor.max():.4f}]")
        
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