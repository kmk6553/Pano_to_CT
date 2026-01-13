"""
3D Diffusion model components with Multi-scale Spatial Conditioning and Position Embedding
Processes 3D latent volumes [B, C, D=3, h, w]

FIXES APPLIED:
1. ConditionEncoder3D - 가운데 슬라이드만 사용하는 대신 3장 슬라이드 평균(Mean) 사용
2. AttentionBlock3D를 middle blocks에 추가하여 Z-axis consistency 향상
3. [NEW v6.0] Condition Latent Concat 옵션 추가 - conditioning 무시 방지
   - use_cond_concat=True로 condition을 입력 채널에 직접 concat
   - UNet이 첫 레이어부터 condition을 "같이 보고" 시작하여 조건 무시가 어려워짐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .vae import ResidualBlock3D, AttentionBlock3D


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SpatialConditionInjector3D(nn.Module):
    """Multi-scale spatial condition injection module for 3D"""
    def __init__(self, cond_channels, target_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(cond_channels, target_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, target_channels),
            nn.SiLU(),
            nn.Conv3d(target_channels, target_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, cond_spatial, target_size):
        """
        Args:
            cond_spatial: [B, C, D, H, W] - spatial condition features
            target_size: (D', H', W') - target spatial resolution
        Returns:
            [B, target_channels, D', H', W'] - injected condition
        """
        if cond_spatial.shape[2:] != target_size:
            cond_spatial = F.interpolate(cond_spatial, size=target_size, 
                                        mode='trilinear', align_corners=False)
        return self.conv(cond_spatial)


class AdaptiveGroupNorm3D(nn.Module):
    """Adaptive Group Normalization for 3D conditioning"""
    def __init__(self, num_groups, in_channels, cond_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.scale_shift = nn.Linear(cond_channels, in_channels * 2)
        self.scale_shift.weight.data.zero_()
        self.scale_shift.bias.data.zero_()
    
    def forward(self, x, cond):
        """
        Args:
            x: [B, C, D, H, W]
            cond: [B, cond_channels]
        """
        normalized = self.norm(x)
        scale_shift = self.scale_shift(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        # Expand for 3D: [B, C] -> [B, C, 1, 1, 1]
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return normalized * (1 + scale) + shift


class ConditionalResBlock3D(nn.Module):
    """3D Conditional Residual Block with time and spatial conditioning"""
    def __init__(self, in_channels, out_channels, time_channels, cond_channels=None, 
                 spatial_cond_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        # Main blocks with conditional normalization
        self.norm1 = AdaptiveGroupNorm3D(32, in_channels, time_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = AdaptiveGroupNorm3D(32, out_channels, time_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Global condition injection via FiLM
        if cond_channels is not None:
            self.cond_proj = nn.Linear(cond_channels, out_channels * 2)
            self.cond_proj.weight.data.zero_()
            self.cond_proj.bias.data.zero_()
        else:
            self.cond_proj = None
        
        # Spatial condition flag
        self.use_spatial_cond = spatial_cond_channels is not None
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb, cond=None, spatial_cond=None):
        """
        Args:
            x: [B, C, D, H, W]
            t_emb: [B, time_channels]
            cond: [B, cond_channels] - global condition
            spatial_cond: [B, C', D, H, W] - spatial condition (same size as x)
        """
        # First block with time conditioning
        h = self.norm1(x, t_emb)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding: [B, C] -> [B, C, 1, 1, 1]
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        
        # Second block with time conditioning
        h = self.norm2(h, t_emb)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Apply global condition via FiLM if provided
        if cond is not None and self.cond_proj is not None:
            scale_shift = self.cond_proj(cond)
            scale, shift = scale_shift.chunk(2, dim=1)
            scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale) + shift
        
        # Apply spatial condition if provided
        if spatial_cond is not None and self.use_spatial_cond:
            if spatial_cond.shape[2:] != h.shape[2:]:
                spatial_cond = F.interpolate(spatial_cond, size=h.shape[2:], 
                                            mode='trilinear', align_corners=False)
            h = h + spatial_cond
        
        return h + self.shortcut(x)


class Downsample3D(nn.Module):
    """3D Downsampling - preserves D, downsamples H and W"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, 
                             stride=(1, 2, 2), padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D Upsampling - preserves D, upsamples H and W"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        b, c, d, h, w = x.shape
        x = F.interpolate(x, size=(d, h * 2, w * 2), mode='nearest')
        return self.conv(x)


class ConditionEncoder3D(nn.Module):
    """
    Encodes 3-channel panorama condition to multi-scale 3D features
    
    Strategy: Process each slice with shared 2D encoder, then stack to 3D
    This is more efficient than full 3D convolution on condition
    
    FIXES APPLIED:
    1. Global Feature 생성 시 가운데 슬라이드([:, 1])만 사용하는 대신 
       3장 슬라이드 정보의 평균(Mean) 사용하여 3D 맥락 유지
    """
    def __init__(self, in_channels=3, base_channels=64, out_channels=512):
        super().__init__()
        
        # Shared 2D encoder for each slice
        self.encoder_2d = nn.ModuleList([
            # Level 0: Original resolution
            nn.Sequential(
                nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, base_channels),
                nn.SiLU(),
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
                nn.GroupNorm(32, base_channels * 2),
                nn.SiLU()
            ),
            # Level 1: 1/2 resolution
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
                nn.GroupNorm(32, base_channels * 4),
                nn.SiLU()
            ),
            # Level 2: 1/4 resolution
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
                nn.GroupNorm(32, base_channels * 8),
                nn.SiLU()
            ),
            # Level 3: 1/8 resolution
            nn.Sequential(
                nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(base_channels * 8, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, out_channels),
                nn.SiLU()
            )
        ])
        
        # Channel dimensions at each level
        self.level_channels = [
            base_channels * 2,   # 128
            base_channels * 4,   # 256
            base_channels * 8,   # 512
            out_channels         # 512
        ]
        
        # Global pooling for global condition
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, cond):
        """
        Args:
            cond: [B, 3, H, W] - 3-channel panorama (prev, curr, next)
                  or [B, 1, 3, H, W] - already in 3D format
        Returns:
            spatial_features: list of [B, C, D=3, h, w] at different scales
            global_cond: [B, out_channels] global condition vector
        """
        # Handle different input formats
        if cond.dim() == 5:
            # Input: [B, 1, D=3, H, W] -> Flatten depth to batch for 2D processing
            b, c, d, h, w = cond.shape
            # Reshape to process each slice: [B*D, 1, H, W]
            cond_flat = cond.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        elif cond.dim() == 4 and cond.shape[1] == 3:
            # [B, 3, H, W] - treat channels as depth slices
            b, d, h, w = cond.shape
            # Reshape: [B, 3, H, W] -> [B*3, 1, H, W]
            cond_flat = cond.reshape(b * d, 1, h, w)
        elif cond.dim() == 4 and cond.shape[1] == 1:
            # [B, 1, H, W] - single channel, replicate to 3
            b, _, h, w = cond.shape
            d = 3
            cond_flat = cond.expand(b, 3, h, w).reshape(b * 3, 1, h, w)
        else:
            raise ValueError(f"Unexpected condition shape: {cond.shape}")
        
        spatial_features = []
        h = cond_flat
        
        for i, encoder in enumerate(self.encoder_2d):
            h = encoder(h)
            # Reshape back to 3D: [B*D, C, h, w] -> [B, C, D, h, w]
            _, c, hh, ww = h.shape
            h_3d = h.reshape(b, d, c, hh, ww).permute(0, 2, 1, 3, 4)  # [B, C, D, h, w]
            spatial_features.append(h_3d)
        
        # ============================================================
        # [수정]: Global condition - 가운데 슬라이드만 사용하는 대신 
        #         3장 슬라이드 정보의 평균(Mean) 사용하여 3D 맥락 유지
        # ============================================================
        # 기존: h_middle = h.reshape(b, d, -1, h.shape[-2], h.shape[-1])[:, 1]
        # 수정: Depth 차원(dim=1)에 대해 평균 Pool 사용
        h_3d_last = h.reshape(b, d, -1, h.shape[-2], h.shape[-1])  # [B, D, C, h, w]
        h_global_feat = h_3d_last.mean(dim=1)  # [B, C, h, w] - 3개 슬라이드 평균
        global_cond = self.global_pool(h_global_feat)  # [B, out_channels]
        
        return spatial_features, global_cond


class ConditionalUNet3D(nn.Module):
    """
    3D Conditional UNet for Diffusion
    
    Processes 3D latent volumes [B, C, D=3, h, w]
    Uses multi-scale spatial conditioning and position embedding
    
    UPDATED: Added AttentionBlock3D to middle blocks for improved Z-axis consistency
    
    [NEW v6.0] Condition Latent Concat 옵션 추가:
    - use_cond_concat=True: condition을 latent 해상도로 내려서 입력 채널에 직접 concat
    - 이렇게 하면 UNet이 첫 레이어부터 condition을 "같이 보고" 시작
    - 기존 feature add(injector) 방식은 학습이 꼬이면 조건 무시 경로가 생길 수 있음
    """
    def __init__(self, in_channels=8, out_channels=8, channels=128, 
                 channel_multipliers=(1, 2, 4, 8), cond_channels=512, 
                 panorama_type='axial', pano_triplet=True, use_self_conditioning=True,
                 use_cond_concat=False, cond_concat_channels=None):
        """
        Args:
            in_channels: Input latent channels (z_channels)
            out_channels: Output channels (same as in_channels)
            channels: Base channel count
            channel_multipliers: Channel multipliers for each level
            cond_channels: Condition feature channels
            panorama_type: Type of panorama extraction
            pano_triplet: Whether condition is 3-slice triplet
            use_self_conditioning: Enable self-conditioning
            use_cond_concat: [NEW v6.0] Enable condition concat to input
            cond_concat_channels: [NEW v6.0] Channels for condition concat (default: in_channels)
        """
        super().__init__()
        
        self.panorama_type = panorama_type
        self.pano_triplet = pano_triplet
        self.use_self_conditioning = use_self_conditioning
        
        # ============================================================
        # [NEW v6.0] Condition Concat 설정
        # ============================================================
        self.use_cond_concat = use_cond_concat
        self.cond_concat_channels = cond_concat_channels if cond_concat_channels is not None else in_channels
        
        # Self-conditioning: double input channels
        actual_in_channels = in_channels * 2 if use_self_conditioning else in_channels
        
        # [NEW v6.0] Condition concat: add cond_concat_channels
        if self.use_cond_concat:
            # Condition을 latent 해상도로 변환하는 모듈
            # cond: [B, 1, 3, H, W] -> [B, cond_concat_channels, 3, h, w]
            self.cond_to_latent = nn.Sequential(
                nn.Conv3d(1, self.cond_concat_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv3d(self.cond_concat_channels, self.cond_concat_channels, kernel_size=3, padding=1),
            )
            actual_in_channels += self.cond_concat_channels
        
        # Time embedding
        time_dim = channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(channels),
            nn.Linear(channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Slice position embedding (for z-axis awareness)
        # Position represents the center slice position in the volume
        self.pos_emb_dim = 128
        self.pos_embedding = nn.Sequential(
            SinusoidalPosEmb(self.pos_emb_dim // 2),
            nn.Linear(self.pos_emb_dim // 2, self.pos_emb_dim),
            nn.SiLU(),
            nn.Linear(self.pos_emb_dim, self.pos_emb_dim)
        )
        
        # Combine time and position embeddings
        self.combined_emb = nn.Linear(time_dim + self.pos_emb_dim, time_dim)
        
        # Panorama condition encoder (outputs 3D features)
        self.cond_encoder = ConditionEncoder3D(
            in_channels=3 if pano_triplet else 1,
            base_channels=64,
            out_channels=cond_channels
        )
        
        # Spatial condition injectors for each resolution level
        self.spatial_injectors = nn.ModuleList([
            SpatialConditionInjector3D(128, channels),           # Level 0
            SpatialConditionInjector3D(256, channels * 2),       # Level 1
            SpatialConditionInjector3D(512, channels * 4),       # Level 2
            SpatialConditionInjector3D(cond_channels, channels * 8)  # Level 3
        ])
        
        # Initial convolution (3D)
        self.conv_in = nn.Conv3d(actual_in_channels, channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_channels = []
        ch = channels
        
        for i, mult in enumerate(channel_multipliers):
            layers = []
            ch_out = channels * mult
            spatial_ch = self.cond_encoder.level_channels[i] if i < 4 else None
            
            # Two residual blocks with spatial conditioning
            layers.append(ConditionalResBlock3D(ch, ch_out, time_dim, cond_channels, spatial_ch))
            ch = ch_out
            layers.append(ConditionalResBlock3D(ch, ch_out, time_dim, cond_channels, spatial_ch))
            
            self.down_channels.append(ch_out)
            
            # Downsample (except last) - only H,W, preserve D
            if i < len(channel_multipliers) - 1:
                layers.append(Downsample3D(ch))
            
            self.down_blocks.append(nn.ModuleList(layers))
        
        # Middle blocks with Self-Attention for Z-axis consistency
        spatial_ch = cond_channels
        self.mid_blocks = nn.ModuleList([
            ConditionalResBlock3D(ch, ch, time_dim, cond_channels, spatial_ch),
            AttentionBlock3D(ch),  # Self-Attention 추가
            ConditionalResBlock3D(ch, ch, time_dim, cond_channels, spatial_ch)
        ])
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        
        for i in reversed(range(len(channel_multipliers))):
            layers = []
            ch_out = channels * channel_multipliers[i]
            skip_ch = self.down_channels[i]
            spatial_ch = self.cond_encoder.level_channels[i] if i < 4 else None
            
            # First residual block with skip connection
            layers.append(ConditionalResBlock3D(ch + skip_ch, ch_out, time_dim, cond_channels, spatial_ch))
            ch = ch_out
            
            # Second residual block
            layers.append(ConditionalResBlock3D(ch, ch_out, time_dim, cond_channels, spatial_ch))
            
            # Upsample (except first/last)
            if i > 0:
                layers.append(Upsample3D(ch))
            
            self.up_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)
        
        # Improved initialization
        self._init_weights()
    
    def _init_weights(self):
        """Improved weight initialization for stable diffusion training"""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        # Zero initialize the final conv layer for better stability
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x, t, cond, slice_pos=None, x_self_cond=None):
        """
        Args:
            x: [B, C, D=3, h, w] - noisy latent volume
            t: [B] - timestep
            cond: [B, 3, H, W] or [B, 1, 3, H, W] - panorama condition
            slice_pos: [B] - normalized position of center slice in [0, 1]
            x_self_cond: [B, C, D=3, h, w] - previous prediction for self-conditioning
        Returns:
            [B, C, D=3, h, w] - predicted noise/v/x0
        """
        # Self-conditioning
        if self.use_self_conditioning:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat([x, x_self_cond], dim=1)
        
        # ============================================================
        # [NEW v6.0] Condition Concat 처리
        # condition을 latent 해상도로 변환하여 입력 채널에 직접 concat
        # ============================================================
        if self.use_cond_concat:
            # cond: [B, 3, H, W] -> [B, 1, 3, H, W]
            if cond.dim() == 4:
                cond_3d = cond.unsqueeze(1)  # [B, 1, 3, H, W]
            else:
                cond_3d = cond  # 이미 [B, 1, 3, H, W] 형태
            
            # Permute to [B, 1, D=3, H, W] - depth를 3번째 차원으로
            # cond_3d는 [B, 1, 3, H, W]이지만 3이 채널이 아닌 depth 역할
            # 따라서 [B, 3, H, W] -> [B, 1, 3, H, W]에서 
            # 실제로는 [B, C=1, D=3, H, W]로 해석해야 함
            # 근데 원래 cond는 [B, 3, H, W]로 3개 슬라이스가 채널로 들어옴
            # 이를 3D로 바꾸면 [B, 1, D=3, H, W]
            
            # cond: [B, 3, H, W] -> cond_3d: [B, 1, D=3, H, W]
            if cond.dim() == 4 and cond.shape[1] == 3:
                cond_3d = cond.unsqueeze(1).permute(0, 1, 2, 3, 4)  # 그대로 [B, 1, 3, H, W]
                # 사실 이미 [B, 3, H, W]에서 unsqueeze(1)하면 [B, 1, 3, H, W]
                # 여기서 dim=1이 채널(C=1), dim=2가 depth(D=3)
            elif cond.dim() == 5:
                cond_3d = cond  # 이미 [B, 1, 3, H, W]
            else:
                # 그 외 케이스
                cond_3d = cond.unsqueeze(1) if cond.dim() == 4 else cond
            
            # Latent 해상도(D, h, w)로 리사이즈
            # x.shape: [B, C*2(self-cond), D=3, h, w]
            target_size = (x.shape[2], x.shape[3], x.shape[4])  # (D, h, w)
            cond_3d_resized = F.interpolate(
                cond_3d, size=target_size,
                mode='trilinear', align_corners=False
            )  # [B, 1, D, h, w]
            
            # cond_to_latent로 채널 확장
            cond_lat = self.cond_to_latent(cond_3d_resized)  # [B, cond_concat_channels, D, h, w]
            
            # x에 concat
            x = torch.cat([x, cond_lat], dim=1)
        
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Position embedding (if provided)
        if slice_pos is not None:
            pos_emb = self.pos_embedding(slice_pos)
            t_emb = self.combined_emb(torch.cat([t_emb, pos_emb], dim=1))
        
        # Extract spatial features from panorama at multiple scales
        spatial_features, cond_global = self.cond_encoder(cond)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Downsampling with skip connections and spatial conditioning
        skip_connections = []
        
        for i, layers in enumerate(self.down_blocks):
            # Get spatial condition for this level
            spatial_cond = None
            if i < len(spatial_features):
                spatial_cond = self.spatial_injectors[i](spatial_features[i], h.shape[2:])
            
            for j, layer in enumerate(layers):
                if isinstance(layer, ConditionalResBlock3D):
                    h = layer(h, t_emb, cond_global, spatial_cond)
                    if j == 1:  # Save after second res block
                        skip_connections.append(h)
                elif isinstance(layer, Downsample3D):
                    h = layer(h)
        
        # Middle with spatial conditioning and Self-Attention
        spatial_cond = self.spatial_injectors[-1](spatial_features[-1], h.shape[2:])
        for layer in self.mid_blocks:
            if isinstance(layer, ConditionalResBlock3D):
                h = layer(h, t_emb, cond_global, spatial_cond)
            elif isinstance(layer, AttentionBlock3D):
                # AttentionBlock3D only takes x as input
                h = layer(h)
            else:
                h = layer(h)
        
        # Upsampling with skip connections and spatial conditioning
        for i, layers in enumerate(self.up_blocks):
            skip_idx = len(skip_connections) - 1 - i
            skip = skip_connections[skip_idx]
            
            # Get spatial condition for this level
            level_idx = len(self.down_blocks) - 1 - i
            spatial_cond = None
            if level_idx < len(spatial_features):
                spatial_cond = self.spatial_injectors[level_idx](spatial_features[level_idx], h.shape[2:])
            
            # Ensure size match for skip connection
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            h = torch.cat([h, skip], dim=1)
            
            for layer in layers:
                if isinstance(layer, ConditionalResBlock3D):
                    h = layer(h, t_emb, cond_global, spatial_cond)
                elif isinstance(layer, Upsample3D):
                    h = layer(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Final size adjustment if needed
        if h.shape[2:] != (x.shape[2] if not self.use_cond_concat else x.shape[2]):
            # x의 원래 shape을 기준으로 맞춤
            # self-cond + cond_concat을 제외한 원래 latent shape
            pass  # 보통 필요 없음
        
        return h


# Backward compatibility alias
ConditionalUNet = ConditionalUNet3D


# ============== Legacy 2D Components (for reference) ==============

class SpatialConditionInjector(nn.Module):
    """2D Multi-scale spatial condition injection - kept for reference"""
    def __init__(self, cond_channels, target_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cond_channels, target_channels, 3, padding=1),
            nn.GroupNorm(32, target_channels),
            nn.SiLU(),
            nn.Conv2d(target_channels, target_channels, 3, padding=1)
        )
        
    def forward(self, cond_spatial, target_size):
        if cond_spatial.shape[2:] != target_size:
            cond_spatial = F.interpolate(cond_spatial, size=target_size, 
                                        mode='bilinear', align_corners=False)
        return self.conv(cond_spatial)