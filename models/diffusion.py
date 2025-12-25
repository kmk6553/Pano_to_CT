"""
Diffusion model components with Multi-scale Spatial Conditioning and Position Embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .vae import ResidualBlock


class SinusoidalPosEmb(nn.Module):
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


class SpatialConditionInjector(nn.Module):
    """Multi-scale spatial condition injection module"""
    def __init__(self, cond_channels, target_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cond_channels, target_channels, 3, padding=1),
            nn.GroupNorm(32, target_channels),
            nn.SiLU(),
            nn.Conv2d(target_channels, target_channels, 3, padding=1)
        )
        
    def forward(self, cond_spatial, target_size):
        """
        Args:
            cond_spatial: [B, C, H, W] - spatial condition features
            target_size: (H', W') - target spatial resolution
        Returns:
            [B, target_channels, H', W'] - injected condition
        """
        if cond_spatial.shape[2:] != target_size:
            cond_spatial = F.interpolate(cond_spatial, size=target_size, 
                                        mode='bilinear', align_corners=False)
        return self.conv(cond_spatial)


class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization for conditioning"""
    def __init__(self, num_groups, in_channels, cond_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.scale_shift = nn.Linear(cond_channels, in_channels * 2)
        self.scale_shift.weight.data.zero_()
        self.scale_shift.bias.data.zero_()
    
    def forward(self, x, cond):
        normalized = self.norm(x)
        scale_shift = self.scale_shift(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return normalized * (1 + scale) + shift


class ConditionalResBlock(nn.Module):
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
        self.norm1 = AdaptiveGroupNorm(32, in_channels, time_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = AdaptiveGroupNorm(32, out_channels, time_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Global condition injection via FiLM
        if cond_channels is not None:
            self.cond_proj = nn.Linear(cond_channels, out_channels * 2)
            self.cond_proj.weight.data.zero_()
            self.cond_proj.bias.data.zero_()
        else:
            self.cond_proj = None
        
        # Spatial condition: SpatialConditionInjector already outputs correct channels
        # So we just need to add it directly (no additional conv needed)
        self.use_spatial_cond = spatial_cond_channels is not None
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb, cond=None, spatial_cond=None):
        # First block with time conditioning
        h = self.norm1(x, t_emb)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_proj(t_emb)[:, :, None, None]
        
        # Second block with time conditioning
        h = self.norm2(h, t_emb)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Apply global condition via FiLM if provided
        if cond is not None and self.cond_proj is not None:
            scale_shift = self.cond_proj(cond)
            scale, shift = scale_shift.chunk(2, dim=1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale) + shift
        
        # Apply spatial condition if provided
        if spatial_cond is not None and self.use_spatial_cond:
            # Ensure spatial_cond matches h's spatial dimensions exactly
            if spatial_cond.shape[2:] != h.shape[2:]:
                spatial_cond = F.interpolate(spatial_cond, size=h.shape[2:], 
                                            mode='bilinear', align_corners=False)
            h = h + spatial_cond
        
        return h + self.shortcut(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=8, channels=128, 
                 channel_multipliers=(1, 2, 4, 8), cond_channels=512, panorama_type='coronal',
                 pano_triplet=False, use_self_conditioning=True):
        super().__init__()
        
        self.panorama_type = panorama_type
        self.pano_triplet = pano_triplet
        self.use_self_conditioning = use_self_conditioning
        
        # Self-conditioning: double input channels
        actual_in_channels = in_channels * 2 if use_self_conditioning else in_channels
        
        # Time embedding
        time_dim = channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(channels),
            nn.Linear(channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Slice position embedding (for z-axis awareness)
        self.pos_emb_dim = 128
        self.pos_embedding = nn.Sequential(
            SinusoidalPosEmb(self.pos_emb_dim // 2),
            nn.Linear(self.pos_emb_dim // 2, self.pos_emb_dim),
            nn.SiLU(),
            nn.Linear(self.pos_emb_dim, self.pos_emb_dim)
        )
        
        # Combine time and position embeddings
        self.combined_emb = nn.Linear(time_dim + self.pos_emb_dim, time_dim)
        
        # Panorama condition encoder - spatial feature extraction
        in_ch = 3 if pano_triplet else 1
        self.cond_encoder_spatial = nn.ModuleList([
            # Level 0: Original resolution
            nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=1),
                nn.SiLU(),
                ResidualBlock(64, 128)
            ),
            # Level 1: 1/2 resolution
            nn.Sequential(
                nn.Conv2d(128, 128, 3, stride=2, padding=1),
                ResidualBlock(128, 256)
            ),
            # Level 2: 1/4 resolution
            nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                ResidualBlock(256, 512)
            ),
            # Level 3: 1/8 resolution
            nn.Sequential(
                nn.Conv2d(512, 512, 3, stride=2, padding=1),
                ResidualBlock(512, cond_channels)
            )
        ])
        
        # Global condition from panorama
        self.cond_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cond_channels, cond_channels)
        )
        
        # Spatial condition injectors for each resolution level
        self.spatial_injectors = nn.ModuleList([
            SpatialConditionInjector(128, channels),  # Level 0
            SpatialConditionInjector(256, channels * 2),  # Level 1
            SpatialConditionInjector(512, channels * 4),  # Level 2
            SpatialConditionInjector(cond_channels, channels * 8)  # Level 3
        ])
        
        # Initial convolution
        self.conv_in = nn.Conv2d(actual_in_channels, channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_channels = []
        ch = channels
        
        for i, mult in enumerate(channel_multipliers):
            layers = []
            ch_out = channels * mult
            spatial_ch = [128, 256, 512, cond_channels][i] if i < 4 else None
            
            # Two residual blocks with spatial conditioning
            layers.append(ConditionalResBlock(ch, ch_out, time_dim, cond_channels, spatial_ch))
            ch = ch_out
            layers.append(ConditionalResBlock(ch, ch_out, time_dim, cond_channels, spatial_ch))
            
            self.down_channels.append(ch_out)
            
            # Downsample (except last)
            if i < len(channel_multipliers) - 1:
                layers.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            
            self.down_blocks.append(nn.ModuleList(layers))
        
        # Middle blocks
        spatial_ch = cond_channels
        self.mid_blocks = nn.ModuleList([
            ConditionalResBlock(ch, ch, time_dim, cond_channels, spatial_ch),
            ConditionalResBlock(ch, ch, time_dim, cond_channels, spatial_ch)
        ])
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        
        for i in reversed(range(len(channel_multipliers))):
            layers = []
            ch_out = channels * channel_multipliers[i]
            skip_ch = self.down_channels[i]
            spatial_ch = [128, 256, 512, cond_channels][i] if i < 4 else None
            
            # First residual block with skip connection and spatial conditioning
            layers.append(ConditionalResBlock(ch + skip_ch, ch_out, time_dim, cond_channels, spatial_ch))
            ch = ch_out
            
            # Second residual block with spatial conditioning
            layers.append(ConditionalResBlock(ch, ch_out, time_dim, cond_channels, spatial_ch))
            
            # Upsample (except first)
            if i > 0:
                layers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(ch, ch, 3, padding=1)
                ))
            
            self.up_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        # Improved initialization
        self._init_weights()
    
    def _init_weights(self):
        """Improved weight initialization for stable diffusion training"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
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
            x: [B, C, H, W] - noisy latent
            t: [B] - timestep
            cond: [B, cond_ch, H, W] - panorama condition
            slice_pos: [B] - normalized slice position in [0, 1]
            x_self_cond: [B, C, H, W] - previous prediction for self-conditioning
        """
        # Self-conditioning
        if self.use_self_conditioning:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat([x, x_self_cond], dim=1)
        
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Position embedding (if provided)
        if slice_pos is not None:
            pos_emb = self.pos_embedding(slice_pos)
            t_emb = self.combined_emb(torch.cat([t_emb, pos_emb], dim=1))
        
        # Extract spatial features from panorama at multiple scales
        if cond.dim() == 3:
            cond = cond.unsqueeze(1)
        
        spatial_features = []
        h_cond = cond
        for encoder in self.cond_encoder_spatial:
            h_cond = encoder(h_cond)
            spatial_features.append(h_cond)
        
        # Global condition
        cond_global = self.cond_global(h_cond)
        
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
                if isinstance(layer, ConditionalResBlock):
                    h = layer(h, t_emb, cond_global, spatial_cond)
                    if j == 1:
                        skip_connections.append(h)
                elif isinstance(layer, nn.Conv2d):
                    h = layer(h)
        
        # Middle with spatial conditioning
        spatial_cond = self.spatial_injectors[-1](spatial_features[-1], h.shape[2:])
        for layer in self.mid_blocks:
            h = layer(h, t_emb, cond_global, spatial_cond)
        
        # Upsampling with skip connections and spatial conditioning
        for i, layers in enumerate(self.up_blocks):
            skip_idx = len(skip_connections) - 1 - i
            skip = skip_connections[skip_idx]
            
            # Get spatial condition for this level
            level_idx = len(self.down_blocks) - 1 - i
            spatial_cond = None
            if level_idx < len(spatial_features):
                spatial_cond = self.spatial_injectors[level_idx](spatial_features[level_idx], h.shape[2:])
            
            # Ensure size match
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            h = torch.cat([h, skip], dim=1)
            
            for layer in layers:
                if isinstance(layer, ConditionalResBlock):
                    h = layer(h, t_emb, cond_global, spatial_cond)
                else:
                    h = layer(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Final size adjustment
        if h.shape[2:] != x.shape[2:]:
            h = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return h
