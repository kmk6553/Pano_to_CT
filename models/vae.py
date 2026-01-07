"""
3D Variational Autoencoder (VAE) components for Slab-based generation
Processes 3-slice windows as volumetric data [B, 1, D=3, H, W]

FIXES APPLIED:
1. AttentionBlock3D 임계값 4096 -> 15000 상향
2. Flash Attention (F.scaled_dot_product_attention) 사용으로 원본 해상도 유지
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class ResidualBlock3D(nn.Module):
    """3D Residual Block with GroupNorm"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class AttentionBlock3D(nn.Module):
    """
    3D Attention Block - processes depth dimension efficiently
    
    FIXES APPLIED:
    1. Downsampling 임계값 4096 -> 15000 상향
    2. Flash Attention (F.scaled_dot_product_attention) 사용으로 원본 해상도 유지
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv3d(channels, channels, kernel_size=1)
        self.k = nn.Conv3d(channels, channels, kernel_size=1)
        self.v = nn.Conv3d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5
    
    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        b, c, d, h, w = q.shape
        spatial_size = d * h * w
        
        # [수정]: Threshold 4096 -> 15000 상향
        if spatial_size > 15000:
            # Approximate attention for very large spatial sizes
            # Downsample for attention computation
            target_size = int(np.cbrt(15000))
            x_small = F.interpolate(h_, size=(min(d, target_size), target_size, target_size), 
                                   mode='trilinear', align_corners=False)
            
            q_small = self.q(x_small)
            k_small = self.k(x_small)
            v_small = self.v(x_small)
            
            b_s, c_s, d_s, h_s, w_s = q_small.shape
            
            # Reshape for attention: [B, C, D*H*W] -> [B, D*H*W, C]
            q_small = q_small.view(b_s, c_s, -1).permute(0, 2, 1)  # [B, N, C]
            k_small = k_small.view(b_s, c_s, -1).permute(0, 2, 1)  # [B, N, C]
            v_small = v_small.view(b_s, c_s, -1).permute(0, 2, 1)  # [B, N, C]
            
            # [수정]: Flash Attention 사용
            out_small = F.scaled_dot_product_attention(q_small, k_small, v_small)
            
            # Reshape back: [B, N, C] -> [B, C, D, H, W]
            out_small = out_small.permute(0, 2, 1).view(b_s, c_s, d_s, h_s, w_s)
            out_small = self.proj_out(out_small)
            
            # Upsample back to original resolution
            out = F.interpolate(out_small, size=(d, h, w), mode='trilinear', align_corners=False)
        else:
            # [수정]: Full Resolution Attention with Flash Attention
            # Reshape for attention: [B, C, D*H*W] -> [B, D*H*W, C]
            q = q.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]
            k = k.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]
            v = v.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]
            
            # [수정]: Flash Attention 사용 (메모리 효율적, 빠름)
            out = F.scaled_dot_product_attention(q, k, v)
            
            # Reshape back: [B, N, C] -> [B, C, D, H, W]
            out = out.permute(0, 2, 1).view(b, c, d, h, w)
            out = self.proj_out(out)
        
        return x + out


class Downsample3D(nn.Module):
    """3D Downsampling - preserves depth (D), downsamples H and W"""
    def __init__(self, channels):
        super().__init__()
        # stride=(1,2,2) to preserve depth dimension
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, 
                             stride=(1, 2, 2), padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D Upsampling - preserves depth (D), upsamples H and W"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Upsample only H and W, keep D
        b, c, d, h, w = x.shape
        x = F.interpolate(x, size=(d, h * 2, w * 2), mode='nearest')
        return self.conv(x)


class Encoder3D(nn.Module):
    """3D Encoder for VAE - processes [B, 1, D=3, H, W] volumes"""
    def __init__(self, in_channels=1, z_channels=8, channels=64, 
                 channel_multipliers=(1, 2, 4, 8)):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv3d(in_channels, channels, kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = channels
        
        for i, mult in enumerate(channel_multipliers):
            block = nn.ModuleList()
            ch_out = channels * mult
            
            # Residual blocks
            for _ in range(2):
                block.append(ResidualBlock3D(ch, ch_out))
                ch = ch_out
            
            # Attention at lower resolutions (last 2 levels)
            if i >= len(channel_multipliers) - 2:
                block.append(AttentionBlock3D(ch))
            
            # Downsample (except last) - only H,W, preserve D
            if i < len(channel_multipliers) - 1:
                block.append(Downsample3D(ch))
            
            self.down_blocks.append(block)
        
        # Middle blocks
        self.mid_blocks = nn.ModuleList([
            ResidualBlock3D(ch, ch),
            AttentionBlock3D(ch),
            ResidualBlock3D(ch, ch)
        ])
        
        # Output layers
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out_mean = nn.Conv3d(ch, z_channels, kernel_size=3, padding=1)
        self.conv_out_logvar = nn.Conv3d(ch, z_channels, kernel_size=3, padding=1)
        
        # Initialize output layers carefully for stable training
        nn.init.xavier_normal_(self.conv_out_mean.weight, gain=0.01)
        nn.init.constant_(self.conv_out_mean.bias, 0)
        nn.init.xavier_normal_(self.conv_out_logvar.weight, gain=0.001)
        nn.init.constant_(self.conv_out_logvar.bias, -3.0)
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, D=3, H, W] - 3-slice CT volume
        Returns:
            mean: [B, z_channels, D=3, h, w] - latent mean
            logvar: [B, z_channels, D=3, h, w] - latent log variance
        """
        # Initial conv
        h = self.conv_in(x)
        
        # Downsampling
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)
        
        # Middle
        for layer in self.mid_blocks:
            h = layer(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        
        # Separate outputs for mean and logvar
        mean = self.conv_out_mean(h)
        logvar = self.conv_out_logvar(h)
        
        # Clamp outputs for stability
        logvar = torch.clamp(logvar, -8.0, 2.0)
        
        return mean, logvar


class Decoder3D(nn.Module):
    """3D Decoder for VAE - reconstructs [B, 1, D=3, H, W] volumes"""
    def __init__(self, out_channels=1, z_channels=8, channels=64, 
                 channel_multipliers=(1, 2, 4, 8)):
        super().__init__()
        
        # Calculate input channels
        ch = channels * channel_multipliers[-1]
        
        # Initial convolution
        self.conv_in = nn.Conv3d(z_channels, ch, kernel_size=3, padding=1)
        
        # Middle blocks
        self.mid_blocks = nn.ModuleList([
            ResidualBlock3D(ch, ch),
            AttentionBlock3D(ch),
            ResidualBlock3D(ch, ch)
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            block = nn.ModuleList()
            ch_out = channels * mult
            
            # Residual blocks
            for _ in range(2):
                block.append(ResidualBlock3D(ch, ch_out))
                ch = ch_out
            
            # Attention at lower resolutions
            if i >= len(channel_multipliers) - 2:
                block.append(AttentionBlock3D(ch))
            
            # Upsample (except first/last level) - only H,W, preserve D
            if i > 0:
                block.append(Upsample3D(ch))
            
            self.up_blocks.append(block)
        
        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)
    
    def forward(self, z):
        """
        Args:
            z: [B, z_channels, D=3, h, w] - latent representation
        Returns:
            [B, 1, D=3, H, W] - reconstructed 3-slice volume
        """
        # Initial conv
        h = self.conv_in(z)
        
        # Middle
        for layer in self.mid_blocks:
            h = layer(h)
        
        # Upsampling
        for block in self.up_blocks:
            for layer in block:
                h = layer(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        h = torch.tanh(h)  # Bound output to [-1, 1]
        
        return h


class VAE3D(nn.Module):
    """3D Variational Autoencoder for Slab-based CT generation
    
    Processes 3-slice windows: [B, 1, D=3, H, W]
    Depth dimension (D=3) is preserved throughout encoding/decoding
    Only spatial dimensions (H, W) are downsampled/upsampled
    """
    def __init__(self, in_channels=1, z_channels=8, channels=64):
        super().__init__()
        self.encoder = Encoder3D(in_channels, z_channels, channels)
        self.decoder = Decoder3D(in_channels, z_channels, channels)
        self.z_channels = z_channels
        
        # Safe weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Safe weight initialization"""
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """
        Encode input volume to latent space
        Args:
            x: [B, 1, D=3, H, W]
        Returns:
            z: [B, z_channels, D=3, h, w]
            mean: [B, z_channels, D=3, h, w]
            logvar: [B, z_channels, D=3, h, w]
        """
        mean, logvar = self.encoder(x)
        return self.sample(mean, logvar), mean, logvar
    
    def sample(self, mean, logvar):
        """Reparameterization trick with numerical stability"""
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-7, max=1e3)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        """
        Decode latent to volume
        Args:
            z: [B, z_channels, D=3, h, w]
        Returns:
            [B, 1, D=3, H, W]
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Full forward pass
        Args:
            x: [B, 1, D=3, H, W]
        Returns:
            recon: [B, 1, D=3, H, W]
            mean: [B, z_channels, D=3, h, w]
            logvar: [B, z_channels, D=3, h, w]
        """
        z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mean, logvar
    
    def get_latent_shape(self, input_shape):
        """
        Calculate latent shape given input shape
        Args:
            input_shape: (D, H, W) tuple, e.g., (3, 200, 200)
        Returns:
            (d, h, w) latent spatial dimensions
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            # Input: [B, C, D, H, W]
            dummy_input = torch.zeros(1, 1, *input_shape, device=device)
            z, _, _ = self.encode(dummy_input)
            return z.shape[2:]  # (D, h, w)
    
    def extract_middle_slice(self, volume):
        """
        Extract middle slice from 3D volume
        Args:
            volume: [B, 1, D=3, H, W]
        Returns:
            [B, 1, H, W] - middle slice
        """
        return volume[:, :, 1, :, :]  # Index 1 is the middle slice


# Backward compatibility alias
VAE = VAE3D


# ============== Legacy 2D Components (for reference/migration) ==============

class ResidualBlock(nn.Module):
    """2D Residual Block - kept for backward compatibility"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    """2D Attention Block - kept for backward compatibility"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
    
    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        
        b, c, h, w = q.shape
        
        if h * w > 4096:
            target_size = int(np.sqrt(4096))
            q_small = F.interpolate(q, size=(target_size, target_size), mode='bilinear', align_corners=False)
            k_small = F.interpolate(k, size=(target_size, target_size), mode='bilinear', align_corners=False)
            v_small = F.interpolate(v, size=(target_size, target_size), mode='bilinear', align_corners=False)
            
            q_small = rearrange(q_small, 'b c h w -> b (h w) c')
            k_small = rearrange(k_small, 'b c h w -> b c (h w)')
            v_small = rearrange(v_small, 'b c h w -> b (h w) c')
            
            attn = torch.bmm(q_small, k_small) * self.scale
            attn = F.softmax(attn, dim=-1)
            
            out = torch.bmm(attn, v_small)
            out = rearrange(out, 'b (h w) c -> b c h w', h=target_size, w=target_size)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        else:
            q = rearrange(q, 'b c h w -> b (h w) c')
            k = rearrange(k, 'b c h w -> b c (h w)')
            v = rearrange(v, 'b c h w -> b (h w) c')
            
            attn = torch.bmm(q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            
            out = torch.bmm(attn, v)
            out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        
        out = self.proj_out(out)
        return x + out