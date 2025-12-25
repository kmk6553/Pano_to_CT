"""
Variational Autoencoder (VAE) components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class ResidualBlock(nn.Module):
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
        
        # Use adaptive pooling for high resolution
        if h * w > 4096:
            # Use bilinear interpolation for deterministic behavior
            target_size = int(np.sqrt(4096))
            q_small = F.interpolate(q, size=(target_size, target_size), mode='bilinear', align_corners=False)
            k_small = F.interpolate(k, size=(target_size, target_size), mode='bilinear', align_corners=False)
            v_small = F.interpolate(v, size=(target_size, target_size), mode='bilinear', align_corners=False)
            
            # Compute attention on smaller resolution
            q_small = rearrange(q_small, 'b c h w -> b (h w) c')
            k_small = rearrange(k_small, 'b c h w -> b c (h w)')
            v_small = rearrange(v_small, 'b c h w -> b (h w) c')
            
            attn = torch.bmm(q_small, k_small) * self.scale
            attn = F.softmax(attn, dim=-1)
            
            out = torch.bmm(attn, v_small)
            out = rearrange(out, 'b (h w) c -> b c h w', h=target_size, w=target_size)
            
            # Upsample back
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        else:
            # Original attention for low resolution
            q = rearrange(q, 'b c h w -> b (h w) c')
            k = rearrange(k, 'b c h w -> b c (h w)')
            v = rearrange(v, 'b c h w -> b (h w) c')
            
            attn = torch.bmm(q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            
            out = torch.bmm(attn, v)
            out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        
        out = self.proj_out(out)
        return x + out


class Encoder(nn.Module):
    def __init__(self, in_channels=1, z_channels=8, channels=64, channel_multipliers=(1, 2, 4, 8)):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = channels
        for i, mult in enumerate(channel_multipliers):
            block = nn.ModuleList()
            ch_out = channels * mult
            
            # Residual blocks
            for _ in range(2):
                block.append(ResidualBlock(ch, ch_out))
                ch = ch_out
            
            # Attention at lower resolutions
            if i >= len(channel_multipliers) - 2:
                block.append(AttentionBlock(ch))
            
            # Downsample (except last)
            if i < len(channel_multipliers) - 1:
                block.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            
            self.down_blocks.append(block)
        
        # Middle blocks
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(ch, ch),
            AttentionBlock(ch),
            ResidualBlock(ch, ch)
        ])
        
        # Output with more stable initialization
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out_mean = nn.Conv2d(ch, z_channels, 3, padding=1)
        self.conv_out_logvar = nn.Conv2d(ch, z_channels, 3, padding=1)
        
        # Initialize output layers carefully
        nn.init.xavier_normal_(self.conv_out_mean.weight, gain=0.01)
        nn.init.constant_(self.conv_out_mean.bias, 0)
        nn.init.xavier_normal_(self.conv_out_logvar.weight, gain=0.001)
        nn.init.constant_(self.conv_out_logvar.bias, -3.0)
    
    def forward(self, x):
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


class Decoder(nn.Module):
    def __init__(self, out_channels=1, z_channels=8, channels=64, channel_multipliers=(1, 2, 4, 8)):
        super().__init__()
        
        # Calculate input channels
        ch = channels * channel_multipliers[-1]
        
        # Initial convolution
        self.conv_in = nn.Conv2d(z_channels, ch, 3, padding=1)
        
        # Middle blocks
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(ch, ch),
            AttentionBlock(ch),
            ResidualBlock(ch, ch)
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_multipliers))):
            block = nn.ModuleList()
            ch_out = channels * mult
            
            # Residual blocks
            for _ in range(2):
                block.append(ResidualBlock(ch, ch_out))
                ch = ch_out
            
            # Attention at lower resolutions
            if i >= len(channel_multipliers) - 2:
                block.append(AttentionBlock(ch))
            
            # Upsample (except first)
            if i > 0:
                block.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            
            self.up_blocks.append(block)
        
        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(self, z):
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


class VAE(nn.Module):
    def __init__(self, in_channels=1, z_channels=8, channels=64):
        super().__init__()
        self.encoder = Encoder(in_channels, z_channels, channels)
        self.decoder = Decoder(in_channels, z_channels, channels)
        self.z_channels = z_channels
        
        # Safe weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Safe weight initialization"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # He initialization with smaller gain
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Xavier initialization with smaller gain
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        mean, logvar = self.encoder(x)
        return self.sample(mean, logvar), mean, logvar
    
    def sample(self, mean, logvar):
        """Reparameterization trick with numerical stability"""
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-7, max=1e3)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mean, logvar
    
    def get_latent_shape(self, input_shape):
        """Calculate latent shape given input shape"""
        with torch.no_grad():
            device = next(self.parameters()).device
            dummy_input = torch.zeros(1, 1, *input_shape, device=device)
            z, _, _ = self.encode(dummy_input)
            return z.shape[2:]
