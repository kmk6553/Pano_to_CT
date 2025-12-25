"""
3D Consistency Network components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialAttention(nn.Module):
    """Axial attention for 3D volumes"""
    def __init__(self, channels, axis):
        super().__init__()
        self.axis = axis
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        self.scale = channels ** -0.5
    
    def forward(self, x):
        # x: [B, C, D, H, W]
        b, c, d, h, w = x.shape
        
        # Apply normalization
        x_norm = self.norm(x.view(b, c, -1).view(b, c, d, h, w))
        
        # Reshape based on axis
        if self.axis == 0:  # Depth axis
            x_reshape = x_norm.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, d)
        elif self.axis == 1:  # Height axis
            x_reshape = x_norm.permute(0, 2, 4, 1, 3).reshape(b * d * w, c, h)
        else:  # Width axis
            x_reshape = x_norm.permute(0, 2, 3, 1, 4).reshape(b * d * h, c, w)
        
        # Compute attention
        q = self.q(x_reshape)
        k = self.k(x_reshape)
        v = self.v(x_reshape)
        
        attn = torch.bmm(q.transpose(1, 2), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.transpose(1, 2))
        out = self.proj_out(out)
        
        # Reshape back
        if self.axis == 0:
            out = out.reshape(b, h, w, c, d).permute(0, 3, 4, 1, 2)
        elif self.axis == 1:
            out = out.reshape(b, d, w, c, h).permute(0, 3, 1, 4, 2)
        else:
            out = out.reshape(b, d, h, c, w).permute(0, 3, 1, 2, 4)
        
        return x + out


class ConsistencyNet3D(nn.Module):
    """Enhanced 3D Consistency Network with Axial Attention"""
    def __init__(self, in_channels=1, features=32, use_axial_attention=True):
        super().__init__()
        self.use_axial_attention = use_axial_attention
        
        # 3D CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, features, 3, padding=1),
            nn.GroupNorm(8, features),
            nn.SiLU(),
            nn.Conv3d(features, features * 2, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, features * 2),
            nn.SiLU(),
            nn.Conv3d(features * 2, features * 4, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, features * 4),
            nn.SiLU()
        )
        
        # Middle blocks with optional axial attention
        if use_axial_attention:
            self.middle = nn.ModuleList([
                nn.Conv3d(features * 4, features * 4, 3, padding=1),
                nn.GroupNorm(8, features * 4),
                nn.SiLU(),
                AxialAttention(features * 4, axis=0),  # Depth attention
                AxialAttention(features * 4, axis=1),  # Height attention
                AxialAttention(features * 4, axis=2),  # Width attention
                nn.Conv3d(features * 4, features * 4, 3, padding=1),
                nn.GroupNorm(8, features * 4),
                nn.SiLU()
            ])
        else:
            self.middle = nn.Sequential(
                nn.Conv3d(features * 4, features * 4, 3, padding=1),
                nn.GroupNorm(8, features * 4),
                nn.SiLU(),
                nn.Conv3d(features * 4, features * 4, 3, padding=1),
                nn.GroupNorm(8, features * 4),
                nn.SiLU()
            )
        
        # 3D CNN decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(features * 4, features * 2, 3, 
                              stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.GroupNorm(8, features * 2),
            nn.SiLU(),
            nn.ConvTranspose3d(features * 2, features, 3, 
                              stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.GroupNorm(8, features),
            nn.SiLU(),
            nn.Conv3d(features, in_channels, 3, padding=1)
        )
        
        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        # Encode
        enc = self.encoder(x)
        
        # Process middle
        if self.use_axial_attention:
            mid = enc
            for layer in self.middle:
                if isinstance(layer, AxialAttention):
                    mid = layer(mid)
                else:
                    mid = layer(mid)
        else:
            mid = self.middle(enc)
        
        # Decode
        out = self.decoder(mid)
        
        # Residual connection with learnable weight
        return x + self.residual_weight * out
