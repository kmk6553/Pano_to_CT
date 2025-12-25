"""
Loss functions implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (L1 with smooth approximation)"""
    def __init__(self, epsilon=1e-2):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()


class SSIMLoss(nn.Module):
    """SSIM Loss for PyTorch"""
    def __init__(self, window_size=11, size_average=True, channel=1, sigma=1.5):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.sigma = sigma
        self.window = self.create_window(window_size, channel, sigma)
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, channel, sigma):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel, self.sigma)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel
        
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)


class GradientDifferenceLoss(nn.Module):
    """Gradient Difference Loss for preserving edges and fine details"""
    def __init__(self):
        super(GradientDifferenceLoss, self).__init__()
        
    def forward(self, pred, target):
        # Compute gradients
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # Compute differences
        dx_diff = torch.abs(pred_dx - target_dx)
        dy_diff = torch.abs(pred_dy - target_dy)
        
        # Return mean absolute difference
        return dx_diff.mean() + dy_diff.mean()


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity Loss"""
    def __init__(self, net='vgg', device='cuda'):
        super(LPIPSLoss, self).__init__()
        self.device = device
        
        # Use VGG16 features
        if net == 'vgg':
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
            
            self.blocks = nn.ModuleList([])
            self.blocks.append(vgg[:4])    # relu1_2
            self.blocks.append(vgg[4:9])   # relu2_2
            self.blocks.append(vgg[9:16])  # relu3_3
            self.blocks.append(vgg[16:23]) # relu4_3
            
            # Weights for different layers
            self.weights = [1.0, 1.0, 1.0, 1.0]
            
        for param in self.parameters():
            param.requires_grad = False
            
    def normalize_tensor(self, x):
        """Normalize tensor to ImageNet statistics"""
        # Assuming input is in [-1, 1], convert to [0, 1]
        x = (x + 1) / 2
        
        # If grayscale, repeat to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std
    
    def forward(self, pred, target):
        # Normalize inputs
        pred_norm = self.normalize_tensor(pred)
        target_norm = self.normalize_tensor(target)
        
        # Extract features
        loss = 0
        x = pred_norm
        y = target_norm
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            # Compute MSE between features
            loss += self.weights[i] * F.mse_loss(x, y)
            
        return loss


class StableKLLoss(nn.Module):
    """Numerically stable KL divergence loss with free bits"""
    def __init__(self, free_bits=0.0):
        super().__init__()
        self.free_bits = free_bits
    
    def forward(self, mean, logvar):
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Clamp inputs for extreme stability
        mean = torch.clamp(mean, min=-10, max=10)
        logvar = torch.clamp(logvar, min=-10, max=2)
        
        # Compute KL term-by-term for better numerical stability
        kl_elementwise = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        
        # Additional safety: clamp individual KL terms
        kl_elementwise = torch.clamp(kl_elementwise, min=-10, max=10)
        
        # Apply free bits to prevent KL from going to zero
        if self.free_bits > 0:
            free_bits_tensor = torch.tensor(self.free_bits, dtype=torch.float32, device=kl_elementwise.device)
            kl_elementwise = torch.maximum(kl_elementwise, free_bits_tensor)
        
        # Take mean over all dimensions except batch
        kl = kl_elementwise.mean(dim=[1, 2, 3])  # [B]
        
        # Average over batch
        return kl.mean()