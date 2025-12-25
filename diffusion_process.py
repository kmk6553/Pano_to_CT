"""
Diffusion process implementation with v-parametrization and CFG
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class DiffusionProcess(nn.Module):
    """Diffusion process with v-parametrization and Classifier-Free Guidance"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule='cosine',
                 prediction_type='v'):  # 'v', 'epsilon', or 'x0'
        super().__init__()
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        
        # Beta schedule
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif schedule == 'cosine':
            s = 0.008
            steps = num_timesteps + 1
            t = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # Register as buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculations for posterior
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', 
                           betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', 
                           (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod))
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_x0_from_eps(self, x_t, t, eps):
        """Predict x0 from epsilon parametrization"""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps
    
    def predict_x0_from_v(self, x_t, t, v):
        """Predict x0 from v-parametrization"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * v
    
    def get_v(self, x_start, noise, t):
        """Get v target from x0 and noise"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start
    
    def predict_eps_from_v(self, x_t, t, v):
        """Convert v prediction to epsilon"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_pred = self.predict_x0_from_v(x_t, t, v)
        return (x_t - sqrt_alphas_cumprod_t * x0_pred) / sqrt_one_minus_alphas_cumprod_t
    
    def predict_x0_from_noise(self, x_t, t, noise):
        """Predict x0 from xt and noise (original method)"""
        return self.predict_x0_from_eps(x_t, t, noise)
    
    def p_mean_variance(self, model, x_t, t, cond, slice_pos=None, x_self_cond=None, 
                       clip_denoised=True, guidance_scale=1.0):
        """Compute mean and variance for reverse process with CFG support"""
        # Model prediction
        model_output = model(x_t, t, cond, slice_pos, x_self_cond)
        
        # Classifier-Free Guidance
        if guidance_scale != 1.0:
            # Unconditional prediction (zero condition)
            uncond = torch.zeros_like(cond)
            model_output_uncond = model(x_t, t, uncond, slice_pos, x_self_cond)
            # Apply guidance
            model_output = model_output_uncond + guidance_scale * (model_output - model_output_uncond)
        
        # Convert prediction to x0
        if self.prediction_type == 'v':
            x_0_pred = self.predict_x0_from_v(x_t, t, model_output)
        elif self.prediction_type == 'epsilon':
            x_0_pred = self.predict_x0_from_eps(x_t, t, model_output)
        elif self.prediction_type == 'x0':
            x_0_pred = model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Posterior mean and variance
        posterior_mean_coef1_t = self.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        posterior_mean_coef2_t = self.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
        
        posterior_mean = posterior_mean_coef1_t * x_0_pred + posterior_mean_coef2_t * x_t
        
        return posterior_mean, posterior_variance_t, x_0_pred
    
    def p_sample(self, model, x_t, t, cond, slice_pos=None, x_self_cond=None, guidance_scale=1.0):
        """Single reverse diffusion step"""
        mean, variance, x0_pred = self.p_mean_variance(
            model, x_t, t, cond, slice_pos, x_self_cond, guidance_scale=guidance_scale
        )
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        
        return mean + nonzero_mask * torch.sqrt(variance) * noise, x0_pred
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond, device, slice_pos=None, guidance_scale=1.0,
                     use_self_conditioning=False):
        """Generate samples from noise with optional self-conditioning and CFG"""
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x_self_cond = None
        
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, x0_pred = self.p_sample(model, x, t, cond, slice_pos, x_self_cond, guidance_scale)
            
            # Self-conditioning: use current prediction for next step
            if use_self_conditioning:
                x_self_cond = x0_pred.detach()
        
        return x
