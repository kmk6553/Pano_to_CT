"""
Diffusion process implementation with v-parametrization and CFG
Supports 3D latent volumes [B, C, D=3, h, w]
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class DiffusionProcess(nn.Module):
    """
    Diffusion process with v-parametrization and Classifier-Free Guidance
    
    Supports both 2D [B, C, H, W] and 3D [B, C, D, H, W] latent representations
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, 
                 schedule='cosine', prediction_type='v'):
        """
        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule: 'linear' or 'cosine'
            prediction_type: 'v', 'epsilon', or 'x0'
        """
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
    
    def _expand_dims(self, tensor, target):
        """
        Expand tensor dimensions to match target
        Works for both 2D [B, C, H, W] and 3D [B, C, D, H, W] targets
        """
        while tensor.dim() < target.dim():
            tensor = tensor.unsqueeze(-1)
        return tensor
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: [B, C, D, H, W] or [B, C, H, W] - clean data
            t: [B] - timesteps
            noise: Optional noise tensor
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._expand_dims(self.sqrt_alphas_cumprod[t], x_start)
        sqrt_one_minus_alphas_cumprod_t = self._expand_dims(self.sqrt_one_minus_alphas_cumprod[t], x_start)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_x0_from_eps(self, x_t, t, eps):
        """Predict x0 from epsilon parametrization"""
        sqrt_recip_alphas_cumprod_t = self._expand_dims(self.sqrt_recip_alphas_cumprod[t], x_t)
        sqrt_recipm1_alphas_cumprod_t = self._expand_dims(self.sqrt_recipm1_alphas_cumprod[t], x_t)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps
    
    def predict_x0_from_v(self, x_t, t, v):
        """Predict x0 from v-parametrization"""
        sqrt_alphas_cumprod_t = self._expand_dims(self.sqrt_alphas_cumprod[t], x_t)
        sqrt_one_minus_alphas_cumprod_t = self._expand_dims(self.sqrt_one_minus_alphas_cumprod[t], x_t)
        return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * v
    
    def get_v(self, x_start, noise, t):
        """Get v target from x0 and noise"""
        sqrt_alphas_cumprod_t = self._expand_dims(self.sqrt_alphas_cumprod[t], x_start)
        sqrt_one_minus_alphas_cumprod_t = self._expand_dims(self.sqrt_one_minus_alphas_cumprod[t], x_start)
        return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start
    
    def predict_eps_from_v(self, x_t, t, v):
        """Convert v prediction to epsilon"""
        sqrt_alphas_cumprod_t = self._expand_dims(self.sqrt_alphas_cumprod[t], x_t)
        sqrt_one_minus_alphas_cumprod_t = self._expand_dims(self.sqrt_one_minus_alphas_cumprod[t], x_t)
        x0_pred = self.predict_x0_from_v(x_t, t, v)
        return (x_t - sqrt_alphas_cumprod_t * x0_pred) / sqrt_one_minus_alphas_cumprod_t
    
    def predict_x0_from_noise(self, x_t, t, noise):
        """Predict x0 from xt and noise (alias for epsilon method)"""
        return self.predict_x0_from_eps(x_t, t, noise)
    
    def p_mean_variance(self, model, x_t, t, cond, slice_pos=None, x_self_cond=None, 
                       clip_denoised=True, guidance_scale=1.0):
        """
        Compute mean and variance for reverse process with CFG support
        
        Args:
            model: Diffusion model
            x_t: [B, C, D, H, W] or [B, C, H, W] - noisy data
            t: [B] - timesteps
            cond: Condition tensor
            slice_pos: [B] - slice positions
            x_self_cond: Previous prediction for self-conditioning
            clip_denoised: Whether to clip predicted x0
            guidance_scale: CFG scale (1.0 = no guidance)
        """
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
        posterior_mean_coef1_t = self._expand_dims(self.posterior_mean_coef1[t], x_t)
        posterior_mean_coef2_t = self._expand_dims(self.posterior_mean_coef2[t], x_t)
        posterior_variance_t = self._expand_dims(self.posterior_variance[t], x_t)
        
        posterior_mean = posterior_mean_coef1_t * x_0_pred + posterior_mean_coef2_t * x_t
        
        return posterior_mean, posterior_variance_t, x_0_pred
    
    def p_sample(self, model, x_t, t, cond, slice_pos=None, x_self_cond=None, guidance_scale=1.0):
        """Single reverse diffusion step"""
        mean, variance, x0_pred = self.p_mean_variance(
            model, x_t, t, cond, slice_pos, x_self_cond, guidance_scale=guidance_scale
        )
        
        noise = torch.randn_like(x_t)
        # Create mask for t != 0
        nonzero_mask = (t != 0).float()
        nonzero_mask = self._expand_dims(nonzero_mask, x_t)
        
        return mean + nonzero_mask * torch.sqrt(variance) * noise, x0_pred
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond, device, slice_pos=None, guidance_scale=1.0,
                     use_self_conditioning=False, show_progress=True):
        """
        Generate samples from noise with optional self-conditioning and CFG
        
        Args:
            model: Diffusion model
            shape: Output shape, e.g., [B, C, D, H, W] for 3D or [B, C, H, W] for 2D
            cond: Condition tensor
            device: Target device
            slice_pos: [B] - slice positions
            guidance_scale: CFG scale
            use_self_conditioning: Whether to use self-conditioning
            show_progress: Whether to show progress bar
        Returns:
            Generated sample with same shape as input
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x_self_cond = None
        
        iterator = reversed(range(self.num_timesteps))
        if show_progress:
            iterator = tqdm(iterator, desc='Sampling', total=self.num_timesteps)
        
        for i in iterator:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, x0_pred = self.p_sample(model, x, t, cond, slice_pos, x_self_cond, guidance_scale)
            
            # Self-conditioning: use current prediction for next step
            if use_self_conditioning:
                x_self_cond = x0_pred.detach()
        
        return x
    
    @torch.no_grad()
    def p_sample_loop_progressive(self, model, shape, cond, device, slice_pos=None, 
                                  guidance_scale=1.0, use_self_conditioning=False,
                                  save_intermediates=False, intermediate_steps=10):
        """
        Generate samples with optional intermediate saving
        
        Args:
            save_intermediates: Whether to save intermediate results
            intermediate_steps: Number of intermediate results to save
        Returns:
            Final sample and optionally list of intermediates
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x_self_cond = None
        
        intermediates = []
        save_interval = self.num_timesteps // intermediate_steps
        
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, x0_pred = self.p_sample(model, x, t, cond, slice_pos, x_self_cond, guidance_scale)
            
            if use_self_conditioning:
                x_self_cond = x0_pred.detach()
            
            if save_intermediates and (i % save_interval == 0 or i == 0):
                intermediates.append(x0_pred.cpu())
        
        if save_intermediates:
            return x, intermediates
        return x
    
    @torch.no_grad()
    def ddim_sample(self, model, shape, cond, device, slice_pos=None, guidance_scale=1.0,
                   use_self_conditioning=False, ddim_steps=50, eta=0.0):
        """
        DDIM sampling for faster generation
        
        Args:
            ddim_steps: Number of DDIM steps (fewer than num_timesteps)
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM)
        """
        batch_size = shape[0]
        
        # Create DDIM timestep schedule
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))
        
        x = torch.randn(shape, device=device)
        x_self_cond = None
        
        for i, t in enumerate(tqdm(timesteps, desc='DDIM Sampling')):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get model prediction
            model_output = model(x, t_tensor, cond, slice_pos, x_self_cond)
            
            # CFG
            if guidance_scale != 1.0:
                uncond = torch.zeros_like(cond)
                model_output_uncond = model(x, t_tensor, uncond, slice_pos, x_self_cond)
                model_output = model_output_uncond + guidance_scale * (model_output - model_output_uncond)
            
            # Predict x0
            if self.prediction_type == 'v':
                x0_pred = self.predict_x0_from_v(x, t_tensor, model_output)
            elif self.prediction_type == 'epsilon':
                x0_pred = self.predict_x0_from_eps(x, t_tensor, model_output)
            else:
                x0_pred = model_output
            
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # Self-conditioning
            if use_self_conditioning:
                x_self_cond = x0_pred.detach()
            
            # Get previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0
            
            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0)
            
            # Expand for broadcasting
            alpha_t = self._expand_dims(alpha_t.unsqueeze(0).expand(batch_size), x)
            alpha_t_prev = self._expand_dims(alpha_t_prev.unsqueeze(0).expand(batch_size), x)
            
            # Predict noise
            if self.prediction_type == 'v':
                eps_pred = self.predict_eps_from_v(x, t_tensor, model_output)
            elif self.prediction_type == 'epsilon':
                eps_pred = model_output
            else:
                eps_pred = (x - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(1 - alpha_t)
            
            # DDIM formula
            sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
            
            pred_x0_coef = torch.sqrt(alpha_t_prev)
            pred_eps_coef = torch.sqrt(1 - alpha_t_prev - sigma ** 2)
            
            x = pred_x0_coef * x0_pred + pred_eps_coef * eps_pred
            
            if t_prev > 0 and eta > 0:
                noise = torch.randn_like(x)
                x = x + sigma * noise
        
        return x