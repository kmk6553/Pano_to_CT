"""
Learning rate schedulers
"""

import numpy as np


class WarmupCosineScheduler:
    """Cosine scheduler with linear warmup"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, base_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing with progress clamping
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler for stable VAE training"""
    def __init__(self, optimizer, initial_lr, target_lr, warmup_epochs=10, stability_threshold=0.1):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.stability_threshold = stability_threshold
        self.current_epoch = 0
        self.loss_history = []
        self.is_stable = False
        
    def step(self, loss):
        self.current_epoch += 1
        self.loss_history.append(loss)
        
        # Stability check
        if len(self.loss_history) >= 5:
            recent_losses = self.loss_history[-5:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            if loss_mean > 0 and loss_std / loss_mean < self.stability_threshold:
                self.is_stable = True
        
        # Adjust learning rate
        if self.current_epoch <= self.warmup_epochs:
            # Initial warmup phase
            lr = self.initial_lr
        elif self.is_stable:
            # Gradual increase after stabilization
            progress = min(1.0, (self.current_epoch - self.warmup_epochs) / self.warmup_epochs)
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * progress
        else:
            # Keep initial LR if unstable
            lr = self.initial_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def state_dict(self):
        """Serializable state"""
        return {
            'current_epoch': self.current_epoch,
            'loss_history': list(self.loss_history),
            'is_stable': self.is_stable,
            'initial_lr': self.initial_lr,
            'target_lr': self.target_lr,
            'warmup_epochs': self.warmup_epochs,
            'stability_threshold': self.stability_threshold
        }
    
    def load_state_dict(self, state_dict):
        """Restore saved state"""
        self.current_epoch = state_dict['current_epoch']
        self.loss_history = state_dict['loss_history']
        self.is_stable = state_dict['is_stable']
        self.initial_lr = state_dict.get('initial_lr', self.initial_lr)
        self.target_lr = state_dict.get('target_lr', self.target_lr)
        self.warmup_epochs = state_dict.get('warmup_epochs', self.warmup_epochs)
        self.stability_threshold = state_dict.get('stability_threshold', self.stability_threshold)
