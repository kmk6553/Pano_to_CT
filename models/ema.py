"""
Exponential Moving Average (EMA) wrapper for models
"""

import torch
from copy import deepcopy


class EMAWrapper:
    """Exponential Moving Average wrapper for models"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self):
        """Update EMA parameters"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def get_model(self):
        """Get EMA model for evaluation"""
        return self.ema_model
