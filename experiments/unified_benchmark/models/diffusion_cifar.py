"""Diffusion model classifier for CIFAR-10."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ScoreNetwork(nn.Module):
    """Score network s_θ(x_t, t) for diffusion model."""
    
    def __init__(self, feature_dim, hidden_dim, time_dim=128):
        super().__init__()
        
        self.time_embed = TimeEmbedding(time_dim)
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        combined = torch.cat([x, t_emb], dim=1)
        return self.net(combined)


class DiffusionClassifier(nn.Module):
    """
    Diffusion-based classifier for CIFAR-10.
    
    Uses class-conditional diffusion model.
    Classification via denoising score matching.
    """
    
    def __init__(self, num_classes=10, hidden_dim=256, n_steps=10):
        super().__init__()
        
        self.num_classes = num_classes
        self.n_steps = n_steps
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.feature_dim = 256 * 4 * 4
        
        # One score network per class
        self.score_networks = nn.ModuleList([
            ScoreNetwork(self.feature_dim, hidden_dim)
            for _ in range(num_classes)
        ])
        
        # Diffusion schedule
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, n_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
    
    def denoise_step(self, x_t, t, score_net):
        """Single denoising step."""
        t_idx = t.long()
        alpha_t = self.alphas_cumprod[t_idx][:, None]
        
        score = score_net(x_t, t)
        x_0_pred = (x_t - (1 - alpha_t).sqrt() * score) / alpha_t.sqrt()
        
        return x_0_pred
    
    def forward(self, x):
        """
        Classification via denoising reconstruction error.
        Class with lowest reconstruction error is predicted.
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # Add noise (forward diffusion)
        noise = torch.randn_like(features)
        t = torch.randint(0, self.n_steps, (features.shape[0],), device=features.device)
        alpha_t = self.alphas_cumprod[t][:, None]
        x_t = alpha_t.sqrt() * features + (1 - alpha_t).sqrt() * noise
        
        # Try denoising with each class-conditional score network
        reconstruction_errors = []
        for score_net in self.score_networks:
            x_0_pred = self.denoise_step(x_t, t.float(), score_net)
            error = F.mse_loss(x_0_pred, features, reduction='none').sum(dim=1)
            reconstruction_errors.append(error)
        
        errors = torch.stack(reconstruction_errors, dim=1)
        
        # Convert errors to logits (lower error = higher probability)
        logits = -errors
        
        return logits
