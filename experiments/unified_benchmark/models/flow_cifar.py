"""Normalizing Flow classifier for CIFAR-10."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        half_dim = dim // 2
        
        self.net = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - half_dim)
        )
        
        self.scale_net = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - half_dim)
        )
    
    def forward(self, x, reverse=False):
        half = x.shape[1] // 2
        x1, x2 = x[:, :half], x[:, half:]
        
        if not reverse:
            shift = self.net(x1)
            scale = torch.sigmoid(self.scale_net(x1) + 2.0)
            x2 = (x2 + shift) * scale
            log_det = torch.sum(torch.log(scale), dim=1)
        else:
            shift = self.net(x1)
            scale = torch.sigmoid(self.scale_net(x1) + 2.0)
            x2 = x2 / scale - shift
            log_det = -torch.sum(torch.log(scale), dim=1)
        
        return torch.cat([x1, x2], dim=1), log_det


class FlowModel(nn.Module):
    """Stack of coupling layers forming a normalizing flow."""
    
    def __init__(self, dim, n_flows=8, hidden_dim=256):
        super().__init__()
        self.flows = nn.ModuleList([
            CouplingLayer(dim, hidden_dim) for _ in range(n_flows)
        ])
    
    def forward(self, x, reverse=False):
        log_det_sum = 0
        
        flows = self.flows if not reverse else reversed(self.flows)
        
        for flow in flows:
            x, log_det = flow(x, reverse=reverse)
            log_det_sum = log_det_sum + log_det
        
        return x, log_det_sum
    
    def log_prob(self, x):
        """Compute log p(x) = log p(z) + log |det df/dz|"""
        z, log_det = self.forward(x, reverse=False)
        
        # Standard Gaussian prior
        log_p_z = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.shape[1] * math.log(2 * math.pi)
        
        return log_p_z + log_det


class FlowClassifier(nn.Module):
    """
    Flow-based classifier for CIFAR-10.
    
    Uses class-conditional normalizing flow:
    p(x | y) learned via maximum likelihood
    Classification: y* = argmax_y p(x | y)
    """
    
    def __init__(self, num_classes=10, n_flows=8):
        super().__init__()
        
        self.num_classes = num_classes
        
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
        
        # One flow per class
        self.flows = nn.ModuleList([
            FlowModel(self.feature_dim, n_flows=n_flows, hidden_dim=512)
            for _ in range(num_classes)
        ])
    
    def forward(self, x):
        """Compute log p(x | y) for all classes."""
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        log_probs = []
        for flow in self.flows:
            log_p = flow.log_prob(features)
            log_probs.append(log_p)
        
        logits = torch.stack(log_probs, dim=1)
        return logits
