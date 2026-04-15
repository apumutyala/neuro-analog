"""Deep Equilibrium Model classifier for CIFAR-10."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DEQLayer(nn.Module):
    """Fixed-point residual layer: z* = f(z*, x)

    Spectral normalization keeps the Lipschitz constant ≤ 1, which guarantees
    the fixed-point map is contractive and iteration converges.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))
    
    def forward(self, z, x):
        """One iteration: z_next = f(z, x)"""
        out = self.fc1(z) + self.fc2(x)
        out = F.relu(out)
        out = self.fc3(out)
        return out


class DEQNet(nn.Module):
    """
    Deep Equilibrium Network for CIFAR-10.
    
    Architecture:
    - Conv feature extractor
    - Fixed-point solver: find z* such that z* = f(z*, x)
    - Linear classifier
    
    DEQ solves: z* = argmin_z ||z - f(z, x)||
    via fixed-point iteration with max_iter steps.
    """
    
    def __init__(self, num_classes=10, hidden_dim=256, max_iter=30, tol=1e-3):
        super().__init__()
        
        self.max_iter = max_iter
        self.tol = tol
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.flatten_dim = 256 * 4 * 4
        self.projection = nn.Linear(self.flatten_dim, hidden_dim)
        
        self.deq_layer = DEQLayer(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        x_proj = self.projection(features)
        
        # Fixed-point iteration
        z = torch.zeros_like(x_proj)
        
        for i in range(self.max_iter):
            z_next = self.deq_layer(z, x_proj)
            
            # Check convergence
            if torch.norm(z_next - z) < self.tol:
                break
            
            z = z_next
        
        logits = self.classifier(z)
        return logits
