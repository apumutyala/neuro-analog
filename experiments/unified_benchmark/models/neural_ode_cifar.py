"""Neural ODE classifier for CIFAR-10."""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class ODEFunc(nn.Module):
    """ODE dynamics function: dh/dt = f(h, t)"""
    
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t, h):
        return self.net(h)


class NeuralODENet(nn.Module):
    """
    Continuous-depth ResNet using Neural ODE.
    
    Architecture:
    - Convolutional feature extractor (downsample 32×32 → 4×4)
    - Flatten to vector
    - Neural ODE block (continuous depth evolution)
    - Linear classifier
    """
    
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16×16
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8×8
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4×4
        )
        
        self.flatten_dim = 256 * 4 * 4
        self.projection = nn.Linear(self.flatten_dim, hidden_dim)
        
        self.ode_func = ODEFunc(hidden_dim)
        self.register_buffer('integration_time', torch.tensor([0.0, 1.0]))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        h0 = self.projection(features)
        
        # Solve ODE: h(t=1) = h(t=0) + ∫[0,1] f(h(t), t) dt
        ht = odeint(self.ode_func, h0, self.integration_time, 
                    method='dopri5', rtol=1e-3, atol=1e-4)
        h1 = ht[-1]  # Final state at t=1
        
        logits = self.classifier(h1)
        return logits
