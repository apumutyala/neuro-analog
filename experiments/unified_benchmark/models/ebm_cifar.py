"""Energy-Based Model classifier for CIFAR-10."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyFunction(nn.Module):
    """Energy function E(x, y) for classification."""
    
    def __init__(self, feature_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features, class_labels):
        """Compute energy E(x, y)."""
        combined = torch.cat([features, class_labels], dim=1)
        energy = self.net(combined)
        return energy.squeeze(-1)


class EBMClassifier(nn.Module):
    """
    Energy-Based Model for CIFAR-10 classification.
    
    Classification via energy minimization:
    y* = argmin_y E(x, y)
    
    Training uses contrastive divergence to learn energy function.
    """
    
    def __init__(self, num_classes=10, hidden_dim=256):
        super().__init__()
        
        self.num_classes = num_classes
        
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
        self.energy_fn = EnergyFunction(self.flatten_dim, hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Inference: find class with minimum energy.
        """
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        features = features.view(batch_size, -1)
        
        # Compute energy for all classes
        energies = []
        for c in range(self.num_classes):
            class_vec = torch.zeros(batch_size, self.num_classes, device=x.device)
            class_vec[:, c] = 1.0
            
            energy = self.energy_fn(features, class_vec)
            energies.append(energy)
        
        energies = torch.stack(energies, dim=1)  # (batch, num_classes)
        
        # Convert energies to logits (lower energy = higher probability)
        logits = -energies
        
        return logits
