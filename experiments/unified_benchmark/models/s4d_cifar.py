"""S4D (diagonal SSM) classifier for CIFAR-10."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class S4DKernel(nn.Module):
    """Diagonal structured state space kernel."""
    
    def __init__(self, d_model, d_state=64, lr_A=1e-3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Diagonal A matrix (log-space for stability)
        # Store log of positive magnitudes; negate in forward for stable negative-real A.
        # A_real = -exp(A_log) ensures eigenvalues are always negative (contractive SSM).
        A_mag = torch.arange(1, d_state + 1, dtype=torch.float32) * 0.5  # positive magnitudes
        self.A_log = nn.Parameter(torch.log(A_mag))
        
        # B and C matrices
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        
        # Discretization step size
        self.log_dt = nn.Parameter(torch.rand(d_model) * -1.0)
    
    def forward(self, x):
        """
        Apply S4D kernel via recurrence.
        x: (batch, length, d_model)
        """
        batch, length, _ = x.shape
        
        # Discretize: A_bar = exp(dt * A) where A = -exp(A_log) (negative-real)
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # [d_model, 1]
        A_bar = torch.exp(dt * (-torch.exp(self.A_log)))  # [d_model, d_state] — stable, |A_bar| < 1     
        # B_bar = dt * B
        B_bar = dt * self.B
        
        # Recurrence: h[t] = A_bar * h[t-1] + B_bar * x[t]
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
        outputs = []
        
        for t in range(length):
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x[:, t, :].unsqueeze(-1)
            y_t = (h * self.C.unsqueeze(0)).sum(dim=-1)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


class S4DBlock(nn.Module):
    """S4D block with skip connection and normalization."""
    
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s4d = S4DKernel(d_model, d_state)
        self.dropout = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.dropout(self.s4d(self.norm(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class S4DNet(nn.Module):
    """
    S4D classifier for CIFAR-10.
    
    Architecture:
    - Flatten image to sequence (32×32 = 1024 tokens)
    - Learned embedding
    - Stack of S4D blocks
    - Global average pooling
    - Classification head
    """
    
    def __init__(self, num_classes=10, d_model=256, n_layers=4, d_state=64):
        super().__init__()
        
        # Treat each pixel as a token
        self.img_size = 32
        self.seq_len = self.img_size * self.img_size
        
        # Embed each RGB pixel
        self.pixel_embed = nn.Linear(3, d_model)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        
        # S4D blocks
        self.blocks = nn.ModuleList([
            S4DBlock(d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # Flatten image: (B, 3, 32, 32) → (B, 1024, 3)
        batch_size = x.shape[0]
        x = x.view(batch_size, 3, -1).transpose(1, 2)
        
        # Embed and add position
        x = self.pixel_embed(x)
        x = x + self.pos_embed
        
        # Apply S4D blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        logits = self.head(x)
        return logits
