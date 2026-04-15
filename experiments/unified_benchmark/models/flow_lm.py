"""Normalizing Flow for autoregressive language modeling."""

import torch
import torch.nn as nn

class CouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        half_dim = hidden_dim // 2
        
        # Transform network for second half
        self.net = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, half_dim * 2)  # Output scale and shift
        )
    
    def forward(self, x):
        """
        Affine coupling transformation.
        
        Args:
            x: Input [batch, seq_len, hidden_dim]
        Returns:
            z: Transformed output [batch, seq_len, hidden_dim]
            log_det: Log determinant of Jacobian (for MLE training)
        """
        half_dim = self.hidden_dim // 2
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        
        # Transform second half based on first half
        params = self.net(x1)
        log_scale, shift = params[..., :half_dim], params[..., half_dim:]
        
        # Affine transformation
        scale = torch.sigmoid(log_scale + 2.0)  # Stabilize around 1
        z2 = x2 * scale + shift
        
        # Combine
        z = torch.cat([x1, z2], dim=-1)
        
        # Log determinant
        log_det = torch.sum(torch.log(scale), dim=-1)
        
        return z, log_det


class FlowLM(nn.Module):
    """
    Normalizing Flow language model.
    
    Uses sequence of invertible coupling layers to refine embeddings
    before prediction.
    """
    
    def __init__(self, vocab_size, hidden_dim=512, n_flows=8, dropout=0.1):
        super().__init__()
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Flow layers
        self.flows = nn.ModuleList([
            CouplingLayer(hidden_dim, dropout)
            for _ in range(n_flows)
        ])
        
        # Output
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: Input token IDs [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embed
        h = self.embed(x)
        h = self.dropout(h)
        
        # Apply flow transformations
        total_log_det = 0
        for flow in self.flows:
            h, log_det = flow(h)
            total_log_det = total_log_det + log_det
        
        # Output projection
        h = self.norm(h)
        logits = self.head(h)
        
        return logits
