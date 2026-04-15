"""Neural ODE for autoregressive language modeling."""

import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """ODE function for continuous-depth transformation."""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, t, x):
        """
        Args:
            t: Time (scalar)
            x: State [batch, seq_len, hidden_dim]
        Returns:
            dx/dt: [batch, seq_len, hidden_dim]
        """
        return self.net(x)


class NeuralODELM(nn.Module):
    """
    Neural ODE language model.
    
    Uses continuous-depth ODE integration over *depth* (not time).
    Sequence is processed in parallel like Transformer.
    """
    
    def __init__(self, vocab_size, hidden_dim=512, n_layers=6, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # ODE function (shared across depth)
        self.ode_func = ODEFunc(hidden_dim, dropout)
        
        # Integration time points (analogous to layer depths)
        self.register_buffer(
            'integration_times',
            torch.linspace(0, 1, n_layers + 1)
        )
        
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
        
        # Integrate ODE over depth (continuous transformation)
        # Note: ODE integrates over "depth" not "time" - sequence is parallel
        h_traj = odeint(
            self.ode_func,
            h,
            self.integration_times,
            method='dopri5',  # Adaptive Runge-Kutta
            rtol=1e-3,
            atol=1e-4
        )
        
        # Take final depth state
        h = h_traj[-1]
        
        # Output projection
        h = self.norm(h)
        logits = self.head(h)
        
        return logits
