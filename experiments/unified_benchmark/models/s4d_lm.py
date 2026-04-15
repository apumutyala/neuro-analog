"""S4D (diagonal SSM) for autoregressive language modeling."""

import torch
import torch.nn as nn
import math

class S4DKernel(nn.Module):
    """Diagonal structured state space kernel."""
    
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Diagonal A matrix (log-space for stability)
        # Store log of positive magnitudes; negate in forward for stable negative-real A.
        A_mag = torch.arange(1, d_state + 1, dtype=torch.float32) * 0.5
        self.A_log = nn.Parameter(torch.log(A_mag))
        
        # B and C matrices
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        
        # Discretization step size
        self.log_dt = nn.Parameter(torch.rand(d_model) * -1.0)
    
    def forward(self, x):
        """
        Apply S4D kernel via recurrence.
        
        Args:
            x: Input sequence [batch, seq_len, d_model]
        Returns:
            Output sequence [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # Discretize: A_bar = exp(dt * A) where A = -exp(A_log) (negative-real, contractive)
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # [d_model, 1]
        A_bar = torch.exp(dt * (-torch.exp(self.A_log)))  # [d_model, d_state]
        
        # B_bar = dt * B
        B_bar = dt * self.B  # [d_model, d_state]
        
        # Recurrence: h[t] = A_bar * h[t-1] + B_bar * x[t]
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x[:, t, :].unsqueeze(-1)
            y_t = (h * self.C.unsqueeze(0)).sum(dim=-1)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


class S4DBlock(nn.Module):
    """S4D block with skip connection and normalization."""
    
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.s4d = S4DKernel(d_model, d_state)
        self.dropout1 = nn.Dropout(dropout)
        
        # FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # S4D layer with residual
        x = x + self.dropout1(self.s4d(self.norm1(x)))
        
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class S4DLM(nn.Module):
    """
    S4D language model.
    
    Uses diagonal state-space recurrence for sequence modeling.
    S4D is designed for long sequences and should excel at language tasks.
    """
    
    def __init__(self, vocab_size, hidden_dim=512, n_layers=6, 
                 d_state=64, dropout=0.1):
        super().__init__()
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # S4D layers
        self.layers = nn.ModuleList([
            S4DBlock(hidden_dim, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
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
        
        # Apply S4D blocks (autoregressive by construction)
        for layer in self.layers:
            h = layer(h)
        
        # Output projection
        h = self.norm(h)
        logits = self.head(h)
        
        return logits
