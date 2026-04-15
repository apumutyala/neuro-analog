"""Energy-Based Model for autoregressive language modeling."""

import torch
import torch.nn as nn

class EnergyFunction(nn.Module):
    """Energy function for EBM."""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, h):
        """
        Compute energy for each token position.
        
        Args:
            h: Hidden states [batch, seq_len, hidden_dim]
        Returns:
            energy: Scalar energy [batch, seq_len]
        """
        return self.net(h).squeeze(-1)


class EBMLM(nn.Module):
    """
    Energy-Based language model.
    
    Minimizes energy of embeddings via Langevin dynamics,
    then predicts next tokens.
    """
    
    def __init__(self, vocab_size, hidden_dim=512, n_steps=10,
                 step_size=0.1, dropout=0.1):
        super().__init__()
        self.n_steps = n_steps
        self.step_size = step_size
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Energy function
        self.energy_fn = EnergyFunction(hidden_dim, dropout)
        
        # Output
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
    
    def minimize_energy(self, h, n_steps=None):
        """
        Minimize energy via Langevin dynamics.

        Args:
            h: Initial states [batch, seq_len, hidden_dim]
            n_steps: Override for number of Langevin steps (default: self.n_steps)
        Returns:
            h_min: Energy-minimized states [batch, seq_len, hidden_dim]
        """
        if n_steps is None:
            n_steps = self.n_steps

        # Detach from training graph; we'll track h manually through each step.
        # Use enable_grad() so this works correctly even when called from within
        # a torch.no_grad() context (e.g. compute_perplexity at eval time).
        h = h.detach()

        for step in range(n_steps):
            h = h.requires_grad_(True)
            with torch.enable_grad():
                energy = self.energy_fn(h).sum()
                grad = torch.autograd.grad(energy, h, create_graph=False)[0]

            # Langevin update: h ← h - step_size * grad + noise
            noise = torch.randn_like(h) * 0.01 if self.training else 0.0
            h = (h - self.step_size * grad + noise).detach()

        return h

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

        # Minimize energy (fewer steps during training for speed)
        train_steps = max(5, self.n_steps // 2) if self.training else self.n_steps
        h_min = self.minimize_energy(h, n_steps=train_steps)
        
        # Output projection
        h_min = self.norm(h_min)
        logits = self.head(h_min)
        
        return logits
