"""DEQ (Deep Equilibrium Model) for autoregressive language modeling."""

import torch
import torch.nn as nn

class DEQLMLayer(nn.Module):
    """Single DEQ layer for fixed-point iteration."""
    
    def __init__(self, hidden_dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, z, x_embed, attn_mask=None):
        """
        DEQ function: z_new = f(z, x_embed)
        
        Args:
            z: Current fixed-point estimate [batch, seq_len, hidden_dim]
            x_embed: Input embeddings [batch, seq_len, hidden_dim]
            attn_mask: Causal mask for autoregressive attention
        Returns:
            z_new: Updated estimate [batch, seq_len, hidden_dim]
        """
        # Self-attention with causal mask
        attn_out, _ = self.self_attn(
            z, z, z,
            attn_mask=attn_mask,
            need_weights=False,
            is_causal=True
        )
        z = self.norm1(z + attn_out)
        
        # FFN
        z = self.norm2(z + self.mlp(z))
        
        # Inject input embedding
        z = z + 0.1 * x_embed
        
        return z


class DEQLM(nn.Module):
    """
    Deep Equilibrium language model.
    
    Finds fixed-point representation z* = f(z*) via iteration,
    then predicts next tokens.
    """
    
    def __init__(self, vocab_size, hidden_dim=512, n_heads=8,
                 max_iter=30, tol=1e-3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.tol = tol
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # DEQ layer (applied iteratively)
        self.deq_layer = DEQLMLayer(hidden_dim, n_heads, dropout)
        
        # Output
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
    
    def find_fixed_point(self, x_embed, max_iter=None):
        """
        Find fixed-point z* via fixed-point iteration.

        Args:
            x_embed: Input embeddings [batch, seq_len, hidden_dim]
            max_iter: Override for iteration count (default: self.max_iter)
        Returns:
            z_star: Fixed-point solution [batch, seq_len, hidden_dim]
        """
        if max_iter is None:
            max_iter = self.max_iter
        _, seq_len, _ = x_embed.shape

        # Initialize z with input embedding
        z = x_embed.clone()

        # Create causal mask
        attn_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x_embed.device
        )

        # Fixed-point iteration
        for i in range(max_iter):
            z_new = self.deq_layer(z, x_embed, attn_mask=attn_mask)

            # Check convergence
            delta = torch.norm(z_new - z) / (torch.norm(z) + 1e-8)
            z = z_new

            if delta < self.tol:
                break

        return z

    def forward(self, x):
        """
        Args:
            x: Input token IDs [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embed
        x_embed = self.embed(x)
        x_embed = self.dropout(x_embed)

        # Find fixed point (fewer iterations during training for speed)
        train_iters = min(15, self.max_iter) if self.training else self.max_iter
        z_star = self.find_fixed_point(x_embed, max_iter=train_iters)
        
        # Output projection
        h = self.norm(z_star)
        logits = self.head(h)
        
        return logits
