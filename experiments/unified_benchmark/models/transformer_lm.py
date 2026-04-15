"""Standard Transformer for autoregressive language modeling."""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor [batch, seq_len]
        Returns:
            Positional encoding [seq_len, d_model]
        """
        return self.pe[:x.size(1), :]

class TransformerLM(nn.Module):
    """GPT-style decoder-only Transformer for language modeling."""
    
    def __init__(self, vocab_size, hidden_dim=512, n_layers=6,
                 n_heads=8, dropout=0.1, max_len=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie weights (common LM trick)
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following GPT-2 scheme."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: Input token IDs [batch, seq_len]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # Embeddings with scaling + positional encoding
        h = self.embed(x) * math.sqrt(self.hidden_dim)
        h = h + self.pos_enc(x).unsqueeze(0)  # Broadcast across batch
        h = self.dropout(h)
        
        # Causal mask for autoregressive generation
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )
        
        # Transformer layers
        h = self.transformer(h, mask=mask, is_causal=True)
        
        # Output projection
        h = self.norm(h)
        logits = self.head(h)
        
        return logits
