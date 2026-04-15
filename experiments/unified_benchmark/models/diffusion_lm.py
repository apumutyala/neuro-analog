"""Diffusion model for autoregressive language modeling."""

import torch
import torch.nn as nn
import math

class ScoreNetwork(nn.Module):
    """Score network for denoising diffusion."""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim * 2),  # +1 for time embedding
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, h, t):
        """
        Predict score (gradient of log-density).
        
        Args:
            h: Noisy states [batch, seq_len, hidden_dim]
            t: Time step (scalar or [batch])
        Returns:
            score: Predicted score [batch, seq_len, hidden_dim]
        """
        # Time embedding
        if isinstance(t, (int, float)):
            t = torch.full((h.shape[0],), t, device=h.device)
        
        t_embed = t.view(-1, 1, 1).expand(h.shape[0], h.shape[1], 1)
        
        # Concatenate h and time
        h_t = torch.cat([h, t_embed], dim=-1)
        
        return self.net(h_t)


class DiffusionLM(nn.Module):
    """
    Diffusion language model.
    
    Uses denoising diffusion to refine embeddings before prediction.
    This is a creative adaptation - diffusion is naturally generative,
    not discriminative.
    """
    
    def __init__(self, vocab_size, hidden_dim=512, n_steps=20, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        
        # Embeddings
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Score network
        self.score_net = ScoreNetwork(hidden_dim, dropout)
        
        # Diffusion schedule (linear for simplicity)
        betas = torch.linspace(0.0001, 0.02, n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # Output
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
    
    def add_noise(self, h, t):
        """
        Add noise at timestep t via forward diffusion.
        
        Args:
            h: Clean embeddings [batch, seq_len, hidden_dim]
            t: Time step (0 to n_steps-1)
        Returns:
            h_noisy: Noisy embeddings
        """
        noise = torch.randn_like(h)
        alpha_t = self.alphas_cumprod[t]
        
        h_noisy = math.sqrt(alpha_t) * h + math.sqrt(1 - alpha_t) * noise
        return h_noisy
    
    def denoise_step(self, h_t, t):
        """
        Single reverse diffusion step.
        
        Args:
            h_t: Noisy state at time t
            t: Current timestep
        Returns:
            h_{t-1}: Denoised state
        """
        # Predict score
        score = self.score_net(h_t, t / self.n_steps)
        
        # Reverse diffusion update
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        
        h_prev = (h_t - beta_t / math.sqrt(1 - self.alphas_cumprod[t]) * score) / math.sqrt(alpha_t)
        
        # Add noise for t > 0
        if t > 0 and self.training:
            noise = torch.randn_like(h_prev) * math.sqrt(beta_t)
            h_prev = h_prev + noise
        
        return h_prev
    
    def reverse_diffusion(self, h, n_steps=None):
        """
        Denoise embeddings via reverse diffusion.

        Args:
            h: Initial (noisy) embeddings [batch, seq_len, hidden_dim]
            n_steps: Override for number of denoising steps (default: self.n_steps)
        Returns:
            h_clean: Denoised embeddings
        """
        if n_steps is None:
            n_steps = self.n_steps
        h_t = h

        # Reverse process: T → T-1 → ... → 0
        for t in reversed(range(n_steps)):
            h_t = self.denoise_step(h_t, t)

        return h_t

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

        # Add noise during training (self-supervised denoising)
        if self.training:
            t = torch.randint(0, self.n_steps, (1,), device=h.device).item()
            h = self.add_noise(h, t)

        # Denoise via reverse diffusion (fewer steps during training for speed)
        train_steps = max(5, self.n_steps // 4) if self.training else self.n_steps
        h_clean = self.reverse_diffusion(h, n_steps=train_steps)
        
        # Output projection
        h_clean = self.norm(h_clean)
        logits = self.head(h_clean)
        
        return logits
