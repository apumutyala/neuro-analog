"""Vision Transformer classifier for CIFAR-10."""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """Convert image to sequence of patch embeddings."""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.position_embeddings
        
        return x


class TransformerBlock(nn.Module):
    """Standard transformer encoder block."""
    
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTClassifier(nn.Module):
    """
    Vision Transformer for CIFAR-10 classification.
    
    Architecture:
    - Patch embedding (32×32 → 64 patches of 4×4)
    - 6 transformer layers
    - Classification head on [CLS] token
    """
    
    def __init__(self, num_classes=10, img_size=32, patch_size=4, 
                 d_model=256, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels=3, embed_dim=d_model
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        
        return logits
