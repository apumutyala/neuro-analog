"""Transformer experiment model — 2-layer classifier on synthetic sequences.

Architecture: 2 transformer layers, dim=24, 2 heads, FFN=48 with ReLU.
Task: Classify whether a 64-token sequence contains adjacent tokens [3, 5].
Metric: Negative cross-entropy loss [higher = better].

IMPORTANT: Built from scratch using nn.Linear directly (NOT nn.TransformerEncoder).
analogize() replaces nn.Linear → AnalogLinear, nn.ReLU → AnalogReLU.
Softmax (in attention) stays digital. LayerNorm stays digital.

We use ReLU (not GELU) since it's analog-native — AnalogReLU uses a diode-connected
transistor model, making the analog fraction higher and the experiment more sensitive.
GELU would be DIGITAL, showing less degradation from the linear layers and more from
the domain-crossing penalty.

Synthetic dataset: sequences of 64 tokens from vocab_size=32.
Pattern: sequence contains tokens [3, 5] adjacent anywhere in the sequence.
Label 1 if present, 0 otherwise. Balanced 50/50 by construction.

seq_len=64, vocab=32: attention heads must scan the full 64-token context to
detect the bigram — the task cannot be solved by local bigram lookup and actually
requires the attention mechanism to work. This avoids the ceiling effect present
in shorter sequences where even random embeddings could learn the pattern.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────────────────────────────────

_VOCAB = 32
_SEQ_LEN = 64
_PATTERN = (3, 5)
_N_TRAIN = 4000
_N_TEST = 1000
_SEED = 42

def _generate_data(n, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    while len(X) < n:
        seq = rng.integers(0, _VOCAB, size=_SEQ_LEN)
        label = 0
        for i in range(_SEQ_LEN - 1):
            if seq[i] == _PATTERN[0] and seq[i+1] == _PATTERN[1]:
                label = 1
                break
        X.append(seq)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def _get_data():
    X_train, y_train = _generate_data(_N_TRAIN, _SEED)
    X_test, y_test = _generate_data(_N_TEST, _SEED + 1)
    return X_train, y_train, X_test, y_test


# ── Model ─────────────────────────────────────────────────────────────────

class _MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention built from nn.Linear (analogizable)."""
    def __init__(self, dim=64, n_heads=2):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class _FFN(nn.Module):
    """Feed-forward network with ReLU (analog-native activation)."""
    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _TransformerLayer(nn.Module):
    def __init__(self, dim=64, n_heads=2, ffn_hidden=128):
        super().__init__()
        self.attn = _MultiHeadSelfAttention(dim, n_heads)
        self.ffn = _FFN(dim, ffn_hidden)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _TransformerClassifier(nn.Module):
    def __init__(self, vocab=16, dim=64, n_heads=2, n_layers=2, ffn_hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([
            _TransformerLayer(dim, n_heads, ffn_hidden) for _ in range(n_layers)
        ])
        self.head = nn.Linear(dim, 2)

    def forward(self, x):
        h = self.embed(x)  # (B, T, D)
        for layer in self.layers:
            h = layer(h)
        return self.head(h.mean(dim=1))  # mean pool → (B, 2)


# ── Standard interface ─────────────────────────────────────────────────────

def create_model() -> nn.Module:
    return _TransformerClassifier(vocab=_VOCAB, dim=24, n_heads=2, n_layers=2, ffn_hidden=48)  # dim=24 operates near capacity on 64-token task


def train_model(model: nn.Module, save_path: str) -> nn.Module:
    X_train, y_train, _, _ = _get_data()
    X_train, y_train = X_train.to(_DEVICE), y_train.to(_DEVICE)
    model = model.to(_DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    batch_size = 128
    n_epochs = 300

    model.train()
    for epoch in range(n_epochs):
        idx = torch.randperm(len(X_train), device=_DEVICE)[:batch_size]
        logits = model(X_train[idx])
        loss = criterion(logits, y_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            acc = evaluate(model)
            print(f"  [Transformer] epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}, test_acc={acc:.3f}")

    torch.save(model.state_dict(), save_path)
    return model


def load_model(save_path: str) -> nn.Module:
    model = create_model()
    model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
    model = model.to(_DEVICE)
    return model


def evaluate(model: nn.Module) -> float:
    """Return negative cross-entropy loss (continuous metric, higher = better).
    
    Cross-entropy measures the full logit distribution, not just argmax.
    Degrades smoothly with analog noise because logit magnitudes matter.
    """
    _, _, X_test, y_test = _get_data()
    X_test, y_test = X_test.to(_DEVICE), y_test.to(_DEVICE)
    model = model.to(_DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        loss = nn.functional.cross_entropy(logits, y_test)
    return -loss.item()  # Negative so higher = better


def evaluate_output_mse(model: nn.Module, digital_baseline: nn.Module) -> float:
    """Compute MSE between analog and digital baseline outputs.
    
    Returns negative MSE so higher = better (consistent with other metrics).
    """
    _, _, X_test, y_test = _get_data()
    X_test = X_test.to(_DEVICE)
    model = model.to(_DEVICE)
    digital_baseline = digital_baseline.to(_DEVICE)
    model.eval()
    digital_baseline.eval()

    with torch.no_grad():
        dig_out = digital_baseline(X_test)
        analog_out = model(X_test)
    
    mse = ((dig_out - analog_out) ** 2).mean().item()
    return -mse


def get_family_name() -> str:
    return "Transformer"
