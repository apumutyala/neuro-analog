"""Diffusion model experiment — Tiny DDPM on 8x8 MNIST.

Architecture: Score network is 3-layer MLP (NOT U-Net).
  Input: 64-dim flattened 8x8 image + 16-dim timestep sinusoidal embedding → 80 dims
  Architecture: Linear(80, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 64)
  T=100 diffusion steps, linear beta schedule beta_1=1e-4, beta_T=0.02.

Metric: Reconstruction quality proxy — generate 500 samples via DDIM,
  compute mean distance from each generated sample to its nearest test neighbor.
  Negate so HIGHER = BETTER (lower distance = higher quality).

DOUBT NOTED: FID requires Inception network features (too heavy for CPU).
We use the "simplified Wasserstein" proxy: for each generated sample x_gen,
find the nearest test sample by L2 distance. Average minimum distance.
This measures whether generated samples cover the data manifold.
Perfect generator: distance → 0. Random noise: distance >> 0.

DOUBT NOTED: The directive says to use DDIM sampling (deterministic ODE).
DDIM reduces T=100 stochastic DDPM steps to ~10 deterministic steps.
This makes the evaluation reproducible across analog trials.
We use 10 DDIM steps for evaluation speed.

DOUBT NOTED: 8x8 MNIST is not in torchvision directly. We downsample 28x28 MNIST
using bilinear interpolation. If torchvision is not installed, we generate
synthetic Gaussian blob data that approximates MNIST's structure.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ── Dataset ───────────────────────────────────────────────────────────────

_IMG_DIM = 64   # 8x8 = 64
_T = 100
_BETA_START = 1e-4
_BETA_END = 0.02

def _get_betas():
    return torch.linspace(_BETA_START, _BETA_END, _T)

def _get_alphas(betas):
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return alphas, alphas_bar

def _load_mnist_8x8(split="train", n=None):
    """Load 8x8 MNIST. Falls back to Gaussian blobs if torchvision unavailable."""
    try:
        import torchvision
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize(8),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        dataset = torchvision.datasets.MNIST(
            root=os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"),
            train=(split == "train"), download=True, transform=transform,
        )
        x = torch.stack([dataset[i][0].flatten() for i in range(len(dataset) if n is None else min(n, len(dataset)))])
        return x
    except Exception:
        pass

    # sklearn digits: 1797 samples of 8x8 images (0–16), normalize to [-1, 1]
    from sklearn.datasets import load_digits
    data = load_digits()
    X = (data.data.astype(np.float32) / 8.0) - 1.0  # [0,16] → [-1, 1]
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    idx = perm[:split_idx] if split == "train" else perm[split_idx:]
    n = n or len(idx)
    return torch.tensor(X[idx[:n]])

def _get_data():
    X_train = _load_mnist_8x8("train", n=5000)
    X_test = _load_mnist_8x8("test", n=500)
    return X_train, X_test


# ── Model ─────────────────────────────────────────────────────────────────

class _SinusoidalEmbed(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch,) integer timesteps 0..T-1
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _ScoreNet(nn.Module):
    """3-layer MLP score network: epsilon_theta(x_t, t) → noise estimate."""
    def __init__(self, img_dim=64, t_embed_dim=16):
        super().__init__()
        self.t_embed = _SinusoidalEmbed(t_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(img_dim + t_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
        )

    def forward(self, x_t, t):
        t_emb = self.t_embed(t)
        h = torch.cat([x_t, t_emb], dim=-1)
        return self.net(h)


# ── Standard interface ─────────────────────────────────────────────────────

def create_model() -> nn.Module:
    return _ScoreNet(img_dim=_IMG_DIM, t_embed_dim=16)


def train_model(model: nn.Module, save_path: str) -> nn.Module:
    X_train, _ = _get_data()
    betas = _get_betas()
    _, alphas_bar = _get_alphas(betas)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    batch_size = 256
    n_epochs = 400
    model.train()

    for epoch in range(n_epochs):
        idx = torch.randperm(len(X_train))[:batch_size]
        x0 = X_train[idx]
        t = torch.randint(0, _T, (batch_size,))
        noise = torch.randn_like(x0)
        alpha_t = alphas_bar[t].unsqueeze(-1)
        x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        pred_noise = model(x_t, t)
        loss = ((pred_noise - noise) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  [Diffusion] epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    return model


def load_model(save_path: str) -> nn.Module:
    model = create_model()
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    return model


def evaluate(model: nn.Module, analog_substrate: str = "classic") -> float:
    """Generate 500 samples, compute negative mean nearest-neighbor distance.
    
    Args:
        model: Score network
        analog_substrate: "classic" (DDIM), "cld" (RLC), or "extropic_dtm" (Gibbs chain)
    """
    _, X_test = _get_data()
    betas = _get_betas()
    alphas, alphas_bar = _get_alphas(betas)

    model.eval()
    n_gen = 500
    n_ddim = 10  # DDIM steps for fast deterministic sampling
    ddim_steps = torch.linspace(_T - 1, 0, n_ddim + 1).long()

    x = torch.randn(n_gen, _IMG_DIM)

    with torch.no_grad():
        if analog_substrate == "classic":
            # Classic DDIM (deterministic ODE)
            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_next = ddim_steps[i + 1].item()
                t_tensor = torch.full((n_gen,), t_curr, dtype=torch.long)

                eps = model(x, t_tensor)
                alpha_curr = alphas_bar[t_curr]
                x0_pred = (x - math.sqrt(1 - alpha_curr) * eps) / math.sqrt(alpha_curr)
                x0_pred = x0_pred.clamp(-1, 1)

                if t_next >= 0:
                    alpha_next = alphas_bar[t_next]
                    x = math.sqrt(alpha_next) * x0_pred + math.sqrt(1 - alpha_next) * eps
                else:
                    x = x0_pred
        elif analog_substrate == "cld":
            # Critically-Damped Langevin Diffusion (RLC circuit mapping)
            # dx = (beta/M)v dt, dv = -beta x dt - Gamma(beta/M)v dt + score + sqrt(2(Gamma)beta) dW
            # But the dW is physically provided by the circuit's thermal noise, so it scales perfectly 
            # and doesn't explicitly compound as algorithmically injected error if perfectly tuned. 
            # We simulate the analog integration using the score function via a coarse Euler step.
            Gamma = 2.0
            M = 1.0
            v = torch.zeros_like(x)
            dt = 1.0 / n_ddim 
            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_tensor = torch.full((n_gen,), t_curr, dtype=torch.long)
                beta_t = betas[t_curr]
                
                # Hardware thermal noise driving the diffusion
                thermal_noise = torch.randn_like(v)
                
                eps = model(x, t_tensor) # score approx
                
                # RLC update
                dx = (beta_t / M) * v * dt
                dv = (-beta_t * x - Gamma * (beta_t / M) * v - eps) * dt + math.sqrt(2 * Gamma * beta_t * dt) * thermal_noise
                
                x = x + dx
                v = v + dv
        elif analog_substrate == "extropic_dtm":
            # Fully connected analog thermodynamic chain (no D/A boundaries)
            # The signal stays in the analog domain, we pass x into score model, step directly
            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_tensor = torch.full((n_gen,), t_curr, dtype=torch.long)
                
                # Direct analog feedback
                score = model(x, t_tensor)
                
                # DTM Gibbs step - simplified continuous update mapped to p-bit energy minimization
                beta_t = betas[t_curr]
                thermal = math.sqrt(beta_t) * torch.randn_like(x)
                x = x - 0.5 * beta_t * x - beta_t * score + thermal


    # Negative mean nearest-neighbor distance (higher = better)
    gen = x.detach()
    test = X_test[:min(len(X_test), 500)]
    # Compute pairwise L2 distances efficiently (chunked to avoid OOM)
    chunk = 50
    min_dists = []
    for i in range(0, len(gen), chunk):
        g_chunk = gen[i:i+chunk]  # (chunk, D)
        dists = ((g_chunk.unsqueeze(1) - test.unsqueeze(0)) ** 2).sum(-1).sqrt()
        min_dists.append(dists.min(dim=1).values)
    mean_dist = torch.cat(min_dists).mean().item()
    return -mean_dist  # higher = better (lower distance = better quality)


def evaluate_output_mse(model: nn.Module, digital_baseline: nn.Module, analog_substrate: str = "classic") -> float:
    """Compute MSE between analog and digital baseline generated samples.
    
    Returns negative MSE so higher = better (consistent with other metrics).
    """
    betas = _get_betas()
    alphas, alphas_bar = _get_alphas(betas)
    
    model.eval()
    digital_baseline.eval()
    
    n_gen = 100
    n_ddim = 10
    ddim_steps = torch.linspace(_T - 1, 0, n_ddim + 1).long()
    
    x_dig = torch.randn(n_gen, _IMG_DIM)
    x_analog = x_dig.clone()
    
    with torch.no_grad():
        if analog_substrate == "classic":
            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_next = ddim_steps[i + 1].item()
                t_tensor = torch.full((n_gen,), t_curr, dtype=torch.long)
                
                eps_dig = digital_baseline(x_dig, t_tensor)
                eps_analog = model(x_analog, t_tensor)
                
                alpha_curr = alphas_bar[t_curr]
                x0_pred_dig = (x_dig - math.sqrt(1 - alpha_curr) * eps_dig) / math.sqrt(alpha_curr)
                x0_pred_analog = (x_analog - math.sqrt(1 - alpha_curr) * eps_analog) / math.sqrt(alpha_curr)
                
                if t_next >= 0:
                    alpha_next = alphas_bar[t_next]
                    x_dig = math.sqrt(alpha_next) * x0_pred_dig + math.sqrt(1 - alpha_next) * eps_dig
                    x_analog = math.sqrt(alpha_next) * x0_pred_analog + math.sqrt(1 - alpha_next) * eps_analog
                else:
                    x_dig = x0_pred_dig
                    x_analog = x0_pred_analog
        elif analog_substrate == "cld":
            Gamma = 2.0
            M = 1.0
            v_dig = torch.zeros_like(x_dig)
            v_analog = torch.zeros_like(x_analog)
            dt = 1.0 / n_ddim 
            
            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_tensor = torch.full((n_gen,), t_curr, dtype=torch.long)
                beta_t = betas[t_curr]
                thermal_noise = torch.randn_like(v_dig)
                
                eps_dig = digital_baseline(x_dig, t_tensor)
                eps_analog = model(x_analog, t_tensor)
                
                # Digital updates
                dx_d = (beta_t / M) * v_dig * dt
                dv_d = (-beta_t * x_dig - Gamma * (beta_t / M) * v_dig - eps_dig) * dt + math.sqrt(2 * Gamma * beta_t * dt) * thermal_noise
                x_dig = x_dig + dx_d
                v_dig = v_dig + dv_d
                
                # Analog updates
                dx_a = (beta_t / M) * v_analog * dt
                dv_a = (-beta_t * x_analog - Gamma * (beta_t / M) * v_analog - eps_analog) * dt + math.sqrt(2 * Gamma * beta_t * dt) * thermal_noise
                x_analog = x_analog + dx_a
                v_analog = v_analog + dv_a
                
        elif analog_substrate == "extropic_dtm":
            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_tensor = torch.full((n_gen,), t_curr, dtype=torch.long)
                
                eps_dig = digital_baseline(x_dig, t_tensor)
                eps_analog = model(x_analog, t_tensor)
                
                beta_t = betas[t_curr]
                thermal = math.sqrt(beta_t) * torch.randn_like(x_dig)
                
                x_dig = x_dig - 0.5 * beta_t * x_dig - beta_t * eps_dig + thermal
                x_analog = x_analog - 0.5 * beta_t * x_analog - beta_t * eps_analog + thermal
    
    mse = ((x_dig - x_analog) ** 2).mean().item()
    return -mse


def get_family_name() -> str:
    return "Diffusion"
