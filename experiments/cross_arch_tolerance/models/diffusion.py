"""Diffusion model experiment — Tiny DDPM on 8x8 MNIST.

Architecture: Score network is 3-layer MLP (NOT U-Net).
  Input: 64-dim flattened 8x8 image + 16-dim timestep sinusoidal embedding → 80 dims
  Architecture: Linear(80, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 64)
  T=100 diffusion steps, linear beta schedule beta_1=1e-4, beta_T=0.02.

Metric: Reconstruction quality proxy — generate 500 samples via DDIM,
  compute mean distance from each generated sample to its nearest test neighbor.
  Negate so higher = better (lower distance = higher quality).

FID requires Inception network features (too heavy for CPU).
We use the "simplified Wasserstein" proxy: for each generated sample x_gen,
find the nearest test sample by L2 distance. Average minimum distance.
This measures whether generated samples cover the data manifold.
Perfect generator: distance → 0. Random noise: distance >> 0.

We use DDIM sampling (deterministic reverse ODE).
DDIM reduces T=100 stochastic DDPM steps to ~10 deterministic steps.
This makes the evaluation reproducible across analog trials.
We use 10 DDIM steps for evaluation speed.

8x8 MNIST is not in torchvision directly. We downsample 28x28 MNIST
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

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    X_train = X_train.to(_DEVICE)
    model = model.to(_DEVICE)
    betas = _get_betas()           # stays on CPU — used for math.sqrt() calls
    _, alphas_bar = _get_alphas(betas)  # stays on CPU

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    batch_size = 256
    n_epochs = 400
    model.train()

    for epoch in range(n_epochs):
        idx = torch.randperm(len(X_train), device=_DEVICE)[:batch_size]
        x0 = X_train[idx]
        t = torch.randint(0, _T, (batch_size,))           # CPU for alphas_bar indexing
        noise = torch.randn_like(x0)
        alpha_t = alphas_bar[t].unsqueeze(-1).to(_DEVICE)
        x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        pred_noise = model(x_t, t.to(_DEVICE))
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
    model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
    model = model.to(_DEVICE)
    return model


_EVAL_SEED = 42  # Fixed seed for initial noise — eliminates sampling variance across trials


def evaluate(model: nn.Module, analog_substrate: str = "classic") -> float:
    """Generate 500 samples, compute negative mean nearest-neighbor distance.

    Uses a fixed seed for the initial noise tensor so the metric is deterministic
    up to analog noise in the model weights — eliminating sampling variance from
    the baseline measurement.

    Args:
        model: Score network
        analog_substrate: "classic" (DDIM) or "cld" (Critically-Damped Langevin / RLC)
    """
    _, X_test = _get_data()
    X_test = X_test.to(_DEVICE)
    model = model.to(_DEVICE)
    betas = _get_betas()                      # stays on CPU — used for math.sqrt() calls
    alphas, alphas_bar = _get_alphas(betas)   # stays on CPU

    model.eval()
    n_gen = 500
    n_ddim = 10  # DDIM steps for fast deterministic sampling
    ddim_steps = torch.linspace(_T - 1, 0, n_ddim + 1).long()  # stays on CPU for indexing

    # Fixed seed: removes z0-sampling variance from baseline, same as Flow fix
    rng = np.random.default_rng(_EVAL_SEED)
    x = torch.tensor(rng.standard_normal((n_gen, _IMG_DIM)), dtype=torch.float32, device=_DEVICE)

    with torch.no_grad():
        if analog_substrate == "classic":
            # Classic DDIM (deterministic reverse ODE)
            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_next = ddim_steps[i + 1].item()
                t_tensor = torch.full((n_gen,), t_curr, dtype=torch.long, device=_DEVICE)

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
            # Critically-Damped Langevin Diffusion — maps to RLC circuit physics.
            # dx = (β/M)v·dt
            # dv = (−β·x − Γ·(β/M)·v − ε_θ)·dt + √(2Γβ·dt)·dW
            # The dW term corresponds to Johnson-Nyquist thermal noise in the resistor R=Γ.
            # This is NOT injected noise — on real hardware it is physically present.
            # The DDPM-trained score function ε_θ approximates ∇log q(x_t), which
            # is compatible with CLD inference as a score-guided SDE.
            Gamma = 2.0
            M = 1.0
            v = torch.zeros_like(x)
            dt = 1.0 / n_ddim
            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_tensor = torch.full((n_gen,), t_curr, dtype=torch.long, device=_DEVICE)
                beta_t = betas[t_curr].item()
                thermal_noise = torch.randn_like(v)
                eps = model(x, t_tensor)
                dx = (beta_t / M) * v * dt
                dv = ((-beta_t * x - Gamma * (beta_t / M) * v - eps) * dt
                      + math.sqrt(2 * Gamma * beta_t * dt) * thermal_noise)
                x = x + dx
                v = v + dv

        elif analog_substrate == "extropic_dtm":
            # Denoising Thermodynamic Model (DTM) — Extropic arXiv:2510.23972
            #
            # Each reverse diffusion step samples from the EBM joint distribution:
            #   P_θ(x^{t-1}|x^t) ∝ exp(-(ℰ^f(x^{t-1},x^t) + ℰ^θ(x^{t-1})))
            #
            # Forward coupling energy (Gaussian DDPM kernel):
            #   ℰ^f(x^{t-1},x^t) = ||x^t - √(1-β_t)·x^{t-1}||² / (2β_t)
            #   ∇_{x^{t-1}} ℰ^f = -√(1-β_t)/β_t · (x^t - √(1-β_t)·x^{t-1})
            #
            # Learned energy gradient via score network:
            #   ε_θ(x,t) ≈ -√(1-ᾱ_t) · ∇_x log p(x_t)
            #   → ∇ℰ^θ ≈ ε_θ(x^{t-1}, t-1) / √(1-ᾱ_{t-1})
            #
            # Langevin MCMC step (K_mix per diffusion step):
            #   x ← x - η(∇ℰ^f + ∇ℰ^θ) + √(2η)·ξ,  ξ ~ N(0,I)
            #
            # Hardware mapping: ξ is thermal/shot noise from subthreshold transistors
            # (not injected — physically present). E_cell ≈ 2 fJ per step (paper §IV).
            # Paper uses K_mix ≈ 250 for convergence; we use 15 for sweep speed.
            # Warm-start from DDIM prediction reduces effective K needed.
            K_mix = 15
            eta = 0.05

            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_prev = max(ddim_steps[i + 1].item(), 0)

                beta_t = betas[t_curr].item()
                sqrt_one_minus_beta = math.sqrt(1.0 - beta_t)
                alpha_bar_prev = alphas_bar[t_prev].item()

                # Warm start: DDIM x0-prediction then re-noise to t_prev
                t_tensor_curr = torch.full((n_gen,), t_curr, dtype=torch.long, device=_DEVICE)
                eps_init = model(x, t_tensor_curr)
                alpha_curr = alphas_bar[t_curr].item()
                x0_pred = (x - math.sqrt(1.0 - alpha_curr) * eps_init) / math.sqrt(alpha_curr)
                x_prev = (math.sqrt(alpha_bar_prev) * x0_pred
                          + math.sqrt(1.0 - alpha_bar_prev) * eps_init)

                # Langevin MCMC targeting P_θ(x^{t-1}|x^t)
                t_tensor_prev = torch.full((n_gen,), max(t_prev, 0), dtype=torch.long, device=_DEVICE)
                score_denom = math.sqrt(max(1.0 - alpha_bar_prev, 1e-8))
                for _ in range(K_mix):
                    # ∇ℰ^f: pulls x_prev toward x / √(1-β_t)
                    grad_f = -(sqrt_one_minus_beta / beta_t) * (x - sqrt_one_minus_beta * x_prev)
                    # ∇ℰ^θ: score network as energy gradient
                    eps = model(x_prev, t_tensor_prev)
                    grad_theta = eps / score_denom
                    # Langevin step — ξ is transistor thermal/shot noise
                    xi = torch.randn_like(x_prev)
                    x_prev = x_prev - eta * (grad_f + grad_theta) + math.sqrt(2.0 * eta) * xi

                x = x_prev

    # Negative mean nearest-neighbor distance (higher = better)
    gen = x.detach()
    test = X_test[:min(len(X_test), 500)]
    chunk = 50
    min_dists = []
    for i in range(0, len(gen), chunk):
        g_chunk = gen[i:i + chunk]
        dists = ((g_chunk.unsqueeze(1) - test.unsqueeze(0)) ** 2).sum(-1).sqrt()
        min_dists.append(dists.min(dim=1).values)
    return -torch.cat(min_dists).mean().item()


def evaluate_output_mse(model: nn.Module, digital_baseline: nn.Module, analog_substrate: str = "classic") -> float:
    """Compute MSE between analog and digital baseline generated samples.
    
    Returns negative MSE so higher = better (consistent with other metrics).
    """
    betas = _get_betas()                      # stays on CPU — used for math.sqrt() calls
    alphas, alphas_bar = _get_alphas(betas)   # stays on CPU

    model = model.to(_DEVICE)
    digital_baseline = digital_baseline.to(_DEVICE)
    model.eval()
    digital_baseline.eval()

    n_gen = 100
    n_ddim = 10
    ddim_steps = torch.linspace(_T - 1, 0, n_ddim + 1).long()  # stays on CPU for indexing

    x_dig = torch.randn(n_gen, _IMG_DIM, device=_DEVICE)
    x_analog = x_dig.clone()

    with torch.no_grad():
        if analog_substrate == "classic":
            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_next = ddim_steps[i + 1].item()
                t_tensor = torch.full((n_gen,), t_curr, dtype=torch.long, device=_DEVICE)

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
                t_tensor = torch.full((n_gen,), t_curr, dtype=torch.long, device=_DEVICE)
                beta_t = betas[t_curr].item()
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
            # DTM — same Langevin formulation as evaluate().
            # Digital baseline and analog model run the same K_mix Langevin steps;
            # MSE between their trajectories isolates analog mismatch contribution.
            # Both share the same thermal noise draw per Langevin step so that
            # the MSE reflects weight-mismatch error, not sampling variance.
            K_mix = 15
            eta = 0.05

            for i in range(n_ddim):
                t_curr = ddim_steps[i].item()
                t_prev = max(ddim_steps[i + 1].item(), 0)

                beta_t = betas[t_curr].item()
                sqrt_one_minus_beta = math.sqrt(1.0 - beta_t)
                alpha_bar_prev = alphas_bar[t_prev].item()
                alpha_curr = alphas_bar[t_curr].item()

                # Warm start for both
                t_tensor_curr = torch.full((n_gen,), t_curr, dtype=torch.long, device=_DEVICE)
                eps_init_dig = digital_baseline(x_dig, t_tensor_curr)
                eps_init_analog = model(x_analog, t_tensor_curr)
                x0_dig = (x_dig - math.sqrt(1.0 - alpha_curr) * eps_init_dig) / math.sqrt(alpha_curr)
                x0_analog = (x_analog - math.sqrt(1.0 - alpha_curr) * eps_init_analog) / math.sqrt(alpha_curr)
                xp_dig = math.sqrt(alpha_bar_prev) * x0_dig + math.sqrt(1.0 - alpha_bar_prev) * eps_init_dig
                xp_analog = math.sqrt(alpha_bar_prev) * x0_analog + math.sqrt(1.0 - alpha_bar_prev) * eps_init_analog

                t_tensor_prev = torch.full((n_gen,), max(t_prev, 0), dtype=torch.long, device=_DEVICE)
                score_denom = math.sqrt(max(1.0 - alpha_bar_prev, 1e-8))
                for _ in range(K_mix):
                    grad_f_dig = -(sqrt_one_minus_beta / beta_t) * (x_dig - sqrt_one_minus_beta * xp_dig)
                    grad_f_analog = -(sqrt_one_minus_beta / beta_t) * (x_analog - sqrt_one_minus_beta * xp_analog)
                    eps_dig_lv = digital_baseline(xp_dig, t_tensor_prev)
                    eps_analog_lv = model(xp_analog, t_tensor_prev)
                    grad_theta_dig = eps_dig_lv / score_denom
                    grad_theta_analog = eps_analog_lv / score_denom
                    # Shared noise: isolates mismatch vs sampling variance
                    xi = torch.randn_like(xp_dig)
                    xp_dig = xp_dig - eta * (grad_f_dig + grad_theta_dig) + math.sqrt(2.0 * eta) * xi
                    xp_analog = xp_analog - eta * (grad_f_analog + grad_theta_analog) + math.sqrt(2.0 * eta) * xi

                x_dig = xp_dig
                x_analog = xp_analog

    mse = ((x_dig - x_analog) ** 2).mean().item()
    return -mse


def get_family_name() -> str:
    return "Diffusion"
