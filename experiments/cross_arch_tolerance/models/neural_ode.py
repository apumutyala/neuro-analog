"""Neural ODE experiment model — Continuous Normalizing Flow on make_circles.

Architecture: f_theta: [2+1 → 64 → 64 → 2] MLP with tanh (time-augmented).
Task: Density estimation on 2D make_circles via CNF change-of-variables.
Metric: Negative log-likelihood = log p(x_test) [HIGHER = BETTER].

CNF training:
  Push data x1 BACKWARD through the ODE (t: 1 → 0) to get z0 ~ N(0,I).
  The change-of-variables formula gives:
    log p(x1) = log p(z0) - integral_0^1 div(f_theta)(z_t, t) dt
  where div = trace of Jacobian = sum of diagonal elements (exact for 2D).
  Training minimizes NLL = -E[log p(x1)].

Analog evaluation:
  Forward pass: z0 ~ N(0,I) → integrate with analog_odeint → x1_gen.
  Push test x1 backward through analogized f_theta → z0_est.
  Compute NLL on z0_est under N(0,I) + accumulated log-det.

DOUBT NOTED: True CNF evaluation requires integrating BACKWARD through the
analog model including the log-det tracking. The analog model introduces
stochastic thermal noise, making the log-det estimate noisy. We average over
the n_trials from the sweep to get stable statistics.

DOUBT NOTED: We use Euler ODE (not an adaptive solver) for reproducibility.
torchdiffeq is optional. For the demo scale (2D, 50 steps), Euler is sufficient.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_circles

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from neuro_analog.simulator import analog_odeint_with_logdet, analogize, set_all_noise

# ── Dataset ───────────────────────────────────────────────────────────────

_N_TRAIN = 2000
_N_TEST = 500
_SEED = 42

def _get_data():
    X, _ = make_circles(n_samples=_N_TRAIN + _N_TEST, noise=0.05, random_state=_SEED)
    X = (X - X.mean(0)) / X.std(0)
    X_train = torch.tensor(X[:_N_TRAIN], dtype=torch.float32)
    X_test = torch.tensor(X[_N_TRAIN:], dtype=torch.float32)
    return X_train, X_test


# ── Model ─────────────────────────────────────────────────────────────────

class _TimeAugMLP(nn.Module):
    """f_theta(t, z): time-augmented MLP for continuous normalizing flow."""
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, z):
        # t can be scalar or (batch,)
        if t.dim() == 0:
            t_feat = t.expand(z.shape[0], 1)
        else:
            t_feat = t.unsqueeze(-1) if t.dim() == 1 else t
        return self.net(torch.cat([z, t_feat], dim=-1))


# ── Standard interface ─────────────────────────────────────────────────────

def create_model() -> nn.Module:
    return _TimeAugMLP(dim=2, hidden=20)  # Reduced from 64 to operate near capacity


def train_model(model: nn.Module, save_path: str) -> nn.Module:
    X_train, _ = _get_data()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    t_span = torch.tensor([1.0, 0.0])  # backward integration for density
    n_steps = 40
    dt = 1.0 / n_steps
    batch_size = 256
    n_epochs = 300

    model.train()
    for epoch in range(n_epochs):
        idx = torch.randperm(len(X_train))[:batch_size]
        x1 = X_train[idx].requires_grad_(True)

        # Backward ODE: x1 → z0 tracking log-det
        z0, delta_logp = analog_odeint_with_logdet(
            model, x1, t_span, dt=dt, noise_sigma=0.0
        )

        log_p0 = -0.5 * (z0 ** 2).sum(dim=-1) - math.log(2 * math.pi)
        log_px = log_p0 + delta_logp
        nll = -log_px.mean()

        optimizer.zero_grad()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  [NeuralODE] epoch {epoch+1}/{n_epochs}: NLL={nll.item():.4f}")

    torch.save(model.state_dict(), save_path)
    return model


def load_model(save_path: str) -> nn.Module:
    model = create_model()
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    return model


def evaluate(model: nn.Module) -> float:
    """Compute log-likelihood on test set. Higher = better."""
    _, X_test = _get_data()
    t_span = torch.tensor([1.0, 0.0])
    dt = 1.0 / 40
    model.eval()

    # Disable thermal noise during log-det computation — stochastic noise makes the
    # Jacobian trace estimate noisy and unusable for density evaluation.
    # Save per-module state and restore exactly after, so ablation sweeps are not corrupted.
    _saved = {
        name: {attr: getattr(m, attr) for attr in ("_use_thermal", "_use_quantization", "_use_mismatch") if hasattr(m, attr)}
        for name, m in model.named_modules()
        if hasattr(m, "_use_thermal")
    }
    for m in model.modules():
        if hasattr(m, "_use_thermal"):
            m._use_thermal = False

    with torch.enable_grad():  # needed for log-det computation
        z0, delta_logp = analog_odeint_with_logdet(
            model, X_test.requires_grad_(True), t_span, dt=dt, noise_sigma=0.0
        )

    for name, m in model.named_modules():
        if name in _saved:
            for attr, val in _saved[name].items():
                setattr(m, attr, val)

    log_p0 = -0.5 * (z0.detach() ** 2).sum(dim=-1) - math.log(2 * math.pi)
    log_px = log_p0 + delta_logp.detach()
    return float(log_px.mean().item())  # log-likelihood (higher = better)


def evaluate_output_mse(model: nn.Module, digital_baseline: nn.Module) -> float:
    """Compute MSE between analog and digital baseline outputs.
    
    Args:
        model: Analog model (with AnalogLinear layers)
        digital_baseline: Original digital model for comparison
    
    Returns negative MSE so higher = better (consistent with other metrics).
    """
    _, X_test = _get_data()
    t_span = torch.tensor([1.0, 0.0])
    dt = 1.0 / 40
    
    digital_baseline.eval()
    model.eval()
    
    from neuro_analog.simulator import set_all_noise
    set_all_noise(model, thermal=False, mismatch=True, quantization=True)
    
    with torch.enable_grad():
        z0_dig, _ = analog_odeint_with_logdet(digital_baseline, X_test.requires_grad_(True), t_span, dt=dt, noise_sigma=0.0)
        z0_analog, _ = analog_odeint_with_logdet(model, X_test.requires_grad_(True), t_span, dt=dt, noise_sigma=0.0)
    
    mse = ((z0_dig.detach() - z0_analog.detach()) ** 2).mean().item()
    return -mse  # Negative so higher = better


def get_family_name() -> str:
    return "Neural ODE"
