"""Flow model experiment — Rectified Flow on make_moons.

Architecture: v_theta: [2+1 → 64 → 64 → 2] MLP with tanh (same as Neural ODE).
Task: 2D generation of make_moons distribution.
Training: Flow matching loss — minimize ||v_theta(x_t, t) - (x1 - x0)||^2
  where x_t = (1-t)*x0 + t*x1, x0 ~ N(0,I), x1 ~ data.
Evaluation: 4 Euler steps (like FLUX-schnell) to generate samples.
Metric: Negative 1D Wasserstein distance between generated and test samples.
  Compute on x-coordinate and y-coordinate separately, average.
  higher = better (smaller distance = better generation quality).

2D Wasserstein is hard to compute exactly (requires solving an
optimal transport problem). We use a 1D proxy: sort generated and test samples
by their x/y coordinates independently and compute average L1 distance between
sorted arrays. This approximates the 1D Earth Mover's Distance.
scipy.stats.wasserstein_distance computes this exactly for 1D distributions.
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from neuro_analog.simulator import analog_odeint

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RC integrator noise (same physical model as neural_ode.py)
_K_B = 1.380649e-23
_TEMP_K = 300.0
_CAP_F = 1e-12
_SIGMA_RC = math.sqrt(_K_B * _TEMP_K / _CAP_F)

try:
    from scipy.stats import wasserstein_distance as _wd
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Dataset ───────────────────────────────────────────────────────────────

_N_TRAIN = 2000
_N_TEST = 500
_SEED = 42

_DATA_CACHE: dict = {}

def _get_data():
    if not _DATA_CACHE:
        from sklearn.datasets import make_moons
        X, _ = make_moons(n_samples=_N_TRAIN + _N_TEST, noise=0.08, random_state=_SEED)
        X = (X - X.mean(0)) / X.std(0)
        _DATA_CACHE["train"] = torch.tensor(X[:_N_TRAIN], dtype=torch.float32)
        _DATA_CACHE["test"] = torch.tensor(X[_N_TRAIN:], dtype=torch.float32)
    return _DATA_CACHE["train"], _DATA_CACHE["test"]


# ── Model ─────────────────────────────────────────────────────────────────

class _FlowMLP(nn.Module):
    """Velocity field v_theta(x_t, t) for rectified flow."""
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t, x):
        # Reshape t for concatenation
        if t.dim() == 0:
            t_feat = t.expand(x.shape[0], 1)
        else:
            t_feat = t.float().unsqueeze(-1) if t.dim() == 1 else t
        return self.net(torch.cat([x, t_feat], dim=-1))


# ── Standard interface ─────────────────────────────────────────────────────

def create_model() -> nn.Module:
    return _FlowMLP(dim=2, hidden=64)


def train_model(model: nn.Module, save_path: str) -> nn.Module:
    X_train, _ = _get_data()
    X_train = X_train.to(_DEVICE)
    model = model.to(_DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 256
    n_epochs = 5000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )
    model.train()

    for epoch in range(n_epochs):
        x1 = X_train[torch.randperm(len(X_train), device=_DEVICE)[:batch_size]]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=_DEVICE)
        x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        target = x1 - x0
        pred = model(t, x_t)
        loss = ((pred - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 500 == 0:
            print(f"  [Flow] epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    return model


def load_model(save_path: str) -> nn.Module:
    model = create_model()
    model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
    model = model.to(_DEVICE)
    return model


_EVAL_SEED = 42  # Fixed seed for z0 sampling — eliminates baseline variance from z0


def evaluate(model: nn.Module, analog_substrate: str = "euler") -> float:
    """Generate samples with 4 Euler steps, compute negative sliced Wasserstein distance.

    Uses sliced Wasserstein (average over 50 random 1D projections) instead of
    marginal-only 1D Wasserstein. This captures 2D joint structure — models that
    generate correct marginals but wrong correlation will now score worse.

    Fixed z0 seed eliminates the z0-sampling variance that inflated baseline
    variance in earlier runs (see §3.1 ‡ footnote in TECHNICAL_NOTE.md).

    Args:
        analog_substrate:
          "euler"         — noiseless integration step (default).
          "rc_integrator" — adds sqrt(kT/C) Johnson-Nyquist noise per Euler step,
                            modeling the integration capacitor of an RC-circuit ODE solver.
    """
    _, X_test = _get_data()
    n_gen = 500
    rng_z = np.random.default_rng(_EVAL_SEED)
    z0 = torch.tensor(rng_z.standard_normal((n_gen, 2)), dtype=torch.float32, device=_DEVICE)
    t_span = torch.tensor([0.0, 1.0])
    noise_sigma = _SIGMA_RC if analog_substrate == "rc_integrator" else 0.0

    model = model.to(_DEVICE)
    model.eval()
    x_gen = analog_odeint(model, z0, t_span, dt=0.01, noise_sigma=noise_sigma).detach().cpu().numpy()
    test_np = X_test.cpu().numpy()

    if _HAS_SCIPY:
        rng_p = np.random.default_rng(_EVAL_SEED)
        directions = rng_p.standard_normal((50, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        distances = [_wd(x_gen @ d, test_np @ d) for d in directions]
        return -float(np.mean(distances))
    else:
        # Fallback to marginal approximation if scipy unavailable
        wd_x = np.abs(np.sort(x_gen[:, 0]) - np.sort(test_np[:n_gen, 0])).mean()
        wd_y = np.abs(np.sort(x_gen[:, 1]) - np.sort(test_np[:n_gen, 1])).mean()
        return -(wd_x + wd_y) / 2.0


def evaluate_sliced_wasserstein(model: nn.Module, n_projections: int = 50, seed: int = 42) -> float:
    """2D-aware generation quality via sliced Wasserstein distance.

    Averages 1D Wasserstein distances over n_projections random unit vectors,
    capturing joint 2D structure that marginal Wasserstein misses. This is the
    recommended metric for future sweeps (see COHERENCE_FIXES.md fix [5]).

    Returns negative sliced Wasserstein (higher = better).
    """
    if not _HAS_SCIPY:
        return evaluate(model)  # fallback

    _, X_test = _get_data()
    n_gen = 500
    z0 = torch.randn(n_gen, 2, device=_DEVICE)
    t_span = torch.tensor([0.0, 1.0])

    model = model.to(_DEVICE)
    model.eval()
    x_gen = analog_odeint(model, z0, t_span, dt=0.01).detach().cpu().numpy()
    test_np = X_test.cpu().numpy()

    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_projections, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    distances = []
    for d in directions:
        proj_gen = x_gen @ d
        proj_test = test_np @ d
        distances.append(_wd(proj_gen, proj_test))

    return -float(np.mean(distances))


def evaluate_output_mse(model: nn.Module, digital_baseline: nn.Module, analog_substrate: str = "euler") -> float:
    """Compute MSE between analog and digital baseline generated samples.
    
    Returns negative MSE so higher = better (consistent with other metrics).
    """
    n_gen = 100
    z0 = torch.randn(n_gen, 2, device=_DEVICE)
    t_span = torch.tensor([0.0, 1.0])

    model = model.to(_DEVICE)
    digital_baseline = digital_baseline.to(_DEVICE)
    model.eval()
    digital_baseline.eval()

    noise_sigma = _SIGMA_RC if analog_substrate == "rc_integrator" else 0.0
    dig_samples = analog_odeint(digital_baseline, z0, t_span, dt=0.01, noise_sigma=0.0)
    analog_samples = analog_odeint(model, z0, t_span, dt=0.01, noise_sigma=noise_sigma)
    
    mse = ((dig_samples.detach() - analog_samples.detach()) ** 2).mean().item()
    return -mse


def get_family_name() -> str:
    return "Flow"
