"""Flow model experiment — Rectified Flow on make_moons.

Architecture: v_theta: [2+1 → 64 → 64 → 2] MLP with tanh (same as Neural ODE).
Task: 2D generation of make_moons distribution.
Training: Flow matching loss — minimize ||v_theta(x_t, t) - (x1 - x0)||^2
  where x_t = (1-t)*x0 + t*x1, x0 ~ N(0,I), x1 ~ data.
Evaluation: 4 Euler steps (like FLUX-schnell) to generate samples.
Metric: Negative 1D Wasserstein distance between generated and test samples.
  Compute on x-coordinate and y-coordinate separately, average.
  HIGHER = BETTER (smaller distance = better generation quality).

DOUBT NOTED: 2D Wasserstein is hard to compute exactly (requires solving an
optimal transport problem). We use a 1D proxy: sort generated and test samples
by their x/y coordinates independently and compute average L1 distance between
sorted arrays. This approximates the 1D Earth Mover's Distance.
scipy.stats.wasserstein_distance computes this exactly for 1D distributions.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from neuro_analog.simulator import analog_odeint

try:
    from scipy.stats import wasserstein_distance as _wd
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Dataset ───────────────────────────────────────────────────────────────

_N_TRAIN = 2000
_N_TEST = 500
_SEED = 42

def _get_data():
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=_N_TRAIN + _N_TEST, noise=0.08, random_state=_SEED)
    X = (X - X.mean(0)) / X.std(0)
    X_train = torch.tensor(X[:_N_TRAIN], dtype=torch.float32)
    X_test = torch.tensor(X[_N_TRAIN:], dtype=torch.float32)
    return X_train, X_test


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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 256
    n_epochs = 400
    model.train()

    for epoch in range(n_epochs):
        x1 = X_train[torch.randperm(len(X_train))[:batch_size]]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size)
        x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        target = x1 - x0
        pred = model(t, x_t)
        loss = ((pred - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  [Flow] epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    return model


def load_model(save_path: str) -> nn.Module:
    model = create_model()
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    return model


def evaluate(model: nn.Module) -> float:
    """Generate samples with 4 Euler steps, compute negative Wasserstein distance."""
    _, X_test = _get_data()
    n_gen = 500
    z0 = torch.randn(n_gen, 2)
    t_span = torch.tensor([0.0, 1.0])

    model.eval()
    x_gen = analog_odeint(model, z0, t_span, dt=0.25)  # 4 steps: 0, 0.25, 0.5, 0.75, 1.0

    gen_np = x_gen.detach().numpy()
    test_np = X_test.numpy()

    if _HAS_SCIPY:
        wd_x = _wd(gen_np[:, 0], test_np[:, 0])
        wd_y = _wd(gen_np[:, 1], test_np[:, 1])
        mean_wd = (wd_x + wd_y) / 2.0
    else:
        # Fallback: sorted L1 distance approximation
        gen_sorted_x = np.sort(gen_np[:n_gen, 0])
        test_sorted_x = np.sort(test_np[:n_gen, 0])
        wd_x = np.abs(gen_sorted_x - test_sorted_x).mean()
        gen_sorted_y = np.sort(gen_np[:n_gen, 1])
        test_sorted_y = np.sort(test_np[:n_gen, 1])
        wd_y = np.abs(gen_sorted_y - test_sorted_y).mean()
        mean_wd = (wd_x + wd_y) / 2.0

    return -mean_wd  # higher = better


def evaluate_output_mse(model: nn.Module, digital_baseline: nn.Module) -> float:
    """Compute MSE between analog and digital baseline generated samples.
    
    Returns negative MSE so higher = better (consistent with other metrics).
    """
    n_gen = 100
    z0 = torch.randn(n_gen, 2)
    t_span = torch.tensor([0.0, 1.0])
    
    model.eval()
    digital_baseline.eval()
    
    from neuro_analog.simulator import analog_odeint
    dig_samples = analog_odeint(digital_baseline, z0, t_span, dt=0.25)
    analog_samples = analog_odeint(model, z0, t_span, dt=0.25)
    
    mse = ((dig_samples.detach() - analog_samples.detach()) ** 2).mean().item()
    return -mse


def get_family_name() -> str:
    return "Flow"
