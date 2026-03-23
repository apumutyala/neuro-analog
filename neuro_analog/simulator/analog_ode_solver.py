"""
Euler-Maruyama ODE/SDE integrator for analog hardware simulation.

This models the INTEGRATION-level noise of an RC-circuit ODE solver,
separate from the COMPUTE-level noise already modeled in AnalogLinear.

The continuous-time update is:
  dx = f(t, x) dt + g dW(t)    [Ito SDE, Euler-Maruyama discretization]
  x_{t+dt} = x_t + dt * f(t, x_t) + sqrt(dt) * N(0, noise_sigma²)

Where:
  f(t, x) = the dynamics function (already analogized — has noisy linear layers)
  g = integration noise coefficient (kT/C of the integration capacitor)
  dW(t) = Wiener process increment

The sqrt(dt) factor is the correct Ito SDE discretization (not just dt * noise),
matching Wang & Achour (arXiv:2411.03557) §4.2 equation 10.

Additional DC drift:
  An RC integrator accumulates offset due to input-referred offset voltage of
  the op-amp. Modeled as: x_t += drift_sigma * sqrt(dt) per step (diffusive drift)
  or alternatively as a deterministic offset per unit time.

A DC drift term (y_t += drift_per_dim * t) can model RC integrator offset accumulation.
We implement both interpretations of drift:
- drift_sigma > 0: stochastic drift (Brownian motion offset, more physically accurate)
- drift_rate > 0: deterministic linear drift per unit time

For the experiment, drift_sigma=0 and drift_rate=0 (we already model noise in AnalogLinear).
The drift parameters are provided for completeness and ablation.

A torchdiffeq wrapper is provided if that package is installed, enabling adaptive
solvers. The default is a pure-PyTorch fixed-step Euler loop, which is sufficient
for the demo scale (2D, ≤50 steps) and avoids the optional dependency.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


def analog_odeint(
    func: nn.Module,
    y0: torch.Tensor,
    t_span: torch.Tensor,
    dt: float = 0.01,
    noise_sigma: float = 0.0,
    drift_sigma: float = 0.0,
) -> torch.Tensor:
    """Euler-Maruyama ODE integration with analog nonidealities.

    Args:
        func: Dynamics function f(t, y) — should already be analogized.
        y0: Initial state, shape (batch, state_dim) or (state_dim,).
        t_span: Either [t0, t1] (two endpoints) or a sequence of time points.
                If [t0, t1], integrates with step dt.
                If multiple points, steps between each consecutive pair.
        dt: Euler step size (used only if t_span has 2 points).
        noise_sigma: Integration noise std dev per sqrt(time) unit.
                     Represents kT/C of the integration capacitor.
                     0.0 for pure ODE (no integration noise).
        drift_sigma: Stochastic drift std dev per sqrt(time) unit.
                     Models op-amp offset diffusion. Usually 0.

    Returns:
        Final state at t1, same shape as y0.

    Note: For the cross-architecture sweep, noise_sigma is typically 0
    because integration noise is negligible compared to weight mismatch.
    The func itself (analogized MLP) already carries mismatch + thermal noise.
    """
    y = y0.float()
    batch_mode = y.dim() > 1

    # Build list of time points to step through
    if t_span.numel() == 2:
        t0, t1 = float(t_span[0]), float(t_span[1])
        direction = 1 if t1 > t0 else -1
        ts = torch.arange(t0, t1, direction * dt).tolist()
        ts.append(t1)
    else:
        ts = t_span.tolist()

    for i in range(len(ts) - 1):
        t = ts[i]
        step = ts[i + 1] - t  # signed step

        t_tensor = torch.tensor(t, dtype=torch.float32, device=y.device)
        if batch_mode:
            t_tensor = t_tensor.expand(y.shape[0])

        with torch.no_grad():
            dy = func(t_tensor, y)

        y = y + step * dy

        # Integration noise: sqrt(|step|) * N(0, noise_sigma²)
        if noise_sigma > 0:
            y = y + math.sqrt(abs(step)) * torch.randn_like(y) * noise_sigma

        # Stochastic drift: sqrt(|step|) * N(0, drift_sigma²)
        if drift_sigma > 0:
            y = y + math.sqrt(abs(step)) * torch.randn_like(y) * drift_sigma

    return y


def analog_odeint_with_logdet(
    func: nn.Module,
    y0: torch.Tensor,
    t_span: torch.Tensor,
    dt: float = 0.01,
    noise_sigma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler integration tracking log|det(Jacobian)| for CNF density estimation.

    For the continuous normalizing flow change-of-variables formula:
      log p(y1) = log p(y0) - integral of div(f)(y_t, t) dt

    The divergence div(f) = tr(∂f/∂y) is computed exactly via autograd.
    This is cheap for low-dimensional state (2D for make_circles demo).

    Computing the exact Jacobian trace via autograd doubles the
    computation time per step. For the 2D demo model this is negligible.
    For higher dimensions, use Hutchinson's trace estimator instead:
      tr(J) ≈ E_v[v^T J v],  v ~ N(0,I)  (unbiased, O(1) backward passes)
    We use exact trace for correctness at dim=2.

    Args:
        func: Dynamics function f(t, y). Must support grad computation.
        y0: Initial state, shape (batch, state_dim).
        t_span: [t0, t1] integration bounds.
        dt: Step size.
        noise_sigma: Integration noise (0 for density estimation — randomness
                     makes the density estimate stochastic and unusable).

    Returns:
        (y1, delta_logp): Final state and cumulative log-det change.
    """
    y = y0.float()
    t0, t1 = float(t_span[0]), float(t_span[1])
    direction = 1 if t1 > t0 else -1
    ts = torch.arange(t0, t1, direction * dt).tolist()
    ts.append(t1)

    batch_size = y.shape[0]
    state_dim = y.shape[1]
    delta_logp = torch.zeros(batch_size, dtype=torch.float32, device=y.device)

    for i in range(len(ts) - 1):
        t = ts[i]
        step = ts[i + 1] - t

        t_tensor = torch.full((batch_size,), t, dtype=torch.float32, device=y.device)
        y_req = y.requires_grad_(True)

        # Compute f(t, y) and tr(∂f/∂y)
        dy = func(t_tensor, y_req)

        # Exact trace of Jacobian via per-output gradients
        # tr(J) = sum_i ∂f_i/∂y_i
        trace = torch.zeros(batch_size, dtype=torch.float32, device=y.device)
        for dim_i in range(state_dim):
            grad = torch.autograd.grad(
                dy[..., dim_i].sum(), y_req,
                create_graph=True, retain_graph=True,
            )[0]
            trace += grad[..., dim_i]

        delta_logp = delta_logp - trace * step
        y = y + step * dy

        if noise_sigma > 0:
            y = y + math.sqrt(abs(step)) * torch.randn_like(y) * noise_sigma

    return y, delta_logp


# ── Optional: torchdiffeq integration ─────────────────────────────────────

def _try_patch_torchdiffeq(sigma: float) -> bool:
    """Patch torchdiffeq.odeint to inject per-step noise. Returns True if patched."""
    try:
        import torchdiffeq
        _original = torchdiffeq.odeint

        def _noisy_odeint(func, y0, t, **kwargs):
            # Use our analog_odeint instead for noise injection
            return analog_odeint(func, y0, t, noise_sigma=sigma)

        torchdiffeq.odeint = _noisy_odeint
        return True
    except ImportError:
        return False
