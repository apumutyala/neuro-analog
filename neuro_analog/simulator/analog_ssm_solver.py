"""
Analog SSM solver — mismatch-aware SSM recurrence for diagonal state-space models.

The standard SSM recurrence is:
    h[t] = A_bar · h[t-1] + B_bar · u[t]     (element-wise complex multiply)
    y[t] = Re(C · h[t]) + D · u[t]

analogize() only replaces nn.Linear modules (B, C, D projections) with AnalogLinear.
It completely misses the A_bar diagonal decay — which is the MOST analog-critical
parameter because it sets the RC time constants of the integrator bank.

This module fills that gap by applying Shem-compatible noise to the full SSM:

  1. A_bar mismatch (Shem §4.1):
     A_bar' = δ_A ⊙ A_bar,  δ_A ~ N(1, σ²)
     In hardware: fabrication variation in RC time constants.
     This is the dominant nonideality for SSMs.

  2. Transient noise on state update (Shem §4.2):
     h[t] = A_bar' · h[t-1] + B_bar · u[t] + σ_transient · √Δt · ξ[t]
     In hardware: Johnson-Nyquist noise on integration capacitors.

  3. B/C projection mismatch:
     Handled by analogize() on the B, C nn.Linear modules (existing path).
     We don't duplicate that here — just apply A_bar mismatch + transient noise.

Usage:
    from neuro_analog.ir import ODESystem
    ode_sys = ssm_extractor.extract_ode_system()
    perturbed = ode_sys.sample_mismatch(sigma=0.05)
    # The perturbed system's A_bar now reflects RC mismatch
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def apply_ssm_mismatch(
    A_bar: torch.Tensor,
    sigma: float = 0.05,
    bounds: Optional[tuple[float, float]] = None,
) -> torch.Tensor:
    """Apply multiplicative mismatch to SSM diagonal state matrix.

    A_bar is complex-valued: A_bar = |A_bar| · exp(jφ).
    Mismatch affects the magnitude (RC time constant) and phase (oscillation frequency)
    independently:
        |A_bar'| = |A_bar| · δ_mag,   δ_mag ~ N(1, σ²)
        φ' = φ + σ · ε_phase,         ε_phase ~ N(0, 1)

    For real-valued A_bar (non-oscillatory modes), this reduces to:
        A_bar' = A_bar · δ,  δ ~ N(1, σ²)

    Args:
        A_bar: Diagonal state matrix entries. Shape (d_state,) or (d_model, d_state).
               Can be real or complex.
        sigma: Mismatch standard deviation (relative).
        bounds: Optional magnitude bounds for clamping.

    Returns:
        Perturbed A_bar with same shape and dtype.
    """
    if sigma <= 0:
        return A_bar.clone()

    if A_bar.is_complex():
        mag = A_bar.abs()
        phase = A_bar.angle()

        # Magnitude mismatch: δ ~ N(1, σ²)
        delta_mag = 1.0 + sigma * torch.randn(mag.shape, device=mag.device, dtype=mag.dtype)
        new_mag = mag * delta_mag

        # Phase mismatch: additive perturbation (smaller effect)
        delta_phase = sigma * torch.randn(phase.shape, device=phase.device, dtype=phase.dtype)
        new_phase = phase + delta_phase

        if bounds is not None:
            lo, hi = bounds
            new_mag = torch.clamp(new_mag, lo, hi)

        return new_mag * torch.exp(1j * new_phase)
    else:
        delta = 1.0 + sigma * torch.randn_like(A_bar)
        result = A_bar * delta
        if bounds is not None:
            lo, hi = bounds
            result = torch.clamp(result, lo, hi)
        return result


def analog_ssm_recurrence(
    A_bar: torch.Tensor,
    Bu: torch.Tensor,
    sigma_transient: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    """Run SSM recurrence with transient noise injection.

    h[t] = A_bar · h[t-1] + Bu[t] + σ · √dt · ξ[t]

    This replaces the standard recurrence loop in _S4DLayer.forward() when
    analog noise simulation is needed.

    Args:
        A_bar: Diagonal state matrix. Shape (d_state,) complex.
        Bu: Pre-computed B·u input. Shape (batch, seq_len, d_state) complex.
        sigma_transient: Transient noise std dev (0 = noiseless).
        dt: Time step for noise scaling (Euler-Maruyama: √dt factor).

    Returns:
        Hidden states h. Shape (batch, seq_len, d_state) complex.
    """
    batch, seq_len, d_state = Bu.shape
    h = torch.zeros(batch, d_state, dtype=Bu.dtype, device=Bu.device)
    hs = []

    sqrt_dt = math.sqrt(abs(dt))

    for t in range(seq_len):
        h = A_bar.unsqueeze(0) * h + Bu[:, t, :]

        # Transient noise: Johnson-Nyquist on integration capacitors
        if sigma_transient > 0:
            if h.is_complex():
                noise_re = sigma_transient * sqrt_dt * torch.randn(
                    batch, d_state, device=h.device, dtype=torch.float32
                )
                noise_im = sigma_transient * sqrt_dt * torch.randn(
                    batch, d_state, device=h.device, dtype=torch.float32
                )
                h = h + torch.complex(noise_re, noise_im)
            else:
                h = h + sigma_transient * sqrt_dt * torch.randn_like(h)

        hs.append(h)

    return torch.stack(hs, dim=1)  # (batch, seq_len, d_state)
