"""
deq.py — DEQ relaxation ODE compiled via Ark CDG.

Physics:  dz/dt = -z + tanh(W_z @ z + W_x @ x + b)

This is the additive_recurrent form with effective bias b_eff = W_x @ x + b.
The circuit converges to the DEQ fixed point z* = f_theta(z*, x).

Local stability of a fixed point z* for the continuous-time ODE requires every
eigenvalue of the Jacobian J = -I + W_z @ diag(1 - tanh^2(W_z z* + b_eff)) to have
NEGATIVE REAL PART. Because tanh is bounded, the state itself can never grow without
bound; when mismatch destabilises the trained fixed point the circuit relaxes to a
different equilibrium / limit cycle (the analog readout no longer matches the digital
model) -- it does not blow up to infinity.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ..core.paradigms import deq_zform
from ..core.compiler import compile_cdg


def build_deq(deq_model, x_input=None, mismatch_sigma: float = 0.0, vectorize: bool = False):
    """
    Compile a PyTorch DEQ model to an Ark BaseAnalogCkt subclass.

    The DEQ is z* = tanh(W_z @ z* + W_x @ x + b).
    We compile the relaxation ODE: dz/dt = -z + tanh(W_z @ z + b_eff)
    where b_eff = W_x @ x + b for a fixed input x.

    Args:
        deq_model: PyTorch model with attributes .W_z, .W_x, .b_x
        x_input: fixed input vector (if None, uses zero)
        mismatch_sigma: per-weight relative mismatch std
        vectorize: passed to OptCompiler (use True for n_state > 8)

    Returns (CktClass, mgr, effective_bias, spectral_radius)
    """
    import torch

    W_z = deq_model.W_z.weight.detach().cpu().numpy()
    W_x = deq_model.W_x.weight.detach().cpu().numpy()
    b = deq_model.W_x.bias.detach().cpu().numpy()
    n_state = W_z.shape[0]

    # Effective bias for fixed x
    if x_input is None:
        x_input = np.zeros(W_x.shape[1])
    b_eff = W_x @ np.array(x_input) + b

    # Spectral radius check (mismatch can push rho > 1)
    # Compute Jacobian at origin: J = -I + W_z (since tanh'(0) = 1)
    J = -np.eye(n_state) + W_z
    eigvals = np.linalg.eigvals(J)
    spectral_radius = float(np.max(np.abs(eigvals)))

    weights = {
        "J": W_z,
        "b": b_eff,
        "activation": jnp.tanh,
    }

    spec = deq_zform(n_state, mismatch_sigma=mismatch_sigma)
    CktClass, mgr = compile_cdg(
        spec=spec,
        weights=weights,
        mismatch_sigma=mismatch_sigma,
        prog_name=f"DEQ_{n_state}d",
        vectorize=vectorize or (n_state > 8),
        normalize_weight=False,
        do_clipping=False,
    )

    return CktClass, mgr, b_eff, spectral_radius


def check_contraction(W_z, sigma: float = 0.0, n_samples: int = 100):
    """
    Estimate the probability that conductance mismatch destabilises the trained
    fixed point (linearised at the origin, the worst case since tanh'(0) = 1).

    For each of n_samples mismatch draws we form W_mis = W_z * (1 + sigma * N(0,1))
    and the origin Jacobian J = -I + W_mis. Continuous-time stability requires
    max(real(eig(J))) < 0; we count a realisation as "destabilised" when
    max(real(eig(J))) >= 0. This is a conservative proxy -- at the true fixed point
    tanh saturation (diag(1 - tanh^2) < 1) only adds stability.
    """
    n = W_z.shape[0]
    diverged = 0
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        delta = 1.0 + sigma * rng.standard_normal((n, n))
        W_mis = W_z * delta
        # Jacobian at origin for continuous-time ODE: J = -I + W_mis
        # ODE stability requires max(real(eigvals(J))) < 0.
        J = -np.eye(n) + W_mis
        max_real = np.max(np.real(np.linalg.eigvals(J)))
        if max_real >= 0.0:
            diverged += 1
    return diverged / n_samples
