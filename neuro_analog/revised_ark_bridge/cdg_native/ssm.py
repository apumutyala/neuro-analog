"""
ssm.py — State-Space Model compiled via Ark CDG or plain fallback.

Physics:  dh/dt = A @ h + B @ u  (continuous-time linear SSM)

CDG path: linear_ssm spec via compile_cdg(). If compilation fails, falls back to
LinearSSMCkt (plain BaseAnalogCkt with trainable A/B).
"""

from __future__ import annotations

import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
from ark.optimization.base_module import BaseAnalogCkt

from ..core.paradigms import linear_ssm
from ..core.compiler import compile_cdg


def spike_test(n: int = 2, sigma: float = 0.0) -> bool:
    """
    Quick spike: can OptCompiler handle a purely linear CDGSpec?

    Uses a simple 2-state diagonal A = [[-0.1, 0], [0, -0.2]] with no input.
    """
    A = np.array([[-0.1, 0.0], [0.0, -0.2]])
    weights = {"A": A}
    try:
        spec = linear_ssm(n, mismatch_sigma=sigma)
        CktClass, mgr = compile_cdg(
            spec=spec,
            weights=weights,
            mismatch_sigma=sigma,
            prog_name="SSMSpike",
            vectorize=False,
            normalize_weight=False,
            do_clipping=False,
        )
        # Try a nominal solve
        init_vals = mgr.get_initial_vals()
        ckt = CktClass(init_trainable=init_vals, is_stochastic=False, solver=diffrax.Tsit5())
        ti = diffrax.TimeInfo(t0=0.0, t1=1.0, dt0=0.1, saveat=jnp.array([1.0]))
        out = ckt(ti, jnp.zeros(n), switch=jnp.array([]), args_seed=0, noise_seed=0)
        return jnp.all(jnp.isfinite(out))
    except Exception:
        return False


class LinearSSMCkt(BaseAnalogCkt):
    """
    Plain BaseAnalogCkt fallback for linear SSM.

    Physics: dh/dt = A @ h + B @ u
    Trainable: flattened A (and B if present) in a_trainable for Shem gradients.
    """

    _A_shape: tuple = eqx.field(init=False, static=True)
    _B_shape: tuple | None = eqx.field(init=False, static=True)
    _has_B: bool = eqx.field(init=False, static=True)
    _u: jax.Array = eqx.field(init=False)
    _sigma: float = eqx.field(init=False, static=True)
    _A_size: int = eqx.field(init=False, static=True)
    _B_size: int = eqx.field(init=False, static=True)

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray | None = None,
        u: np.ndarray | None = None,
        mismatch_sigma: float = 0.0,
        solver=diffrax.Tsit5(),
    ):
        # Flatten A and B into a_trainable for Shem compatibility
        flat = [jnp.array(A).flatten()]
        if B is not None:
            flat.append(jnp.array(B).flatten())
        a_trainable = jnp.concatenate(flat)
        super().__init__(init_trainable=a_trainable, is_stochastic=False, solver=solver)
        self._A_shape = A.shape
        self._B_shape = B.shape if B is not None else None
        self._has_B = B is not None
        # u must match B's input dimension, not A's state dimension
        u_dim = B.shape[1] if B is not None else A.shape[0]
        self._u = jnp.array(u) if u is not None else jnp.zeros(u_dim)
        self._sigma = mismatch_sigma
        # Precompute Python int sizes to avoid ConcretizationTypeError under JIT
        self._A_size = int(np.prod(A.shape))
        self._B_size = int(np.prod(B.shape)) if B is not None else 0

    def _parse_args(self, args):
        """Reshape flat trainable back to A (and B)."""
        A = args[:self._A_size].reshape(self._A_shape)
        if self._has_B:
            B = args[self._A_size:].reshape(self._B_shape)
            return A, B
        return A, None

    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):
        args = self.a_trainable
        if self._sigma > 0.0:
            key = jax.random.PRNGKey(int(mismatch_seed))
            keys = jax.random.split(key, 2 if self._has_B else 1)
            flat_A = args[:self._A_size]
            delta_A = 1.0 + self._sigma * jax.random.normal(keys[0], (self._A_size,))
            new_args = [flat_A * delta_A]
            if self._has_B:
                flat_B = args[self._A_size:]
                delta_B = 1.0 + self._sigma * jax.random.normal(keys[1], (self._B_size,))
                new_args.append(flat_B * delta_B)
            return jnp.concatenate(new_args)
        return args

    def ode_fn(self, t, h, args):
        A, B = self._parse_args(args)
        dhdt = A @ h
        if B is not None:
            dhdt += B @ self._u
        return dhdt

    def noise_fn(self, t, h, args):
        return jnp.zeros_like(h)

    def readout(self, y):
        # Return the full trajectory over saveat (shape [T, d_state]), matching the
        # behaviour of the compiled DEQ/EBM circuits so callers can plot h(t).
        # (Previously returned only the final state y[-1]; that discarded the
        # transient, which is exactly what we want to visualise for a linear SSM.)
        return y


def build_ssm(ssm_model, mismatch_sigma: float = 0.0, u_input=None, force_plain: bool = False):
    """
    Compile a PyTorch SSM model to Ark.

    Tries CDG first (spike test). Falls back to plain LinearSSMCkt if needed.

    Args:
        ssm_model: PyTorch model with .A [d_state, d_state] (continuous-time)
        mismatch_sigma: per-coefficient mismatch std
        u_input: external input vector (if None, zero)
        force_plain: skip CDG attempt, use plain class directly

    Returns (CktClass_or_instance, mgr_or_None, used_cdg: bool)
    """
    import torch

    # S4D model: extract continuous-time A from first layer's log_A parameters.
    # The trained model is a stack of _S4DLayer modules inside ssm_model.layers.
    layer = ssm_model.layers[0]  # use first layer for demo
    A_real = -torch.exp(layer.log_A_real).detach().cpu().numpy()
    A_imag = layer.log_A_imag.detach().cpu().numpy()
    # Real-only approximation for analog circuit (drops oscillatory imag part)
    A = np.diag(A_real)
    d_state = A.shape[0]

    B = None
    if hasattr(layer, 'B'):
        # B is a nn.Linear(d_model, d_state*2); take first d_state columns as real projection
        B_full = layer.B.weight.detach().cpu().numpy()  # [d_state*2, d_model]
        B = B_full[:d_state, :]  # [d_state, d_model] real part

    if u_input is None:
        u_input = np.zeros(B.shape[1]) if B is not None else np.zeros(d_state)

    if not force_plain:
        try:
            ok = spike_test(d_state, sigma=mismatch_sigma)
            if ok:
                weights = {"A": A}
                if B is not None:
                    weights["B"] = B
                spec = linear_ssm(d_state, mismatch_sigma=mismatch_sigma)
                CktClass, mgr = compile_cdg(
                    spec=spec,
                    weights=weights,
                    mismatch_sigma=mismatch_sigma,
                    prog_name=f"SSM_{d_state}d",
                    vectorize=(d_state > 8),
                    normalize_weight=False,
                    do_clipping=False,
                )
                return CktClass, mgr, True
        except Exception:
            pass

    # Fallback: plain class
    ckt = LinearSSMCkt(A, B, u_input, mismatch_sigma=mismatch_sigma)
    return ckt, None, False
