"""
mlp_field.py — Plain BaseAnalogCkt for MLP vector-field dynamics.

Covers both Neural ODE and Normalizing Flow (identical MLP structure, different
weights). Time-augmented input: the MLP takes [z, t] concatenated.

Physics: dz/dt = MLP([z, t]; theta)
Analog primitive: crossbar MVMs for linear layers, differential-pair tanh activations.
Mismatched via multiplicative delta in make_args.
"""

from __future__ import annotations

import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo


class MLPFieldCkt(BaseAnalogCkt):
    """
    Hand-written BaseAnalogCkt for an MLP-defined vector field.

    The MLP has structure: [z_dim + 1] -> hidden -> hidden -> z_dim,
    with time t concatenated as an extra input feature.

    This is a *plain* class (not CDG-compiled) because time-augmentation and
the sequential MVM+activation pipeline are awkward to express in Ark's static
CDG grammar where edge parameters are fixed at build time.
    """

    _keys: list = eqx.field(init=False, static=True)
    _sigma: float = eqx.field(init=False)
    _shapes: list = eqx.field(init=False, static=True)
    _sizes: list = eqx.field(init=False, static=True)

    def __init__(
        self,
        weights: dict[str, jnp.ndarray],
        mismatch_sigma: float = 0.0,
        solver=diffrax.Tsit5(),
    ):
        """
        Args:
            weights: dict with keys 'W1', 'b1', 'W2', 'b2', 'W3', 'b3'
                     Shapes: W1 [h, z+1], b1 [h], W2 [h, h], b2 [h],
                            W3 [z, h], b3 [z]
            mismatch_sigma: per-weight relative mismatch std
            solver: diffrax solver
        """
        # Flatten for a_trainable (needed for Shem gradient compatibility)
        flat = jnp.concatenate([w.flatten() for w in weights.values()])
        super().__init__(init_trainable=flat, is_stochastic=False, solver=solver)
        object.__setattr__(self, '_keys', list(weights.keys()))
        object.__setattr__(self, '_sigma', mismatch_sigma)
        object.__setattr__(self, '_shapes', [w.shape for w in weights.values()])
        # Precompute Python int sizes to avoid ConcretizationTypeError under JIT
        object.__setattr__(self, '_sizes', [int(np.prod(w.shape)) for w in weights.values()])

    def _parse_args(self, args):
        """Unpack flat args into per-layer (W, b) using stored shapes."""
        idx = 0
        parsed = {}
        for name, shape, size in zip(self._keys, self._shapes, self._sizes):
            parsed[name] = args[idx: idx + size].reshape(shape)
            idx += size
        return parsed

    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):
        """Apply multiplicative mismatch if sigma > 0."""
        args = self.a_trainable
        if self._sigma > 0.0:
            key = jax.random.PRNGKey(int(mismatch_seed))
            keys = jax.random.split(key, len(self._keys))
            idx = 0
            new_args = []
            for k, size in zip(keys, self._sizes):
                flat = args[idx: idx + size]
                delta = 1.0 + self._sigma * jax.random.normal(k, (size,))
                new_args.append(flat * delta)
                idx += size
            return jnp.concatenate(new_args)
        return args

    def ode_fn(self, t, z, args):
        """MLP([z, t]) -> dz/dt."""
        w = self._parse_args(args)
        zt = jnp.concatenate([z, jnp.atleast_1d(jnp.asarray(t, dtype=z.dtype))])
        h1 = jnp.tanh(w["W1"] @ zt + w["b1"])
        h2 = jnp.tanh(w["W2"] @ h1 + w["b2"])
        dzdt = w["W3"] @ h2 + w["b3"]
        return dzdt

    def noise_fn(self, t, z, args):
        return jnp.zeros_like(z)

    def readout(self, y):
        return y[-1] if y.ndim >= 1 and y.shape[0] > 1 else y


def build_neural_ode(ode_model, mismatch_sigma: float = 0.0):
    """
    Extract weights from PyTorch Neural ODE and build MLPFieldCkt.

    Expects ode_model.net to be Sequential(Linear, Tanh, Linear, Tanh, Linear).
    """
    import torch

    layers = [m for m in ode_model.net if isinstance(m, torch.nn.Linear)]
    assert len(layers) == 3, "Expected 3 Linear layers in MLP"

    weights = {}
    for idx, linear in enumerate(layers):
        weights[f"W{idx + 1}"] = jnp.array(linear.weight.detach().cpu().numpy())
        if linear.bias is not None:
            weights[f"b{idx + 1}"] = jnp.array(linear.bias.detach().cpu().numpy())
        else:
            out_dim = linear.out_features
            weights[f"b{idx + 1}"] = jnp.zeros(out_dim)

    return MLPFieldCkt(weights, mismatch_sigma=mismatch_sigma)


def build_flow(flow_model, mismatch_sigma: float = 0.0):
    """
    Extract weights from Flow MLP and build MLPFieldCkt.

    Same MLP structure as Neural ODE; only the trained weights differ.
    """
    return build_neural_ode(flow_model, mismatch_sigma)
