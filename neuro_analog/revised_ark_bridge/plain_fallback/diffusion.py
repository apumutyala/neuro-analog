"""
diffusion.py — Critically-Damped Langevin Dynamics (CLD) SDE via plain BaseAnalogCkt.

Physics (2nd-order damped oscillator with thermal noise):
    dx/dt = (beta/M) * v
    dv/dt = -beta * x - (Gamma * beta / M) * v - score_theta(x, t)
              + sqrt(2 * Gamma * beta) * eta(t)

Where:
    beta(t)   = noise schedule (time-dependent)
    Gamma     = damping coefficient
    M         = mass (inertia)
    score_theta = trained score network (small MLP)
    eta(t)    = white noise (Johnson-Nyquist thermal noise on v)

This is the analog-native form of diffusion. The DDIM sampler is digital;
we model the physical continuous-time dynamics that the analog circuit implements.

Analog primitive: RC integrator for x, LC resonator for v, crossbar MVM for score net.
"""

from __future__ import annotations

import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
from ark.optimization.base_module import BaseAnalogCkt


class CLDCkt(BaseAnalogCkt):
    """
    Plain BaseAnalogCkt for Critically-Damped Langevin Dynamics.

    State = [x, v] where x is position/data and v is velocity/momentum.
    is_stochastic=True enables diffrax MultiTerm solve with noise_fn.
    """

    _keys: list = eqx.field(init=False, static=True)
    _beta: jax.Array = eqx.field(init=False)
    _Gamma: float = eqx.field(init=False, static=True)
    _M: float = eqx.field(init=False, static=True)
    _sigma: float = eqx.field(init=False, static=True)
    _n_beta: int = eqx.field(init=False, static=True)
    _shapes: list = eqx.field(init=False, static=True)
    _sizes: list = eqx.field(init=False, static=True)

    def __init__(
        self,
        score_weights: dict[str, jnp.ndarray],
        beta_schedule: np.ndarray,
        Gamma: float = 1.0,
        M: float = 1.0,
        mismatch_sigma: float = 0.0,
        solver=diffrax.Tsit5(),
    ):
        # Flatten score net weights for a_trainable
        flat = jnp.concatenate([w.flatten() for w in score_weights.values()])
        super().__init__(init_trainable=flat, is_stochastic=True, solver=solver)
        self._keys = list(score_weights.keys())
        self._beta = jnp.array(beta_schedule)
        self._Gamma = float(Gamma)
        self._M = float(M)
        self._sigma = mismatch_sigma
        self._n_beta = len(beta_schedule)
        self._shapes = [w.shape for w in score_weights.values()]
        # Precompute Python int sizes to avoid ConcretizationTypeError under JIT
        self._sizes = [int(np.prod(w.shape)) for w in score_weights.values()]

    def _parse_score_args(self, args):
        """Unpack flat args into score MLP (W1, b1, W2, b2)."""
        idx = 0
        parsed = {}
        for name, shape, size in zip(self._keys, self._shapes, self._sizes):
            parsed[name] = args[idx: idx + size].reshape(shape)
            idx += size
        return parsed

    def _beta_at_t(self, t):
        """Index beta schedule by time."""
        idx = jnp.clip(jnp.astype(t * self._n_beta, int), 0, self._n_beta - 1)
        return self._beta[idx]

    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):
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

    def ode_fn(self, t, state, args):
        """
        Deterministic drift of CLD.
        state = [x, v]
        """
        x_dim = state.shape[0] // 2
        x = state[:x_dim]
        v = state[x_dim:]

        beta = self._beta_at_t(t)
        w = self._parse_score_args(args)

        # Score network: PyTorch model concatenates x with sinusoidal t_embed.
        # The trained net is Linear(x_dim + t_embed_dim, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, x_dim).
        # For the analog CLD we fuse t into the first layer by pre-pending t features.
        # Build a simple sinusoidal t_embed (same dim as PyTorch _SinusoidalEmbed, default 16).
        t_embed_dim = 16
        half = t_embed_dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / (half - 1))
        args_sin = t * freqs
        t_emb = jnp.concatenate([jnp.sin(args_sin), jnp.cos(args_sin)])
        xt = jnp.concatenate([x, t_emb])

        h = jnp.tanh(w["W1"] @ xt + w["b1"])
        h2 = jnp.tanh(w["W2"] @ h + w["b2"])
        score = w["W3"] @ h2 + w["b3"]

        dx = (beta / self._M) * v
        dv = -beta * x - (self._Gamma * beta / self._M) * v - score

        return jnp.concatenate([dx, dv])

    def noise_fn(self, t, state, args):
        """
        Stochastic diffusion: sqrt(2 * Gamma * beta) on v only.
        Johnson-Nyquist thermal noise.
        """
        x_dim = state.shape[0] // 2
        beta = self._beta_at_t(t)
        noise_amp = jnp.sqrt(2.0 * self._Gamma * beta)
        return jnp.concatenate([
            jnp.zeros(x_dim),
            jnp.ones(x_dim) * noise_amp,
        ])

    def readout(self, y):
        """Return x component (first half of state)."""
        if y.ndim >= 2:
            x_dim = y.shape[-1] // 2
            return y[:, :x_dim]
        x_dim = y.shape[0] // 2
        return y[:x_dim]


def build_diffusion(diffusion_model, mismatch_sigma: float = 0.0, n_beta_steps: int = 100):
    """
    Extract score network from PyTorch diffusion model and build CLDCkt.

    Args:
        diffusion_model: PyTorch _ScoreNet (small MLP: t_embed + net Sequential)
        mismatch_sigma: per-weight relative mismatch std
        n_beta_steps: number of discrete beta steps in the noise schedule (default 100)

    Returns CLDCkt instance (already instantiated, not a class).
    """
    import torch

    # Extract score net weights from the Sequential inside _ScoreNet
    layers = [m for m in diffusion_model.net if isinstance(m, torch.nn.Linear)]
    weights = {}
    for idx, linear in enumerate(layers):
        weights[f"W{idx + 1}"] = jnp.array(linear.weight.detach().cpu().numpy())
        weights[f"b{idx + 1}"] = jnp.array(linear.bias.detach().cpu().numpy())

    # Beta schedule: linear from 1e-4 to 0.02 over n_beta_steps
    betas = np.linspace(1e-4, 0.02, n_beta_steps)
    # CLD hyperparameters not present in PyTorch model; use defaults
    Gamma = 1.0
    M = 1.0

    return CLDCkt(
        score_weights=weights,
        beta_schedule=betas,
        Gamma=Gamma,
        M=M,
        mismatch_sigma=mismatch_sigma,
    )
