"""
Diffusion model CDG bridge — VP-SDE probability flow ODE.

The DDPM reverse process expressed as a deterministic ODE (DDIM):

    dx/dt = (T-1) * [-beta(k)/2 * x + beta(k)/(2*sqrt(1-alpha_bar_k)) * eps_theta(x,t)]

where t ∈ [0,1] maps to diffusion step k = round((1-t)*(T-1)):
    t=0 → k=T-1  (fully noisy, start of reverse)
    t=1 → k=0    (clean, end of reverse)

The score network is a 3-layer MLP:
    Linear(img_dim + t_embed_dim, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, img_dim)

with sinusoidal timestep embedding matching _SinusoidalEmbed in models/diffusion.py.

Why this maps to Ark:
    dx/dt = f(x, t, params)  — standard ODE form
    beta(k) lookup uses embedded schedule array (constant during inference)
    Sinusoidal embedding of k is computed inside ode_fn from k = round((1-t)*(T-1))
    No U-Net, no cross-attention — pure MLP, 100% analog FLOP share
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def export_diffusion_to_ark(
    score_net,
    betas_np: np.ndarray,
    output_path,
    mismatch_sigma: float = 0.05,
    class_name: str = "DiffusionAnalogCkt",
    img_dim: int = 64,
    t_embed_dim: int = 16,
) -> str:
    """Generate a standalone Ark-compatible BaseAnalogCkt for DDPM reverse ODE.

    Extracts weights from a PyTorch _ScoreNet and writes a standalone .py file
    that implements the VP-SDE probability flow ODE as a BaseAnalogCkt subclass.

    ODE: dx/dt = (T-1) * [-beta(k)/2 * x + beta(k)/(2*sqrt(1-ab_k)) * eps_theta(x,k)]
    where k = round((1 - t) * (T-1)), t in [0,1].

    Args:
        score_net:       PyTorch _ScoreNet module (already trained).
        betas_np:        (T,) numpy array of beta schedule values.
        output_path:     path to write the generated .py file.
        mismatch_sigma:  per-weight relative mismatch std for make_args().
        class_name:      name of the generated BaseAnalogCkt subclass.
        img_dim:         image dimension (default 64 for 8x8 MNIST).
        t_embed_dim:     sinusoidal time embedding dimension (default 16).

    Returns:
        Generated source code as a string.
    """
    import torch

    # Extract MLP weights from Sequential: net.0, net.2, net.4
    state = score_net.state_dict()
    W0 = state["net.0.weight"].cpu().float().numpy()   # (256, 80)
    b0 = state["net.0.bias"].cpu().float().numpy()     # (256,)
    W1 = state["net.2.weight"].cpu().float().numpy()   # (256, 256)
    b1 = state["net.2.bias"].cpu().float().numpy()     # (256,)
    W2 = state["net.4.weight"].cpu().float().numpy()   # (64, 256)
    b2 = state["net.4.bias"].cpu().float().numpy()     # (64,)

    # Compute alphas_bar from betas
    alphas = 1.0 - betas_np
    alphas_bar = np.cumprod(alphas)

    T = len(betas_np)
    in_dim = img_dim + t_embed_dim   # 80
    h1_dim = W1.shape[0]             # 256
    h2_dim = W2.shape[0]             # 64 = img_dim

    # Shape metadata (in_dim rows x cols for each W)
    shapes_w0 = W0.shape   # (256, 80)
    shapes_w1 = W1.shape   # (256, 256)
    shapes_w2 = W2.shape   # (64, 256)

    # Flat param counts for static offset slices in make_args
    n_W0 = W0.size                                     # 20480
    n_b0 = b0.size                                     # 256
    n_W1 = W1.size                                     # 65536
    n_b1 = b1.size                                     # 256
    n_W2 = W2.size                                     # 16384
    n_b2 = b2.size                                     # 64
    total_mlp = n_W0 + n_b0 + n_W1 + n_b1 + n_W2 + n_b2  # 102976

    # Cumulative offsets
    off0 = 0
    off1 = off0 + n_W0                       # 20480
    off2 = off1 + n_b0                       # 20736
    off3 = off2 + n_W1                       # 86272
    off4 = off3 + n_b1                       # 86528
    off5 = off4 + n_W2                       # 102912
    off6 = off5 + n_b2                       # 102976

    # Half for sinusoidal embedding
    half_embed = t_embed_dim // 2  # 8

    lines = [
        '"""',
        f"Ark-compatible diffusion model — {class_name}.",
        f"ODE: dx/dt = (T-1)*[-beta(k)/2*x + beta(k)/(2*sqrt(1-ab_k))*eps_theta(x,k)]",
        f"Architecture: Linear({in_dim},{h1_dim}) -> ReLU -> Linear({h1_dim},{h1_dim}) -> ReLU -> Linear({h1_dim},{h2_dim})",
        f"Trained on: 8x8 MNIST (img_dim={img_dim}), T={T}, linear beta schedule",
        "",
        "Usage:",
        "    from ark.optimization.base_module import TimeInfo",
        f"    ckt = {class_name}()",
        "    # Integrate from t=0 (noisy) to t=1 (clean)",
        "    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.001, saveat=jnp.array([1.0]))",
        f"    x0 = jnp.array(your_noisy_image)  # shape ({img_dim},)",
        "    result = ckt(time_info, x0, switch=jnp.array([]), args_seed=42, noise_seed=43)",
        f"    # result shape: ({img_dim},) — denoised image",
        '"""',
        "",
        "import jax",
        "import jax.numpy as jnp",
        "import jax.random as jrandom",
        "import diffrax",
        "from ark.optimization.base_module import BaseAnalogCkt, TimeInfo",
        "",
        f"class {class_name}(BaseAnalogCkt):",
        f'    """Analog circuit for DDPM denoising — {class_name} (BaseAnalogCkt subclass).',
        "",
        f"    a_trainable holds all MLP weights concatenated flat ({total_mlp} params).",
        "    betas and alphas_bar are embedded as class attributes (non-trainable constants).",
        "    make_args applies multiplicative mismatch (delta ~ N(1, sigma^2)) to MLP weights.",
        '    """',
        "    betas: jnp.ndarray",
        "    alphas_bar: jnp.ndarray",
        "",
        "    def __init__(self):",
        f"        # Beta schedule: {T} steps, linear {betas_np[0]:.4f} -> {betas_np[-1]:.4f}",
        f"        _betas = jnp.array({betas_np.tolist()}, dtype=jnp.float32)",
        f"        _alphas_bar = jnp.array({alphas_bar.tolist()}, dtype=jnp.float32)",
        "",
    ]

    # Emit weight arrays
    for name, arr in [("_W0", W0), ("_b0", b0), ("_W1", W1), ("_b1", b1), ("_W2", W2), ("_b2", b2)]:
        lines.append(f"        {name} = jnp.array({arr.tolist()}, dtype=jnp.float32)")

    lines += [
        "        a_trainable = jnp.concatenate([",
        "            _W0.flatten(), _b0.flatten(), _W1.flatten(), _b1.flatten(), _W2.flatten(), _b2.flatten()",
        "        ])",
        "        # Store non-trainable constants before super().__init__ freezes the module",
        "        object.__setattr__(self, 'betas', _betas)",
        "        object.__setattr__(self, 'alphas_bar', _alphas_bar)",
        "        super().__init__(",
        "            init_trainable=a_trainable,",
        "            is_stochastic=False,",
        "            solver=diffrax.Heun(),  # Heun provides error estimates for PIDController",
        "        )",
        "",
        "    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):",
        "        key = jrandom.PRNGKey(mismatch_seed)",
        "        keys = jrandom.split(key, 6)",
        f"        sigma = {mismatch_sigma}",
        f"        _p0 = (self.a_trainable[{off0}:{off1}] * (1.0 + sigma * jrandom.normal(keys[0], ({n_W0},)))).reshape({shapes_w0})",
        f"        _p1 = (self.a_trainable[{off1}:{off2}] * (1.0 + sigma * jrandom.normal(keys[1], ({n_b0},)))).reshape(({n_b0},))",
        f"        _p2 = (self.a_trainable[{off2}:{off3}] * (1.0 + sigma * jrandom.normal(keys[2], ({n_W1},)))).reshape({shapes_w1})",
        f"        _p3 = (self.a_trainable[{off3}:{off4}] * (1.0 + sigma * jrandom.normal(keys[3], ({n_b1},)))).reshape(({n_b1},))",
        f"        _p4 = (self.a_trainable[{off4}:{off5}] * (1.0 + sigma * jrandom.normal(keys[4], ({n_W2},)))).reshape({shapes_w2})",
        f"        _p5 = (self.a_trainable[{off5}:{off6}] * (1.0 + sigma * jrandom.normal(keys[5], ({n_b2},)))).reshape(({n_b2},))",
        "        return (_p0, _p1, _p2, _p3, _p4, _p5, self.betas, self.alphas_bar)",
        "",
        "    def ode_fn(self, t, x, args):",
        "        W0, b0, W1, b1, W2, b2, betas, alphas_bar = args",
        f"        T = {T}",
        "        # Map Ark t in [0,1] to diffusion step: t=0->k=T-1 (noisy), t=1->k=0 (clean)",
        "        k = jnp.clip(jnp.round((1.0 - t) * (T - 1)).astype(jnp.int32), 0, T - 1)",
        "        beta_k = betas[k]",
        "        alpha_bar_k = alphas_bar[k]",
        "",
        f"        # Sinusoidal time embedding (dim={t_embed_dim}, matches _SinusoidalEmbed)",
        f"        half = {half_embed}",
        "        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / jnp.maximum(half - 1, 1))",
        "        k_f = k.astype(jnp.float32)",
        "        t_embed = jnp.concatenate([jnp.sin(k_f * freqs), jnp.cos(k_f * freqs)])",
        "",
        "        # Score MLP forward pass: eps_theta(x, k)",
        "        h0 = jax.nn.relu(W0 @ jnp.concatenate([x, t_embed]) + b0)",
        "        h1 = jax.nn.relu(W1 @ h0 + b1)",
        "        eps = W2 @ h1 + b2",
        "",
        "        # VP-SDE probability flow ODE (reverse, Ark forward time t=0->1 = noisy->clean)",
        "        # dx/dt = (T-1) * [-beta(k)/2 * x + beta(k)/(2*sqrt(1-alpha_bar_k)) * eps]",
        "        score_scale = beta_k / (2.0 * jnp.sqrt(jnp.maximum(1.0 - alpha_bar_k, 1e-8)))",
        "        drift = (T - 1) * (-0.5 * beta_k * x + score_scale * eps)",
        "        return drift",
        "",
        "    def noise_fn(self, t, x, args):",
        "        # is_stochastic=False, so noise_fn is never called.",
        "        # For stochastic VP-SDE reverse (full DDPM): return sqrt(beta(k)) * jnp.ones_like(x)",
        "        return jnp.zeros_like(x)",
        "",
        "    def readout(self, y):",
        "        # y shape: (len(saveat), img_dim) — return denoised image at t=1",
        "        return y[-1]",
        "",
        "if __name__ == '__main__':",
        f"    ckt = {class_name}()",
        "    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.001, saveat=jnp.array([1.0]))",
        f"    x0 = jnp.zeros(({img_dim},))",
        "    switch = jnp.array([])",
        "    result = ckt(time_info, x0, switch, args_seed=42, noise_seed=43)",
        "    print(f'Result shape: {result.shape}')",
        "",
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding="utf-8")
    return code
