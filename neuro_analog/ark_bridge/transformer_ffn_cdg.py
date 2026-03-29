"""
Transformer FFN CDG bridge — depth-as-ODE-time.

Each transformer residual FFN block:
    h <- h + FFN(LayerNorm(h))  ≈ Euler step of  dh/dt = W2 * relu(W1 * h + b1) + b2

N stacked FFN blocks = N steps of this ODE.  The Ark circuit integrates from
t=0 to t=N, using the k-th block's weights at time step k = floor(t).

Why this maps to Ark:
    - dh/dt = f(h, t, params) — standard first-order ODE
    - LayerNorm is omitted (reasonable for short-depth stacks; add if needed)
    - Attention stays DIGITAL (dynamic matmul not expressible in Ark grammar)
    - FFN is fully ANALOG: two static-weight MVMs + ReLU

Grammar note on attention:
    Full self-attention requires phi(q_i) * phi(k_j)^T * v_j where both phi(k_j)
    and v_j are dynamic node states.  Current Ark production rules can express
    EDGE.g * VAR(SRC) (fixed edge param × one dynamic state) but NOT the product
    of two independent dynamic signals.  This is a genuine grammar limitation —
    not a near-term extension.  The FFN-only export is the correct scope.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_ffn_to_ark(
    transformer_model,
    output_path,
    mismatch_sigma: float = 0.05,
    class_name: str = "TransformerFFNAnalogCkt",
) -> str:
    """Generate a standalone Ark-compatible BaseAnalogCkt for transformer FFN blocks.

    Extracts the FFN weights from all transformer layers and writes a file
    implementing dh/dt = W2_k * relu(W1_k * h + b1_k) + b2_k where k = floor(t).

    LayerNorm is omitted — the ODE approximation holds without it for short stacks.
    Attention is excluded (digital dynamic matmul; not expressible in Ark grammar).

    Args:
        transformer_model:  PyTorch _TransformerClassifier module (already trained).
        output_path:        path to write the generated .py file.
        mismatch_sigma:     per-weight relative mismatch std for make_args().
        class_name:         name of the generated BaseAnalogCkt subclass.

    Returns:
        Generated source code as a string.
    """
    # Extract FFN weights from each transformer layer
    ffn_weights = []
    for i, layer in enumerate(transformer_model.layers):
        W1 = layer.ffn.fc1.weight.detach().cpu().cpu().float().numpy()   # (ffn_dim, dim)
        b1 = layer.ffn.fc1.bias.detach().cpu().cpu().float().numpy()     # (ffn_dim,)
        W2 = layer.ffn.fc2.weight.detach().cpu().cpu().float().numpy()   # (dim, ffn_dim)
        b2 = layer.ffn.fc2.bias.detach().cpu().cpu().float().numpy()     # (dim,)
        ffn_weights.append((W1, b1, W2, b2))

    N = len(ffn_weights)
    dim = ffn_weights[0][0].shape[1]       # input/output dim (24)
    ffn_dim = ffn_weights[0][0].shape[0]   # hidden dim (48)

    # Compute flat param counts and offsets for each layer
    n_W1 = ffn_weights[0][0].size          # ffn_dim * dim
    n_b1 = ffn_weights[0][1].size          # ffn_dim
    n_W2 = ffn_weights[0][2].size          # dim * ffn_dim
    n_b2 = ffn_weights[0][3].size          # dim
    n_per_layer = n_W1 + n_b1 + n_W2 + n_b2
    total_params = N * n_per_layer

    lines = [
        '"""',
        f"Ark-compatible transformer FFN ODE — {class_name}.",
        f"ODE: dh/dt = W2_k * relu(W1_k * h + b1_k) + b2_k  where k = floor(t)",
        f"Architecture: {N} layers, dim={dim}, ffn_dim={ffn_dim}, ReLU",
        f"t in [0, {N}]:  k=0 uses layer-0 weights, k=1 uses layer-1 weights, ...",
        "",
        "Attention is excluded — dynamic Q*K^T matmul is not expressible in Ark grammar.",
        "LayerNorm is omitted in the ODE approximation.",
        "",
        "Usage:",
        "    from ark.optimization.base_module import TimeInfo",
        f"    ckt = {class_name}()",
        f"    time_info = TimeInfo(t0=0.0, t1={float(N)}, dt0=0.1, saveat=jnp.array([{float(N)}]))",
        f"    h0 = jnp.zeros(({dim},))  # initial hidden state (e.g. post-embedding)",
        "    result = ckt(time_info, h0, switch=jnp.array([]), args_seed=42, noise_seed=43)",
        f"    # result shape: ({dim},) — final hidden state after {N} FFN blocks",
        '"""',
        "",
        "import jax",
        "import jax.numpy as jnp",
        "import jax.random as jrandom",
        "import diffrax",
        "from ark.optimization.base_module import BaseAnalogCkt, TimeInfo",
        "",
        f"class {class_name}(BaseAnalogCkt):",
        f'    """Analog circuit for transformer FFN ODE — {class_name} (BaseAnalogCkt subclass).',
        "",
        f"    a_trainable holds all FFN weights from {N} layers ({total_params} params total).",
        "    dh/dt = W2_k * relu(W1_k * h + b1_k) + b2_k  where k = floor(t).",
        "    make_args applies multiplicative mismatch (delta ~ N(1, sigma^2)) per weight.",
        '    """',
        "",
        "    def __init__(self):",
        "        arrays = []",
    ]

    # Emit each layer's weight arrays
    for i, (W1, b1, W2, b2) in enumerate(ffn_weights):
        lines += [
            f"        # Layer {i}",
            f"        _W1_{i} = jnp.array({W1.tolist()}, dtype=jnp.float32)",
            f"        _b1_{i} = jnp.array({b1.tolist()}, dtype=jnp.float32)",
            f"        _W2_{i} = jnp.array({W2.tolist()}, dtype=jnp.float32)",
            f"        _b2_{i} = jnp.array({b2.tolist()}, dtype=jnp.float32)",
            f"        arrays += [_W1_{i}.flatten(), _b1_{i}, _W2_{i}.flatten(), _b2_{i}]",
        ]

    lines += [
        "        a_trainable = jnp.concatenate(arrays)",
        "        super().__init__(",
        "            init_trainable=a_trainable,",
        "            is_stochastic=False,",
        "            solver=diffrax.Heun(),",
        "        )",
        "",
        "    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):",
        "        key = jrandom.PRNGKey(mismatch_seed)",
        f"        keys = jrandom.split(key, {N * 4})",
        f"        sigma = {mismatch_sigma}",
    ]

    # Emit make_args slices
    ki = 0
    for i in range(N):
        base = i * n_per_layer
        off_W1 = base
        off_b1 = base + n_W1
        off_W2 = base + n_W1 + n_b1
        off_b2 = base + n_W1 + n_b1 + n_W2
        w1_shape = (ffn_dim, dim)
        w2_shape = (dim, ffn_dim)
        lines += [
            f"        _w1_{i} = (self.a_trainable[{off_W1}:{off_b1}] * (1.0 + sigma * jrandom.normal(keys[{ki}], ({n_W1},)))).reshape({w1_shape})",
            f"        _bb1_{i} = (self.a_trainable[{off_b1}:{off_W2}] * (1.0 + sigma * jrandom.normal(keys[{ki+1}], ({n_b1},))))",
            f"        _w2_{i} = (self.a_trainable[{off_W2}:{off_b2}] * (1.0 + sigma * jrandom.normal(keys[{ki+2}], ({n_W2},)))).reshape({w2_shape})",
            f"        _bb2_{i} = (self.a_trainable[{off_b2}:{off_b2+n_b2}] * (1.0 + sigma * jrandom.normal(keys[{ki+3}], ({n_b2},))))",
        ]
        ki += 4

    # Stack into arrays for ode_fn indexing
    lines += [
        f"        W1s = jnp.stack([{', '.join(f'_w1_{i}' for i in range(N))}])",
        f"        b1s = jnp.stack([{', '.join(f'_bb1_{i}' for i in range(N))}])",
        f"        W2s = jnp.stack([{', '.join(f'_w2_{i}' for i in range(N))}])",
        f"        b2s = jnp.stack([{', '.join(f'_bb2_{i}' for i in range(N))}])",
        "        return (W1s, b1s, W2s, b2s)",
        "",
        "    def ode_fn(self, t, x, args):",
        "        W1s, b1s, W2s, b2s = args",
        f"        # k = floor(t), clamped to [0, {N-1}]",
        f"        k = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, {N - 1})",
        "        W1 = W1s[k]; b1 = b1s[k]; W2 = W2s[k]; b2 = b2s[k]",
        "        # dh/dt = W2 * relu(W1 * h + b1) + b2",
        "        h = jax.nn.relu(W1 @ x + b1)",
        "        return W2 @ h + b2",
        "",
        "    def noise_fn(self, t, x, args):",
        "        return jnp.zeros_like(x)",
        "",
        "    def readout(self, y):",
        f"        # y shape: (len(saveat), {dim}) — return final hidden state",
        "        return y[-1]",
        "",
        "if __name__ == '__main__':",
        f"    ckt = {class_name}()",
        f"    time_info = TimeInfo(t0=0.0, t1={float(N)}, dt0=0.1, saveat=jnp.array([{float(N)}]))",
        f"    h0 = jnp.zeros(({dim},))",
        "    switch = jnp.array([])",
        "    result = ckt(time_info, h0, switch, args_seed=42, noise_seed=43)",
        "    print(f'Result shape: {result.shape}')",
        "",
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding="utf-8")
    return code
