"""
EBM (Hopfield / Boltzmann machine) CDG bridge.

Hopfield Langevin mean-field ODE:
    dx/dt = -x + tanh(W_sym @ x + b)

This is structurally identical to the Neural ODE CANN paradigm already
in neural_ode_cdg.py — compile_hopfield_cdg() is a thin wrapper around
compile_neural_ode_cdg() with symmetry enforcement and no InpNode.

For RBMs: augment state z = [v; h] and use make_rbm_hopfield_weights()
to construct the block-symmetric W matrix. The augmented ODE is the same
Hopfield form with n = n_v + n_h.

Why Langevin mean-field works for Hopfield:
    -  Gibbs sampling equilibrium distribution is exactly the Boltzmann
       distribution P(x) ∝ exp(-E(x)/T).
    -  The mean-field (deterministic limit T→0) converges to the same
       energy minima — which ARE the stored patterns.
    -  Analog RC circuits naturally implement this ODE via current summation
       (dx/dt = -x/RC + tanh(Wx+b)/RC), running until capacitor settles.
    -  The stochastic case (Langevin) adds thermal noise to the production
       rules — expressible in Ark's grammar if needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .neural_ode_cdg import compile_neural_ode_cdg


def compile_hopfield_cdg(
    W: np.ndarray,
    b: np.ndarray,
    mismatch_sigma: float = 0.05,
    prog_name: str = "HopfieldCkt",
):
    """Compile a Hopfield network to a BaseAnalogCkt subclass via Ark OptCompiler.

    Maps dx/dt = -x + tanh(W_sym @ x + b) to the CANN CDGSpec.
    Enforces weight symmetry (required by Hopfield energy function).

    Args:
        W:               (n, n) weight matrix. Symmetrised before compilation.
        b:               (n,) bias vector.
        mismatch_sigma:  relative per-weight mismatch std (0 = ideal).
        prog_name:       name of the generated BaseAnalogCkt subclass.

    Returns:
        (CktClass, trainable_mgr)

    Example:
        W = np.array([[0, 0.8, -0.5], [0.8, 0, 0.4], [-0.5, 0.4, 0]])
        b = np.zeros(3)
        CktClass, mgr = compile_hopfield_cdg(W, b, mismatch_sigma=0.05)
        import diffrax
        ckt = CktClass(init_trainable=mgr.get_initial_vals(),
                       is_stochastic=False, solver=diffrax.Tsit5())
    """
    W_sym = (W + W.T) / 2.0
    return compile_neural_ode_cdg(
        J=W_sym, b=b, K=None,
        mismatch_sigma=mismatch_sigma,
        prog_name=prog_name,
    )


def make_rbm_hopfield_weights(
    W_rbm: np.ndarray,
    b_v: np.ndarray,
    b_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert RBM weights to augmented Hopfield form with state z = [v; h].

    The Langevin ODE on the augmented state z ∈ R^{n_v + n_h}:
        dz/dt = -z + tanh(W_block @ z + b_aug)

    where W_block = [[0, W^T], [W, 0]] is symmetric by construction.

    Args:
        W_rbm:  (n_h, n_v) RBM weight matrix.
        b_v:    (n_v,) visible unit biases.
        b_h:    (n_h,) hidden unit biases.

    Returns:
        (W_block, b_aug) where W_block ∈ R^{(n_v+n_h)×(n_v+n_h)},
        b_aug ∈ R^{n_v+n_h}.
    """
    n_h, n_v = W_rbm.shape
    n = n_v + n_h
    W_block = np.zeros((n, n))
    W_block[:n_v, n_v:] = W_rbm.T   # top-right block: W^T
    W_block[n_v:, :n_v] = W_rbm     # bottom-left block: W
    b_aug = np.concatenate([b_v, b_h])
    return W_block, b_aug


def export_hopfield_to_ark(
    W: np.ndarray,
    b: np.ndarray,
    output_path,
    mismatch_sigma: float = 0.05,
    class_name: str = "HopfieldAnalogCkt",
) -> str:
    """Generate a standalone Ark-compatible BaseAnalogCkt for a Hopfield network.

    Produces a Python file containing a BaseAnalogCkt subclass that can be
    imported, run, and passed to Ark's adjoint optimizer.

    ODE:  dx/dt = -x + tanh(W_sym @ x + b)
    The circuit settles to energy minima of E(x) = -½ x^T W x - b^T tanh(x).

    Args:
        W:               (n, n) weight matrix. Symmetrised at export time.
        b:               (n,) bias vector.
        output_path:     path to write the generated .py file.
        mismatch_sigma:  per-weight relative mismatch std for make_args().
        class_name:      name of the generated BaseAnalogCkt subclass.

    Returns:
        Generated source code as a string.
    """
    W_sym = (W + W.T) / 2.0
    n = len(b)
    n_sq = n * n

    lines = [
        '"""',
        f"Ark-compatible Hopfield network — {class_name}.",
        f"ODE: dx/dt = -x + tanh(W_sym @ x + b)",
        f"n={n}, mismatch_sigma={mismatch_sigma}",
        "",
        "Usage:",
        "    from ark.optimization.base_module import TimeInfo",
        f"    ckt = {class_name}()",
        "    time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.01, saveat=jnp.array([5.0]))",
        f"    x0 = jnp.zeros(({n},))",
        "    result = ckt(time_info, x0, switch=jnp.array([]), args_seed=42, noise_seed=43)",
        '"""',
        "",
        "import jax.numpy as jnp",
        "import jax.random as jrandom",
        "import diffrax",
        "from ark.optimization.base_module import BaseAnalogCkt, TimeInfo",
        "",
        f"class {class_name}(BaseAnalogCkt):",
        f'    """Hopfield analog circuit.',
        f"",
        f"    ODE: dx/dt = -x + tanh(W_sym @ x + b)",
        f"    n={n}, mismatch_sigma={mismatch_sigma}",
        f"",
        f"    a_trainable = [W.flatten() | b] — mismatch applied in make_args().",
        f'    """',
        "",
        "    def __init__(self):",
        f"        _W = jnp.array({W_sym.tolist()})",
        f"        _b = jnp.array({b.tolist()})",
        f"        a_trainable = jnp.concatenate([_W.flatten(), _b])",
        "        super().__init__(",
        "            init_trainable=a_trainable,",
        "            is_stochastic=False,",
        "            solver=diffrax.Heun(),",
        "        )",
        "",
        "    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):",
        "        key = jrandom.PRNGKey(mismatch_seed)",
        "        keys = jrandom.split(key, 2)",
        f"        sigma = {mismatch_sigma}",
        f"        W = (self.a_trainable[:{n_sq}]"
        f" * (1.0 + sigma * jrandom.normal(keys[0], ({n_sq},)))).reshape(({n}, {n}))",
        f"        b = (self.a_trainable[{n_sq}:{n_sq + n}]"
        f" * (1.0 + sigma * jrandom.normal(keys[1], ({n},))))",
        "        return (W, b)",
        "",
        "    def ode_fn(self, t, x, args):",
        "        W, b = args",
        "        return -x + jnp.tanh(W @ x + b)",
        "",
        "    def noise_fn(self, t, x, args):",
        "        # Zero by default — Hopfield settles deterministically to energy minimum.",
        "        # For Langevin sampling add:  jnp.ones_like(x) * jnp.sqrt(2.0 / beta)",
        "        return jnp.zeros_like(x)",
        "",
        "    def readout(self, y):",
        "        # y shape: (len(saveat), n) — return state at final time",
        "        return y[-1]",
        "",
        "if __name__ == '__main__':",
        f"    ckt = {class_name}()",
        "    time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.01, saveat=jnp.array([5.0]))",
        f"    x0 = jnp.zeros(({n},))",
        "    switch = jnp.array([])",
        "    result = ckt(time_info, x0, switch, args_seed=42, noise_seed=43)",
        "    print(f'Result shape: {result.shape}')",
        "",
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding="utf-8")
    return code
