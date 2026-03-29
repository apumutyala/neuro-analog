"""
DEQ CDG bridge — gradient-flow ODE for Deep Equilibrium Models.

The fixed-point equation z* = f_theta(z*, x_input) is reformulated as the
gradient-flow ODE:

    dz/dt = f_theta(z, x_input) - z

At equilibrium (dz/dt = 0): z* = f_theta(z*, x_input).  An analog RC feedback
circuit naturally settles to this equilibrium on its RC time constant — no
digital iteration required.

f_theta for our experiment model (_DEQClassifier):
    f_theta(z, x) = tanh(W_z @ z + W_x @ x + b_x)

    W_z: (z_dim, z_dim) — spectrally normalized recurrent weight
    W_x: (z_dim, x_dim) — input injection weight (with bias b_x)

State augmentation:
    y = [z, x_input]   (dim = z_dim + x_dim)
    dz/dt = tanh(W_z @ z + W_x @ x_input + b_x) - z
    dx_input/dt = 0   (hold input constant; it was converted to analog at injection)

D/A boundaries: 2 total — one DAC at x_input injection, one ADC at z* readout.
The fixed-point loop itself is purely analog (no ADC/DAC inside).

Integration time: t_final ~ 5.0 gives ~50 steps (dt0=0.1) for convergence.
Spectral radius rho(df/dz) < 1 guarantees z* is reached before t_final.

Mismatch model: multiplicative delta ~ N(1, sigma^2) on all weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_deq_to_ark(
    deq_model,
    output_path,
    mismatch_sigma: float = 0.05,
    class_name: str = "DEQAnalogCkt",
    z_dim: int = 64,
    x_dim: int = 64,
) -> str:
    """Generate a standalone Ark-compatible BaseAnalogCkt for the DEQ gradient-flow ODE.

    Extracts weights from a trained _DEQClassifier and writes a .py file
    implementing dz/dt = tanh(W_z @ z + W_x @ x_input + b_x) - z.

    The digital readout (W_read @ z* + b_read → class logits) is NOT exported —
    it stays digital.  The Ark circuit's job is to produce z*; classification
    is done downstream.

    Args:
        deq_model:       Trained _DEQClassifier instance (from models/deq.py).
        output_path:     Path to write the generated .py file.
        mismatch_sigma:  Per-weight relative mismatch std for make_args().
        class_name:      Name of the generated BaseAnalogCkt subclass.
        z_dim:           Fixed-point state dimension (default 64).
        x_dim:           Input injection dimension (default 64).

    Returns:
        Generated source code as a string.
    """
    # --- Weight extraction -------------------------------------------------
    # model.W_z.weight gives the spectrally normalised effective weight —
    # the actual tensor used in f_theta's forward pass, not the raw parameter.
    # (nn.utils.parametrizations.spectral_norm exposes this via the .weight property.)
    W_z = deq_model.W_z.weight.detach().cpu().float().numpy()   # (z_dim, z_dim)
    W_x = deq_model.W_x.weight.detach().cpu().float().numpy()   # (z_dim, x_dim)
    b_x = deq_model.W_x.bias.detach().cpu().float().numpy()     # (z_dim,)

    # --- Static parameter layout ------------------------------------------
    n_Wz = W_z.size            # z_dim * z_dim = 4096
    n_Wx = W_x.size            # z_dim * x_dim = 4096
    n_bx = b_x.size            # z_dim = 64
    total_params = n_Wz + n_Wx + n_bx  # 8256

    off_Wz_start = 0
    off_Wz_end   = n_Wz
    off_Wx_start = off_Wz_end
    off_Wx_end   = off_Wx_start + n_Wx
    off_bx_start = off_Wx_end
    off_bx_end   = off_bx_start + n_bx

    aug_dim = z_dim + x_dim    # augmented state dim: [z, x_input]

    lines = [
        '"""',
        f"Ark-compatible DEQ gradient-flow ODE — {class_name}.",
        f"dz/dt = tanh(W_z @ z + W_x @ x_input + b_x) - z",
        f"State augmented: y = [z, x_input]  (dim={aug_dim}),  dx_input/dt = 0.",
        f"",
        f"At equilibrium (dz/dt = 0): z* = f_theta(z*, x_input).",
        f"rho(df_theta/dz) < 1 guarantees convergence (spectral norm enforced).",
        f"",
        f"Architecture: z_dim={z_dim}, x_dim={x_dim}, tanh activation.",
        f"Total trainable params: {total_params}  (W_z:{n_Wz}, W_x:{n_Wx}, b_x:{n_bx})",
        f"Mismatch sigma={mismatch_sigma} on all weight matrices.",
        f"",
        f"Usage:",
        f"    from ark.optimization.base_module import TimeInfo",
        f"    import jax.numpy as jnp",
        f"    ckt = {class_name}()",
        f"    # y0 = [z0, x_input] — z0=zeros, x_input=your analog input",
        f"    y0 = jnp.zeros({aug_dim})",
        f"    time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.1, saveat=jnp.array([5.0]))",
        f"    z_star = ckt(time_info, y0, switch=jnp.array([]), args_seed=42, noise_seed=43)",
        f'"""',
        f"",
        f"import jax",
        f"import jax.numpy as jnp",
        f"import jax.random as jrandom",
        f"import diffrax",
        f"from ark.optimization.base_module import BaseAnalogCkt, TimeInfo",
        f"",
        f"",
        f"class {class_name}(BaseAnalogCkt):",
        f'    """DEQ gradient-flow ODE: dz/dt = f_theta(z, x_input) - z.',
        f"",
        f"    State augmented: y = [z, x_input]  (dim={aug_dim}).",
        f"    dx_input/dt = 0: input is held constant (injected once via DAC).",
        f"    At equilibrium: z* = tanh(W_z @ z* + W_x @ x_input + b_x).",
        f"    rho(df/dz) < 1 guaranteed by spectral normalisation on W_z.",
        f"",
        f"    D/A boundaries: 2 total (DAC at x injection, ADC at z* readout).",
        f"    Digital readout (W_read @ z* -> logits) NOT included — stays digital.",
        f'    """',
        f"",
        f"    def __init__(self):",
    ]

    # Emit weight literals
    lines.append(f"        _Wz = jnp.array({W_z.tolist()}, dtype=jnp.float32)")
    lines.append(f"        _Wx = jnp.array({W_x.tolist()}, dtype=jnp.float32)")
    lines.append(f"        _bx = jnp.array({b_x.tolist()}, dtype=jnp.float32)")
    lines += [
        f"        a_trainable = jnp.concatenate([",
        f"            _Wz.flatten(), _Wx.flatten(), _bx.flatten(),",
        f"        ])",
        f"        super().__init__(",
        f"            init_trainable=a_trainable,",
        f"            is_stochastic=False,",
        f"            solver=diffrax.Heun(),",
        f"        )",
        f"",
        f"    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):",
        f'        """Sample multiplicative mismatch delta ~ N(1, sigma^2) per weight."""',
        f"        key = jrandom.PRNGKey(mismatch_seed)",
        f"        k1, k2, k3 = jrandom.split(key, 3)",
        f"        sigma = {mismatch_sigma}",
        f"        _Wz = (self.a_trainable[{off_Wz_start}:{off_Wz_end}]"
        f" * (1.0 + sigma * jrandom.normal(k1, ({n_Wz},)))).reshape(({z_dim}, {z_dim}))",
        f"        _Wx = (self.a_trainable[{off_Wx_start}:{off_Wx_end}]"
        f" * (1.0 + sigma * jrandom.normal(k2, ({n_Wx},)))).reshape(({z_dim}, {x_dim}))",
        f"        _bx = (self.a_trainable[{off_bx_start}:{off_bx_end}]"
        f" * (1.0 + sigma * jrandom.normal(k3, ({n_bx},))))",
        f"        return (_Wz, _Wx, _bx)",
        f"",
        f"    def ode_fn(self, t, y, args):",
        f'        """dy/dt = [dz/dt, 0]  where dz/dt = tanh(W_z@z + W_x@x + b_x) - z."""',
        f"        Wz, Wx, bx = args",
        f"        z = y[:{z_dim}]",
        f"        x = y[{z_dim}:]",
        f"        dz = jnp.tanh(Wz @ z + Wx @ x + bx) - z",
        f"        dx = jnp.zeros({x_dim})",
        f"        return jnp.concatenate([dz, dx])",
        f"",
        f"    def noise_fn(self, t, y, args):",
        f"        return jnp.zeros({aug_dim})",
        f"",
        f"    def readout(self, y):",
        f'        """Return z* from the final saved time point."""',
        f"        return y[-1, :{z_dim}]",
        f"",
        f"",
        f'if __name__ == "__main__":',
        f"    import jax",
        f"    ckt = {class_name}()",
        f"    print(f'Is BaseAnalogCkt: {{issubclass({class_name}, BaseAnalogCkt)}}')",
        f"    print(f'a_trainable shape: {{ckt.a_trainable.shape}}')",
        f"    y0 = jnp.zeros({aug_dim})",
        f"    time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.1, saveat=jnp.array([5.0]))",
        f"    z_star = ckt(time_info, y0, switch=jnp.array([]), args_seed=42, noise_seed=43)",
        f"    print(f'z* shape: {{z_star.shape}}')  # ({z_dim},)",
        f"    print(f'z* norm:  {{jnp.linalg.norm(z_star):.4f}}')",
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding="utf-8")
    return code
