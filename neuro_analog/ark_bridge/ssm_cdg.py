"""
SSM CDG bridge — continuous-time S4D dynamics kernel.

The diagonal S4D SSM is discretised via bilinear (Tustin) transform for
digital computation, but its underlying continuous-time form is:

    dh/dt = A_c * h + B_c * u(t)     [element-wise complex, diagonal A]
    y(t)  = Re(C * h(t)) + D * u(t)  [real-valued output]

where A_c = -exp(log_A_real) + i * log_A_imag is diagonal complex (stable:
Re(A_c) < 0 always).

Ark works with real-valued ODEs.  We split h = h_re + i*h_im into 2*d_state
real values and expand the complex multiply into real arithmetic:

    dh_re/dt = A_re * h_re - A_im * h_im + Bu_re
    dh_im/dt = A_im * h_re + A_re * h_im + Bu_im

where:
    A_re = -exp(log_A_real)   (shape: d_state, all negative — RC decay rates)
    A_im = log_A_imag         (shape: d_state — oscillation frequencies)
    Bu = B_weight @ u         (shape: 2*d_state) — re/im split from B linear map

This export targets a SINGLE _S4DLayer (the per-layer analog dynamics kernel).
The surrounding classifier (embedding, residual stacking, head) stays digital.
Exporting a single layer is the correct scope: the continuous-time ODE is the
dynamics kernel; the sequence processing and residual connections are discrete.

Autonomous mode (u=0):
    The exported circuit runs in autonomous mode: u=0, so Bu=0 and the ODE
    reduces to pure RC decay + oscillation.  The circuit shows the transient
    response of the analog eigenvalue bank from initial conditions h0.

    To process a non-zero input embedding u, embed it in the initial state or
    extend ode_fn — documented as a forward-compatibility note in the output.

Mismatch model:
    A_re, A_im: hardware mismatch is on the RC time constants (magnitude) and
    oscillation frequency (phase).  We model this as complex polar perturbation:
        |A_c'| = |A_c| * (1 + sigma * eps_mag)
        angle(A_c') = angle(A_c) + sigma * eps_phase
    expanded back to real/imag.

    B_weight: multiplicative delta ~ N(1, sigma^2) per element (crossbar MVM).

Trainable params: A_re (d_state), A_im (d_state), B_flat (2*d_state * d_model)
C_weight, D_weight: stored as class attributes — readout stays digital.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def export_s4d_to_ark(
    s4d_layer,
    output_path,
    mismatch_sigma: float = 0.05,
    class_name: str = "SSMAnalogCkt",
    d_model: int = 16,
    d_state: int = 8,
) -> str:
    """Generate a standalone Ark-compatible BaseAnalogCkt for the S4D dynamics kernel.

    Extracts log_A_real, log_A_imag, B.weight from a trained _S4DLayer and
    writes a .py file implementing the real/imag split continuous-time SSM ODE.

    C.weight and D.weight are embedded as non-trainable class attributes for
    reference; they are not mismatch-tagged because they are part of the digital
    readout stage, not the core analog dynamics.

    Args:
        s4d_layer:      Trained _S4DLayer instance (from models/ssm.py).
        output_path:    Path to write the generated .py file.
        mismatch_sigma: Per-weight relative mismatch std for make_args().
        class_name:     Name of the generated BaseAnalogCkt subclass.
        d_model:        Input/output dimension (default 16 for experiment model).
        d_state:        SSM state dimension (complex, default 8 → 16 real values).

    Returns:
        Generated source code as a string.
    """
    # --- Weight extraction -------------------------------------------------
    # Continuous-time A parameters (computed from the learned log-parametrisation)
    A_re_np = -np.exp(s4d_layer.log_A_real.detach().cpu().float().numpy())  # (d_state,) all <0
    A_im_np = s4d_layer.log_A_imag.detach().cpu().float().numpy()           # (d_state,)

    B_weight_np = s4d_layer.B.weight.detach().cpu().float().numpy()         # (2*d_state, d_model)
    C_weight_np = s4d_layer.C.weight.detach().cpu().float().numpy()         # (d_model, 2*d_state)
    D_weight_np = s4d_layer.D.weight.detach().cpu().float().numpy()         # (d_model, d_model)

    state_dim = 2 * d_state   # 16 real values (h_re, h_im)

    # --- Static parameter layout (only A and B are mismatch-tagged) --------
    n_Are = d_state                    # 8
    n_Aim = d_state                    # 8
    n_B   = B_weight_np.size          # 2*d_state * d_model = 256

    off_Are_start = 0
    off_Are_end   = n_Are
    off_Aim_start = off_Are_end
    off_Aim_end   = off_Aim_start + n_Aim
    off_B_start   = off_Aim_end
    off_B_end     = off_B_start + n_B
    total_params  = off_B_end         # 272

    lines = [
        '"""',
        f"Ark-compatible S4D dynamics kernel — {class_name}.",
        f"",
        f"Continuous-time SSM (real/imag split, autonomous u=0):",
        f"    dh_re/dt = A_re * h_re - A_im * h_im",
        f"    dh_im/dt = A_im * h_re + A_re * h_im",
        f"",
        f"State: h = [h_re, h_im]  (dim={state_dim}),",
        f"    h_re, h_im each in R^{d_state} = Re/Im of complex SSM state.",
        f"",
        f"A_re = -exp(log_A_real)  (RC decay rates, all negative)",
        f"A_im = log_A_imag        (oscillation frequencies)",
        f"",
        f"Autonomous mode (u=0): B_weight is embedded but not active.",
        f"To inject input u: extend ode_fn with Bu = B @ u.",
        f"",
        f"Architecture: d_model={d_model}, d_state={d_state} (complex) = {state_dim} real.",
        f"Trainable: A_re({n_Are}), A_im({n_Aim}), B_flat({n_B})  total={total_params}",
        f"Non-trainable (digital readout): C ({C_weight_np.shape}), D ({D_weight_np.shape})",
        f"Mismatch sigma={mismatch_sigma}:",
        f"  A: complex polar (magnitude + phase separately)",
        f"  B: multiplicative delta ~ N(1, sigma^2)",
        f"",
        f"Usage:",
        f"    from ark.optimization.base_module import TimeInfo",
        f"    import jax.numpy as jnp",
        f"    ckt = {class_name}()",
        f"    h0 = jnp.zeros({state_dim})",
        f"    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))",
        f"    h_final = ckt(time_info, h0, switch=jnp.array([]), args_seed=42, noise_seed=43)",
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
        f'    """S4D continuous-time dynamics kernel (BaseAnalogCkt subclass).',
        f"",
        f"    Real/imag split ODE over h = [h_re, h_im] (dim={state_dim}):",
        f"        dh_re/dt = A_re * h_re - A_im * h_im",
        f"        dh_im/dt = A_im * h_re + A_re * h_im",
        f"",
        f"    A_re mismatch: RC time constant drift (magnitude perturbation).",
        f"    A_im mismatch: oscillation frequency drift (phase perturbation).",
        f"    B mismatch:    input crossbar MVM (multiplicative).",
        f"",
        f"    C_weight, D_weight stored as class attributes (digital readout).",
        f'    """',
        f"",
        f"    C_weight: jnp.ndarray",
        f"    D_weight: jnp.ndarray",
        f"",
        f"    def __init__(self):",
        f"        # Continuous-time A parameters (computed from log parametrisation)",
        f"        _Are = jnp.array({A_re_np.tolist()}, dtype=jnp.float32)",
        f"        _Aim = jnp.array({A_im_np.tolist()}, dtype=jnp.float32)",
        f"        _B   = jnp.array({B_weight_np.tolist()}, dtype=jnp.float32)",
        f"        # Digital readout matrices (non-trainable, no mismatch)",
        f"        _C   = jnp.array({C_weight_np.tolist()}, dtype=jnp.float32)",
        f"        _D   = jnp.array({D_weight_np.tolist()}, dtype=jnp.float32)",
        f"        a_trainable = jnp.concatenate([_Are, _Aim, _B.flatten()])",
        f"        object.__setattr__(self, 'C_weight', _C)",
        f"        object.__setattr__(self, 'D_weight', _D)",
        f"        super().__init__(",
        f"            init_trainable=a_trainable,",
        f"            is_stochastic=False,",
        f"            solver=diffrax.Heun(),",
        f"        )",
        f"",
        f"    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):",
        f'        """Sample hardware mismatch for A (complex polar) and B (multiplicative)."""',
        f"        key = jrandom.PRNGKey(mismatch_seed)",
        f"        k1, k2, k3 = jrandom.split(key, 3)",
        f"        sigma = {mismatch_sigma}",
        f"",
        f"        _Are = self.a_trainable[{off_Are_start}:{off_Are_end}]",
        f"        _Aim = self.a_trainable[{off_Aim_start}:{off_Aim_end}]",
        f"        _B   = self.a_trainable[{off_B_start}:{off_B_end}].reshape(({state_dim}, {d_model}))",
        f"",
        f"        # A mismatch: complex polar — magnitude drift + phase drift",
        f"        # Matches _S4DLayer.resample_mismatch() convention.",
        f"        mag   = jnp.sqrt(_Are ** 2 + _Aim ** 2)",
        f"        angle = jnp.arctan2(_Aim, _Are)",
        f"        new_mag   = mag   * (1.0 + sigma * jrandom.normal(k1, ({d_state},)))",
        f"        new_angle = angle +         sigma * jrandom.normal(k2, ({d_state},))",
        f"        _Are_m = new_mag * jnp.cos(new_angle)",
        f"        _Aim_m = new_mag * jnp.sin(new_angle)",
        f"",
        f"        # B mismatch: multiplicative per-element (crossbar MVM)",
        f"        _B_m = _B * (1.0 + sigma * jrandom.normal(k3, ({state_dim}, {d_model})))",
        f"",
        f"        return (_Are_m, _Aim_m, _B_m)",
        f"",
        f"    def ode_fn(self, t, h, args):",
        f'        """dh/dt = [A_re*h_re - A_im*h_im, A_im*h_re + A_re*h_im]  (u=0 autonomous)."""',
        f"        A_re, A_im, B = args",
        f"        h_re = h[:{d_state}]",
        f"        h_im = h[{d_state}:]",
        f"        dh_re = A_re * h_re - A_im * h_im",
        f"        dh_im = A_im * h_re + A_re * h_im",
        f"        # To inject input u (d_model-dim): uncomment and pass u in initial state",
        f"        # Bu = B @ u;  dh_re += Bu[:{d_state}];  dh_im += Bu[{d_state}:]",
        f"        return jnp.concatenate([dh_re, dh_im])",
        f"",
        f"    def noise_fn(self, t, h, args):",
        f"        return jnp.zeros({state_dim})",
        f"",
        f"    def readout(self, y):",
        f'        """Return raw h = [h_re, h_im] at final time. Apply self.C_weight for y."""',
        f"        return y[-1]   # shape: ({state_dim},)",
        f"",
        f"",
        f'if __name__ == "__main__":',
        f"    ckt = {class_name}()",
        f"    print(f'Is BaseAnalogCkt: {{issubclass({class_name}, BaseAnalogCkt)}}')",
        f"    print(f'a_trainable shape: {{ckt.a_trainable.shape}}')",
        f"    h0 = jnp.zeros({state_dim})",
        f"    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))",
        f"    h_final = ckt(time_info, h0, switch=jnp.array([]), args_seed=42, noise_seed=43)",
        f"    print(f'h_final shape: {{h_final.shape}}')  # ({state_dim},)",
        f"    print(f'h_re norm: {{jnp.linalg.norm(h_final[:{d_state}]):.4f}}')",
        f"    print(f'h_im norm: {{jnp.linalg.norm(h_final[{d_state}:]):.4f}}')",
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding="utf-8")
    return code
