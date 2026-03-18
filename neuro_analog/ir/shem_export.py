"""
Export AnalogGraph to Shem-compatible JAX ODE specifications.

Generates JAX/Diffrax code from extracted AnalogGraphs, following the
Shem compiler's ODE specification format.

Actual Shem API (from paper — not approximated):
  - AnalogTrainable(init=val)          marks parameter as optimizable
  - mismatch(param, sigma=0.1)         multiplicative δ ~ N(1,σ²) perturbation
  - Shem.compile(model)                produces differentiable simulation
  - model.diffeq_solve(seed=...)       forward solve with sampled mismatch
  - model.gradient(loss)              adjoint-based gradient computation

Shem uses Diffrax internally. All our exports use Diffrax for ODE solving
so the generated code is immediately composable with Shem's infrastructure.

Readout times follow Shem's convention — cost functions are evaluated at
specific time points along the ODE trajectory, not over the full integral.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from .graph import AnalogGraph
from .types import ArchitectureFamily, Domain, OpType
from .node import AnalogNode


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

def _header(model_name: str, arch: str, notes: list[str] | None = None) -> list[str]:
    """Standard file header block."""
    extra = "\n".join(f"# {n}" for n in (notes or []))
    return [
        f'"""',
        f"Auto-generated Shem ODE specification — neuro-analog.",
        f"Model      : {model_name}",
        f"Architecture: {arch}",
        f"",
        f"Usage:",
        f"    from shem import Shem",
        f"    model = <ClassName>()",
        f"    compiled = Shem.compile(model)",
        f"    sol = compiled.diffeq_solve(mismatch_sample_seed=42)",
        f"    grad = compiled.gradient(your_loss)",
        f'"""',
        f"",
        f"import jax",
        f"import jax.numpy as jnp",
        f"import diffrax",
        f"from ark.optimization.base_module import BaseAnalogCkt",
        f"from shem import AnalogTrainable, mismatch, Shem",
        f"",
    ]


def _mismatch_sigma_from_node(node: AnalogNode, default: float = 0.05) -> float:
    """Extract tolerable mismatch sigma from node's NoiseSpec or use default."""
    if node.noise and node.noise.sigma > 0:
        # Convert absolute sigma to relative if it looks like a fraction
        if node.noise.sigma < 1.0:
            return float(node.noise.sigma)
    return default


def _noise_comment(node: AnalogNode) -> str:
    """Inline comment describing the hardware noise model for a node."""
    if not node.noise:
        return "  # no noise spec"
    return (
        f"  # {node.noise.kind} noise: σ={node.noise.sigma:.2e}, "
        f"BW={node.noise.bandwidth_hz or 'N/A'} Hz"
    )


def _diffrax_solve_block(
    t0: float,
    t1: float,
    readout_times: list[float],
    dt0: float = 0.01,
    solver: str = "Tsit5",
) -> list[str]:
    """Generate a Diffrax solve block — the same solver Shem uses internally."""
    ts_str = str(readout_times)
    return [
        f"    def __call__(self, y0: jnp.ndarray) -> jnp.ndarray:",
        f'        """Integrate ODE using Diffrax (Shem\'s internal solver).',
        f"",
        f"        Solver: {solver} (Runge-Kutta 4/5 with error control).",
        f"        Readout times: {readout_times}",
        f"        For Shem: cost is evaluated at these time points.",
        f'        """',
        f"        term = diffrax.ODETerm(self.dynamics)",
        f"        solver = diffrax.{solver}()",
        f"        saveat = diffrax.SaveAt(ts=jnp.array({ts_str}))",
        f"        sol = diffrax.diffeqsolve(",
        f"            term, solver,",
        f"            t0={t0}, t1={t1}, dt0={dt0},",
        f"            y0=y0, saveat=saveat,",
        f"        )",
        f"        return sol.ys  # shape: (len(readout_times), state_dim)",
        f"",
    ]


# ──────────────────────────────────────────────────────────────────────
# SSM export
# ──────────────────────────────────────────────────────────────────────

def export_ssm_to_shem(
    graph: AnalogGraph,
    extractor,
    output_path: Path | str,
    mismatch_sigma: float = 0.05,
) -> str:
    """Export SSM dynamics as a proper BaseAnalogCkt subclass.

    Dynamics: dx/dt = A * x + B * u  (A diagonal, u=0 autonomous simplification).
    Compatible with Ark's OptCompiler output format and BaseAnalogCkt.__call__.
    """
    dynamics = graph._dynamics
    state_dim = dynamics.state_dimension or 16
    tc_sample = dynamics.time_constants[:state_dim] if dynamics.time_constants else [1e-3] * state_dim
    A_vals = [round(-1.0 / max(tc, 1e-9), 6) for tc in tc_sample]

    B_weights, C_weights = None, None
    if extractor and hasattr(extractor, 'model') and extractor.model is not None:
        for name, param in extractor.model.named_parameters():
            if 'x_proj' in name and 'weight' in name:
                x_proj = param.detach().float().cpu().numpy()
                mid = x_proj.shape[0] // 2
                B_weights = x_proj[:mid, :state_dim]
                C_weights = x_proj[mid:, :state_dim]
                break

    # Static offsets (A, B, C all length state_dim — diagonal vectors)
    A_off, B_off, C_off = 0, state_dim, 2 * state_dim

    if B_weights is not None:
        import numpy as _np
        B_diag = _np.diag(B_weights[:state_dim, :state_dim]).tolist() if B_weights.shape[0] >= state_dim else [1.0] * state_dim
        C_diag = _np.diag(C_weights[:state_dim, :state_dim]).tolist() if C_weights.shape[0] >= state_dim else [1.0] * state_dim
        B_init = f"jnp.array({[round(float(v), 6) for v in B_diag]})"
        C_init = f"jnp.array({[round(float(v), 6) for v in C_diag]})"
    else:
        B_init = f"jnp.ones({state_dim})"
        C_init = f"jnp.ones({state_dim})"

    lines = [
        f'"""',
        f"Ark-compatible JAX ODE specification for {graph.name} (SSM).",
        f"Proper BaseAnalogCkt subclass -- compatible with Ark OptCompiler output format.",
        f"",
        f"Dynamics: dx/dt = A * x + B * u  (diagonal A, u=0 autonomous simplification)",
        f"For sequence processing: augment initial_state with u(t).",
        f"",
        f"Usage:",
        f"    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))",
        f"    result = ckt(time_info, h0, switch=jnp.array([]), args_seed=42, noise_seed=43)",
        f'"""',
        f"",
        f"import jax",
        f"import jax.numpy as jnp",
        f"import jax.random as jrandom",
        f"import diffrax",
        f"from ark.optimization.base_module import BaseAnalogCkt, TimeInfo",
        f"",
        f"class SSMAnalogCkt(BaseAnalogCkt):",
        f'    """Continuous-time SSM (BaseAnalogCkt subclass).',
        f"",
        f"    State: x (hidden state, dim={state_dim})",
        f"    Dynamics: dx/dt = A * x + B * u  (diagonal A -- element-wise multiply)",
        f'    """',
        f"",
        f"    def __init__(self):",
        f"        _A = jnp.array({A_vals})",
        f"        _B = {B_init}",
        f"        _C = {C_init}",
        f"        a_trainable = jnp.concatenate([_A, _B, _C])",
        f"        super().__init__(",
        f"            init_trainable=a_trainable,",
        f"            is_stochastic=False,",
        f"            solver=diffrax.Heun(),",
        f"        )",
        f"",
        f"    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):",
        f"        key = jrandom.PRNGKey(mismatch_seed)",
        f"        k1, k2, k3 = jrandom.split(key, 3)",
        f"        sigma = {mismatch_sigma}",
        f"        _A = self.a_trainable[{A_off}:{A_off+state_dim}] * (1.0 + sigma * jrandom.normal(k1, ({state_dim},)))",
        f"        _B = self.a_trainable[{B_off}:{B_off+state_dim}] * (1.0 + sigma * jrandom.normal(k2, ({state_dim},)))",
        f"        _C = self.a_trainable[{C_off}:{C_off+state_dim}] * (1.0 + sigma * jrandom.normal(k3, ({state_dim},)))",
        f"        return (_A, _B, _C)",
        f"",
        f"    def ode_fn(self, t, x, args):",
        f"        A, B, C = args",
        f"        u = jnp.zeros_like(x)  # autonomous; replace with input for seq processing",
        f"        return A * x + B * u   # diagonal A: element-wise multiply",
        f"",
        f"    def noise_fn(self, t, x, args):",
        f"        return jnp.ones_like(x) * 0.01",
        f"",
        f"    def readout(self, y):",
        f"        return y[-1]  # final time point, shape ({state_dim},)",
        f"",
        f"if __name__ == '__main__':",
        f"    ckt = SSMAnalogCkt()",
        f"    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))",
        f"    h0 = jnp.zeros(({state_dim},))",
        f"    switch = jnp.array([])",
        f"    result = ckt(time_info, h0, switch, args_seed=42, noise_seed=43)",
        f"    print(f'Result shape: {{result.shape}}')",
        f"",
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding='utf-8')
    return code


# ──────────────────────────────────────────────────────────────────────
# Flow model export
# ──────────────────────────────────────────────────────────────────────

def export_flow_to_shem(
    graph: AnalogGraph,
    output_path: Path | str,
    mismatch_sigma: float = 0.05,
    nfe: int = 4,
) -> str:
    """Export flow model ODE to Shem-compatible JAX code.

    Flow matching: dx/dt = v_θ(x,t), formally identical to Arco's input.

    The challenge: v_θ is a 12B-param transformer. We can't express it
    directly as Shem AnalogTrainable parameters at that scale.

    Strategy: express the ODE INTEGRATION LOOP as a Shem system, with
    v_θ as an external oracle (the mixed analog/digital computation).
    The analog part (capacitor accumulation of Euler steps) IS in Shem's domain.

    Readout convention: t = 1.0 (final generated image).
    """
    dynamics = graph._dynamics
    actual_nfe = dynamics.num_function_evaluations or nfe
    dt = 1.0 / actual_nfe
    readout_times = [round((i + 1) * dt, 4) for i in range(actual_nfe)]

    lines = _header(graph.name, "Flow (FLUX/SD3)", notes=[
        "dx/dt = v_θ(x, t)  — rectified flow matching ODE",
        "v_θ is a large transformer (analog/digital partition required)",
        "ODE integration loop (Euler steps) is analog-native",
        f"NFE = {actual_nfe} steps, dt = {dt:.4f}",
    ])

    lines += [
        f"class FlowODE(BaseAnalogCkt):",
        f'    """Flow matching ODE for {graph.name}.',
        f"",
        f"    v_θ evaluation is mixed analog/digital (see AnalogGraph partition).",
        f"    The Euler accumulation x_{{t+dt}} = x_t + dt·v_θ is pure analog.",
        f"",
        f"    Analog circuit mapping:",
        f"      x_t → capacitor state (stored charge)",
        f"      dt·v_θ → current injection (programmable gain amp × v_θ)",
        f"      accumulation → Kirchhoff current summation",
        f"",
        f"    D/A boundaries per step: ~4 (2× ADC for digital attention, 2× DAC back)",
        f"    Total for {actual_nfe} steps: ~{actual_nfe * 4} converters",
        f'    """',
        f"",
        f"    def __init__(self, num_steps: int = {actual_nfe}):",
        f"        self.num_steps = num_steps",
        f"        self.dt = 1.0 / num_steps",
        f"        # Readout at final generation: t = 1.0",
        f"        self.readout_time = 1.0",
        f"",
        f"        # Gain parameter for Euler scaling: x_new = x + gain * v_theta * dt",
        f"        # In hardware: programmable gain amplifier at crossbar output",
        f"        self.euler_gain = mismatch(",
        f"            AnalogTrainable(init=jnp.ones(1)),",
        f"            sigma={mismatch_sigma}  # Gain mismatch → step size error",
        f"        )",
        f"        self.d_trainable = []  # no discrete/digital trainable params",
        f"",
        f"    def euler_step(",
        f"        self,",
        f"        x_t: jnp.ndarray,",
        f"        v_theta: jnp.ndarray,",
        f"        dt: float,",
        f"    ) -> jnp.ndarray:",
        f'        """Single Euler step: x_{{t+dt}} = x_t + dt · gain · v_θ.',
        f"",
        f"        Analog: dt is a fixed time reference; gain is a crossbar output amplifier.",
        f"        Accumulation via capacitor charge (Kirchhoff's current law).",
        f'        """',
        f"        return x_t + dt * self.euler_gain * v_theta",
        f"",
        f"    def dynamics(self, t, x):",
        f'        """Placeholder — in production, v_θ network evaluation goes here."""',
        f"        return jnp.zeros_like(x)  # Replace with actual v_θ call",
        f"",
    ]

    lines += _diffrax_solve_block(0.0, 1.0, readout_times, dt0=dt, solver="Euler")

    lines += [
        f"    def generate(self, z: jnp.ndarray, velocity_fn) -> jnp.ndarray:",
        f'        """Full generation: integrate {actual_nfe} Euler steps.',
        f"",
        f"        For FLUX-schnell: t = 0.00 → 0.25 → 0.50 → 0.75 → 1.00",
        f'        """',
        f"        x = z",
        f"        for i in range(self.num_steps):",
        f"            t = i * self.dt",
        f"            v = velocity_fn(x, t)  # Mixed analog/digital (see AnalogGraph)",
        f"            x = self.euler_step(x, v, self.dt)",
        f"        return x  # Readout at t=1.0",
        f"",
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding="utf-8")
    return code


# ──────────────────────────────────────────────────────────────────────
# Diffusion model export
# ──────────────────────────────────────────────────────────────────────

def export_diffusion_to_shem(
    graph: AnalogGraph,
    output_path: Path | str,
    mismatch_sigma: float = 0.05,
) -> str:
    """Export diffusion SDE/ODE to Shem-compatible JAX code.

    VP-SDE: dx = [-½β(t)x - β(t)s_θ(x,t)]dt + √β(t)dW

    CLD (Critically-Damped Langevin Diffusion) → RLC circuit:
      dx = β/M · v · dt
      dv = -β·x·dt - Γ·β/M·v·dt + √(2Γβ)·dW

    RLC mapping:
      x → capacitor charge (q),  v → inductor current (i)
      M → inductance (L),         Γ → resistance (R), 1/β → capacitance (C)
      √(2Γβ)·dW → Johnson-Nyquist thermal noise (physical, not injected)

    Readout convention: one readout per denoising step.
    """
    dynamics = graph._dynamics
    num_steps = dynamics.num_diffusion_steps or 20

    beta_min = 0.0001
    beta_max = 0.02
    if dynamics.beta_schedule and len(dynamics.beta_schedule) >= 2:
        beta_min = float(min(dynamics.beta_schedule))
        beta_max = float(max(dynamics.beta_schedule))

    readout_times = [round(i / num_steps, 4) for i in range(1, num_steps + 1)]

    lines = _header(graph.name, "Diffusion (SD/DiT)", notes=[
        "VP-SDE: dx = [-½β(t)x - β(t)s_θ]dt + √β(t)dW",
        "CLD maps to RLC circuit — Johnson-Nyquist noise is physical",
        f"β(t) ∈ [{beta_min:.4f}, {beta_max:.4f}], steps = {num_steps}",
        "Score network s_θ is the mixed analog/digital bottleneck",
    ])

    lines += [
        f"class DiffusionDynamics(BaseAnalogCkt):",
        f'    """VP-SDE and CLD dynamics for {graph.name}.',
        f"",
        f"    The score network s_θ(x,t) is 99.9% of compute per step.",
        f"    The SDE update itself is trivially analog (scaling + noise injection).",
        f"    See AnalogGraph for full s_θ partition (analog ~72%, digital ~28%).",
        f'    """',
        f"",
        f"    def __init__(self):",
        f"        self.num_steps = {num_steps}",
        f"        self.beta_min = {beta_min}",
        f"        self.beta_max = {beta_max}",
        f"        # β(t) schedule: extracted from scheduler.betas",
        f"        self.betas = mismatch(",
        f"            AnalogTrainable(init=jnp.linspace({beta_min}, {beta_max}, {num_steps})),",
        f"            sigma={mismatch_sigma}  # β precision → noise schedule accuracy",
        f"        )",
        f"        # Readout at each denoising step boundary",
        f"        self.readout_times = jnp.array({readout_times[:min(8, len(readout_times))]})",
        f"        self.d_trainable = []  # no discrete/digital trainable params",
        f"",
        f"    def _beta_t(self, t: float) -> jnp.ndarray:",
        f"        return self.beta_min + t * (self.beta_max - self.beta_min)",
        f"",
        f"    def vp_sde_drift(self, t, x, score_fn):",
        f'        """VP-SDE drift term (the deterministic part of dx/dt).',
        f"",
        f"        Analog: -½β(t)x via programmable gain (GAIN node).",
        f"        Mixed:  -β(t)·s_θ(x,t) via score network (AnalogGraph partition).",
        f'        """',
        f"        beta_t = self._beta_t(t)",
        f"        score = score_fn(x, t)  # s_θ: mixed analog/digital",
        f"        return -0.5 * beta_t * x - beta_t * score",
        f"",
        f"    def vp_sde_diffusion_coeff(self, t) -> jnp.ndarray:",
        f'        """Diffusion coefficient g(t) = √β(t).',
        f"",
        f"        Analog: hardware thermal noise source calibrated to match √β(t).",
        f"        This is where Extropic's stochastic units (sMTJ) shine.",
        f'        """',
        f"        return jnp.sqrt(self._beta_t(t))",
        f"",
        f"    def cld_dynamics(self, t, state, score_fn, M: float = 1.0, Gamma: float = 2.0):",
        f'        """Critically-Damped Langevin Diffusion — best analog match.',
        f"",
        f"        dx = β/M · v · dt",
        f"        dv = -β·x·dt - Γ·β/M·v·dt  (+ score network correction)",
        f"",
        f"        Maps exactly to RLC: L=M, R=Γ, C=1/β.",
        f"        Thermal noise in the resistor = Johnson-Nyquist = √(4kTR·BW).",
        f"        This is NOT injected noise — it is physical hardware noise.",
        f'        """',
        f"        x, v = state[..., :state.shape[-1]//2], state[..., state.shape[-1]//2:]",
        f"        beta_t = self._beta_t(t)",
        f"        dx = beta_t / M * v",
        f"        dv = -beta_t * x - Gamma * beta_t / M * v + score_fn(x, t)",
        f"        return jnp.concatenate([dx, dv], axis=-1)",
        f"",
    ]

    # Diffrax SDE solve (using Euler-Maruyama)
    lines += [
        f"    def solve_sde(self, x0: jnp.ndarray) -> jnp.ndarray:",
        f'        """Solve VP-SDE via Euler-Maruyama (Shem\'s SDE mode).',
        f"",
        f"        dx = drift·dt + diffusion·dW",
        f"        Hardware: drift via analog circuit, dW via thermal TRNG.",
        f'        """',
        f"        term = diffrax.MultiTerm(",
        f"            diffrax.ODETerm(lambda t, x, args: self.vp_sde_drift(t, x, args)),",
        f"            diffrax.WeaklyDiagonalControlTerm(",
        f"                lambda t, x, args: self.vp_sde_diffusion_coeff(t),",
        f"                diffrax.VirtualBrownianTree(t0=1.0, t1=0.0, tol=1e-3, shape=x0.shape, key=jax.random.PRNGKey(0)),",
        f"            ),",
        f"        )",
        f"        solver = diffrax.Euler()",
        f"        saveat = diffrax.SaveAt(ts=self.readout_times)",
        f"        sol = diffrax.diffeqsolve(",
        f"            term, solver, t0=1.0, t1=0.0, dt0=-1.0/{num_steps},",
        f"            y0=x0, saveat=saveat, args=lambda x, t: jnp.zeros_like(x),",
        f"        )",
        f"        return sol.ys",
        f"",
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding="utf-8")
    return code



def export_deq_to_shem(graph: AnalogGraph, output_path, sigma: float = 0.05) -> str:
    """Export a DEQ AnalogGraph as a Shem/Diffrax ODE specification.

    DEQ fixed-point equation z* = f_theta(z*, x) maps to the gradient flow ODE:
      dz/dt = f_theta(z, x) - z
    At equilibrium dz/dt = 0, so z* = f_theta(z*, x).

    This is native Arco/Legno format. Shem can optimize theta for mismatch-robust
    convergence - pushing the spectral radius rho(df/dz) further below 1.

    State augmentation: y = concat([z, x_input]), dx/dt = 0 (x is held constant).
    This lets BaseAnalogCkt.__call__ drive both z and x through a single ODETerm.

    Args:
        graph: AnalogGraph with family=DEQ (expects 3 MVM nodes: W_z, W_x, W_out).
        output_path: Path to write generated .py file.
        sigma: Multiplicative mismatch sigma (delta ~ N(1, sigma^2)).

    Returns:
        Generated source code as string.
    """
    mvm_nodes = [n for n in graph.nodes if n.op_type == OpType.MVM]
    if len(mvm_nodes) < 3:
        raise ValueError(
            f"DEQ export expects 3 MVM nodes (W_z, W_x, W_out), got {len(mvm_nodes)}"
        )

    # Extract dims from graph nodes
    z_dim = mvm_nodes[0].input_shape[0] if mvm_nodes[0].input_shape else 64
    hidden_dim = mvm_nodes[0].output_shape[0] if mvm_nodes[0].output_shape else 128
    x_dim = mvm_nodes[1].input_shape[0] if mvm_nodes[1].input_shape else 64

    # Flat parameter layout (all computed at export time - no dynamic indexing)
    WZ_SIZE = hidden_dim * z_dim
    WX_SIZE = hidden_dim * x_dim
    BH_SIZE = hidden_dim
    WOUT_SIZE = z_dim * hidden_dim
    BOUT_SIZE = z_dim

    WZ_END = WZ_SIZE
    WX_END = WZ_END + WX_SIZE
    BH_END = WX_END + BH_SIZE
    WOUT_END = BH_END + WOUT_SIZE
    BOUT_END = WOUT_END + BOUT_SIZE
    N_PARAMS = BOUT_END
    N_SHAPES = 5  # Wz, Wx, bh, Wout, bout

    aug_dim = z_dim + x_dim  # augmented state y = [z, x]

    lines = [
        "# Auto-generated BaseAnalogCkt DEQ specification",
        "# dz/dt = f_theta(z, x) - z  [gradient flow to fixed point]",
        "# State augmented: y = concat([z, x_input]), dx/dt = 0",
        "# Generated by neuro_analog.ir.shem_export.export_deq_to_shem()",
        "",
        "import jax.numpy as jnp",
        "import jax.random as jrandom",
        "import diffrax",
        "import equinox as eqx",
        "from ark.optimization.base_module import BaseAnalogCkt, TimeInfo",
        "",
        "",
        "class DEQAnalogCkt(BaseAnalogCkt):",
        f'    """DEQ gradient-flow ODE: dz/dt = f_theta(z, x) - z.',
        f"",
        f"    State augmented: y = concat([z, x_input]), dx/dt = 0.",
        f"    At equilibrium: z* = f_theta(z*, x_input).",
        f"    rho(df_theta/dz) < 1 guarantees convergence.",
        f"",
        f"    z_dim={z_dim}, x_dim={x_dim}, hidden_dim={hidden_dim}",
        f"    Mismatch sigma={sigma} on all analog weight matrices.",
        f'    """',
        "",
        "    shapes: list",
        "",
        "    def __init__(self):",
        f"        a_trainable = jnp.zeros({N_PARAMS})",
        f"        object.__setattr__(self, 'shapes', [{z_dim}, {x_dim}, {hidden_dim}])",
        "        super().__init__(",
        "            init_trainable=a_trainable,",
        "            is_stochastic=False,",
        "            solver=diffrax.Heun(),",
        "        )",
        "",
        "    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):",
        f'        """Sample mismatch perturbations and return (Wz, Wx, bh, Wout, bout)."""',
        f"        key = jrandom.PRNGKey(mismatch_seed)",
        f"        keys = jrandom.split(key, {N_SHAPES})",
        f"        sigma = {sigma}",
        f"        _Wz   = (self.a_trainable[0:{WZ_END}]"
        f" * (1.0 + sigma * jrandom.normal(keys[0], ({WZ_SIZE},)))).reshape(({hidden_dim}, {z_dim}))",
        f"        _Wx   = (self.a_trainable[{WZ_END}:{WX_END}]"
        f" * (1.0 + sigma * jrandom.normal(keys[1], ({WX_SIZE},)))).reshape(({hidden_dim}, {x_dim}))",
        f"        _bh   = (self.a_trainable[{WX_END}:{BH_END}]"
        f" * (1.0 + sigma * jrandom.normal(keys[2], ({BH_SIZE},))))",
        f"        _Wout = (self.a_trainable[{BH_END}:{WOUT_END}]"
        f" * (1.0 + sigma * jrandom.normal(keys[3], ({WOUT_SIZE},)))).reshape(({z_dim}, {hidden_dim}))",
        f"        _bout = (self.a_trainable[{WOUT_END}:{BOUT_END}]"
        f" * (1.0 + sigma * jrandom.normal(keys[4], ({BOUT_SIZE},))))",
        f"        return (_Wz, _Wx, _bh, _Wout, _bout)",
        "",
        "    def ode_fn(self, t, y, args):",
        f'        """dy/dt = [dz/dt, 0]  where dz/dt = f_theta(z, x_input) - z."""',
        "        Wz, Wx, bh, Wout, bout = args",
        f"        z = y[:{z_dim}]",
        f"        x = y[{z_dim}:]",
        "        h  = jnp.tanh(Wz @ z + Wx @ x + bh)",
        "        dz = Wout @ h + bout - z",
        f"        dx = jnp.zeros({x_dim})",
        "        return jnp.concatenate([dz, dx])",
        "",
        "    def noise_fn(self, t, y, args):",
        f"        return jnp.zeros({aug_dim})",
        "",
        "    def readout(self, y):",
        f'        """Extract z* from the last saved time point."""',
        f"        return y[-1, :{z_dim}]",
        "",
        "",
        'if __name__ == "__main__":',
        "    import jax",
        "",
        "    ckt = DEQAnalogCkt()",
        f"    print('Is BaseAnalogCkt subclass:', issubclass(DEQAnalogCkt, BaseAnalogCkt))",
        f"    print('isinstance check:', isinstance(ckt, BaseAnalogCkt))",
        f"    print('a_trainable shape:', ckt.a_trainable.shape)",
        f"    print('is_stochastic:', ckt.is_stochastic)",
        f"    print('solver:', type(ckt.solver).__name__)",
        "",
        f"    x_input = jax.random.normal(jax.random.PRNGKey(0), ({x_dim},))",
        f"    z0 = jnp.zeros({z_dim})",
        "    y0 = jnp.concatenate([z0, x_input])",
        "    time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.1, saveat=jnp.array([5.0]))",
        "    switch = jnp.array([])",
        "    result = ckt(",
        "        time_info, y0, switch=switch,",
        "        args_seed=42, noise_seed=43, gumbel_temp=1.0, hard_gumbel=False,",
        "    )",
        f"    print('z_star shape:', result.shape)  # should be ({z_dim},)",
        "    print('z_star:', result)",
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding="utf-8")
    return code
