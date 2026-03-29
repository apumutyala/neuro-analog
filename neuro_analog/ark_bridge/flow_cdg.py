"""
Flow matching CDG bridge — analog/digital partition analysis document.

Flow matching (dx/dt = v_θ(x, t)) CANNOT be expressed as a BaseAnalogCkt
subclass: v_θ is a 10B+ parameter transformer (FLUX, SD3) with no fixed-size
ODE state vector.  This module generates an analysis document that records the
analog/digital boundary and circuit mapping for co-design purposes.

Analog-native part: the Euler accumulation step
    x_{t+dt} = x_t + dt · v_θ(x, t)
maps cleanly to a capacitor + current injection (one analog accumulator per
dimension, identical to the Neural ODE RC integrator primitive).

The velocity network v_θ has crossbar-compatible linear projections but a
digital softmax in multi-head attention — so v_θ is mixed, not fully analog.

For the small-scale experiment model (experiments/cross_arch_tolerance), the
velocity field IS a fully analog MLP (no attention, no softmax).  Its export
uses NeuralODEExtractor → export_neural_ode_to_ark() directly.  This module
covers the conceptual FLUX/SD3-scale case.
"""

from __future__ import annotations

from pathlib import Path


def _analysis_header(model_name: str, arch: str, notes: list[str] | None = None) -> list[str]:
    """Header block for analysis-only exports — not runnable Ark code."""
    note_lines = [f"#   {n}" for n in (notes or [])]
    return [
        '"""',
        "Analog hardware analysis — neuro-analog.  NOT a runnable Ark module.",
        f"Model      : {model_name}",
        f"Architecture: {arch}",
        "",
        "Flow and diffusion architectures cannot be expressed as BaseAnalogCkt",
        "subclasses: their score/velocity networks are 10B+ parameter transformers",
        "with no fixed-size ODE state vector.  This file documents the analog/digital",
        "partition, circuit mapping, and D/A boundary analysis for use in co-design.",
        "",
        "For architectures that DO map to BaseAnalogCkt (Neural ODE, SSM, DEQ),",
        "see neuro_analog.extractors and neuro_analog.ark_bridge.",
        '"""',
        "",
        "import jax.numpy as jnp",
        "import diffrax",
        "",
    ] + note_lines + ([""] if note_lines else [])


def export_flow_to_ark(
    graph,
    output_path: Path | str,
    mismatch_sigma: float = 0.05,
    nfe: int = 4,
) -> str:
    """Export flow model analog/digital partition as an analysis document.

    Flow matching (dx/dt = v_θ(x,t)) CANNOT be expressed as a BaseAnalogCkt
    subclass: v_θ is a 10B+ parameter transformer with no fixed ODE state
    vector.  This export documents the analog/digital boundary and circuit
    mapping for co-design purposes — it is not runnable Ark code.

    For the small experiment model (FlowMLP, 2D), use FlowMLPExtractor whose
    export_to_ark() delegates to export_neural_ode_to_ark() — a proper
    BaseAnalogCkt subclass is generated because that velocity field IS an MLP.

    Args:
        graph:          AnalogGraph for the flow model (or any object with
                        .name and ._dynamics.num_function_evaluations).
        output_path:    Path to write the generated analysis document.
        mismatch_sigma: Nominal mismatch for the analysis header.
        nfe:            Fallback number of function evaluations if not in graph.

    Returns:
        Generated source code as a string.
    """
    dynamics = graph._dynamics
    actual_nfe = dynamics.num_function_evaluations or nfe
    dt = 1.0 / actual_nfe

    lines = _analysis_header(graph.name, "Flow (FLUX/SD3)", notes=[
        "dx/dt = v_θ(x, t)  — rectified flow matching ODE",
        "v_θ is a large transformer: NOT expressible as BaseAnalogCkt",
        "Analog-native part: Euler accumulation (capacitor + current injection)",
        f"NFE = {actual_nfe} steps, dt = {dt:.4f}",
        f"D/A boundaries per step: ~4   Total: ~{actual_nfe * 4}",
    ])

    lines += [
        "# FlowODE is an ANALYSIS class, not a BaseAnalogCkt subclass.",
        "class FlowODE:",
        f'    """Flow matching ODE analysis for {graph.name}.',
        "",
        "    NOT a BaseAnalogCkt subclass — v_θ transformer cannot be expressed",
        "    as a fixed-size CDG.  Documents analog/digital co-design boundary.",
        "",
        "    Analog-native:  x_{t+dt} = x_t + dt·v_θ  (capacitor + current injection)",
        "    Mixed:          v_θ linear projections → crossbar MVM",
        "                    attention softmax → digital",
        f"    D/A per step:   ~4   Total for {actual_nfe} steps: ~{actual_nfe * 4}",
        f"    Mismatch sigma: {mismatch_sigma}",
        '    """',
        "",
        f"    num_steps: int = {actual_nfe}",
        f"    dt: float = {dt:.4f}",
        "",
        "    def euler_step(self, x_t, v_theta, dt, gain=1.0):",
        '        """x_{t+dt} = x_t + dt · gain · v_θ  (analog accumulation)."""',
        "        return x_t + dt * gain * v_theta",
        "",
        "    def generate(self, z, velocity_fn):",
        f'        """Integrate {actual_nfe} Euler steps.  velocity_fn is mixed analog/digital."""',
        "        x = z",
        "        for i in range(self.num_steps):",
        "            t = i * self.dt",
        "            x = self.euler_step(x, velocity_fn(x, t), self.dt)",
        "        return x",
        "",
        "# To build a runnable Ark circuit: implement v_θ linear projections as CDG",
        "# crossbar nodes (see neuro_analog.ark_bridge), keep softmax digital.",
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding="utf-8")
    return code
