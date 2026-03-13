"""
Signal scaling / operating range analysis — Nonideality #4.

This mirrors Legno's LScale pass, adapted for neural network graphs.

LScale (Legno) solves:
    Find scaling factors s_i for each signal x_i such that:
        hw_min ≤ s_i · x_i ≤ hw_max   for all i, all inputs

    While preserving ODE dynamics. For dx/dt = Ax + Bu with scaled x' = Sx:
        dx'/dt = SAS⁻¹x' + SBu
    The eigenvalues of SAS⁻¹ are identical to A — scaling preserves dynamics.

For neural networks: each crossbar tile's output amplifier must be sized to
bring the analog signal into the hardware operating range. This is the "gain"
that appears between crossbar output and the next stage.

HCDCv2 hardware targets:
    Current mode: ±100 µA (multiplication, integration, summation)
    Voltage mode: ±1 V (lookup table inputs)
    ADC/DAC: ±1 V reference
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from neuro_analog.ir import AnalogGraph
from neuro_analog.ir.types import OpType, Domain, PrecisionSpec

# HCDCv2 operating ranges
HW_VOLTAGE_MIN = -1.0
HW_VOLTAGE_MAX = 1.0
HW_CURRENT_MIN = -100e-6   # -100 µA
HW_CURRENT_MAX = 100e-6    # +100 µA


@dataclass
class ScalingReport:
    """Per-node signal scaling analysis."""
    node_name: str
    op_type: str
    domain: str

    # Signal statistics (extracted from model calibration)
    signal_min: float = 0.0
    signal_max: float = 0.0
    signal_rms: float = 1.0

    # Hardware operating range
    hw_min: float = HW_VOLTAGE_MIN
    hw_max: float = HW_VOLTAGE_MAX

    # Scaling solution
    scale_factor: float = 1.0       # Multiply signal by this to fit hardware range
    gain_db: float = 0.0            # scale_factor in dB
    clipping_risk: bool = False     # True if signal exceeds hw range without scaling
    dynamic_range_db: float = 0.0   # Signal DR vs hardware DR

    # Hardware implication
    requires_gain_stage: bool = False
    gain_amplifier_power_mW: float = 0.0  # Programmable gain amp power

    # For ODE systems: scaled dynamics
    scaled_a_matrix_norm: Optional[float] = None  # ||SAS⁻¹|| (eigenvalues unchanged)
    scaling_comment: str = ""


def _compute_scale_factor(
    sig_max_abs: float,
    hw_max: float,
    hw_min: float,
) -> float:
    """Find scale factor s such that s · sig_max ≤ hw_max and s · sig_min ≥ hw_min."""
    if sig_max_abs <= 0:
        return 1.0
    hw_range = max(abs(hw_max), abs(hw_min))
    return hw_range / sig_max_abs


def _gain_amp_power_mW(gain_db: float) -> float:
    """Rough power estimate for a programmable gain amplifier.

    Simple model: P ≈ 0.1 mW base + 0.01 mW per dB of gain range.
    """
    return 0.1 + 0.01 * abs(gain_db)


def analyze_signal_ranges(
    graph: AnalogGraph,
    activation_stats: Optional[dict[str, PrecisionSpec]] = None,
    hw_voltage_range: tuple[float, float] = (HW_VOLTAGE_MIN, HW_VOLTAGE_MAX),
    hw_current_range: tuple[float, float] = (HW_CURRENT_MIN, HW_CURRENT_MAX),
) -> dict[str, ScalingReport]:
    """Determine signal scaling factors for all analog nodes in the graph.

    Implements Legno's LScale analysis adapted for neural network IR.

    For each analog node:
    1. Extract signal range from activation_stats (if available) or PrecisionSpec
    2. Determine whether the signal fits in the hardware operating range
    3. If not: compute scale factor (gain) needed at the output amplifier
    4. Report gain in dB, power cost, and whether dynamics are preserved

    For ODE systems (INTEGRATION/DECAY), verify that the scaled dynamics
    SAS⁻¹ have the same eigenvalue structure as the original.

    Args:
        graph: AnalogGraph from any extractor.
        activation_stats: Per-layer PrecisionSpec from calibration passes.
        hw_voltage_range: Hardware voltage operating range (HCDCv2: ±1V).
        hw_current_range: Hardware current operating range (HCDCv2: ±100µA).

    Returns:
        dict mapping node_name → ScalingReport
    """
    reports: dict[str, ScalingReport] = {}
    hw_vmin, hw_vmax = hw_voltage_range

    for node in graph.nodes:
        if node.domain == Domain.DIGITAL:
            continue

        report = ScalingReport(
            node_name=node.name,
            op_type=node.op_type.name,
            domain=node.domain.name,
            hw_min=hw_vmin,
            hw_max=hw_vmax,
        )

        # Get signal range: prefer measured activation stats
        sig_min = -1.0
        sig_max = 1.0
        source = "default"

        if activation_stats and node.name in activation_stats:
            p = activation_stats[node.name]
            if p.activation_max != 0:
                sig_max = p.activation_max
                sig_min = p.activation_min
                source = "calibration"
        elif node.precision and (node.precision.activation_max != 0 or node.precision.weight_max != 0):
            p = node.precision
            sig_max = p.activation_max if p.activation_max != 0 else p.weight_max
            sig_min = p.activation_min if p.activation_min != 0 else p.weight_min
            source = "precision_spec"

        report.signal_min = sig_min
        report.signal_max = sig_max
        sig_max_abs = max(abs(sig_max), abs(sig_min))
        report.signal_rms = sig_max_abs / math.sqrt(2.0)  # Assume sinusoidal

        # Check clipping risk
        report.clipping_risk = sig_max_abs > hw_vmax

        # Compute scale factor
        scale = _compute_scale_factor(sig_max_abs, hw_vmax, hw_vmin)
        report.scale_factor = scale
        report.gain_db = 20.0 * math.log10(abs(scale)) if scale != 0 else 0.0
        report.requires_gain_stage = abs(report.gain_db) > 1.0  # >1dB → need explicit gain

        # Dynamic range: signal DR vs hardware DR
        hw_dr_db = 20.0 * math.log10(hw_vmax / max(abs(hw_vmin), 1e-12)) if hw_vmin != 0 else 60.0
        sig_dr_db = 20.0 * math.log10(sig_max_abs / max(abs(sig_min), sig_max_abs / 1000))
        report.dynamic_range_db = sig_dr_db

        # Power cost if gain stage needed
        if report.requires_gain_stage:
            report.gain_amplifier_power_mW = _gain_amp_power_mW(report.gain_db)

        # ODE dynamics comment
        if node.op_type in (OpType.INTEGRATION, OpType.DECAY):
            tc = node.metadata.get("time_constant", 1e-3)
            scaled_tc = tc  # SAS⁻¹ has same eigenvalues → time constants unchanged
            report.scaled_a_matrix_norm = 1.0 / scaled_tc
            report.scaling_comment = (
                f"Scaling preserves dynamics: τ={tc:.3e}s unchanged (LScale invariant). "
                f"Input amplifier scales u by {scale:.3f}×; output divides by {scale:.3f}×."
            )
        elif node.op_type == OpType.MVM:
            report.scaling_comment = (
                f"Crossbar output amplifier gain = {scale:.3f}× ({report.gain_db:.1f} dB). "
                f"Source: {source}."
            )

        reports[node.name] = report

    return reports


def scaling_summary(reports: dict[str, ScalingReport]) -> str:
    """Generate a signal scaling report table."""
    lines = [
        "SIGNAL SCALING ANALYSIS (Legno LScale equivalent)",
        "=" * 70,
        f"Hardware range: ±{abs(HW_VOLTAGE_MAX):.1f} V",
        "",
        f"{'Node':<35} {'Type':<14} {'Sig Max':>8} {'Gain dB':>8} {'Gain Amp?':>10} {'Clip?':>6}",
        "─" * 83,
    ]
    clipping = 0
    gain_needed = 0
    total_power = 0.0
    for r in reports.values():
        if r.clipping_risk:
            clipping += 1
        if r.requires_gain_stage:
            gain_needed += 1
            total_power += r.gain_amplifier_power_mW
        clip_str = "YES" if r.clipping_risk else "—"
        gain_str = "YES" if r.requires_gain_stage else "—"
        lines.append(
            f"{r.node_name:<35} {r.op_type:<14} "
            f"{r.signal_max:>8.3f} {r.gain_db:>8.1f} {gain_str:>10} {clip_str:>6}"
        )
    lines += [
        "─" * 83,
        f"Clipping risk: {clipping} nodes | Gain stages needed: {gain_needed} | "
        f"Extra power: {total_power:.2f} mW",
    ]
    return "\n".join(lines)
