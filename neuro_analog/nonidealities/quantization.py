"""
DAC/ADC precision requirements — Nonideality #3.

Differentiable discrete optimization (Wang & Achour arXiv:2411.03557 §4.3) uses
Gumbel-Softmax relaxation for gradient-based tuning of quantized crossbar conductances.

Our role upstream: determine WHAT DAC resolution is needed at each D/A
boundary so that Ark's discrete optimizer can parameterize it correctly.

Key formula:
    SNR_dB = 6.02 · ENOB + 1.76   (ideal ADC/DAC)
    ENOB = (SNR_dB - 1.76) / 6.02

For a target SNR of 40dB: ENOB ≈ 6.4 → 7-bit minimum.

Dynamic range requirement:
    DR_db = 20 · log10(V_max / V_min)
    ENOB_required = DR_db / 6.02

Both constraints must be satisfied: we use the maximum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from neuro_analog.ir import AnalogGraph
from neuro_analog.ir.types import OpType, Domain, PrecisionSpec
from neuro_analog.ir.graph import DABoundary


@dataclass
class QuantizationReport:
    """Per-boundary ADC/DAC precision requirement."""
    boundary_id: str          # "src_node → tgt_node"
    direction: str            # "ADC" or "DAC"

    # From dynamic range analysis
    signal_max_abs: float = 0.0
    signal_min_abs: float = 0.0
    dynamic_range_db: float = 0.0

    # From SNR requirement
    target_snr_db: float = 40.0
    snr_limited_enob: float = 0.0

    # Final recommendation
    required_enob: float = 0.0
    required_bits: int = 8          # Ceiling of required_enob
    quantization_noise_variance: float = 0.0
    achieved_snr_db: float = 0.0

    # Gumbel-Softmax discrete levels (§4.3)
    num_discrete_levels: int = 256  # 2^required_bits

    # Hardware cost
    estimated_power_mW: float = 0.0
    estimated_area_um2: float = 0.0


def _enob_from_snr(target_snr_db: float) -> float:
    """ENOB from SNR target: ENOB = (SNR_dB - 1.76) / 6.02."""
    return (target_snr_db - 1.76) / 6.02


def _enob_from_dynamic_range(max_abs: float, min_abs: float) -> float:
    """ENOB from signal dynamic range: ENOB = DR_dB / 6.02."""
    if min_abs <= 0 or max_abs <= 0:
        return 8.0
    dr_db = 20.0 * math.log10(max_abs / min_abs)
    return dr_db / 6.02


def _quantization_snr(enob: int, signal_rms: float = 1.0, v_ref: float = 1.0) -> float:
    """Achieved SNR for an ideal n-bit converter."""
    return 6.02 * enob + 1.76


def _adc_power_mW(bits: int) -> float:
    """Rough power estimate for a SAR ADC (Walden figure-of-merit scaling).

    FOM = P / (f_s · 2^ENOB) ≈ 50 fJ/conversion (state-of-art 2024).
    At 1 MSPS: P ≈ 50e-15 · 1e6 · 2^bits [W]
    """
    return 50e-15 * 1e6 * (2 ** bits) * 1e3  # mW


def _adc_area_um2(bits: int) -> float:
    """Rough area estimate for a SAR ADC at 28nm node.

    Scales roughly as 2^bits · 0.5 µm² per bit (capacitor DAC).
    """
    return 2 ** bits * 0.5


def compute_precision_requirements(
    graph: AnalogGraph,
    weight_stats: Optional[dict[str, PrecisionSpec]] = None,
    activation_stats: Optional[dict[str, PrecisionSpec]] = None,
    target_snr_db: float = 40.0,
) -> dict[str, QuantizationReport]:
    """Determine ADC/DAC resolution at every D/A boundary in the graph.

    Method:
    1. Find all D/A boundaries via graph.find_da_boundaries()
    2. For each boundary, look up the signal dynamic range from activation_stats
       (or use PrecisionSpec from the source/target node)
    3. Compute ENOB from max(SNR requirement, dynamic range requirement)
    4. Report required bits, quantization noise, and discrete levels

    Args:
        graph: AnalogGraph from any extractor.
        weight_stats: Per-layer weight PrecisionSpec (from BaseExtractor).
        activation_stats: Per-layer activation PrecisionSpec (from calibration).
        target_snr_db: Minimum SNR requirement (40dB ≈ 6.5 ENOB).

    Returns:
        dict mapping boundary_id → QuantizationReport
    """
    boundaries = graph.find_da_boundaries()
    reports: dict[str, QuantizationReport] = {}

    for boundary in boundaries:
        src_node = graph.get_node(boundary.source_node_id)
        tgt_node = graph.get_node(boundary.target_node_id)
        boundary_id = f"{boundary.source_node_id} → {boundary.target_node_id}"

        report = QuantizationReport(
            boundary_id=boundary_id,
            direction=boundary.direction,
            target_snr_db=target_snr_db,
        )

        # Get signal statistics from the source node's precision spec
        signal_max = 1.0
        signal_min = 1e-3  # Default: assume 60dB dynamic range
        if src_node and src_node.precision:
            p = src_node.precision
            if p.activation_max != 0 and p.activation_min != 0:
                signal_max = abs(p.activation_max)
                signal_min = abs(p.activation_min) if p.activation_min != 0 else signal_max / 1000
            elif p.weight_max != 0:
                signal_max = abs(p.weight_max)
                signal_min = abs(p.weight_min) if p.weight_min != 0 else signal_max / 1000

        # Override with activation_stats if available (more accurate)
        if activation_stats and src_node and src_node.name in activation_stats:
            ps = activation_stats[src_node.name]
            if ps.activation_max != 0:
                signal_max = abs(ps.activation_max)
                signal_min = abs(ps.activation_min) if ps.activation_min != 0 else signal_max / 1000

        # Dynamic range in dB
        dr_db = 20.0 * math.log10(max(signal_max, 1e-9) / max(signal_min, 1e-12))
        report.signal_max_abs = signal_max
        report.signal_min_abs = signal_min
        report.dynamic_range_db = dr_db

        # ENOB from two constraints
        enob_snr = _enob_from_snr(target_snr_db)
        enob_dr = dr_db / 6.02
        required_enob = max(enob_snr, enob_dr)

        report.snr_limited_enob = enob_snr
        report.required_enob = required_enob
        report.required_bits = math.ceil(required_enob)

        # Quantization noise at recommended precision
        v_lsb = (2.0 * signal_max) / (2 ** report.required_bits)
        report.quantization_noise_variance = (v_lsb ** 2) / 12.0
        report.achieved_snr_db = _quantization_snr(report.required_bits)
        report.num_discrete_levels = 2 ** report.required_bits

        # Hardware cost estimates
        report.estimated_power_mW = _adc_power_mW(report.required_bits)
        report.estimated_area_um2 = _adc_area_um2(report.required_bits)

        reports[boundary_id] = report

    return reports


def quantization_summary(reports: dict[str, QuantizationReport]) -> str:
    """Format a table of converter requirements."""
    lines = [
        "ADC/DAC PRECISION REQUIREMENTS",
        "=" * 70,
        f"{'Boundary':<45} {'Dir':>4} {'Bits':>5} {'SNR':>7} {'Levels':>8} {'Power':>8}",
        "─" * 78,
    ]
    total_power = 0.0
    for r in reports.values():
        total_power += r.estimated_power_mW
        lines.append(
            f"{r.boundary_id:<45} {r.direction:>4} {r.required_bits:>5} "
            f"{r.achieved_snr_db:>6.1f}dB {r.num_discrete_levels:>8,} "
            f"{r.estimated_power_mW:>7.3f}mW"
        )
    lines += [
        "─" * 78,
        f"Total converter overhead: {total_power:.3f} mW across {len(reports)} boundaries",
        f"",
        f"Ark note: use Gumbel-Softmax(num_levels) for differentiable discrete optimization",
    ]
    return "\n".join(lines)
