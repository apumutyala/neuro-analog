"""
Thermal noise budget — Nonideality #2.

Wang & Achour (arXiv:2411.03557) model transient noise as an SDE: dx = f(x,θ,t)dt + g(x,θ,t)dW(t).
Our framework estimates the diffusion coefficient g at each node from
first-principles physics, then propagates noise power through the graph.

Physics models:
  kT/C   — thermal noise floor for capacitor-based integrators (op-amp integrators)
  Shot   — current-mode circuits: I_noise = sqrt(2qI·BW)
  ADC    — quantization noise: σ² = V_ref² / (12 · 4^ENOB)
  Crossbar — conductance variation noise + readout amplifier kT/C

All models reference Legno's HCDCv2 chip targets:
  Voltage range: ±1V    (current mode: ±100µA)
  Bandwidth: ~250kHz (limited by op-amp GBW product)
  Temperature: 300K  (room temperature)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from neuro_analog.ir import AnalogGraph
from neuro_analog.ir.types import OpType, Domain, NoiseSpec

# Physical constants
K_B = 1.380649e-23   # Boltzmann constant (J/K)
Q_E = 1.602176634e-19  # Elementary charge (C)

# HCDCv2 target parameters (Legno hardware)
HW_BANDWIDTH_HZ = 250e3       # 250 kHz op-amp bandwidth
HW_CAPACITANCE_F = 1e-12      # 1 pF on-chip integrating capacitor
HW_BIAS_CURRENT_A = 1e-6      # 1 µA bias current (current-mode circuits)
HW_V_REF = 1.0                # ±1V reference for ADC


@dataclass
class NoiseBudget:
    """Per-node noise power budget."""
    node_name: str
    op_type: str
    domain: str

    # Noise power (variance) in equivalent input units
    thermal_noise_variance: float = 0.0     # kT/C (integrators, RC)
    shot_noise_variance: float = 0.0        # 2qI·BW (current-mode)
    quantization_noise_variance: float = 0.0  # ADC/DAC quantization
    total_noise_variance: float = 0.0

    # SNR (undefined for stochastic-native nodes — use calibration_error instead)
    signal_rms: float = 1.0
    snr_db: float = float("inf")
    meets_target_snr: bool = True

    # Calibration error for SAMPLE and NOISE_INJECTION nodes.
    # These nodes intentionally inject noise — the metric is whether the actual σ
    # matches the target σ, not whether noise is small relative to signal.
    # calibration_error = |actual_σ - target_σ| / target_σ  (nan if no target_σ set)
    calibration_error: float = float("nan")
    meets_calibration: bool = True

    # SDE parameters (Wang & Achour §4.2 noise model)
    sde_diffusion_coeff: float = 0.0  # g in dx = f·dt + g·dW
    noise_bandwidth_hz: float = HW_BANDWIDTH_HZ

    # Hardware model details
    temperature_K: float = 300.0
    capacitance_F: float = HW_CAPACITANCE_F
    enob: Optional[int] = None  # Effective number of bits (converters)


def _thermal_noise_variance(temperature_K: float = 300.0, capacitance_F: float = HW_CAPACITANCE_F) -> float:
    """kT/C thermal noise variance for a capacitor-based integrator.

    This is the fundamental lower bound for any RC-based analog memory.
    σ² = kT/C  (units: V²)

    At T=300K, C=1pF: σ = sqrt(4.14e-9) ≈ 64 µV RMS.
    """
    return K_B * temperature_K / capacitance_F


def _shot_noise_variance(bias_current_A: float = HW_BIAS_CURRENT_A, bandwidth_hz: float = HW_BANDWIDTH_HZ) -> float:
    """Shot noise variance for current-mode circuits.

    I_noise² = 2 · q · I_bias · BW  (units: A²)
    Normalized to input units: σ² = I_noise² / I_signal²
    """
    return 2.0 * Q_E * bias_current_A * bandwidth_hz


def _quantization_noise_variance(enob: int = 8, v_ref: float = HW_V_REF) -> float:
    """ADC/DAC quantization noise variance.

    σ² = (2·V_ref)² / (12 · 4^ENOB)
    = V_LSB² / 12
    where V_LSB = 2·V_ref / 2^ENOB.

    SNR formula: SNR_dB = 6.02·ENOB + 1.76 dB.
    """
    v_lsb = (2.0 * v_ref) / (2 ** enob)
    return (v_lsb ** 2) / 12.0


def _snr_db(signal_rms: float, noise_variance: float) -> float:
    if noise_variance <= 0:
        return float("inf")
    return 20.0 * math.log10(signal_rms / math.sqrt(noise_variance))


def compute_noise_budget(
    graph: AnalogGraph,
    temperature_K: float = 300.0,
    signal_rms: float = 1.0,
    target_snr_db: float = 40.0,
    calibration_tolerance: float = 0.10,  # 10% relative error acceptable for stochastic nodes
    capacitance_F: float = HW_CAPACITANCE_F,
    bias_current_A: float = HW_BIAS_CURRENT_A,
    bandwidth_hz: float = HW_BANDWIDTH_HZ,
    v_ref: float = HW_V_REF,
) -> dict[str, NoiseBudget]:
    """Compute thermal noise power at each analog node using physics-based models.

    Node-type noise models:
      MVM (crossbar):
        - Conductance variation: modeled via existing NoiseSpec if present
        - Readout amplifier: kT/C thermal noise at sense node
        - Total: max(NoiseSpec.sigma², kT/C)

      INTEGRATION / DECAY (RC circuits, op-amp integrators):
        - Dominant: kT/C thermal noise on integrating capacitor
        - σ² = kT/C

      NOISE_INJECTION / SAMPLE (stochastic nodes):
        - Shot noise: 2qI·BW
        - Modeled as SDE diffusion term g·dW

      Converters (ADC/DAC at D/A boundaries):
        - Quantization: σ² = V_LSB² / 12
        - SNR formula: SNR = 6.02·ENOB + 1.76 dB

    Noise propagation:
        output_σ² = Σ_inputs (gain_i² · input_σ²_i) + local_σ²
        For MVM: gain ≈ ||W||_F / sqrt(input_dim) (Frobenius norm)

    Args:
        graph: AnalogGraph from any extractor.
        temperature_K: Operating temperature (default 300K room temp).
        signal_rms: Nominal signal amplitude (normalization reference).
        target_snr_db: SNR target for flagging violations (40dB ≈ 6.5 ENOB).
        capacitance_F: On-chip capacitor size for kT/C calculation.
        bias_current_A: Bias current for shot noise calculation.
        bandwidth_hz: Analog circuit bandwidth limit.
        v_ref: ADC/DAC reference voltage.

    Returns:
        dict mapping node_name → NoiseBudget
    """
    budgets: dict[str, NoiseBudget] = {}
    thermal_var = _thermal_noise_variance(temperature_K, capacitance_F)
    shot_var = _shot_noise_variance(bias_current_A, bandwidth_hz)

    for node in graph.nodes:
        if node.domain == Domain.DIGITAL:
            continue

        budget = NoiseBudget(
            node_name=node.name,
            op_type=node.op_type.name,
            domain=node.domain.name,
            signal_rms=signal_rms,
            temperature_K=temperature_K,
            capacitance_F=capacitance_F,
            noise_bandwidth_hz=bandwidth_hz,
        )

        if node.op_type == OpType.MVM:
            # Crossbar: kT/C from readout amplifier + any existing NoiseSpec
            spec_var = (node.noise.sigma ** 2) if node.noise and node.noise.sigma > 0 else 0.0
            budget.thermal_noise_variance = thermal_var
            budget.total_noise_variance = max(spec_var, thermal_var)

        elif node.op_type in (OpType.INTEGRATION, OpType.DECAY):
            # RC integrator / op-amp: kT/C is the fundamental floor
            budget.thermal_noise_variance = thermal_var
            budget.total_noise_variance = thermal_var
            # SDE diffusion coefficient: g = sqrt(2·kT/C / τ)
            tc = node.metadata.get("time_constant", 1e-3)
            budget.sde_diffusion_coeff = math.sqrt(2.0 * thermal_var / max(tc, 1e-9))

        elif node.op_type in (OpType.NOISE_INJECTION, OpType.SAMPLE):
            # Stochastic-native nodes: noise is the intended signal, not a nonideality.
            # SNR is therefore undefined (high SNR = node never fires = broken device).
            # Correct metric: calibration error |actual_σ - target_σ| / target_σ.
            #
            # SAMPLE (p-bit/sMTJ): switching σ must equal sqrt(T) at operating point.
            #   target_sigma set by EBMExtractor._build_rbm_graph() from cfg.temperature.
            # NOISE_INJECTION (diffusion): injected σ must equal sqrt(β_t) per schedule.
            #   target_sigma set by StableDiffusionExtractor.build_graph() from betas.
            target_sigma = node.metadata.get("target_sigma")
            actual_sigma = (node.noise.sigma if (node.noise and node.noise.sigma > 0)
                            else math.sqrt(shot_var))
            if target_sigma and target_sigma > 0:
                budget.calibration_error = abs(actual_sigma - target_sigma) / target_sigma
                budget.meets_calibration = budget.calibration_error < calibration_tolerance
            budget.sde_diffusion_coeff = actual_sigma
            budget.total_noise_variance = actual_sigma ** 2

        elif node.op_type == OpType.GIBBS_STEP:
            # DTCA subthreshold CMOS RNG (Jelinčič et al. 2025, §II.B, Fig. 3).
            # Thermal noise IS the computation — not a nonideality to be minimized.
            # Appendix E: E_rng ≈ 350 aJ/bit; Appendix K: τ_rng ≈ 100 ns.
            # SDE diffusion coefficient scales with kT and τ_rng.
            tau_rng_s = node.metadata.get("tau_rng_s", 100e-9)
            budget.sde_diffusion_coeff = math.sqrt(K_B * temperature_K * tau_rng_s)
            budget.total_noise_variance = 0.0  # SNR undefined for thermodynamic sampler

        elif node.op_type in (OpType.ACCUMULATION, OpType.SKIP_CONNECTION):
            # Current summation: shot noise from each branch
            budget.shot_noise_variance = shot_var
            budget.total_noise_variance = shot_var

        elif node.op_type in (OpType.GAIN, OpType.ELEMENTWISE_MUL):
            # Gain stage: kT/C of output node
            budget.thermal_noise_variance = thermal_var
            budget.total_noise_variance = thermal_var

        else:
            # Hybrid nodes: conservative kT/C estimate
            budget.thermal_noise_variance = thermal_var
            budget.total_noise_variance = thermal_var

        # SNR is undefined for stochastic-native nodes (noise IS the computation/signal).
        # GIBBS_STEP: thermal noise drives Gibbs — calibration-free by circuit physics.
        # SAMPLE / NOISE_INJECTION: intentional noise injection — use calibration_error.
        if node.op_type in (OpType.GIBBS_STEP, OpType.NOISE_INJECTION, OpType.SAMPLE):
            budget.snr_db = float("inf")
            budget.meets_target_snr = True
        else:
            budget.snr_db = _snr_db(signal_rms, budget.total_noise_variance)
            budget.meets_target_snr = budget.snr_db >= target_snr_db
        budgets[node.name] = budget

    return budgets


_SNR_EXEMPT = frozenset({"GIBBS_STEP", "NOISE_INJECTION", "SAMPLE"})


def noise_budget_summary(budgets: dict[str, NoiseBudget], target_snr_db: float = 40.0) -> str:
    """Generate a formatted noise budget report."""
    lines = [
        "THERMAL NOISE BUDGET",
        "=" * 72,
        f"Target SNR: {target_snr_db:.0f} dB  |  Calibration tolerance: 10%",
        f"kT/C (1pF, 300K): {math.sqrt(_thermal_noise_variance()):.2e} V_rms",
        "",
        f"{'Node':<35} {'Type':<16} {'SNR/Cal':>9} {'OK?':>5} {'SDE g':>10}",
        "─" * 79,
    ]
    snr_violations = 0
    cal_violations = 0
    n_snr = 0
    n_cal = 0
    for b in budgets.values():
        sde = f"{b.sde_diffusion_coeff:.2e}" if b.sde_diffusion_coeff > 0 else "—"
        if b.op_type == "GIBBS_STEP":
            metric_str = "thermo"
            ok = "~"   # N/A — calibration-free by physics
        elif b.op_type in ("NOISE_INJECTION", "SAMPLE"):
            n_cal += 1
            if not math.isnan(b.calibration_error):
                metric_str = f"{b.calibration_error * 100:.1f}% err"
                ok = "✓" if b.meets_calibration else "✗"
                if not b.meets_calibration:
                    cal_violations += 1
            else:
                metric_str = "no target"
                ok = "?"
        else:
            n_snr += 1
            metric_str = f"{b.snr_db:.1f}" if b.snr_db != float("inf") else "∞"
            ok = "✓" if b.meets_target_snr else "✗"
            if not b.meets_target_snr:
                snr_violations += 1
        lines.append(f"{b.node_name:<35} {b.op_type:<16} {metric_str:>9} {ok:>5} {sde:>10}")
    lines += [
        "",
        f"SNR violations:         {snr_violations} / {n_snr} nodes",
        f"Calibration violations: {cal_violations} / {n_cal} stochastic nodes",
    ]
    return "\n".join(lines)
