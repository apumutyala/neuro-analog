"""
Energy and latency modeling for analog vs digital compute.

This module provides:
- HardwareProfile: Configurable hardware specification for energy/latency estimation
- estimate_node_cost(): OpType-specific energy/latency mapping
- compute_amenability_score(): Empirical amenability scoring based on sweep results
- dynamics_penalty(): Dynamics-aware penalty for iterative architectures

Constants are sourced from:
- IBM PCM array modeling (Nature Computational Science 2024)
- SRAM IMC benchmarking papers
- High-precision AIMC surveys
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

from .types import (
    AnalogAmenabilityProfile, DynamicsProfile, OpType, Domain,
    HardwareEstimate, CrossbarSpec, IntegratorSpec, ConverterSpec,
)
from .node import AnalogNode


@dataclass
class HardwareProfile:
    """Hardware specification for energy/latency estimation.

    All energy values in picojoules (pJ), latency in nanoseconds (ns).
    Throughput in operations per second.

    Published silicon data sources:
    - IBM HERMES PCM (Nature Electronics 2023, arXiv:2212.02872):
      * 8-bit 1-phase (LP): 9.76 TOPS/W  -> 0.102 pJ/MAC (end-to-end)
      * 8-bit 4-phase (HP): 2.48 TOPS/W -> 0.403 pJ/MAC (end-to-end)
      * HERMES Core JSSC 2022: 10.5 TOPS/W -> 0.095 pJ/MAC
      * MVM latency: 130 ns (O(1) parallel operation)
      * Crossbar size: 256×256 typical
    - imec/KU Leuven SRAM AIMC (ESSCIRC 2023, arXiv:2305.18335):
      * 16nm DIMC 8b: 23.8 TOPS/W -> 0.042 pJ/MAC
      * SRAM AIMC core (LP): ~1000 TOPS/W -> 0.001 pJ/MAC (core only)
    - HKUST/ACCESS SRAM CIM (VLSI 2023): 137.5 TOPS/W -> 0.007 pJ/MAC
    - Digital baseline: Modern 16nm accelerators achieve ~10 fJ/MAC (0.01 pJ/MAC)
      Legacy GPU baseline: 100 pJ/MAC (conservative)
    - ADC energy scaling: Exponential with resolution ~100 fJ for 5-bit, 2^(bits-5)
    - DAC energy: ~15 fJ per conversion
    """
    # Crossbar (analog MAC) - Updated to research-validated defaults
    analog_mac_energy_pJ: float = 0.10  # Default: IBM HERMES end-to-end 8-bit PCM
    analog_mac_throughput: float = 1e12  # MAC/s (1 TOPS) - legacy, prefer crossbar_read_latency_ns

    # Integrator (ODE/SDE state update)
    integrator_energy_pJ_per_state: float = 0.5  # Energy per state variable
    integrator_time_constant_s: float = 1e-3  # RC time constant

    # ADC/DAC (domain conversion)
    # When using end-to-end measured presets (e.g., ibm_hermes_*), these are
    # zeroed by calibrate_from_reference(is_end_to_end=True) to avoid double counting.
    # Research-validated values: ADC ~100 fJ for 5-bit with exponential scaling 2^(bits-5)
    adc_energy_pJ: float = 0.10  # Per-element ADC conversion [pJ] - 100 fJ for 5-bit baseline
    adc_latency_ns: float = 250.0  # IBM HERMES CCO-based ADC ~250 ns (300 ps/LSB @ 8b)
    adc_resolution_bits: int = 8  # ADC resolution, affects energy (exponential scaling)
    dac_energy_pJ: float = 0.015  # Per-element DAC conversion [pJ] - 15 fJ from research
    dac_latency_ns: float = 100.0  # Latency per conversion

    # Digital baseline - Updated to match modern 16nm accelerators
    # Digital baseline — system-level comparison for analog/digital energy/latency
    # Default 10.0 pJ/MAC matches modern edge NPU with on-chip SRAM (system-level).
    # This is NOT a core-only digital MAC figure; core-only 16nm digital MAC is ~0.01 pJ,
    # but system-level comparison must include memory access, control, and data movement.
    # References:
    #   - IBM HERMES PCM: 0.10 pJ/MAC end-to-end (Nature Electronics 2023)
    #   - Edge NPU 8-bit: ~10 pJ/MAC system-level (SRAM + MAC + control)
    #   - GPU+HBM: 100-1000 pJ/MAC system-level
    digital_mac_energy_pJ: float = 10.0  # System-level digital MAC energy [pJ]
    digital_mac_throughput: float = 1e11  # MAC/s (100 GOPS) - legacy
    digital_mac_latency_ns: float = 1.0  # 1 ns per MAC (modern accelerators)
    digital_memory_access_pJ: float = 10.0  # Energy per memory access

    # Thermodynamic sampling (EBM, DTM)
    thermodynamic_sample_energy_pJ: float = 0.35  # Energy per sampled bit (350 aJ)
    rng_latency_ns: float = 100.0  # RNG flip time

    # Sequence-aware and physical latency parameters
    # Research-validated: MVM is O(1) parallel operation, typical 130ns from IBM HERMES
    crossbar_read_latency_ns: float = 130.0  # Time per MVM operation (O(1) parallel)
    use_tops_latency: bool = False  # If True, revert to MAC/throughput legacy latency model
    integrator_settling_time_constants: float = 5.0  # Number of time constants for SSM integrator to settle
    num_parallel_converters: int = 0  # 0 means fully parallel across hidden dim, >0 models limited ADC/DAC parallelization

    # ── Calibration API ────────────────────────────────────────────────────

    _PRESETS: dict[str, dict] = field(default_factory=lambda: {
        "ibm_hermes_pcm_lp": {
            "analog_mac_energy_pJ": 1.0 / 9.76,
            "adc_energy_pJ": 0.0,
            "dac_energy_pJ": 0.0,
            "description": "IBM HERMES 64-core PCM, 8-bit 1-phase (low-precision), end-to-end",
        },
        "ibm_hermes_pcm_hp": {
            "analog_mac_energy_pJ": 1.0 / 2.48,
            "adc_energy_pJ": 0.0,
            "dac_energy_pJ": 0.0,
            "description": "IBM HERMES 64-core PCM, 8-bit 4-phase (high-precision), end-to-end",
        },
        "ibm_hermes_core": {
            "analog_mac_energy_pJ": 1.0 / 10.5,
            "adc_energy_pJ": 0.0,
            "dac_energy_pJ": 0.0,
            "description": "IBM HERMES Core 256x256 PCM, JSSC 2022, end-to-end",
        },
        "imec_dimc_8b": {
            "analog_mac_energy_pJ": 1.0 / 23.8,
            "adc_energy_pJ": 0.0,
            "dac_energy_pJ": 0.0,
            "description": "imec 16nm digital IMC (DIMC), 8-bit MAC, ESSCIRC 2023",
        },
        "hkust_sram_cim": {
            "analog_mac_energy_pJ": 1.0 / 137.5,
            "adc_energy_pJ": 0.0,
            "dac_energy_pJ": 0.0,
            "description": "HKUST/ACCESS SRAM CIM, 137.5 TOPS/W, VLSI 2023",
        },
        "digital_gpu_fp16": {
            "digital_mac_energy_pJ": 100.0,
            "description": "Conservative GPU FP16 baseline (100 pJ/MAC)",
        },
        "digital_edge_8b": {
            "digital_mac_energy_pJ": 10.0,
            "description": "Edge NPU 8-bit baseline (10 pJ/MAC)",
        },
    }, repr=False)

    @classmethod
    def from_preset(cls, preset_name: str) -> "HardwareProfile":
        """Load a HardwareProfile from a named preset of published silicon data.

        Args:
            preset_name: One of the keys in _PRESETS (e.g., 'ibm_hermes_pcm_lp').

        Returns:
            HardwareProfile with calibrated constants.
        """
        inst = cls()
        inst.load_preset(preset_name)
        return inst

    def load_preset(self, preset_name: str) -> None:
        """Apply a named preset in-place."""
        presets = self._PRESETS
        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}"
            )
        data = presets[preset_name].copy()
        data.pop("description", None)
        for k, v in data.items():
            setattr(self, k, v)

    def calibrate_from_reference(
        self,
        tops_per_w: float | None = None,
        energy_pj_per_mac: float | None = None,
        is_end_to_end: bool = True,
    ) -> None:
        """Set analog_mac_energy_pJ from a published TOPS/W or direct pJ/MAC figure.

        Derivation: 1 TOPS = 1e12 ops/s. 1 W = 1 J/s.
        Therefore 1 TOPS/W = 1e12 ops/J = 1 op / 1e-12 J = 1 pJ/op.
        So energy_per_MAC [pJ] = 1 / TOPS_per_W.

        Args:
            tops_per_w: Published throughput per Watt (TOPS/W).
            energy_pj_per_mac: Direct energy per MAC in picojoules.
            is_end_to_end: If True, zero out adc_energy_pJ and dac_energy_pJ
                to avoid double-counting converter energy already included in
                the published figure.
        """
        if energy_pj_per_mac is not None:
            self.analog_mac_energy_pJ = float(energy_pj_per_mac)
        elif tops_per_w is not None:
            if tops_per_w <= 0:
                raise ValueError("tops_per_w must be positive")
            self.analog_mac_energy_pJ = 1.0 / float(tops_per_w)
        else:
            raise ValueError("Provide either tops_per_w or energy_pj_per_mac")

        if is_end_to_end:
            self.adc_energy_pJ = 0.0
            self.dac_energy_pJ = 0.0

    @classmethod
    def from_config(cls, config_path: str | Path) -> "HardwareProfile":
        """Load HardwareProfile from YAML config file.
        
        Args:
            config_path: Path to YAML config file.
            
        Returns:
            HardwareProfile with values from config (defaults used for missing fields).
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return cls(**config)
    
    def to_config(self, config_path: str | Path) -> None:
        """Save HardwareProfile to YAML config file.
        
        Args:
            config_path: Path to save YAML config file.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            yaml.dump({
                "analog_mac_energy_pJ": self.analog_mac_energy_pJ,
                "analog_mac_throughput": self.analog_mac_throughput,
                "integrator_energy_pJ_per_state": self.integrator_energy_pJ_per_state,
                "integrator_time_constant_s": self.integrator_time_constant_s,
                "adc_energy_pJ": self.adc_energy_pJ,
                "adc_latency_ns": self.adc_latency_ns,
                "dac_energy_pJ": self.dac_energy_pJ,
                "dac_latency_ns": self.dac_latency_ns,
                "digital_mac_energy_pJ": self.digital_mac_energy_pJ,
                "digital_mac_throughput": self.digital_mac_throughput,
                "digital_memory_access_pJ": self.digital_memory_access_pJ,
                "thermodynamic_sample_energy_pJ": self.thermodynamic_sample_energy_pJ,
                "rng_latency_ns": self.rng_latency_ns,
                "crossbar_read_latency_ns": self.crossbar_read_latency_ns,
                "use_tops_latency": self.use_tops_latency,
                "integrator_settling_time_constants": self.integrator_settling_time_constants,
                "num_parallel_converters": self.num_parallel_converters,
            }, f, default_flow_style=False)


def estimate_node_cost(node: AnalogNode, profile: HardwareProfile) -> HardwareEstimate:
    """Estimate energy and latency for a single AnalogNode.
    
    Args:
        node: AnalogNode to estimate.
        profile: HardwareProfile with energy/latency constants.
        
    Returns:
        HardwareEstimate with power, area, latency, energy.
    """
    # Determine if node runs in analog or digital domain
    is_analog = node.domain == Domain.ANALOG or node.domain == Domain.HYBRID
    
    energy_pJ = 0.0
    latency_ns = 0.0
    
    if is_analog:
        # Analog-native operations
        if node.op_type == OpType.MVM:
            # Crossbar array MAC operations
            mac_count = node.flops // 2  # FLOPs = 2 * MACs (multiply + accumulate)
            energy_pJ = mac_count * profile.analog_mac_energy_pJ
            if profile.use_tops_latency:
                latency_ns = mac_count / profile.analog_mac_throughput * 1e9  # Convert to ns
            else:
                latency_ns = node.seq_len * profile.crossbar_read_latency_ns
            
        elif node.op_type in (OpType.INTEGRATION, OpType.DECAY):
            # RC integrator / decay circuit
            # output_shape is per-token, volume is dim or dim * state_dim
            state_dim = node.flops // (node.seq_len * 2) if node.flops > 0 else 1
            energy_pJ = state_dim * node.seq_len * profile.integrator_energy_pJ_per_state
            # Latency dominated by RC time constant (per sequence step)
            latency_ns = node.seq_len * profile.integrator_time_constant_s * 1e9 * profile.integrator_settling_time_constants
            
        elif node.op_type == OpType.ACCUMULATION:
            # Kirchhoff current addition (negligible energy)
            energy_pJ = node.flops * 0.1  # Very low energy
            latency_ns = node.seq_len * profile.crossbar_read_latency_ns
            
        elif node.op_type == OpType.ELEMENTWISE_MUL:
            # Gilbert cell / analog multiplier
            energy_pJ = node.flops * profile.analog_mac_energy_pJ
            latency_ns = node.seq_len * profile.crossbar_read_latency_ns
            
        elif node.op_type in (OpType.ANALOG_SIGMOID, OpType.ANALOG_EXP, OpType.ANALOG_RELU):
            # Subthreshold MOSFET differential pair / current mirror
            energy_pJ = node.flops * 0.5  # Low energy analog activation
            latency_ns = node.seq_len * profile.crossbar_read_latency_ns
            
        elif node.op_type == OpType.NOISE_INJECTION:
            # Thermal/shot noise source (TRNG)
            energy_pJ = node.flops * profile.thermodynamic_sample_energy_pJ
            latency_ns = node.flops * profile.rng_latency_ns  # Assuming serial RNG
            
        elif node.op_type in (OpType.SAMPLE, OpType.GIBBS_STEP):
            # p-bit / sMTJ / subthreshold CMOS RNG
            energy_pJ = node.flops * profile.thermodynamic_sample_energy_pJ
            latency_ns = node.flops * profile.rng_latency_ns
            
        elif node.op_type == OpType.SKIP_CONNECTION:
            # Current summation (negligible)
            energy_pJ = node.flops * 0.1
            latency_ns = node.seq_len * profile.crossbar_read_latency_ns
            
        elif node.op_type == OpType.GAIN:
            # Programmable gain amplifier
            energy_pJ = node.flops * 0.5
            latency_ns = node.seq_len * profile.crossbar_read_latency_ns
            
        else:
            # Fallback for other analog ops
            energy_pJ = node.flops * profile.analog_mac_energy_pJ
            latency_ns = node.seq_len * profile.crossbar_read_latency_ns
    else:
        # Digital-required operations
        if node.op_type in (OpType.SOFTMAX, OpType.LAYER_NORM, OpType.GROUP_NORM,
                           OpType.RMS_NORM, OpType.BATCH_NORM):
            # Precision-critical operations
            energy_pJ = node.flops * profile.digital_mac_energy_pJ
            # Use research-validated digital MAC latency (1 ns per operation)
            latency_ns = node.flops * profile.digital_mac_latency_ns
            
        elif node.op_type in (OpType.SOFTPLUS, OpType.SILU, OpType.GELU, OpType.ADALN):
            # Activation functions
            energy_pJ = node.flops * profile.digital_mac_energy_pJ
            latency_ns = node.flops * profile.digital_mac_latency_ns
            
        elif node.op_type == OpType.DYNAMIC_MATMUL:
            # Data-dependent matrix multiply (Q.K^T, attn.V)
            mac_count = node.flops // 2
            energy_pJ = mac_count * profile.digital_mac_energy_pJ
            latency_ns = mac_count * profile.digital_mac_latency_ns
            
        elif node.op_type == OpType.EMBEDDING:
            # Embedding lookup (memory-bound)
            energy_pJ = node.param_count * profile.digital_memory_access_pJ
            latency_ns = node.param_count * profile.digital_mac_latency_ns
            
        elif node.op_type == OpType.MAX_POOL:
            # Spatial downsampling
            energy_pJ = node.flops * profile.digital_mac_energy_pJ
            latency_ns = node.flops * profile.digital_mac_latency_ns
            
        elif node.op_type == OpType.DROPOUT:
            # Zero-compute at inference
            energy_pJ = 0.0
            latency_ns = 0.0
            
        elif node.op_type == OpType.RESHAPE:
            # Zero-compute routing
            energy_pJ = 0.0
            latency_ns = 0.0
            
        else:
            # Fallback for other digital ops
            energy_pJ = node.flops * profile.digital_mac_energy_pJ
            latency_ns = node.flops * profile.digital_mac_latency_ns
    
    # Compute power (energy / latency)
    power_mW = energy_pJ / latency_ns * 1e3 if latency_ns > 0 else 0.0
    
    # Area estimate (rough approximation)
    area_mm2 = 0.0  # Would need detailed tile mapping
    
    return HardwareEstimate(
        power_mW=power_mW,
        area_mm2=area_mm2,
        latency_ns=latency_ns,
        energy_pJ=energy_pJ,
    )


def dynamics_penalty(dynamics: DynamicsProfile) -> float:
    """Compute dynamics penalty for amenability scoring (0-1, higher = worse).
    
    Args:
        dynamics: DynamicsProfile from the architecture.
        
    Returns:
        Penalty in [0, 1], where 0 = single-pass, 1 = highly iterative.
    """
    if not dynamics.has_dynamics:
        return 0.0
    
    penalty = 0.0
    
    # Penalty for fixed-point iteration (DEQ)
    if dynamics.dynamics_type == "implicit_equilibrium":
        # DEQ: iterative convergence compounds error
        penalty += 0.5
        # Additional penalty for high iteration counts (if available)
        # This would need iteration count from DynamicsProfile
    
    # Penalty for diffusion steps
    if dynamics.num_diffusion_steps and dynamics.num_diffusion_steps > 1:
        # More steps = more error accumulation
        penalty += min(dynamics.num_diffusion_steps / 100.0, 0.5)
    
    # Penalty for ODE solver steps
    if dynamics.num_function_evaluations and dynamics.num_function_evaluations > 1:
        penalty += min(dynamics.num_function_evaluations / 50.0, 0.3)
    
    # Penalty for stochastic dynamics (requires precise calibration)
    if dynamics.is_stochastic and dynamics.dynamics_type != "thermodynamic_gibbs":
        penalty += 0.2
    
    # Cap at 1.0
    return min(penalty, 1.0)


def compute_amenability_score(profile: AnalogAmenabilityProfile) -> float:
    """Compute empirical amenability score based on sweep results and IR analysis.
    
    Features (normalized 0-1):
    - f1: analog FLOP fraction (reward more analog compute)
    - f2: D/A boundary density (penalize more boundaries)
    - f3: sigma_10pct normalized (reward higher noise tolerance)
    - f4: dynamics penalty (penalize iterative/multi-step)
    - f5: precision penalty (reward lower precision needs)
    
    Score formula:
        score = 0.3*f1 + 0.3*f3 + 0.1*f5 - 0.2*f2 - 0.1*f4
    
    Args:
        profile: AnalogAmenabilityProfile with IR metrics and sigma_10pct.
        
    Returns:
        Amenity score in [0, 1], higher = more analog-amenable.
    """
    # f1: analog FLOP fraction
    f1 = profile.analog_flop_fraction
    
    # f2: normalized D/A boundaries (per layer)
    layer_count = profile.layer_count if profile.layer_count > 0 else 1
    f2 = min(profile.da_boundary_count / layer_count, 1.0)
    
    # f3: sigma_10pct normalized to max tested (15%)
    f3 = profile.sigma_10pct / 0.15 if profile.sigma_10pct > 0 else 0.0
    
    # f4: dynamics penalty
    f4 = dynamics_penalty(profile.dynamics)
    
    # f5: precision penalty (lower precision needed = better)
    # Normalize from 2-8 bits to 0-1
    f5 = 1.0 - (profile.min_weight_precision_bits - 2) / (8 - 2)
    f5 = max(0.0, min(1.0, f5))
    
    # Weighted score
    score = (
        0.3 * f1      # reward more analog compute
        + 0.3 * f3    # reward higher noise tolerance
        + 0.1 * f5    # reward lower precision needs
        - 0.2 * f2    # penalize more D/A boundaries
        - 0.1 * f4    # penalize iterative/multi-step pipelines
    )
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))
