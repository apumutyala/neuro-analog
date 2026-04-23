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
    AnalogNode, AnalogAmenabilityProfile, DynamicsProfile, OpType, Domain,
    HardwareEstimate, CrossbarSpec, IntegratorSpec, ConverterSpec,
)


@dataclass
class HardwareProfile:
    """Hardware specification for energy/latency estimation.
    
    All energy values in picojoules (pJ), latency in nanoseconds (ns).
    Throughput in operations per second.
    
    Sources:
    - Crossbar: IBM PCM arrays ~5 pJ/MAC (Nature Comp. Sci. 2024)
    - ADC: 0.5-1 pJ/conversion at 1 MSPS (AIMC surveys)
    - Digital MAC: 100-500 pJ/MAC (GPU/SRAM-IMC baselines)
    """
    # Crossbar (analog MAC)
    analog_mac_energy_pJ: float = 5.0  # Energy per MAC operation
    analog_mac_throughput: float = 1e12  # MAC/s (1 TOPS)
    
    # Integrator (ODE/SDE state update)
    integrator_energy_pJ_per_state: float = 0.5  # Energy per state variable
    integrator_time_constant_s: float = 1e-3  # RC time constant
    
    # ADC/DAC (domain conversion)
    adc_energy_pJ: float = 0.8  # Energy per ADC conversion
    adc_latency_ns: float = 1000.0  # Latency per conversion
    dac_energy_pJ: float = 0.8  # Energy per DAC conversion
    dac_latency_ns: float = 1000.0  # Latency per conversion
    
    # Digital baseline (GPU/SRAM-IMC)
    digital_mac_energy_pJ: float = 100.0  # Energy per digital MAC
    digital_mac_throughput: float = 1e11  # MAC/s (100 GOPS)
    digital_memory_access_pJ: float = 10.0  # Energy per memory access
    
    # Thermodynamic sampling (EBM, DTM)
    thermodynamic_sample_energy_pJ: float = 0.35  # Energy per sampled bit (350 aJ)
    rng_latency_ns: float = 100.0  # RNG flip time
    
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
            latency_ns = mac_count / profile.analog_mac_throughput * 1e9  # Convert to ns
            
        elif node.op_type in (OpType.INTEGRATION, OpType.DECAY):
            # RC integrator / decay circuit
            state_dim = node.output_shape[0] if node.output_shape else node.flops
            energy_pJ = state_dim * profile.integrator_energy_pJ_per_state
            # Latency dominated by RC time constant
            latency_ns = profile.integrator_time_constant_s * 1e9 * 5  # 5*tau for settling
            
        elif node.op_type == OpType.ACCUMULATION:
            # Kirchhoff current addition (negligible energy)
            energy_pJ = node.flops * 0.1  # Very low energy
            latency_ns = node.flops / profile.analog_mac_throughput * 1e9
            
        elif node.op_type == OpType.ELEMENTWISE_MUL:
            # Gilbert cell / analog multiplier
            energy_pJ = node.flops * profile.analog_mac_energy_pJ
            latency_ns = node.flops / profile.analog_mac_throughput * 1e9
            
        elif node.op_type in (OpType.ANALOG_SIGMOID, OpType.ANALOG_EXP, OpType.ANALOG_RELU):
            # Subthreshold MOSFET differential pair / current mirror
            energy_pJ = node.flops * 0.5  # Low energy analog activation
            latency_ns = node.flops / profile.analog_mac_throughput * 1e9
            
        elif node.op_type == OpType.NOISE_INJECTION:
            # Thermal/shot noise source (TRNG)
            energy_pJ = node.flops * profile.thermodynamic_sample_energy_pJ
            latency_ns = node.flops * profile.rng_latency_ns
            
        elif node.op_type in (OpType.SAMPLE, OpType.GIBBS_STEP):
            # p-bit / sMTJ / subthreshold CMOS RNG
            energy_pJ = node.flops * profile.thermodynamic_sample_energy_pJ
            latency_ns = node.flops * profile.rng_latency_ns
            
        elif node.op_type == OpType.SKIP_CONNECTION:
            # Current summation (negligible)
            energy_pJ = node.flops * 0.1
            latency_ns = node.flops / profile.analog_mac_throughput * 1e9
            
        elif node.op_type == OpType.GAIN:
            # Programmable gain amplifier
            energy_pJ = node.flops * 0.5
            latency_ns = node.flops / profile.analog_mac_throughput * 1e9
            
        else:
            # Fallback for other analog ops
            energy_pJ = node.flops * profile.analog_mac_energy_pJ
            latency_ns = node.flops / profile.analog_mac_throughput * 1e9
    else:
        # Digital-required operations
        if node.op_type in (OpType.SOFTMAX, OpType.LAYER_NORM, OpType.GROUP_NORM,
                           OpType.RMS_NORM, OpType.BATCH_NORM):
            # Precision-critical operations
            energy_pJ = node.flops * profile.digital_mac_energy_pJ
            latency_ns = node.flops / profile.digital_mac_throughput * 1e9
            
        elif node.op_type in (OpType.SOFTPLUS, OpType.SILU, OpType.GELU, OpType.ADALN):
            # Activation functions
            energy_pJ = node.flops * profile.digital_mac_energy_pJ
            latency_ns = node.flops / profile.digital_mac_throughput * 1e9
            
        elif node.op_type == OpType.DYNAMIC_MATMUL:
            # Data-dependent matrix multiply (Q.K^T, attn.V)
            mac_count = node.flops // 2
            energy_pJ = mac_count * profile.digital_mac_energy_pJ
            latency_ns = mac_count / profile.digital_mac_throughput * 1e9
            
        elif node.op_type == OpType.EMBEDDING:
            # Embedding lookup (memory-bound)
            energy_pJ = node.param_count * profile.digital_memory_access_pJ
            latency_ns = node.param_count / profile.digital_mac_throughput * 1e9
            
        elif node.op_type == OpType.MAX_POOL:
            # Spatial downsampling
            energy_pJ = node.flops * profile.digital_mac_energy_pJ
            latency_ns = node.flops / profile.digital_mac_throughput * 1e9
            
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
            latency_ns = node.flops / profile.digital_mac_throughput * 1e9
    
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
