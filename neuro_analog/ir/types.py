"""
Core type definitions for the neuro-analog intermediate representation.

Every neural network operation is classified into one of three domains
(ANALOG, DIGITAL, HYBRID) and one of ~20 primitive operation types.
These primitives span the full set of operations found across SSMs,
diffusion models, flow models, EBMs, and transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, Optional


# ──────────────────────────────────────────────────────────────────────
# Domain classification
# ──────────────────────────────────────────────────────────────────────

class Domain(Enum):
    """Whether an operation runs in analog, digital, or could go either way."""
    ANALOG = auto()
    DIGITAL = auto()
    HYBRID = auto()  # Analog-possible with accuracy tradeoff


# ──────────────────────────────────────────────────────────────────────
# Operation type taxonomy
# ──────────────────────────────────────────────────────────────────────

class OpType(Enum):
    """Primitive operation types spanning all five architecture families.
    
    ANALOG_NATIVE: Operations that map directly to analog circuit primitives.
    DIGITAL_REQUIRED: Operations that require digital computation.
    HYBRID: Operations that can be approximated in analog with quality loss.
    """
    
    # ── Analog-native operations ─────────────────────────────────────
    MVM = auto()                # Matrix-vector multiply → crossbar array
    INTEGRATION = auto()        # ODE state update → op-amp integrator
    DECAY = auto()              # Exponential decay → RC circuit
    ACCUMULATION = auto()       # Signal summation → Kirchhoff current addition
    ELEMENTWISE_MUL = auto()    # Element-wise multiply → Gilbert cell / analog multiplier
    ANALOG_SIGMOID = auto()     # Sigmoid → subthreshold MOSFET differential pair
    ANALOG_EXP = auto()         # Exponential → BJT collector current
    ANALOG_RELU = auto()        # ReLU → current mirror
    NOISE_INJECTION = auto()    # Gaussian noise → thermal/shot noise source (TRNG)
    ANALOG_FIR = auto()         # Short convolution → delay line + summing amplifier
    SAMPLE = auto()             # Probabilistic sampling → p-bit / sMTJ
    SKIP_CONNECTION = auto()    # Residual add → current summation
    GAIN = auto()               # Scalar multiply → programmable gain amplifier
    GIBBS_STEP = auto()         # Chromatic Gibbs update → subthreshold CMOS RNG (DTCA §II.B, Fig. 3)
    RESISTOR_DAC = auto()       # Bias loading → DAC-driven resistor network (DTCA Appendix E)
    
    # ── Digital-required operations ──────────────────────────────────
    SOFTMAX = auto()            # exp + sum + division (precision-critical)
    LAYER_NORM = auto()         # mean + variance + sqrt + division
    GROUP_NORM = auto()         # per-group normalization
    RMS_NORM = auto()           # root mean square normalization
    SOFTPLUS = auto()           # log(1 + exp(x))
    SILU = auto()               # x · σ(x) — SiLU/Swish
    GELU = auto()               # x · Φ(x)
    ADALN = auto()              # Adaptive layer norm (scale, shift, gate)
    DYNAMIC_MATMUL = auto()     # Data-dependent matrix multiply (Q·K^T, attn·V)
    RESHAPE = auto()            # Tensor reshape/permute (zero-compute, routing)
    EMBEDDING = auto()          # Token/position embedding lookup
    
    # ── Hybrid operations ────────────────────────────────────────────
    KERNEL_ATTENTION = auto()   # FAVOR+ kernel approximation of softmax attention
    APPROX_NORM = auto()        # Simplified normalization (L1 norm, fixed stats)
    PIECEWISE_SILU = auto()     # Piecewise-linear SiLU approximation
    ANALOG_SOFTMAX = auto()     # Subthreshold exponential + current summing


# ──────────────────────────────────────────────────────────────────────
# Hardware specification dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PrecisionSpec:
    """Precision requirements for an operation, extracted from pretrained weights."""
    weight_bits: int = 8           # Minimum weight precision for <1% quality loss
    activation_bits: int = 8       # Minimum activation precision
    accumulator_bits: int = 16     # Accumulator precision (weight_bits + activation_bits + log2(fan_in))
    
    # Extracted statistics
    weight_min: float = 0.0
    weight_max: float = 0.0
    weight_std: float = 0.0
    activation_min: float = 0.0
    activation_max: float = 0.0
    activation_std: float = 0.0
    
    @property
    def weight_dynamic_range_db(self) -> float:
        """Weight dynamic range in dB: 20·log10(|max/min|)."""
        if self.weight_min == 0:
            return float('inf')
        return 20 * __import__('math').log10(abs(self.weight_max / self.weight_min))

    @property
    def activation_dynamic_range_db(self) -> float:
        """Activation crest-factor in dB: 20·log10(activation_max / activation_std).

        This is the quantity that determines ADC bit requirements — it measures how
        many dB of headroom the ADC must cover above the typical signal level.
        6 dB ≈ 1 bit.  A value of ~20 dB (crest ≈ 10) implies an 8-bit ADC is
        adequate, matching the empirical analog crossbar standard.
        """
        if self.activation_std <= 0 or self.activation_max <= 0:
            return 0.0
        return 20 * __import__('math').log10(self.activation_max / self.activation_std)


@dataclass
class HardwareEstimate:
    """Power, area, and latency estimates for one operation."""
    power_mW: float = 0.0
    area_mm2: float = 0.0
    latency_ns: float = 0.0
    energy_pJ: float = 0.0  # power × latency


@dataclass
class CrossbarSpec:
    """Specification for a crossbar array implementing an MVM."""
    rows: int = 256
    cols: int = 256
    precision_bits: int = 8
    technology: str = "RRAM"  # RRAM, PCM, SRAM, NOR_FLASH, CAPACITIVE
    
    # Derived estimates (technology-dependent)
    power_per_mac_pJ: float = 5.0
    area_per_cell_um2: float = 0.01
    
    @property
    def total_area_mm2(self) -> float:
        return self.rows * self.cols * self.area_per_cell_um2 * 1e-6
    
    @property
    def total_macs(self) -> int:
        return self.rows * self.cols


@dataclass 
class IntegratorSpec:
    """Specification for an op-amp integrator implementing ODE integration."""
    time_constant_s: float = 1e-3   # RC time constant
    bandwidth_hz: float = 250e3     # 3dB bandwidth
    precision_bits: int = 8         # Effective precision (limited by offset, drift)
    drift_rate_mV_per_ms: float = 0.1  # DC drift rate


@dataclass
class ConverterSpec:
    """ADC or DAC specification for a domain boundary."""
    direction: str = "ADC"          # "ADC" or "DAC"
    resolution_bits: int = 8
    sample_rate_MSPS: float = 1.0
    power_mW: float = 0.5
    area_mm2: float = 0.001
    
    @property
    def energy_per_conversion_pJ(self) -> float:
        return self.power_mW * 1e9 / (self.sample_rate_MSPS * 1e6)


@dataclass
class NoiseSpec:
    """Hardware noise specification for an analog operation limit."""
    kind: Literal["none", "thermal", "shot", "adc", "composite"]
    sigma: float                       # noise std-dev in equivalent input units
    mean: float = 0.0                  # bias / systematic offset
    corr_length: Optional[float] = None   # temporal correlation length (s)
    corr_spatial: Optional[float] = None  # spatial correlation across rows/cols
    bandwidth_hz: Optional[float] = None  # effective noise bandwidth


# ──────────────────────────────────────────────────────────────────────
# Architecture-level profiles
# ──────────────────────────────────────────────────────────────────────

class ArchitectureFamily(Enum):
    """Top-level architecture classification."""
    SSM = "ssm"
    DIFFUSION = "diffusion"
    FLOW = "flow"
    EBM = "ebm"
    TRANSFORMER = "transformer"
    NEURAL_ODE = "neural_ode"  # Chen et al. 2018; IS an ODE, strongest Ark fit
    DEQ = "deq"               # Bai et al. 2019; implicit equilibrium = feedback analog circuit
    DTM = "dtm"               # Denoising Thermodynamic Model (Jelinčič et al. 2025, Extropic Corp)


@dataclass
class DynamicsProfile:
    """Continuous-time dynamics characterization (for ODE/SDE architectures)."""
    has_dynamics: bool = False
    dynamics_type: str = ""  # "LTI_ODE", "time_varying_ODE", "SDE", "energy_minimization"
    
    # SSM-specific
    time_constants: Optional[list[float]] = None      # 1/|a_i| for diagonal A
    time_constant_spread: Optional[float] = None      # max/min ratio
    state_dimension: Optional[int] = None
    
    # Diffusion-specific
    beta_schedule: Optional[list[float]] = None        # β(t) values
    beta_dynamic_range: Optional[float] = None         # max(β)/min(β)
    num_diffusion_steps: Optional[int] = None
    is_stochastic: bool = False
    
    # Flow-specific
    lipschitz_constant: Optional[float] = None         # Velocity field Lipschitz bound
    flow_straightness: Optional[float] = None          # 0=curved, 1=perfectly straight
    num_function_evaluations: Optional[int] = None

    # Stiffness
    stiffness_ratio: Optional[float] = None            # Condition number of dynamics

    # DTM-specific (Jelinčič et al. 2025, Extropic Corp — Appendix D, E, K)
    grid_side_L: Optional[int] = None                  # DTCA grid side length L (paper default: 70)
    connectivity_degree: Optional[int] = None          # G_k neighbor count k ∈ {8,12,16,20,24} (Table II)
    gibbs_steps_K: Optional[int] = None                # Gibbs sweeps per denoising step K (Appendix E)
    denoising_steps_T: Optional[int] = None            # DTM denoising chain length T (Section III)
    tau_rng_ns: Optional[float] = None                 # RNG flip time in ns (Appendix K: ~100 ns)
    energy_per_sample_J: Optional[float] = None        # Energy per sampled bit (Appendix E: ~350 aJ)


@dataclass
class AnalogAmenabilityProfile:
    """Complete analog amenability assessment for one architecture instance."""
    architecture: ArchitectureFamily
    model_name: str
    model_params: int  # Total parameter count
    
    # Core metrics
    analog_flop_fraction: float = 0.0      # % of FLOPs in analog domain
    digital_flop_fraction: float = 0.0     # % of FLOPs in digital domain
    hybrid_flop_fraction: float = 0.0      # % of FLOPs in hybrid domain
    
    da_boundary_count: int = 0             # Mandatory D/A transitions per inference
    da_boundary_count_per_step: int = 0    # Per ODE/SDE step (for iterative models)
    
    # Hardware requirements
    crossbar_tiles_needed: int = 0
    total_analog_area_mm2: float = 0.0
    total_digital_area_mm2: float = 0.0
    analog_power_mW: float = 0.0
    digital_power_mW: float = 0.0
    converter_power_mW: float = 0.0
    
    # Dynamics characterization
    dynamics: DynamicsProfile = field(default_factory=DynamicsProfile)
    
    # Precision
    min_weight_precision_bits: int = 8
    min_activation_precision_bits: int = 8
    
    # Composite scores (0-1, higher = more analog-friendly)
    dynamics_score: float = 0.0     # Natural fit to ODE/SDE circuits
    precision_score: float = 0.0    # Tolerance to low precision
    boundary_score: float = 0.0     # Few D/A conversions needed
    noise_score: float = 0.0        # Fraction of analog FLOPs meeting SNR targets
    overall_score: float = 0.0      # Weighted composite
    
    def compute_scores(self, weights: dict[str, float] | None = None):
        """Compute composite scores from raw metrics."""
        if weights is None:
            weights = {"dynamics": 0.25, "precision": 0.25, "boundary": 0.15, "noise": 0.20, "analog_frac": 0.15}
        
        # Noise score: stochastic architectures requiring precise TRNG calibration score lower.
        # Exception: DTCA thermodynamic_gibbs — thermal noise IS the computational resource
        # (Jelinčič et al. 2025, Section I: "thermal fluctuations as a computational resource").
        # For DTCA, uncalibrated thermal noise is not a nonideality — it is the desired randomness.
        if self.dynamics.dynamics_type == "thermodynamic_gibbs":
            self.noise_score = 1.0
        elif self.dynamics.is_stochastic:
            self.noise_score = 0.7
        else:
            self.noise_score = 1.0

        # Dynamics score: 1.0 for native ODE/SDE, 0.0 for no dynamics
        if self.dynamics.has_dynamics:
            self.dynamics_score = 0.8
            if self.dynamics.dynamics_type in ("LTI_ODE", "continuous_ODE", "time_varying_ODE"):
                # All continuous ODE types map naturally to RC integrators.
                # Neural ODE (time_varying_ODE) IS Ark's native input format → top score.
                self.dynamics_score = 1.0
            elif self.dynamics.dynamics_type == "energy_minimization":
                self.dynamics_score = 0.95
            elif self.dynamics.dynamics_type == "SDE":
                self.dynamics_score = 0.7
            elif self.dynamics.dynamics_type == "implicit_equilibrium":
                # DEQ: feedback circuit settles to z* physically.
                # Natural analog paradigm but convergence is fragile under mismatch
                # (spectral radius may exceed 1). Score between energy_min and SDE.
                self.dynamics_score = 0.85
        
        # Precision score: joint weight + activation demand (0 = needs high precision, 1 = tolerates 4-bit)
        # Weight score: how well the weight distribution fits low-precision crossbar storage.
        # Activation score: how well the activation dynamic range fits the ADC range
        #   (determined by crest factor; low crest = fewer ADC bits needed = better).
        # Weights 60/40: weight precision is the dominant cost at the crossbar; activation
        # precision drives the ADC, which is shared across many layers.
        # When activation calibration has not been run, min_activation_precision_bits stays
        # at the default of 8 — which maps to act_score=0.67, a neutral mid-range value
        # that neither rewards nor heavily penalises uncalibrated profiles.
        weight_score = max(0.0, 1.0 - (self.min_weight_precision_bits - 4) / 12)
        act_score = max(0.0, 1.0 - (self.min_activation_precision_bits - 4) / 12)
        self.precision_score = 0.6 * weight_score + 0.4 * act_score
        
        # Boundary score: fewer boundaries = better
        self.boundary_score = max(0, 1.0 - self.da_boundary_count / 100)
        
        # Overall
        self.overall_score = (
            weights["dynamics"] * self.dynamics_score
            + weights["precision"] * self.precision_score
            + weights["boundary"] * self.boundary_score
            + weights.get("noise", 0.0) * self.noise_score
            + weights["analog_frac"] * self.analog_flop_fraction
        )
