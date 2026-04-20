"""
AnalogNode: a single operation in the neuro-analog intermediate representation.

Each node represents one primitive operation (MVM, integration, softmax, etc.)
annotated with its domain classification, precision requirements, and hardware
estimates. Nodes are assembled into AnalogGraph objects for full-model analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import uuid

from .types import OpType, Domain, TargetBackend, PrecisionSpec, HardwareEstimate, NoiseSpec


# ──────────────────────────────────────────────────────────────────────
# Domain assignment rules
# ──────────────────────────────────────────────────────────────────────

# Default domain for each operation type
_DEFAULT_DOMAIN: dict[OpType, Domain] = {
    # Analog-native
    OpType.MVM: Domain.ANALOG,
    OpType.INTEGRATION: Domain.ANALOG,
    OpType.DECAY: Domain.ANALOG,
    OpType.ACCUMULATION: Domain.ANALOG,
    OpType.ELEMENTWISE_MUL: Domain.ANALOG,
    OpType.ANALOG_SIGMOID: Domain.ANALOG,
    OpType.ANALOG_EXP: Domain.ANALOG,
    OpType.ANALOG_RELU: Domain.ANALOG,
    OpType.NOISE_INJECTION: Domain.ANALOG,
    OpType.ANALOG_FIR: Domain.ANALOG,
    OpType.SAMPLE: Domain.ANALOG,
    OpType.SKIP_CONNECTION: Domain.ANALOG,
    OpType.GAIN: Domain.ANALOG,
    
    # Digital-required
    OpType.SOFTMAX: Domain.DIGITAL,
    OpType.LAYER_NORM: Domain.DIGITAL,
    OpType.GROUP_NORM: Domain.DIGITAL,
    OpType.RMS_NORM: Domain.DIGITAL,
    OpType.BATCH_NORM: Domain.DIGITAL,
    OpType.SOFTPLUS: Domain.DIGITAL,
    OpType.SILU: Domain.DIGITAL,
    OpType.GELU: Domain.DIGITAL,
    OpType.ADALN: Domain.DIGITAL,
    OpType.DYNAMIC_MATMUL: Domain.DIGITAL,
    OpType.RESHAPE: Domain.DIGITAL,
    OpType.EMBEDDING: Domain.DIGITAL,
    OpType.MAX_POOL: Domain.DIGITAL,
    OpType.DROPOUT: Domain.DIGITAL,
    
    # Hybrid
    OpType.KERNEL_ATTENTION: Domain.HYBRID,
    OpType.APPROX_NORM: Domain.HYBRID,
    OpType.PIECEWISE_SILU: Domain.HYBRID,
    OpType.ANALOG_SOFTMAX: Domain.HYBRID,
}


def default_domain(op_type: OpType) -> Domain:
    """Get the default domain classification for an operation type."""
    return _DEFAULT_DOMAIN.get(op_type, Domain.DIGITAL)


# ──────────────────────────────────────────────────────────────────────
# AnalogNode
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AnalogNode:
    """A single operation in the neuro-analog IR graph.
    
    Attributes:
        node_id: Unique identifier for this node.
        name: Human-readable name (e.g., "layer_0.ffn.linear1").
        op_type: Primitive operation type from taxonomy.
        domain: ANALOG, DIGITAL, or HYBRID classification.
        input_shape: Shape of input tensor(s).
        output_shape: Shape of output tensor.
        weight_shape: Shape of weight tensor (None for non-parametric ops).
        flops: Estimated floating-point operations.
        precision: Precision requirements extracted from pretrained weights.
        analog_estimate: Hardware estimates for analog implementation.
        digital_estimate: Hardware estimates for digital implementation.
        inputs: List of predecessor node IDs.
        outputs: List of successor node IDs.
        metadata: Architecture-specific extra information.
    """
    
    # Identity
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    op_type: OpType = OpType.MVM
    domain: Domain = Domain.ANALOG
    
    # Tensor shapes
    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()
    weight_shape: Optional[tuple[int, ...]] = None
    
    # Compute cost
    flops: int = 0  # FLOPs per sample (batch_size=1). Multiply by actual batch_size for total inference FLOPs.
    param_count: int = 0  # Number of learnable parameters
    
    # Hardware requirements
    precision: PrecisionSpec = field(default_factory=PrecisionSpec)
    analog_estimate: HardwareEstimate = field(default_factory=HardwareEstimate)
    digital_estimate: HardwareEstimate = field(default_factory=HardwareEstimate)
    
    # Graph connectivity
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    
    # Architecture-specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Noise model
    noise: Optional[NoiseSpec] = None
    
    # Backend targeting (optional; None = default analog-circuit behavior)
    target_backend: Optional[TargetBackend] = None
    
    # Mixed-signal boundary flag (separate from Domain.HYBRID semantic)
    # True if this node is at an analog/digital RTL domain crossing point
    is_mixed_signal_boundary: bool = False
    
    def __post_init__(self):
        """Set default domain based on op_type if not explicitly provided."""
        if self.domain is None:
            self.domain = default_domain(self.op_type)
    
    @property
    def is_analog(self) -> bool:
        return self.domain == Domain.ANALOG
    
    @property
    def is_digital(self) -> bool:
        return self.domain == Domain.DIGITAL
    
    @property
    def is_hybrid(self) -> bool:
        return self.domain == Domain.HYBRID
    
    @property
    def is_parametric(self) -> bool:
        """Whether this operation has learnable weights."""
        return self.weight_shape is not None and self.param_count > 0
    
    @property
    def has_boundary_with(self) -> bool:
        """Whether this node's domain differs from any of its predecessors.
        (Actual boundary detection requires the full graph; this is a placeholder.)
        """
        return False  # Computed at graph level
    
    def summary(self) -> str:
        """One-line summary string."""
        domain_icon = {"ANALOG": "⚡", "DIGITAL": "💻", "HYBRID": "🔄"}
        icon = domain_icon.get(self.domain.name, "?")
        params = f" ({self.param_count:,} params)" if self.param_count > 0 else ""
        return f"{icon} {self.name}: {self.op_type.name} [{self.domain.name}]{params}"


# ──────────────────────────────────────────────────────────────────────
# Factory functions for common operations
# ──────────────────────────────────────────────────────────────────────

def make_mvm_node(
    name: str,
    in_features: int,
    out_features: int,
    noise: Optional[NoiseSpec] = None,
    **kwargs,
) -> AnalogNode:
    """Create an MVM node for a linear layer / convolution."""
    return AnalogNode(
        name=name,
        op_type=OpType.MVM,
        domain=Domain.ANALOG,
        input_shape=(in_features,),
        output_shape=(out_features,),
        weight_shape=(in_features, out_features),
        flops=2 * in_features * out_features,  # multiply + accumulate
        param_count=in_features * out_features,
        noise=noise or NoiseSpec(kind="adc", sigma=1/256), # 8-bit ADC LSB noise
        **kwargs,
    )


def make_norm_node(
    name: str,
    dim: int,
    norm_type: str = "layer_norm",
    **kwargs,
) -> AnalogNode:
    """Create a normalization node (LayerNorm, GroupNorm, RMSNorm)."""
    op_map = {
        "layer_norm": OpType.LAYER_NORM,
        "group_norm": OpType.GROUP_NORM,
        "rms_norm": OpType.RMS_NORM,
    }
    return AnalogNode(
        name=name,
        op_type=op_map.get(norm_type, OpType.LAYER_NORM),
        domain=Domain.DIGITAL,
        input_shape=(dim,),
        output_shape=(dim,),
        flops=5 * dim,  # mean + var + sqrt + scale + shift
        param_count=2 * dim,  # gamma + beta
        **kwargs,
    )


def make_activation_node(
    name: str,
    dim: int,
    activation: str = "silu",
    **kwargs,
) -> AnalogNode:
    """Create an activation function node."""
    op_map = {
        "silu": OpType.SILU,
        "gelu": OpType.GELU,
        "relu": OpType.ANALOG_RELU,
        "sigmoid": OpType.ANALOG_SIGMOID,
        "softplus": OpType.SOFTPLUS,
    }
    op_type = op_map.get(activation, OpType.SILU)
    domain = default_domain(op_type)
    
    return AnalogNode(
        name=name,
        op_type=op_type,
        domain=domain,
        input_shape=(dim,),
        output_shape=(dim,),
        flops=dim,  # ~1 op per element
        **kwargs,
    )


def make_integration_node(
    name: str,
    state_dim: int,
    time_constant: float = 1e-3,
    noise: Optional[NoiseSpec] = None,
    **kwargs,
) -> AnalogNode:
    """Create an ODE integration node (for SSMs and flow model steps)."""
    return AnalogNode(
        name=name,
        op_type=OpType.INTEGRATION,
        domain=Domain.ANALOG,
        input_shape=(state_dim,),
        output_shape=(state_dim,),
        flops=2 * state_dim,  # multiply + accumulate per state variable
        metadata={"time_constant": time_constant},
        noise=noise or NoiseSpec(kind="thermal", sigma=1e-4, bandwidth_hz=250e3),
        **kwargs,
    )


def make_noise_node(
    name: str,
    dim: int,
    noise_type: str = "gaussian",
    noise: Optional[NoiseSpec] = None,
    **kwargs,
) -> AnalogNode:
    """Create a noise injection node (for diffusion SDE steps)."""
    return AnalogNode(
        name=name,
        op_type=OpType.NOISE_INJECTION,
        domain=Domain.ANALOG,
        input_shape=(dim,),
        output_shape=(dim,),
        flops=dim,
        metadata={"noise_type": noise_type},
        noise=noise or NoiseSpec(kind="shot", sigma=1e-3),
        **kwargs,
    )


def make_batch_norm_node(
    name: str,
    num_features: int,
    **kwargs,
) -> AnalogNode:
    """Create a batch normalization node."""
    return AnalogNode(
        name=name,
        op_type=OpType.BATCH_NORM,
        domain=Domain.DIGITAL,
        input_shape=(num_features,),
        output_shape=(num_features,),
        flops=5 * num_features,  # mean + var + sqrt + scale + shift
        param_count=2 * num_features,  # gamma + beta
        **kwargs,
    )


def make_max_pool_node(
    name: str,
    kernel_size: int,
    input_shape: tuple[int, ...],
    **kwargs,
) -> AnalogNode:
    """Create a max pooling node."""
    return AnalogNode(
        name=name,
        op_type=OpType.MAX_POOL,
        domain=Domain.DIGITAL,
        input_shape=input_shape,
        output_shape=input_shape,  # Simplified: same shape, actual shape depends on stride/padding
        flops=input_shape[0] * input_shape[1] * input_shape[2],  # ~1 compare per element
        param_count=0,
        metadata={"kernel_size": kernel_size},
        **kwargs,
    )


def make_dropout_node(
    name: str,
    dim: int,
    dropout_rate: float = 0.1,
    **kwargs,
) -> AnalogNode:
    """Create a dropout node (training-only, zero-compute at inference)."""
    return AnalogNode(
        name=name,
        op_type=OpType.DROPOUT,
        domain=Domain.DIGITAL,
        input_shape=(dim,),
        output_shape=(dim,),
        flops=0,  # Zero-compute at inference (training-only stochastic masking)
        param_count=0,
        metadata={"dropout_rate": dropout_rate},
        **kwargs,
    )
