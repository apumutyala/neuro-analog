"""Regression tests for backend extension: ensure existing analog-only paths are preserved.

This test suite validates that adding TargetBackend enum and backend-specific scoring
does not corrupt existing analog amenability analysis. Per the council transcript:
"Extend test_ir.py and test_ark_export.py to assert that all 6 current Ark-export
graphs pass their analog fidelity checks end-to-end."

These tests capture the current analog-only behavior as a baseline. After the IR
extension, these tests must pass unchanged.
"""

import pytest

from neuro_analog.ir.types import (
    Domain, OpType, ArchitectureFamily, AnalogAmenabilityProfile, DynamicsProfile,
)
from neuro_analog.ir.node import (
    AnalogNode, make_mvm_node, make_norm_node, make_activation_node,
    make_integration_node,
)
from neuro_analog.ir.graph import AnalogGraph


# ──────────────────────────────────────────────────────────────────────
# Baseline: HYBRID domain semantics (approximation quality, not RTL)
# ──────────────────────────────────────────────────────────────────────

def test_hybrid_domain_means_approximation_quality():
    """HYBRID must mean 'analog-possible with accuracy tradeoff', not 'mixed-signal RTL'.
    
    This is the semantic collision identified in the council transcript. Domain.HYBRID
    is used in amenability scoring as an approximation quality signal. Reusing it to
    mean RTL domain-crossing would silently corrupt all existing scoring logic.
    """
    # Create a HYBRID node (e.g., KERNEL_ATTENTION)
    hybrid_node = AnalogNode(
        name="kernel_attn",
        op_type=OpType.KERNEL_ATTENTION,
        domain=Domain.HYBRID,
        input_shape=(128,),
        output_shape=(128,),
        flops=256,
    )
    
    # HYBRID nodes are treated as ANALOG in boundary detection (graph.py:94-95)
    assert hybrid_node.domain == Domain.HYBRID
    assert hybrid_node.is_hybrid
    
    # HYBRID contributes to hybrid_flop_fraction in amenability profile
    g = AnalogGraph("test", ArchitectureFamily.TRANSFORMER)
    g.add_node(hybrid_node)
    profile = g.analyze()
    
    assert profile.hybrid_flop_fraction == 1.0
    assert profile.analog_flop_fraction == 0.0
    assert profile.digital_flop_fraction == 0.0


def test_hybrid_treated_as_analog_in_boundary_detection():
    """graph.find_da_boundaries() treats HYBRID as ANALOG (line 94-95).
    
    This is existing behavior that must be preserved. HYBRID nodes are
    considered analog-possible, so HYBRID→DIGITAL creates an ADC boundary.
    """
    g = AnalogGraph("test", ArchitectureFamily.SSM)
    hybrid_node = AnalogNode(
        name="approx_norm", op_type=OpType.APPROX_NORM, domain=Domain.HYBRID,
        input_shape=(64,), output_shape=(64,), flops=64,
    )
    digital_node = make_norm_node("layer_norm", 64)
    
    g.add_node(hybrid_node)
    g.add_node(digital_node)
    g.add_edge("approx_norm", "layer_norm")
    
    boundaries = g.find_da_boundaries()
    # HYBRID (treated as ANALOG) → DIGITAL creates boundary
    assert len(boundaries) == 1
    # But DABoundary.direction returns "HYBRID" because source_domain is HYBRID (not ANALOG)
    assert boundaries[0].direction == "HYBRID"
    assert boundaries[0].source_domain == Domain.HYBRID


# ──────────────────────────────────────────────────────────────────────
# Baseline: Analog amenability scoring without backend awareness
# ──────────────────────────────────────────────────────────────────────

def test_analog_scoring_baseline_neural_ode():
    """Neural ODE analog amenability score baseline (architecture with dynamics).
    
    This captures the existing scoring behavior for a dynamics-bearing architecture.
    After backend extension, nodes without target_backend specified must produce
    identical scores.
    """
    g = AnalogGraph("neural_ode", ArchitectureFamily.NEURAL_ODE, model_params=10000)
    g.add_node(make_mvm_node("W_in", 64, 128))
    g.add_node(make_integration_node("ode_step", 128))
    g.add_node(make_mvm_node("W_out", 128, 10))
    g.add_edge("W_in", "ode_step")
    g.add_edge("ode_step", "W_out")
    
    # Set dynamics profile (Neural ODE = time_varying_ODE)
    g.set_dynamics(DynamicsProfile(
        has_dynamics=True,
        dynamics_type="time_varying_ODE",
        state_dimension=128,
    ))
    
    profile = g.analyze()
    
    # Baseline assertions (from compute_scores logic)
    assert profile.architecture == ArchitectureFamily.NEURAL_ODE
    assert profile.analog_flop_fraction > 0.95  # All ops are analog-native
    assert profile.dynamics_score == 1.0  # time_varying_ODE gets top score (line 290)
    assert profile.noise_score == 1.0  # Non-stochastic (line 282)
    assert profile.overall_score > 0.7  # Should be high-scoring
    
    # Preserve these exact values as regression baseline
    return profile  # Return for future comparison if needed


def test_analog_scoring_baseline_transformer():
    """Transformer analog amenability score baseline (no dynamics).
    
    Captures scoring for architecture without native dynamics. Should score
    lower than Neural ODE due to lack of dynamics and high D/A boundary count.
    """
    g = AnalogGraph("transformer_ffn", ArchitectureFamily.TRANSFORMER, model_params=5000)
    g.add_node(make_mvm_node("qkv_proj", 512, 1536))
    g.add_node(make_activation_node("gelu", 1536, "gelu"))  # DIGITAL
    g.add_node(make_norm_node("layer_norm", 512))  # DIGITAL
    g.add_edge("qkv_proj", "gelu")
    g.add_edge("gelu", "layer_norm")
    
    # No dynamics for Transformer
    g.set_dynamics(DynamicsProfile(has_dynamics=False))
    
    profile = g.analyze()
    
    assert profile.architecture == ArchitectureFamily.TRANSFORMER
    assert profile.dynamics_score == 0.0  # No dynamics (line 300)
    assert profile.noise_score == 1.0  # Non-stochastic
    # Only 1 boundary: MVM (ANALOG) → GELU (DIGITAL). GELU→LayerNorm is both DIGITAL, no boundary.
    assert profile.da_boundary_count == 1
    # High analog_flop_fraction (MVM dominates) and low boundary count boost score despite no dynamics
    # Actual baseline: ~0.665. Still lower than Neural ODE's >0.7.
    assert 0.6 < profile.overall_score < 0.7


def test_analog_scoring_stochastic_penalty():
    """Stochastic architectures get noise_score = 0.7 penalty (TRNG requirement).
    
    Baseline behavior from lines 279-280. Diffusion models require precise
    noise calibration, so they score lower on noise amenability.
    """
    g = AnalogGraph("diffusion", ArchitectureFamily.DIFFUSION, model_params=3000)
    g.add_node(make_mvm_node("score_net", 64, 64))
    g.set_dynamics(DynamicsProfile(has_dynamics=True, is_stochastic=True))
    
    profile = g.analyze()
    
    assert profile.noise_score == 0.7  # Stochastic penalty
    assert profile.dynamics_score == 0.8  # has_dynamics but not top-tier type


def test_precision_score_baseline():
    """Precision score calculation baseline (lines 310-312).
    
    Default min_weight_precision_bits = 8, min_activation_precision_bits = 8.
    weight_score = max(0, 1 - (8-4)/12) = 1 - 4/12 = 0.667
    act_score = 0.667
    precision_score = 0.6*0.667 + 0.4*0.667 = 0.667
    """
    g = AnalogGraph("test", ArchitectureFamily.SSM)
    g.add_node(make_mvm_node("linear", 64, 64))
    profile = g.analyze()
    
    # Default precision bits = 8
    assert profile.min_weight_precision_bits == 8
    # precision_score calculation (line 310-312)
    expected = 0.6 * (1 - (8-4)/12) + 0.4 * (1 - (8-4)/12)
    assert abs(profile.precision_score - expected) < 1e-6


def test_boundary_score_baseline():
    """Boundary score: max(0, 1 - da_boundary_count/100) (line 315)."""
    g = AnalogGraph("test", ArchitectureFamily.EBM)
    g.add_node(make_mvm_node("mvm1", 64, 64))
    g.add_node(make_norm_node("norm", 64))  # Creates boundary
    g.add_edge("mvm1", "norm")
    
    profile = g.analyze()
    
    assert profile.da_boundary_count == 1
    # boundary_score = max(0, 1 - 1/100) = 0.99
    assert abs(profile.boundary_score - 0.99) < 1e-6


# ──────────────────────────────────────────────────────────────────────
# Baseline: Overall score computation (weighted composite)
# ──────────────────────────────────────────────────────────────────────

def test_overall_score_default_weights():
    """Overall score uses default weights (line 271, 318-324).
    
    Default: dynamics=0.25, precision=0.25, boundary=0.15, noise=0.20, analog_frac=0.15
    """
    g = AnalogGraph("test", ArchitectureFamily.NEURAL_ODE)
    g.add_node(make_mvm_node("mvm", 64, 64))
    g.set_dynamics(DynamicsProfile(has_dynamics=True, dynamics_type="LTI_ODE"))
    
    profile = g.analyze()
    
    # Manual calculation with default weights
    expected = (
        0.25 * profile.dynamics_score
        + 0.25 * profile.precision_score
        + 0.15 * profile.boundary_score
        + 0.20 * profile.noise_score
        + 0.15 * profile.analog_flop_fraction
    )
    
    assert abs(profile.overall_score - expected) < 1e-6


def test_overall_score_all_analog_high():
    """All-analog graph with dynamics should score highly."""
    g = AnalogGraph("all_analog", ArchitectureFamily.EBM, model_params=1000)
    g.add_node(make_mvm_node("mvm1", 128, 128))
    g.add_node(make_mvm_node("mvm2", 128, 10))
    g.add_edge("mvm1", "mvm2")
    g.set_dynamics(DynamicsProfile(has_dynamics=True, dynamics_type="energy_minimization"))
    
    profile = g.analyze()
    
    assert profile.analog_flop_fraction == 1.0
    assert profile.da_boundary_count == 0
    assert profile.dynamics_score == 0.95  # energy_minimization (line 292)
    assert profile.overall_score > 0.7


# ──────────────────────────────────────────────────────────────────────
# Serialization baseline (to_dict must preserve all fields)
# ──────────────────────────────────────────────────────────────────────

def test_graph_to_dict_baseline():
    """AnalogGraph.to_dict() serialization baseline (graph.py:142-150).
    
    After adding target_backend field, serialization must include it when present
    but not break when absent (backward compatibility).
    """
    g = AnalogGraph("test_model", ArchitectureFamily.SSM, model_params=1234)
    g.add_node(make_mvm_node("layer1", 64, 128))
    g.add_node(make_activation_node("relu", 128, "relu"))
    g.add_edge("layer1", "relu")
    
    d = g.to_dict()
    
    # Baseline fields that must always be present
    assert d["name"] == "test_model"
    assert d["family"] == "ssm"
    assert d["model_params"] == 1234
    assert len(d["nodes"]) == 2
    assert len(d["edges"]) == 1
    
    # Node fields baseline
    node = d["nodes"][0]
    assert "id" in node
    assert "name" in node
    assert "op_type" in node
    assert "domain" in node
    assert "flops" in node
    assert "param_count" in node
    
    # After backend extension: "target_backend" should appear if set, but not required
    # Test will validate that nodes without target_backend don't break serialization


# ──────────────────────────────────────────────────────────────────────
# Guard against future scoring contamination
# ──────────────────────────────────────────────────────────────────────

def test_compute_scores_is_idempotent():
    """Calling compute_scores() multiple times produces same result.
    
    Validates that scoring doesn't have hidden state that could be corrupted
    by backend-specific logic paths.
    """
    g = AnalogGraph("test", ArchitectureFamily.DEQ)
    g.add_node(make_mvm_node("W", 64, 64))
    g.set_dynamics(DynamicsProfile(has_dynamics=True, dynamics_type="implicit_equilibrium"))
    
    profile1 = g.analyze()
    score1 = profile1.overall_score
    
    # Re-compute
    profile1.compute_scores()
    score2 = profile1.overall_score
    
    assert abs(score1 - score2) < 1e-9


def test_analog_only_graph_no_backend_field():
    """Nodes without target_backend field must work (backward compatibility).
    
    After adding AnalogNode.target_backend, existing code that creates nodes
    without specifying backend must continue to work and default to analog behavior.
    """
    # Create node without target_backend (will be added as optional field)
    node = make_mvm_node("test", 64, 64)
    
    # Should not raise, should default to analog-circuit behavior
    g = AnalogGraph("test", ArchitectureFamily.NEURAL_ODE)
    g.add_node(node)
    profile = g.analyze()
    
    # Should produce valid analog amenability scores
    assert 0.0 <= profile.overall_score <= 1.0
    assert profile.analog_flop_fraction == 1.0


# ────────────────────────────────────────────────────────────────────
# Backend extension tests (new IR fields)
# ────────────────────────────────────────────────────────────────────

def test_target_backend_field_exists():
    """AnalogNode.target_backend field should be available and optional."""
    from neuro_analog.ir.types import TargetBackend
    
    node = make_mvm_node("test", 64, 64)
    assert hasattr(node, "target_backend")
    assert node.target_backend is None  # Default: unset
    
    # Can be set
    node.target_backend = TargetBackend.FPGA_INFERENCE
    assert node.target_backend == TargetBackend.FPGA_INFERENCE


def test_is_mixed_signal_boundary_field_exists():
    """AnalogNode.is_mixed_signal_boundary should exist and default to False."""
    node = make_mvm_node("test", 64, 64)
    assert hasattr(node, "is_mixed_signal_boundary")
    assert node.is_mixed_signal_boundary is False
    
    # Can be set
    node.is_mixed_signal_boundary = True
    assert node.is_mixed_signal_boundary is True


def test_backend_gating_rejects_fpga():
    """compute_scores() must reject FPGA_INFERENCE backend (not analog-physics)."""
    from neuro_analog.ir.types import TargetBackend
    
    g = AnalogGraph("fpga_graph", ArchitectureFamily.SSM)
    g.add_node(make_mvm_node("mvm", 64, 64))
    
    # Attempting to score as FPGA backend should raise ValueError
    with pytest.raises(ValueError, match="analog-physics.*FPGA_INFERENCE"):
        g.analyze(target_backend=TargetBackend.FPGA_INFERENCE)


def test_backend_gating_rejects_asic():
    """compute_scores() must reject ASIC_RTL backend."""
    from neuro_analog.ir.types import TargetBackend
    
    g = AnalogGraph("asic_graph", ArchitectureFamily.NEURAL_ODE)
    g.add_node(make_mvm_node("mvm", 64, 64))
    
    with pytest.raises(ValueError, match="analog-physics.*ASIC_RTL"):
        g.analyze(target_backend=TargetBackend.ASIC_RTL)


def test_backend_gating_accepts_analog_circuit():
    """compute_scores() should accept ANALOG_CIRCUIT backend explicitly."""
    from neuro_analog.ir.types import TargetBackend
    
    g = AnalogGraph("analog_graph", ArchitectureFamily.EBM)
    g.add_node(make_mvm_node("mvm", 128, 128))
    g.set_dynamics(DynamicsProfile(has_dynamics=True, dynamics_type="energy_minimization"))
    
    # Should not raise
    profile = g.analyze(target_backend=TargetBackend.ANALOG_CIRCUIT)
    assert profile.overall_score > 0.0


def test_backend_gating_none_is_analog_default():
    """target_backend=None should default to analog-circuit behavior (backward compat)."""
    g = AnalogGraph("legacy_graph", ArchitectureFamily.NEURAL_ODE)
    g.add_node(make_mvm_node("mvm", 64, 64))
    g.set_dynamics(DynamicsProfile(has_dynamics=True, dynamics_type="LTI_ODE"))
    
    # target_backend=None is the legacy path, should work
    profile = g.analyze(target_backend=None)
    assert profile.dynamics_score == 1.0
    assert profile.overall_score > 0.0


def test_serialization_includes_target_backend():
    """to_dict() must include target_backend when set."""
    from neuro_analog.ir.types import TargetBackend
    
    g = AnalogGraph("test", ArchitectureFamily.SSM)
    node = make_mvm_node("fpga_node", 64, 64)
    node.target_backend = TargetBackend.FPGA_INFERENCE
    g.add_node(node)
    
    d = g.to_dict()
    assert d["nodes"][0]["target_backend"] == "FPGA_INFERENCE"


def test_serialization_omits_unset_target_backend():
    """to_dict() must omit target_backend when None (backward compatibility)."""
    g = AnalogGraph("test", ArchitectureFamily.EBM)
    g.add_node(make_mvm_node("mvm", 64, 64))  # target_backend=None
    
    d = g.to_dict()
    assert "target_backend" not in d["nodes"][0]


def test_serialization_includes_mixed_signal_boundary():
    """to_dict() must include is_mixed_signal_boundary when True."""
    g = AnalogGraph("test", ArchitectureFamily.TRANSFORMER)
    node = make_mvm_node("boundary_node", 64, 64)
    node.is_mixed_signal_boundary = True
    g.add_node(node)
    
    d = g.to_dict()
    assert d["nodes"][0]["is_mixed_signal_boundary"] is True


def test_serialization_omits_false_mixed_signal_boundary():
    """to_dict() must omit is_mixed_signal_boundary when False (compactness)."""
    g = AnalogGraph("test", ArchitectureFamily.SSM)
    g.add_node(make_mvm_node("mvm", 64, 64))  # is_mixed_signal_boundary=False
    
    d = g.to_dict()
    assert "is_mixed_signal_boundary" not in d["nodes"][0]


def test_hybrid_domain_not_conflated_with_mixed_signal():
    """Domain.HYBRID and is_mixed_signal_boundary are separate concepts.
    
    Domain.HYBRID = approximation quality signal (analog-possible with tradeoff)
    is_mixed_signal_boundary = RTL domain-crossing flag
    They must not be conflated.
    """
    from neuro_analog.ir.types import TargetBackend
    
    # Create HYBRID domain node (approximation quality)
    hybrid_node = AnalogNode(
        name="kernel_attn",
        op_type=OpType.KERNEL_ATTENTION,
        domain=Domain.HYBRID,
        input_shape=(128,),
        output_shape=(128,),
        flops=256,
    )
    # is_mixed_signal_boundary should be False (separate semantic)
    assert hybrid_node.domain == Domain.HYBRID
    assert hybrid_node.is_mixed_signal_boundary is False
    
    # Can set mixed-signal boundary independently
    hybrid_node.is_mixed_signal_boundary = True
    assert hybrid_node.domain == Domain.HYBRID  # Domain unchanged
    assert hybrid_node.is_mixed_signal_boundary is True
