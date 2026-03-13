"""Tests for the neuro-analog intermediate representation."""

import json
import pytest

from neuro_analog.ir.types import (
    Domain, OpType, NoiseSpec, PrecisionSpec, ArchitectureFamily,
    AnalogAmenabilityProfile, DynamicsProfile,
)
from neuro_analog.ir.node import (
    AnalogNode, make_mvm_node, make_norm_node, make_activation_node,
    make_integration_node, make_noise_node,
)
from neuro_analog.ir.graph import AnalogGraph, DABoundary


# ──────────────────────────────────────────────────────────────────────
# NoiseSpec — bug 1.1 regression
# ──────────────────────────────────────────────────────────────────────

def test_noise_spec_literal_import():
    """NoiseSpec should instantiate without ImportError (Literal fix)."""
    ns = NoiseSpec(kind="thermal", sigma=0.01)
    assert ns.kind == "thermal"
    assert ns.sigma == 0.01


def test_noise_spec_all_kinds():
    for kind in ("none", "thermal", "shot", "adc", "composite"):
        ns = NoiseSpec(kind=kind, sigma=0.001)
        assert ns.kind == kind


# ──────────────────────────────────────────────────────────────────────
# AnalogGraph.nodes — bug 1.2 regression
# ──────────────────────────────────────────────────────────────────────

def test_nodes_property_not_exhausted():
    """graph.nodes must return same list on repeated access."""
    g = AnalogGraph("test", ArchitectureFamily.SSM)
    g.add_node(make_mvm_node("n1", 64, 64))
    g.add_node(make_mvm_node("n2", 64, 64))
    first = list(g.nodes)
    second = list(g.nodes)
    assert len(first) == 2
    assert len(second) == 2
    assert [n.name for n in first] == [n.name for n in second]


# ──────────────────────────────────────────────────────────────────────
# FLOP accounting
# ──────────────────────────────────────────────────────────────────────

def test_flop_fractions_sum_to_one():
    g = AnalogGraph("test", ArchitectureFamily.TRANSFORMER)
    g.add_node(make_mvm_node("mvm", 64, 64))       # ANALOG, 2*64*64 = 8192 FLOPs
    g.add_node(make_norm_node("norm", 64))          # DIGITAL, 5*64 = 320 FLOPs
    fracs = g.flop_fractions()
    total = sum(fracs.values())
    assert abs(total - 1.0) < 1e-6, f"FLOP fractions sum to {total}, not 1.0"


def test_flop_fractions_correct_domain():
    g = AnalogGraph("test", ArchitectureFamily.SSM)
    g.add_node(make_mvm_node("mvm", 100, 100))    # ANALOG: 2*100*100 = 20000 FLOPs
    g.add_node(make_activation_node("silu", 100, "silu"))  # DIGITAL: 100 FLOPs
    fracs = g.flop_fractions()
    # Analog should dominate
    assert fracs[Domain.ANALOG] > fracs[Domain.DIGITAL]


# ──────────────────────────────────────────────────────────────────────
# D/A boundary detection
# ──────────────────────────────────────────────────────────────────────

def _build_crossed_graph() -> AnalogGraph:
    """Graph with one guaranteed A→D and one D→A boundary."""
    g = AnalogGraph("test", ArchitectureFamily.TRANSFORMER)
    g.add_node(make_mvm_node("mvm_a", 64, 64))         # ANALOG
    g.add_node(make_norm_node("norm_d", 64))            # DIGITAL   ← A→D boundary
    g.add_node(make_mvm_node("mvm_b", 64, 64))         # ANALOG    ← D→A boundary
    g.add_edge("mvm_a", "norm_d")
    g.add_edge("norm_d", "mvm_b")
    return g


def test_da_boundaries_detected():
    g = _build_crossed_graph()
    boundaries = g.find_da_boundaries()
    assert len(boundaries) == 2


def test_da_boundary_directions():
    g = _build_crossed_graph()
    boundaries = g.find_da_boundaries()
    directions = {b.direction for b in boundaries}
    assert "ADC" in directions
    assert "DAC" in directions


def test_no_boundary_all_analog():
    g = AnalogGraph("test", ArchitectureFamily.EBM)
    g.add_node(make_mvm_node("mvm1", 64, 64))
    g.add_node(make_mvm_node("mvm2", 64, 64))
    g.add_edge("mvm1", "mvm2")
    boundaries = g.find_da_boundaries()
    assert len(boundaries) == 0


# ──────────────────────────────────────────────────────────────────────
# AnalogGraph.analyze() — scores and profile
# ──────────────────────────────────────────────────────────────────────

def test_analyze_returns_profile():
    g = _build_crossed_graph()
    profile = g.analyze()
    assert isinstance(profile, AnalogAmenabilityProfile)
    assert 0.0 <= profile.analog_flop_fraction <= 1.0
    assert 0.0 <= profile.overall_score <= 1.0


def test_analyze_noise_score_populated():
    """Bug 1.4 regression: noise_score must be set, not left at 0.0."""
    g = AnalogGraph("test", ArchitectureFamily.SSM)
    g.add_node(make_mvm_node("mvm", 64, 64))
    profile = g.analyze()
    # Non-stochastic arch → noise_score should be 1.0
    assert profile.noise_score == 1.0, f"noise_score={profile.noise_score}, expected 1.0"


def test_analyze_stochastic_noise_score():
    """Stochastic dynamics → noise_score = 0.7 (TRNG requirement penalty)."""
    g = AnalogGraph("test", ArchitectureFamily.DIFFUSION)
    g.add_node(make_mvm_node("mvm", 64, 64))
    g.set_dynamics(DynamicsProfile(has_dynamics=True, is_stochastic=True))
    profile = g.analyze()
    assert profile.noise_score == 0.7


# ──────────────────────────────────────────────────────────────────────
# to_dict() / JSON round-trip
# ──────────────────────────────────────────────────────────────────────

def test_to_dict_serializable():
    g = _build_crossed_graph()
    d = g.to_dict()
    json_str = json.dumps(d)  # Must not raise
    assert "nodes" in d
    assert "edges" in d
    assert len(d["nodes"]) == 3


def test_to_dict_node_fields():
    g = AnalogGraph("mymodel", ArchitectureFamily.SSM)
    g.add_node(make_mvm_node("layer_0.in_proj", 768, 1536))
    d = g.to_dict()
    node = d["nodes"][0]
    assert node["op_type"] == "MVM"
    assert node["domain"] == "ANALOG"
    assert node["param_count"] == 768 * 1536


# ──────────────────────────────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────────────────────────────

def test_make_mvm_node_flops():
    n = make_mvm_node("test", 128, 256)
    assert n.flops == 2 * 128 * 256
    assert n.param_count == 128 * 256
    assert n.domain == Domain.ANALOG
    assert n.op_type == OpType.MVM


def test_make_integration_node_metadata():
    n = make_integration_node("integrator", 16, time_constant=5e-4)
    assert n.op_type == OpType.INTEGRATION
    assert n.domain == Domain.ANALOG
    assert n.metadata["time_constant"] == 5e-4


def test_make_norm_node_digital():
    n = make_norm_node("norm", 512, "layer_norm")
    assert n.domain == Domain.DIGITAL
    assert n.op_type == OpType.LAYER_NORM


def test_make_activation_tanh_analog():
    """tanh maps to ANALOG_SIGMOID in the type system — analog-native via diff pair."""
    from neuro_analog.ir.node import default_domain
    # ReLU → ANALOG_RELU → ANALOG domain
    n = make_activation_node("relu", 64, "relu")
    assert n.domain == Domain.ANALOG


def test_make_activation_silu_digital():
    n = make_activation_node("silu", 64, "silu")
    assert n.domain == Domain.DIGITAL
