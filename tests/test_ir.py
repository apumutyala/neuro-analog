"""Tests for the neuro-analog intermediate representation."""

import json
import pytest

from neuro_analog.analysis.taxonomy import AnalogTaxonomy, TaxonomyEntry
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


# ──────────────────────────────────────────────────────────────────────
# AnalogTaxonomy — cross-architecture amenability
# ──────────────────────────────────────────────────────────────────────

def test_taxonomy_empty_table():
    t = AnalogTaxonomy()
    assert t.comparison_table() == "No profiles added to taxonomy."
    assert t.rank_by_analog_amenability() == []


def test_taxonomy_add_reference_profiles_families():
    """add_reference_profiles() must populate all four reference families."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    families = {e.family for e in t.entries}
    assert ArchitectureFamily.NEURAL_ODE in families
    assert ArchitectureFamily.EBM in families
    assert ArchitectureFamily.DEQ in families
    assert ArchitectureFamily.TRANSFORMER in families


def test_taxonomy_no_duplicate_neural_ode():
    """Calling add_reference_profiles() twice must not create a second Neural ODE entry."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    t.add_reference_profiles()
    neural_ode_entries = [e for e in t.entries if e.family == ArchitectureFamily.NEURAL_ODE]
    assert len(neural_ode_entries) == 1


def test_taxonomy_all_scores_bounded():
    """Every profile overall_score and noise_score must lie in [0, 1]."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    for e in t.entries:
        assert 0.0 <= e.profile.overall_score <= 1.0, (
            f"{e.model_name} overall_score={e.profile.overall_score} out of bounds"
        )
        assert 0.0 <= e.profile.noise_score <= 1.0, (
            f"{e.model_name} noise_score={e.profile.noise_score} out of bounds"
        )


def test_taxonomy_ranking_neural_ode_first():
    """Core research claim: Neural ODE is the most analog-amenable architecture."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    ranked = t.rank_by_analog_amenability()
    assert ranked[0].family == ArchitectureFamily.NEURAL_ODE


def test_taxonomy_ranking_transformer_last():
    """Transformer should rank below all dynamics-bearing architectures."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    ranked = t.rank_by_analog_amenability()
    assert ranked[-1].family == ArchitectureFamily.TRANSFORMER


def test_taxonomy_dynamics_annotations():
    """DEQ must be annotated as dynamics-bearing; Transformer must not."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    by_family = {e.family: e for e in t.entries}
    assert by_family[ArchitectureFamily.DEQ].has_native_dynamics is True
    assert by_family[ArchitectureFamily.TRANSFORMER].has_native_dynamics is False


def test_taxonomy_to_dict_serializable():
    """to_dict() must be JSON-serializable and contain required fields."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    data = t.to_dict()
    json_str = json.dumps(data)  # Must not raise
    assert len(data) == 4
    required_keys = {
        "family", "model", "analog_flop_fraction", "da_boundary_count",
        "has_dynamics", "noise_score", "overall_score",
    }
    for entry in data:
        assert required_keys <= entry.keys(), f"Missing keys in {entry['family']}"


def test_taxonomy_to_dict_round_trip():
    """Scores survive JSON round-trip without precision loss."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    data = t.to_dict()
    reloaded = json.loads(json.dumps(data))
    for orig, rt in zip(data, reloaded):
        assert abs(orig["overall_score"] - rt["overall_score"]) < 1e-9


def test_taxonomy_save_load(tmp_path):
    """save() writes valid JSON that can be loaded back."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    out = tmp_path / "taxonomy.json"
    t.save(out)
    loaded = json.loads(out.read_text())
    assert len(loaded) == 4
    families = {e["family"] for e in loaded}
    assert "neural_ode" in families
    assert "transformer" in families


def test_taxonomy_comparison_table_contains_all_families():
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    table = t.comparison_table()
    for family in (ArchitectureFamily.NEURAL_ODE, ArchitectureFamily.EBM,
                   ArchitectureFamily.DEQ, ArchitectureFamily.TRANSFORMER):
        assert family.value in table, f"{family.value} missing from comparison table"


def test_taxonomy_ebm_high_analog_fraction():
    """EBM reference profile must have analog_flop_fraction >= 0.9 (near-native compute)."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    ebm = next(e for e in t.entries if e.family == ArchitectureFamily.EBM)
    assert ebm.profile.analog_flop_fraction >= 0.9


def test_taxonomy_deq_low_da_boundaries():
    """DEQ feedback loop stays analog — da_boundary_count should be ≤ 2."""
    t = AnalogTaxonomy()
    t.add_reference_profiles()
    deq = next(e for e in t.entries if e.family == ArchitectureFamily.DEQ)
    assert deq.profile.da_boundary_count <= 2
