"""Tests for architecture extractors."""

import pytest
import torch
import torch.nn as nn

from neuro_analog.ir.types import ArchitectureFamily, Domain, OpType
from neuro_analog.ir.graph import AnalogGraph
from neuro_analog.extractors.neural_ode import NeuralODEExtractor, export_neural_ode_to_ark
from neuro_analog.extractors.ebm import EBMExtractor, EBMConfig
from neuro_analog.extractors.transformer import TransformerExtractor


# ──────────────────────────────────────────────────────────────────────
# Neural ODE extractor
# ──────────────────────────────────────────────────────────────────────

class TestNeuralODEExtractor:
    def test_demo_builds_graph(self):
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=32, num_layers=2)
        graph = ext.build_graph()
        assert graph.node_count > 0
        assert graph.family == ArchitectureFamily.NEURAL_ODE

    def test_demo_has_integration_node(self):
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=32)
        graph = ext.build_graph()
        int_nodes = [n for n in graph.nodes if n.op_type == OpType.INTEGRATION]
        assert len(int_nodes) == 1

    def test_demo_has_mvm_nodes(self):
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=64, num_layers=3)
        graph = ext.build_graph()
        mvm_nodes = [n for n in graph.nodes if n.op_type == OpType.MVM]
        assert len(mvm_nodes) >= 3  # At least one per linear layer

    def test_tanh_activation_is_analog(self):
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=32, activation="tanh")
        graph = ext.build_graph()
        act_nodes = [n for n in graph.nodes if n.op_type == OpType.ANALOG_SIGMOID]
        assert len(act_nodes) > 0
        for n in act_nodes:
            assert n.domain == Domain.ANALOG

    def test_from_module(self):
        f_theta = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 2))
        ext = NeuralODEExtractor.from_module(f_theta, state_dim=2)
        graph = ext.build_graph()
        assert graph.node_count > 0

    def test_run_returns_profile(self):
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=16)
        profile = ext.run()
        assert 0.0 <= profile.analog_flop_fraction <= 1.0
        assert profile.architecture == ArchitectureFamily.NEURAL_ODE

    def test_analog_fraction_high_with_tanh(self):
        """tanh activations + MVMs → high analog fraction."""
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=64, activation="tanh")
        profile = ext.run()
        assert profile.analog_flop_fraction > 0.5

    def test_extract_dynamics_has_nfe(self):
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=32)
        ext.load_model()
        dynamics = ext.extract_dynamics()
        assert dynamics.has_dynamics is True
        assert dynamics.num_function_evaluations is not None
        assert dynamics.num_function_evaluations > 0

    def test_weight_stats_have_required_bits(self):
        ext = NeuralODEExtractor.demo(state_dim=4, hidden_dim=32)
        ext.load_model()
        f_theta = ext._get_f_theta()
        stats = ext.extract_weight_stats(f_theta)
        assert len(stats) > 0
        for name, s in stats.items():
            assert "required_bits" in s
            assert 4 <= s["required_bits"] <= 16


# ──────────────────────────────────────────────────────────────────────
# Ark export
# ──────────────────────────────────────────────────────────────────────

class TestArkExport:
    def test_neural_ode_export_syntactically_valid(self, tmp_path):
        """Generated Ark code must parse as valid Python."""
        import ast
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=32, num_layers=2)
        ext.load_model()
        ext.build_graph()
        output = tmp_path / "neural_ode_ark.py"
        code = export_neural_ode_to_ark(ext, output_path=output)
        # Must be valid Python
        ast.parse(code)

    def test_neural_ode_export_has_mismatch(self, tmp_path):
        """Generated code must apply multiplicative mismatch (δ ~ N(1, σ²)) to weights."""
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=16, num_layers=2)
        ext.load_model()
        ext.build_graph()
        output = tmp_path / "out.py"
        code = export_neural_ode_to_ark(ext, output_path=output, mismatch_sigma=0.05)
        # Generated code applies mismatch as: param * (1.0 + sigma * jrandom.normal(...))
        assert "jrandom.normal" in code
        assert "sigma" in code

    def test_neural_ode_export_has_diffrax(self, tmp_path):
        """Export embeds a diffrax solver in __init__ (Ark runtime calls it via BaseAnalogCkt)."""
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=16)
        ext.load_model()
        ext.build_graph()
        output = tmp_path / "out.py"
        code = export_neural_ode_to_ark(ext, output_path=output)
        assert "import diffrax" in code
        assert "solver=diffrax." in code   # e.g. solver=diffrax.Heun()

    def test_neural_ode_export_has_saveat(self, tmp_path):
        """Export must use TimeInfo with saveat (Ark's ODE time specification)."""
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=16, t_span=(0.0, 2.0))
        ext.load_model()
        ext.build_graph()
        output = tmp_path / "out.py"
        code = export_neural_ode_to_ark(ext, output_path=output)
        assert "saveat" in code
        assert "2.0" in code  # t_end = 2.0

    def test_ssm_export_syntactically_valid(self, tmp_path):
        import ast
        from neuro_analog.ir.ark_export import export_ssm_to_ark
        from neuro_analog.ir.types import DynamicsProfile
        g = AnalogGraph("test_ssm", ArchitectureFamily.SSM)
        g.set_dynamics(DynamicsProfile(
            has_dynamics=True,
            dynamics_type="LTI_ODE",
            time_constants=[1e-3, 2e-3, 5e-3, 1e-2],
            time_constant_spread=10.0,
            state_dimension=4,
        ))
        output = tmp_path / "ssm_ark.py"
        code = export_ssm_to_ark(g, None, output)  # extractor=None → B/C fall back to jnp.ones
        ast.parse(code)
        assert "jrandom.normal" in code
        assert "diffrax" in code

    def test_flow_export_has_diffrax(self, tmp_path):
        import ast
        from neuro_analog.ir.ark_export import export_flow_to_ark
        from neuro_analog.ir.types import DynamicsProfile
        g = AnalogGraph("flux_test", ArchitectureFamily.FLOW)
        g.set_dynamics(DynamicsProfile(
            has_dynamics=True, dynamics_type="time_varying_ODE",
            num_function_evaluations=4,
        ))
        output = tmp_path / "flow_ark.py"
        code = export_flow_to_ark(g, output)
        ast.parse(code)
        assert "diffrax" in code


# ──────────────────────────────────────────────────────────────────────
# EBM extractor
# ──────────────────────────────────────────────────────────────────────

class TestEBMExtractor:
    def test_rbm_graph_has_sample_nodes(self):
        ext = EBMExtractor.rbm(n_visible=64, n_hidden=64)
        ext.load_model()
        graph = ext.build_graph()
        sample_nodes = [n for n in graph.nodes if n.op_type == OpType.SAMPLE]
        assert len(sample_nodes) >= 2  # hidden + visible p-bits

    def test_rbm_all_analog(self):
        ext = EBMExtractor.rbm(n_visible=64, n_hidden=64)
        ext.load_model()
        graph = ext.build_graph()
        boundaries = graph.find_da_boundaries()
        assert len(boundaries) == 0  # No D/A boundaries in RBM steady state

    def test_hopfield_has_analog_mvm(self):
        ext = EBMExtractor.hopfield(pattern_dim=128, num_patterns=32)
        ext.load_model()
        graph = ext.build_graph()
        mvm_nodes = [n for n in graph.nodes if n.op_type == OpType.MVM]
        assert len(mvm_nodes) >= 2  # X^T query + X retrieve

    def test_dtm_has_denoising_chain(self):
        steps = 5
        ext = EBMExtractor.extropic_dtm(dim=64, denoising_steps=steps)
        ext.load_model()
        graph = ext.build_graph()
        # DTCA uses GIBBS_STEP (subthreshold CMOS RNG), not SAMPLE (sMTJ p-bit)
        gibbs_nodes = [n for n in graph.nodes if n.op_type == OpType.GIBBS_STEP]
        assert len(gibbs_nodes) == steps
        # Each step: ACCUMULATION (G_k field) + RESISTOR_DAC (bias) + GIBBS_STEP
        assert len(graph.nodes) == steps * 3
        assert graph.family == ArchitectureFamily.DTM

    def test_dtm_dynamics(self):
        ext = EBMExtractor.extropic_dtm(dim=64, denoising_steps=3)
        ext.load_model()
        dynamics = ext.extract_dynamics()
        assert dynamics.dynamics_type == "thermodynamic_gibbs"
        assert dynamics.tau_rng_ns == 100.0
        assert dynamics.energy_per_sample_J == 350e-18

    def test_ebm_dynamics_stochastic(self):
        ext = EBMExtractor.rbm()
        ext.load_model()
        dynamics = ext.extract_dynamics()
        assert dynamics.is_stochastic is True
        assert dynamics.dynamics_type == "energy_minimization"


# ──────────────────────────────────────────────────────────────────────
# Transformer extractor (reference, no download)
# ──────────────────────────────────────────────────────────────────────

class TestTransformerExtractor:
    def test_reference_builds_graph(self):
        ext = TransformerExtractor.reference(dim=256, n_layers=4, heads=4)
        assert ext._graph is not None
        assert ext._graph.node_count > 0

    def test_reference_has_mvm_nodes(self):
        ext = TransformerExtractor.reference(dim=256, n_layers=2, heads=4)
        g = ext._graph
        mvm_nodes = [n for n in g.nodes if n.op_type == OpType.MVM]
        assert len(mvm_nodes) > 0

    def test_reference_digital_softmax(self):
        ext = TransformerExtractor.reference(dim=128, n_layers=2, use_favor_plus=False)
        g = ext._graph
        softmax_nodes = [n for n in g.nodes if n.op_type == OpType.SOFTMAX]
        assert len(softmax_nodes) > 0
        for n in softmax_nodes:
            assert n.domain == Domain.DIGITAL

    def test_favor_plus_replaces_softmax(self):
        ext_std = TransformerExtractor.reference(dim=128, n_layers=2, use_favor_plus=False)
        ext_fvp = TransformerExtractor.reference(dim=128, n_layers=2, use_favor_plus=True)
        softmax_std = [n for n in ext_std._graph.nodes if n.op_type == OpType.SOFTMAX]
        softmax_fvp = [n for n in ext_fvp._graph.nodes if n.op_type == OpType.SOFTMAX]
        kernel_fvp = [n for n in ext_fvp._graph.nodes if n.op_type == OpType.KERNEL_ATTENTION]
        assert len(softmax_std) > 0
        assert len(softmax_fvp) == 0
        assert len(kernel_fvp) > 0

    def test_favor_plus_fewer_da_boundaries(self):
        """FAVOR+ linear attention avoids the softmax D/A crossing — fewer or equal boundaries."""
        ext_std = TransformerExtractor.reference(dim=256, n_layers=4, use_favor_plus=False)
        ext_fvp = TransformerExtractor.reference(dim=256, n_layers=4, use_favor_plus=True)
        b_std = len(ext_std._graph.find_da_boundaries())
        b_fvp = len(ext_fvp._graph.find_da_boundaries())
        assert b_fvp <= b_std, (
            f"FAVOR+ should have <= D/A boundaries vs standard attention "
            f"({b_fvp} vs {b_std})"
        )


# ──────────────────────────────────────────────────────────────────────
# DEQ extractor
# ──────────────────────────────────────────────────────────────────────

class TestDEQExtractor:
    """DEQExtractor.reference() requires no model download — load_model() is a no-op."""

    def _make(self, z_dim=16, x_dim=16, hidden_dim=32):
        from neuro_analog.extractors.deq import DEQExtractor
        ext = DEQExtractor.reference(z_dim=z_dim, x_dim=x_dim, hidden_dim=hidden_dim)
        ext.load_model()
        return ext

    def test_reference_builds_graph(self):
        graph = self._make().build_graph()
        assert graph.node_count > 0
        assert graph.family == ArchitectureFamily.DEQ

    def test_graph_has_three_mvm_nodes(self):
        """W_z + W_x + W_out = exactly 3 MVM crossbars."""
        graph = self._make().build_graph()
        mvm_nodes = [n for n in graph.nodes if n.op_type == OpType.MVM]
        assert len(mvm_nodes) == 3

    def test_graph_has_feedback_integration_node(self):
        """Analog feedback path is modelled as an INTEGRATION node (state holder)."""
        graph = self._make().build_graph()
        int_nodes = [n for n in graph.nodes if n.op_type == OpType.INTEGRATION]
        assert len(int_nodes) == 1, f"Expected 1 integration node, got {len(int_nodes)}"

    def test_graph_has_da_boundaries(self):
        """Input DAC + output ADC — feedback loop itself is purely analog."""
        graph = self._make().build_graph()
        boundaries = graph.find_da_boundaries()
        assert len(boundaries) >= 1

    def test_dynamics_implicit_equilibrium(self):
        ext = self._make()
        dyn = ext.extract_dynamics()
        assert dyn.has_dynamics is True
        assert dyn.dynamics_type == "implicit_equilibrium"
        assert dyn.is_stochastic is False
        assert dyn.state_dimension == 16

    def test_run_returns_profile(self):
        from neuro_analog.extractors.deq import DEQExtractor
        ext = DEQExtractor.reference(z_dim=16, x_dim=16, hidden_dim=32)
        profile = ext.run()
        assert 0.0 <= profile.analog_flop_fraction <= 1.0
        assert profile.architecture == ArchitectureFamily.DEQ


# ──────────────────────────────────────────────────────────────────────
# Mamba inter-layer edges — bug 1.5 regression
# ──────────────────────────────────────────────────────────────────────

class TestMambaEdges:
    def test_inter_layer_edges_present(self):
        """After bug fix, MambaExtractor.build_graph() must wire inter-layer edges."""
        from neuro_analog.extractors.ssm import MambaExtractor
        import unittest.mock as mock

        # Build a minimal mock model that satisfies the extractor's config lookup
        class FakeConfig:
            d_model = 64
            d_state = 4
            n_layer = 3
            expand = 2

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeConfig()

            def named_parameters(self, recurse=True):
                return []

        ext = MambaExtractor.__new__(MambaExtractor)
        ext.model_name = "fake_mamba"
        ext.device = "cpu"
        ext.model = FakeModel()
        ext._graph = None

        graph = ext.build_graph()
        # Expect edges from layer_0.residual → layer_1.in_proj, etc.
        edge_set = set(graph._edges)
        assert ("layer_0.residual", "layer_1.in_proj") in edge_set
        assert ("layer_1.residual", "layer_2.in_proj") in edge_set
