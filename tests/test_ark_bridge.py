"""Tests for neuro_analog.ark_bridge (CDG bridge for Hopfield Neural ODE).

Validates:
1. CDGSpec builds correctly (node types, edge types, production rules)
2. CDG topology from weight matrices (node/edge counts, no-K case)
3. Compile pipeline produces a valid BaseAnalogCkt subclass
4. Forward pass returns correct shape for nominal and mismatch modes
5. Ideal (sigma=0) and mismatch (sigma>0) branches both work
6. K (input weight) matrix is wired correctly
"""

import pytest
import numpy as np

# ── Optional dependency guard ──────────────────────────────────────────────────

ark_available = True
try:
    import diffrax
    import jax.numpy as jnp
    from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
except ImportError:
    ark_available = False

pytestmark = pytest.mark.skipif(
    not ark_available,
    reason="ark_bridge tests require Ark (pip install -e ./Ark) and diffrax/jax",
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def small_weights():
    """2×2 symmetric J with spectral radius < 1, stable bias b."""
    rng = np.random.default_rng(42)
    n = 2
    A = rng.standard_normal((n, n)) * 0.5
    J = (A + A.T) / (2 * n)
    b = rng.standard_normal(n) * 0.1
    return n, J, b


@pytest.fixture
def medium_weights():
    """4×4 J with input matrix K (4×3)."""
    rng = np.random.default_rng(7)
    n, m = 4, 3
    A = rng.standard_normal((n, n)) * 0.5
    J = (A + A.T) / (2 * n)
    b = rng.standard_normal(n) * 0.1
    K = rng.standard_normal((n, m)) * 0.3
    return n, m, J, b, K


@pytest.fixture
def time_info():
    return TimeInfo(t0=0.0, t1=1.0, dt0=0.1, saveat=jnp.array([1.0]))


# ── 1. CDGSpec ─────────────────────────────────────────────────────────────────

class TestMakeSpec:

    def test_returns_six_tuple(self):
        from neuro_analog.ark_bridge import make_neural_ode_spec
        result = make_neural_ode_spec(0.05)
        assert len(result) == 6

    def test_spec_name(self):
        from neuro_analog.ark_bridge import make_neural_ode_spec
        spec, *_ = make_neural_ode_spec(0.0)
        assert spec.name == "neural_ode"

    def test_node_types_registered(self):
        from neuro_analog.ark_bridge import make_neural_ode_spec
        spec, StateVar, OutUnit, InpNode, MapEdge, FlowEdge = make_neural_ode_spec(0.0)
        type_names = {t.name for t in spec.node_types()}
        assert "StateVar" in type_names
        assert "OutUnit" in type_names
        assert "InpNode" in type_names

    def test_edge_types_registered(self):
        from neuro_analog.ark_bridge import make_neural_ode_spec
        spec, StateVar, OutUnit, InpNode, MapEdge, FlowEdge = make_neural_ode_spec(0.0)
        type_names = {t.name for t in spec.edge_types()}
        assert "MapEdge" in type_names
        assert "FlowEdge" in type_names

    def test_five_production_rules(self):
        # ReadOut, SelfDecay, JWeight, KWeight, InpHold
        from neuro_analog.ark_bridge import make_neural_ode_spec
        spec, *_ = make_neural_ode_spec(0.0)
        assert len(spec.production_rules()) == 5

    def test_mismatch_branch_uses_attrdefmismatch(self):
        from neuro_analog.ark_bridge import make_neural_ode_spec
        from ark.specification.attribute_def import AttrDefMismatch
        _, _, _, _, _, FlowEdge = make_neural_ode_spec(mismatch_sigma=0.05)
        g_attr = FlowEdge.attr_def["g"]
        assert isinstance(g_attr, AttrDefMismatch)
        assert g_attr.rstd == pytest.approx(0.05)

    def test_ideal_branch_uses_attrdef(self):
        from neuro_analog.ark_bridge import make_neural_ode_spec
        from ark.specification.attribute_def import AttrDef, AttrDefMismatch
        _, _, _, _, _, FlowEdge = make_neural_ode_spec(mismatch_sigma=0.0)
        g_attr = FlowEdge.attr_def["g"]
        assert isinstance(g_attr, AttrDef)
        assert not isinstance(g_attr, AttrDefMismatch)

    def test_analog_attr_has_range(self):
        """AnalogAttr must carry a range so Trainable.check_valid passes."""
        from neuro_analog.ark_bridge import make_neural_ode_spec
        _, StateVar, _, _, _, FlowEdge = make_neural_ode_spec(0.05)
        b_attr_type = StateVar.attr_def["b"].attr_type
        assert b_attr_type.has_range, "StateVar.b AnalogAttr must have a range"
        g_attr_type = FlowEdge.attr_def["g"].attr_type
        assert g_attr_type.has_range, "FlowEdge.g AnalogAttr must have a range"


# ── 2. CDG topology ────────────────────────────────────────────────────────────

class TestNeuralODEToCDG:

    def test_node_count_no_K(self, small_weights):
        from neuro_analog.ark_bridge import neural_ode_to_cdg
        n, J, b = small_weights
        cdg, *_ = neural_ode_to_cdg(J, b, K=None, mismatch_sigma=0.0)
        # n StateVar + n OutUnit
        assert len(cdg.nodes) == 2 * n

    def test_node_count_with_K(self, medium_weights):
        from neuro_analog.ark_bridge import neural_ode_to_cdg
        n, m, J, b, K = medium_weights
        cdg, *_ = neural_ode_to_cdg(J, b, K=K, mismatch_sigma=0.0)
        # n StateVar + n OutUnit + m InpNode
        assert len(cdg.nodes) == 2 * n + m

    def test_map_edges_no_K(self, small_weights):
        from neuro_analog.ark_bridge import neural_ode_to_cdg
        n, J, b = small_weights
        cdg, *_, _ = neural_ode_to_cdg(J, b, K=None, mismatch_sigma=0.0)
        # 2n MapEdges: n SelfDecay + n ReadOut
        map_edges = [e for e in cdg.edges if type(e).__name__ == "MapEdge"]
        assert len(map_edges) == 2 * n

    def test_flow_edges_match_nonzero_weights(self, small_weights):
        from neuro_analog.ark_bridge import neural_ode_to_cdg
        n, J, b = small_weights
        cdg, *_ = neural_ode_to_cdg(J, b, K=None, mismatch_sigma=0.0)
        expected = sum(1 for i in range(n) for j in range(n) if abs(J[i, j]) >= 1e-12)
        flow_edges = [e for e in cdg.edges if type(e).__name__ == "FlowEdge"]
        assert len(flow_edges) == expected

    def test_trainable_count_no_K(self, small_weights):
        from neuro_analog.ark_bridge import neural_ode_to_cdg
        n, J, b = small_weights
        _, _, mgr, _, _ = neural_ode_to_cdg(J, b, K=None, mismatch_sigma=0.05)
        # n bias + non-zero J weight edges
        nz_J = sum(1 for i in range(n) for j in range(n) if abs(J[i, j]) >= 1e-12)
        assert len(mgr.analog) == n + nz_J

    def test_inp_nodes_empty_without_K(self, small_weights):
        from neuro_analog.ark_bridge import neural_ode_to_cdg
        n, J, b = small_weights
        _, _, _, _, inp_nodes = neural_ode_to_cdg(J, b, K=None, mismatch_sigma=0.0)
        assert inp_nodes == []


# ── 3. Compile pipeline ────────────────────────────────────────────────────────

class TestCompileNeuralODECDG:

    def test_returns_base_analog_ckt_subclass(self, small_weights):
        from neuro_analog.ark_bridge import compile_neural_ode_cdg
        n, J, b = small_weights
        CktClass, _ = compile_neural_ode_cdg(J, b, mismatch_sigma=0.05)
        assert issubclass(CktClass, BaseAnalogCkt)

    def test_instantiate_nominal(self, small_weights):
        from neuro_analog.ark_bridge import compile_neural_ode_cdg
        n, J, b = small_weights
        CktClass, mgr = compile_neural_ode_cdg(J, b, mismatch_sigma=0.05)
        ckt = CktClass(
            init_trainable=mgr.get_initial_vals(),
            is_stochastic=False,
            solver=diffrax.Tsit5(),
        )
        assert not ckt.is_stochastic

    def test_a_trainable_shape(self, small_weights):
        from neuro_analog.ark_bridge import compile_neural_ode_cdg
        n, J, b = small_weights
        nz = sum(1 for i in range(n) for j in range(n) if abs(J[i, j]) >= 1e-12)
        CktClass, mgr = compile_neural_ode_cdg(J, b, mismatch_sigma=0.05)
        ckt = CktClass(
            init_trainable=mgr.get_initial_vals(),
            is_stochastic=False,
            solver=diffrax.Tsit5(),
        )
        assert ckt.a_trainable.shape == (n + nz,)

    def test_ideal_sigma_zero(self, small_weights):
        from neuro_analog.ark_bridge import compile_neural_ode_cdg
        n, J, b = small_weights
        CktClass, mgr = compile_neural_ode_cdg(J, b, mismatch_sigma=0.0)
        ckt = CktClass(
            init_trainable=mgr.get_initial_vals(),
            is_stochastic=False,
            solver=diffrax.Tsit5(),
        )
        assert issubclass(type(ckt), BaseAnalogCkt)


# ── 4. Forward pass ────────────────────────────────────────────────────────────

class TestForwardPass:

    def _make_ckt(self, J, b, sigma, stochastic=False):
        from neuro_analog.ark_bridge import compile_neural_ode_cdg
        CktClass, mgr = compile_neural_ode_cdg(J, b, mismatch_sigma=sigma)
        return CktClass(
            init_trainable=mgr.get_initial_vals(),
            is_stochastic=stochastic,
            solver=diffrax.Tsit5(),
        )

    def test_output_shape(self, small_weights, time_info):
        n, J, b = small_weights
        ckt = self._make_ckt(J, b, sigma=0.05)
        y0 = jnp.zeros(n)
        result = ckt(time_info, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        # saveat has 1 point → shape (1, n)
        assert result.shape == (1, n)

    def test_output_finite(self, small_weights, time_info):
        n, J, b = small_weights
        ckt = self._make_ckt(J, b, sigma=0.05)
        y0 = jnp.zeros(n)
        result = ckt(time_info, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        assert jnp.all(jnp.isfinite(result))

    def test_deterministic_when_not_stochastic(self, small_weights, time_info):
        """Same circuit called twice (is_stochastic=False) must return identical results."""
        n, J, b = small_weights
        ckt = self._make_ckt(J, b, sigma=0.05)
        y0 = jnp.zeros(n)
        r1 = ckt(time_info, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        r2 = ckt(time_info, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        np.testing.assert_array_equal(r1, r2)

    def test_multiple_saveat_points(self, small_weights):
        from neuro_analog.ark_bridge import compile_neural_ode_cdg
        n, J, b = small_weights
        CktClass, mgr = compile_neural_ode_cdg(J, b, mismatch_sigma=0.0)
        ckt = CktClass(
            init_trainable=mgr.get_initial_vals(),
            is_stochastic=False,
            solver=diffrax.Tsit5(),
        )
        ti = TimeInfo(t0=0.0, t1=2.0, dt0=0.1, saveat=jnp.array([0.5, 1.0, 2.0]))
        result = ckt(ti, jnp.zeros(n), switch=jnp.array([]), args_seed=0, noise_seed=0)
        assert result.shape == (3, n)

    def test_different_args_seeds_differ_when_stochastic(self, small_weights, time_info):
        """is_stochastic=True + different args_seed samples different per-weight mismatch
        realizations and should (almost always) give different outputs."""
        n, J, b = small_weights
        from neuro_analog.ark_bridge import compile_neural_ode_cdg
        CktClass, mgr = compile_neural_ode_cdg(J, b, mismatch_sigma=0.05)
        # Use Heun solver — Tsit5 warns about SDE convergence
        ckt = CktClass(
            init_trainable=mgr.get_initial_vals(),
            is_stochastic=True,
            solver=diffrax.Heun(),
        )
        y0 = jnp.zeros(n)
        r1 = ckt(time_info, y0, switch=jnp.array([]), args_seed=0,  noise_seed=0)
        r2 = ckt(time_info, y0, switch=jnp.array([]), args_seed=99, noise_seed=0)
        assert not jnp.allclose(r1, r2), "Different args_seeds should give different mismatch samples"

    def test_k_matrix_forward_pass(self, medium_weights, time_info):
        """K-wired CDG (with external input nodes) runs a full forward pass.

        InpNodes have order=1 in the spec so they are integrated by the ODE
        solver alongside StateVars.  The full ODE state is x ∈ R^(n+m), so
        y0 must have size n+m.  readout_nodes=state_nodes covers only
        StateVar (n nodes), so output shape is (saveat_len=1, n).
        """
        from neuro_analog.ark_bridge import compile_neural_ode_cdg
        n, m, J, b, K = medium_weights
        CktClass, mgr = compile_neural_ode_cdg(J, b, K=K, mismatch_sigma=0.05)
        ckt = CktClass(
            init_trainable=mgr.get_initial_vals(),
            is_stochastic=False,
            solver=diffrax.Tsit5(),
        )
        # ODE state covers StateVar (n) + InpNode (m) — both have order=1
        y0 = jnp.zeros(n + m)
        result = ckt(time_info, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        assert result.shape == (1, n), f"Expected (1, {n}), got {result.shape}"
        assert jnp.all(jnp.isfinite(result)), "K-wired forward pass produced non-finite values"
