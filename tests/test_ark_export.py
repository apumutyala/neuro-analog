"""Tests for ark_export code generation (Ark BaseAnalogCkt export).

Two export tiers are tested:

  Runnable BaseAnalogCkt subclasses (Neural ODE, SSM, DEQ):
    - Valid Python syntax
    - Correct class hierarchy (inherits BaseAnalogCkt)
    - Required interface methods present (make_args, ode_fn, noise_fn, readout)
    - Embeds solver in __init__ (no solver arg needed at instantiation)

  Analysis-only documents (Flow, Diffusion):
    - Valid Python syntax
    - Plain Python class, NOT a BaseAnalogCkt subclass
    - Has expected analysis methods / attributes
"""

import ast
import pytest
from pathlib import Path

import torch
import torch.nn as nn

jax_available = True
diffrax_available = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax_available = False
try:
    import diffrax
except ImportError:
    diffrax_available = False

pytestmark = pytest.mark.skipif(
    not (jax_available and diffrax_available),
    reason="ark export tests require jax and diffrax",
)

from neuro_analog.extractors.neural_ode import NeuralODEExtractor, export_neural_ode_to_ark
from neuro_analog.extractors.ssm import MambaExtractor
from neuro_analog.extractors.deq import DEQExtractor
from neuro_analog.ir.ark_export import (
    export_ssm_to_ark,
    export_flow_to_ark,
    export_diffusion_to_ark,
    export_deq_to_ark,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse(code: str) -> ast.Module:
    """Parse code and fail the test with a readable message on SyntaxError."""
    try:
        return ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"SyntaxError in generated code: {e}\n\n{code[:500]}")


def _has_class_inheriting(code: str, class_name: str, base: str) -> bool:
    tree = _parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for b in node.bases:
                if isinstance(b, ast.Name) and b.id == base:
                    return True
                if isinstance(b, ast.Attribute) and b.attr == base:
                    return True
    return False


def _has_method(code: str, method_name: str) -> bool:
    tree = _parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return True
    return False


# ── Shared fake models ─────────────────────────────────────────────────────────

class TinySSM(nn.Module):
    def __init__(self, d_model=32, d_state=8, expand=1):
        super().__init__()
        d_inner = d_model * expand
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.x_proj = nn.Linear(d_inner, 2 * d_state)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.d_model = d_model
        self.d_state = d_state


# ── Neural ODE export (BaseAnalogCkt) ─────────────────────────────────────────

class TestNeuralODEExport:

    @pytest.fixture
    def neural_ode_code(self, tmp_path):
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=16, num_layers=2)
        ext.load_model()
        return export_neural_ode_to_ark(ext, tmp_path / "ode.py", mismatch_sigma=0.05)

    def test_syntax_valid(self, neural_ode_code):
        _parse(neural_ode_code)

    def test_is_base_analog_ckt_subclass(self, neural_ode_code):
        assert "class NeuralODEAnalogCkt(BaseAnalogCkt)" in neural_ode_code

    def test_has_make_args(self, neural_ode_code):
        assert _has_method(neural_ode_code, "make_args")

    def test_has_ode_fn(self, neural_ode_code):
        assert _has_method(neural_ode_code, "ode_fn")

    def test_has_noise_fn(self, neural_ode_code):
        assert _has_method(neural_ode_code, "noise_fn")

    def test_has_readout(self, neural_ode_code):
        assert _has_method(neural_ode_code, "readout")

    def test_embeds_solver_in_init(self, neural_ode_code):
        assert "solver=diffrax." in neural_ode_code

    def test_has_jax_imports(self, neural_ode_code):
        assert "import jax" in neural_ode_code
        assert "import diffrax" in neural_ode_code

    def test_has_real_weights(self, neural_ode_code):
        # Weight arrays are embedded as jnp.array([...])
        assert "jnp.array([" in neural_ode_code

    def test_code_length_reasonable(self, neural_ode_code):
        assert len(neural_ode_code) > 500

    def test_instantiable_and_runnable(self, neural_ode_code, tmp_path):
        """Load the generated file and run a forward pass."""
        import importlib.util
        path = tmp_path / "ode.py"
        path.write_text(neural_ode_code, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("NeuralODEAnalogCkt", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckt = mod.NeuralODEAnalogCkt()
        from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
        assert isinstance(ckt, BaseAnalogCkt)
        y0 = jnp.zeros(2)
        ti = TimeInfo(t0=0.0, t1=1.0, dt0=0.1, saveat=jnp.array([1.0]))
        result = ckt(ti, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        assert result.shape == (2,)     # readout returns y[-1], shape (state_dim,)
        assert jnp.all(jnp.isfinite(result))


# ── SSM export (BaseAnalogCkt) ────────────────────────────────────────────────

class TestSSMExport:

    @pytest.fixture
    def ssm_extractor(self):
        model = TinySSM(d_model=32, d_state=8)
        ext = MambaExtractor(model_name="test_ssm", device="cpu")
        ext.model = model
        ext.model.eval()
        ext.extract_dynamics()
        return ext

    @pytest.fixture
    def ssm_code(self, ssm_extractor, tmp_path):
        graph = ssm_extractor.build_graph()
        return export_ssm_to_ark(graph, ssm_extractor, tmp_path / "ssm.py", mismatch_sigma=0.05)

    def test_syntax_valid(self, ssm_code):
        _parse(ssm_code)

    def test_is_base_analog_ckt_subclass(self, ssm_code):
        assert "class SSMAnalogCkt(BaseAnalogCkt)" in ssm_code

    def test_has_required_methods(self, ssm_code):
        for method in ("make_args", "ode_fn", "noise_fn", "readout"):
            assert _has_method(ssm_code, method), f"Missing method: {method}"

    def test_embeds_solver_in_init(self, ssm_code):
        assert "solver=diffrax." in ssm_code

    def test_has_state_matrices(self, ssm_code):
        assert "_A = jnp.array(" in ssm_code

    def test_has_jax_imports(self, ssm_code):
        assert "import jax" in ssm_code
        assert "import diffrax" in ssm_code

    def test_instantiable_and_runnable(self, ssm_code, tmp_path):
        import importlib.util
        path = tmp_path / "ssm.py"
        path.write_text(ssm_code, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("SSMAnalogCkt", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckt = mod.SSMAnalogCkt()
        from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
        assert isinstance(ckt, BaseAnalogCkt)
        state_dim = ckt.a_trainable.shape[0] // 3   # A+B+C concatenated, each state_dim
        y0 = jnp.zeros(state_dim)
        ti = TimeInfo(t0=0.0, t1=1.0, dt0=0.1, saveat=jnp.array([1.0]))
        result = ckt(ti, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        assert result.shape == (state_dim,)          # readout returns y[-1]
        assert jnp.all(jnp.isfinite(result))


# ── DEQ export (BaseAnalogCkt) ────────────────────────────────────────────────

class TestDEQExport:

    @pytest.fixture
    def deq_code(self, tmp_path):
        ext = DEQExtractor.reference(z_dim=16, x_dim=16, hidden_dim=32)
        ext.load_model()
        graph = ext.build_graph()
        return export_deq_to_ark(graph, tmp_path / "deq.py", sigma=0.05)

    def test_syntax_valid(self, deq_code):
        _parse(deq_code)

    def test_is_base_analog_ckt_subclass(self, deq_code):
        assert "class DEQAnalogCkt(BaseAnalogCkt)" in deq_code

    def test_has_required_methods(self, deq_code):
        for method in ("make_args", "ode_fn", "noise_fn", "readout"):
            assert _has_method(deq_code, method), f"Missing method: {method}"

    def test_embeds_solver_in_init(self, deq_code):
        assert "solver=diffrax." in deq_code

    def test_has_jax_imports(self, deq_code):
        assert "import jax" in deq_code
        assert "import diffrax" in deq_code

    def test_instantiable_and_runnable(self, deq_code, tmp_path):
        import importlib.util
        path = tmp_path / "deq.py"
        path.write_text(deq_code, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("DEQAnalogCkt", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckt = mod.DEQAnalogCkt()
        from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
        assert isinstance(ckt, BaseAnalogCkt)
        z_dim, x_dim = 16, 16
        y0 = jnp.zeros(z_dim + x_dim)
        ti = TimeInfo(t0=0.0, t1=5.0, dt0=0.1, saveat=jnp.array([5.0]))
        result = ckt(ti, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        assert result.shape == (z_dim,)             # readout returns y[-1, :z_dim]
        assert jnp.all(jnp.isfinite(result))

    def test_multi_step_saveat(self, deq_code, tmp_path):
        """readout = y[-1, :z_dim]: with multiple saveat points the output is still
        (z_dim,) because readout always slices the last saved time point."""
        import importlib.util
        path = tmp_path / "deq_multi.py"
        path.write_text(deq_code, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("DEQAnalogCkt", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ckt = mod.DEQAnalogCkt()
        from ark.optimization.base_module import TimeInfo
        z_dim, x_dim = 16, 16
        y0 = jnp.zeros(z_dim + x_dim)
        ti = TimeInfo(t0=0.0, t1=5.0, dt0=0.1, saveat=jnp.array([1.0, 2.0, 5.0]))
        result = ckt(ti, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        # readout does y[-1, :z_dim] — always the final saveat slice, shape (z_dim,)
        assert result.shape == (z_dim,), f"Expected ({z_dim},), got {result.shape}"
        assert jnp.all(jnp.isfinite(result))


# ── Flow export (analysis-only, NOT BaseAnalogCkt) ────────────────────────────

class TestFlowExport:

    @pytest.fixture
    def flow_code(self, tmp_path):
        # FLUXExtractor.build_graph() works without load_model() — model=None
        # falls back to a known parameter count (12B) for FLUX.1 architecture.
        from neuro_analog.extractors.flow import FLUXExtractor
        ext = FLUXExtractor("FLUX.1-dev", device="cpu")
        graph = ext.build_graph()
        return export_flow_to_ark(graph, tmp_path / "flow.py", mismatch_sigma=0.05)

    def test_syntax_valid(self, flow_code):
        _parse(flow_code)

    def test_does_not_inherit_base_analog_ckt(self, flow_code):
        # "BaseAnalogCkt" appears in comments — check it is NOT used as base class
        assert not _has_class_inheriting(flow_code, "FlowODE", "BaseAnalogCkt")

    def test_is_plain_class(self, flow_code):
        assert "class FlowODE:" in flow_code

    def test_has_analysis_methods(self, flow_code):
        assert _has_method(flow_code, "euler_step")
        assert _has_method(flow_code, "generate")

    def test_has_analysis_header(self, flow_code):
        assert "NOT a runnable Ark" in flow_code or "ANALYSIS" in flow_code

    def test_has_jax_imports(self, flow_code):
        assert "import jax" in flow_code or "jnp" in flow_code


# ── Diffusion export (analysis-only, NOT BaseAnalogCkt) ──────────────────────

class TestDiffusionExport:

    @pytest.fixture
    def diffusion_code(self, tmp_path):
        # StableDiffusionExtractor.build_graph() requires load_model() (downloads 4GB).
        # Build a minimal AnalogGraph directly — export only needs graph._dynamics and name.
        from neuro_analog.ir.graph import AnalogGraph
        from neuro_analog.ir.types import ArchitectureFamily, DynamicsProfile
        graph = AnalogGraph(name="test_diffusion", family=ArchitectureFamily.DIFFUSION)
        graph._dynamics = DynamicsProfile(num_diffusion_steps=20, has_dynamics=True)
        return export_diffusion_to_ark(graph, tmp_path / "diff.py", mismatch_sigma=0.05)

    def test_syntax_valid(self, diffusion_code):
        _parse(diffusion_code)

    def test_does_not_inherit_base_analog_ckt(self, diffusion_code):
        # "BaseAnalogCkt" appears in comments — check it is NOT used as base class
        assert not _has_class_inheriting(diffusion_code, "DiffusionDynamics", "BaseAnalogCkt")

    def test_is_plain_class(self, diffusion_code):
        assert "class DiffusionDynamics:" in diffusion_code

    def test_has_analysis_methods(self, diffusion_code):
        assert _has_method(diffusion_code, "vp_sde_drift")
        assert _has_method(diffusion_code, "cld_dynamics")

    def test_has_beta_schedule(self, diffusion_code):
        assert "self.betas = jnp.linspace(" in diffusion_code

    def test_has_analysis_header(self, diffusion_code):
        assert "NOT a runnable Ark" in diffusion_code or "ANALYSIS" in diffusion_code
