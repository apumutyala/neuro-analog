"""Tests for ark_export code generation (Ark BaseAnalogCkt export).

Two export tiers are tested:

  Runnable BaseAnalogCkt subclasses (Neural ODE, SSM, DEQ, Diffusion):
    - Valid Python syntax
    - Correct class hierarchy (inherits BaseAnalogCkt)
    - Required interface methods present (make_args, ode_fn, noise_fn, readout)
    - Embeds solver in __init__ (no solver arg needed at instantiation)

  Analysis-only documents (Flow):
    - Valid Python syntax
    - Plain Python class, NOT a BaseAnalogCkt subclass
    - Has expected analysis methods / attributes
"""

import ast
import pytest
import numpy as np
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
from neuro_analog.ark_bridge.ssm_cdg import export_s4d_to_ark
from neuro_analog.ark_bridge.deq_cdg import export_deq_to_ark
from neuro_analog.ark_bridge.flow_cdg import export_flow_to_ark
from neuro_analog.ark_bridge.diffusion_cdg import export_diffusion_to_ark


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


# ── Fake model helpers ─────────────────────────────────────────────────────────

class FakeS4DLayer(nn.Module):
    """Minimal _S4DLayer-compatible module for testing export_s4d_to_ark."""
    def __init__(self, d_state: int = 4, d_model: int = 8):
        super().__init__()
        self.log_A_real = nn.Parameter(torch.zeros(d_state))
        self.log_A_imag = nn.Parameter(torch.zeros(d_state))
        self.B = nn.Linear(d_model, 2 * d_state, bias=False)
        self.C = nn.Linear(2 * d_state, d_model, bias=False)
        self.D = nn.Linear(d_model, d_model, bias=False)


class FakeDEQModel(nn.Module):
    """Minimal _DEQClassifier-compatible module for testing export_deq_to_ark."""
    def __init__(self, z_dim: int = 16, x_dim: int = 16):
        super().__init__()
        self.W_z = nn.Linear(z_dim, z_dim, bias=False)
        self.W_x = nn.Linear(x_dim, z_dim, bias=True)


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
        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))


# ── SSM export (BaseAnalogCkt) ────────────────────────────────────────────────

class TestSSMExport:
    # FakeS4DLayer params: d_state=4 → state_dim=8 real values, d_model=8
    _D_STATE = 4
    _D_MODEL = 8

    @pytest.fixture
    def ssm_layer(self):
        layer = FakeS4DLayer(d_state=self._D_STATE, d_model=self._D_MODEL)
        layer.eval()
        return layer

    @pytest.fixture
    def ssm_code(self, ssm_layer, tmp_path):
        return export_s4d_to_ark(
            ssm_layer, tmp_path / "ssm.py",
            d_model=self._D_MODEL, d_state=self._D_STATE,
        )

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
        # Real/imag split — A_re and A_im are the embedded state matrices
        assert "_A_re = jnp.array(" in ssm_code
        assert "_A_im = jnp.array(" in ssm_code

    def test_has_jax_imports(self, ssm_code):
        assert "import jax" in ssm_code
        assert "import diffrax" in ssm_code

    def test_has_polar_mismatch(self, ssm_code):
        # Complex polar mismatch on A: magnitude + phase independently
        assert "new_mag" in ssm_code
        assert "new_angle" in ssm_code

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
        state_dim = 2 * self._D_STATE  # real/imag split → 8 values
        y0 = jnp.zeros(state_dim)
        ti = TimeInfo(t0=0.0, t1=1.0, dt0=0.1, saveat=jnp.array([1.0]))
        result = ckt(ti, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        assert result.shape == (state_dim,)
        assert jnp.all(jnp.isfinite(result))


# ── DEQ export (BaseAnalogCkt) ────────────────────────────────────────────────

class TestDEQExport:
    _Z_DIM = 16
    _X_DIM = 16

    @pytest.fixture
    def deq_model(self):
        return FakeDEQModel(z_dim=self._Z_DIM, x_dim=self._X_DIM)

    @pytest.fixture
    def deq_code(self, deq_model, tmp_path):
        return export_deq_to_ark(
            deq_model, tmp_path / "deq.py",
            mismatch_sigma=0.05,
            z_dim=self._Z_DIM, x_dim=self._X_DIM,
        )

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

    def test_ode_is_gradient_flow(self, deq_code):
        # dz/dt = tanh(...) - z  — the defining DEQ ODE form
        assert "jnp.tanh" in deq_code
        assert "- z" in deq_code

    def test_augmented_state(self, deq_code):
        # State = [z, x_input]; dx/dt = 0 holds input constant
        assert "x_input" in deq_code
        assert f"z_dim = {self._Z_DIM}" in deq_code

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
        y0 = jnp.zeros(self._Z_DIM + self._X_DIM)
        ti = TimeInfo(t0=0.0, t1=5.0, dt0=0.1, saveat=jnp.array([5.0]))
        result = ckt(ti, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        assert result.shape == (self._Z_DIM,)
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
        y0 = jnp.zeros(self._Z_DIM + self._X_DIM)
        ti = TimeInfo(t0=0.0, t1=5.0, dt0=0.1, saveat=jnp.array([1.0, 2.0, 5.0]))
        result = ckt(ti, y0, switch=jnp.array([]), args_seed=0, noise_seed=0)
        assert result.shape == (self._Z_DIM,), f"Expected ({self._Z_DIM},), got {result.shape}"
        assert jnp.all(jnp.isfinite(result))


# ── Flow export (analysis-only, NOT BaseAnalogCkt) ────────────────────────────

class TestFlowExport:

    @pytest.fixture
    def flow_code(self, tmp_path):
        from neuro_analog.extractors.flow import FLUXExtractor
        ext = FLUXExtractor("FLUX.1-dev", device="cpu")
        graph = ext.build_graph()
        return export_flow_to_ark(graph, tmp_path / "flow.py", mismatch_sigma=0.05)

    def test_syntax_valid(self, flow_code):
        _parse(flow_code)

    def test_does_not_inherit_base_analog_ckt(self, flow_code):
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


# ── Diffusion export (BaseAnalogCkt) ─────────────────────────────────────────

class TestDiffusionExport:
    # Small fake matching the MLP structure export_diffusion_to_ark expects:
    # net.0, net.2, net.4 are the Linear layers in a nn.Sequential
    _IMG_DIM = 8
    _T_EMBED = 4   # sinusoidal embed dim
    _H_DIM = 16
    _T = 10        # diffusion steps

    @pytest.fixture
    def diffusion_code(self, tmp_path):
        in_dim = self._IMG_DIM + self._T_EMBED
        score_net = nn.Sequential(
            nn.Linear(in_dim, self._H_DIM),
            nn.ReLU(),
            nn.Linear(self._H_DIM, self._H_DIM),
            nn.ReLU(),
            nn.Linear(self._H_DIM, self._IMG_DIM),
        )
        betas_np = np.linspace(1e-4, 0.02, self._T, dtype=np.float32)
        return export_diffusion_to_ark(
            score_net, betas_np, tmp_path / "diff.py",
            img_dim=self._IMG_DIM, t_embed_dim=self._T_EMBED,
        )

    def test_syntax_valid(self, diffusion_code):
        _parse(diffusion_code)

    def test_is_base_analog_ckt_subclass(self, diffusion_code):
        assert "class DiffusionAnalogCkt(BaseAnalogCkt)" in diffusion_code

    def test_has_required_methods(self, diffusion_code):
        for method in ("make_args", "ode_fn", "noise_fn", "readout"):
            assert _has_method(diffusion_code, method), f"Missing method: {method}"

    def test_embeds_solver_in_init(self, diffusion_code):
        assert "solver=diffrax." in diffusion_code

    def test_has_beta_schedule(self, diffusion_code):
        assert "betas" in diffusion_code
        assert "alphas_bar" in diffusion_code

    def test_has_jax_imports(self, diffusion_code):
        assert "import jax" in diffusion_code
        assert "import diffrax" in diffusion_code

    def test_has_real_weights(self, diffusion_code):
        assert "jnp.array(" in diffusion_code
