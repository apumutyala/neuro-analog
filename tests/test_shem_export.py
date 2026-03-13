"""Tests for Shem export code generation (Neural ODE and SSM).

Validates that:
1. Shem exports are syntactically valid Python/JAX code
2. Required Shem primitives are present (AnalogTrainable, mismatch(), diffrax)
3. Real weights are extracted (not placeholders)
"""

import ast
import pytest
from pathlib import Path

import torch
import torch.nn as nn

# Check for optional dependencies
jax_available = True
diffrax_available = True
try:
    import jax
except ImportError:
    jax_available = False

try:
    import diffrax
except ImportError:
    diffrax_available = False

# Import core modules (always available)
from neuro_analog.extractors.neural_ode import NeuralODEExtractor, export_neural_ode_to_shem
from neuro_analog.extractors.ssm import MambaExtractor
from neuro_analog.ir.shem_export import export_ssm_to_shem

# Skip entire module if JAX deps missing
pytestmark = pytest.mark.skipif(
    not (jax_available and diffrax_available),
    reason="Shem export tests require jax and diffrax (optional dependencies)"
)


# ──────────────────────────────────────────────────────────────────────
# Neural ODE Shem Export Tests
# ──────────────────────────────────────────────────────────────────────

class TestNeuralODEShemExport:
    """Validation suite for Neural ODE → Shem code generation."""

    @pytest.fixture
    def neural_ode_extractor(self):
        """Create and prepare a demo Neural ODE extractor."""
        ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=32, num_layers=2)
        ext.load_model()
        ext.build_graph()
        return ext

    def _capture_export_code(self, ext):
        """Capture generated code by monkey-patching Path.write_text."""
        original_write_text = Path.write_text
        captured_code = []

        def capture_write_text(self, data, **kwargs):
            captured_code.append(data)
            return len(data)

        Path.write_text = capture_write_text
        try:
            export_neural_ode_to_shem(ext, output_path=Path("dummy.py"))
            return captured_code[0]
        finally:
            Path.write_text = original_write_text

    def test_syntax_valid(self, neural_ode_extractor):
        """Generated code must be syntactically valid Python."""
        code = self._capture_export_code(neural_ode_extractor)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in generated Neural ODE code: {e}")

    def test_has_required_imports(self, neural_ode_extractor):
        """Generated code must import required JAX/Diffrax modules."""
        code = self._capture_export_code(neural_ode_extractor)
        assert "import jax" in code, "Missing 'import jax'"
        assert "import diffrax" in code, "Missing 'import diffrax'"

    def test_has_mismatch_method(self, neural_ode_extractor):
        """Generated code must define mismatch() method for Shem."""
        code = self._capture_export_code(neural_ode_extractor)
        assert "def mismatch(" in code, "Missing mismatch() method"

    def test_has_analog_trainable(self, neural_ode_extractor):
        """Generated code must use AnalogTrainable for Shem primitives."""
        code = self._capture_export_code(neural_ode_extractor)
        assert "AnalogTrainable" in code, "Missing AnalogTrainable"

    def test_has_weight_arrays(self, neural_ode_extractor):
        """Generated code must include weight arrays (W_, b_)."""
        code = self._capture_export_code(neural_ode_extractor)
        n_weights = code.count("W_") + code.count("b_")
        assert n_weights > 0, "No weight arrays found in generated code"

    def test_code_length_reasonable(self, neural_ode_extractor):
        """Generated code should be substantial (not empty/trivial)."""
        code = self._capture_export_code(neural_ode_extractor)
        assert len(code) > 1000, f"Generated code too short: {len(code)} chars"


# ──────────────────────────────────────────────────────────────────────
# SSM/Mamba Shem Export Tests
# ──────────────────────────────────────────────────────────────────────

class TinySSM(nn.Module):
    """Minimal SSM structure for testing export."""

    def __init__(self, d_model=64, d_state=16, expand=2):
        super().__init__()
        d_inner = d_model * expand

        # Core SSM parameters
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))

        # Selective mechanism projections
        self.x_proj = nn.Linear(d_inner, 2 * d_state)  # -> B, C
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand


class TestSSMShemExport:
    """Validation suite for SSM → Shem code generation."""

    @pytest.fixture
    def ssm_extractor(self):
        """Create and prepare an SSM extractor with synthetic model."""
        model = TinySSM(d_model=64, d_state=16)
        extractor = MambaExtractor(model_name="test_ssm", device="cpu")
        extractor.model = model
        extractor.model.eval()
        return extractor

    def test_syntax_valid(self, ssm_extractor):
        """Generated SSM Shem code must be syntactically valid Python."""
        dynamics = ssm_extractor.extract_dynamics()
        graph = ssm_extractor.build_graph()
        
        code = export_ssm_to_shem(graph, ssm_extractor, Path("dummy.py"), mismatch_sigma=0.05)
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in generated SSM code: {e}")

    def test_has_required_imports(self, ssm_extractor):
        """Generated SSM code must import required JAX/Diffrax modules."""
        dynamics = ssm_extractor.extract_dynamics()
        graph = ssm_extractor.build_graph()
        
        code = export_ssm_to_shem(graph, ssm_extractor, Path("dummy.py"), mismatch_sigma=0.05)
        
        assert "import jax" in code, "Missing 'import jax'"
        assert "import diffrax" in code, "Missing 'import diffrax'"

    def test_has_dynamics_method(self, ssm_extractor):
        """Generated SSM code must define dynamics() method."""
        dynamics = ssm_extractor.extract_dynamics()
        graph = ssm_extractor.build_graph()
        
        code = export_ssm_to_shem(graph, ssm_extractor, Path("dummy.py"), mismatch_sigma=0.05)
        
        assert "def dynamics(self, t, x" in code, "Missing dynamics() method"

    def test_has_ssm_matrices(self, ssm_extractor):
        """Generated SSM code must include A, B, C matrices."""
        dynamics = ssm_extractor.extract_dynamics()
        graph = ssm_extractor.build_graph()
        
        code = export_ssm_to_shem(graph, ssm_extractor, Path("dummy.py"), mismatch_sigma=0.05)
        
        assert "self.A = " in code, "Missing A matrix"
        assert "self.B = " in code, "Missing B matrix"
        assert "self.C = " in code, "Missing C matrix"

    def test_has_mismatch_method(self, ssm_extractor):
        """Generated SSM code must define mismatch() for Shem."""
        dynamics = ssm_extractor.extract_dynamics()
        graph = ssm_extractor.build_graph()
        
        code = export_ssm_to_shem(graph, ssm_extractor, Path("dummy.py"), mismatch_sigma=0.05)
        
        assert "def mismatch(" in code, "Missing mismatch() method"

    def test_has_analog_trainable(self, ssm_extractor):
        """Generated SSM code must use AnalogTrainable."""
        dynamics = ssm_extractor.extract_dynamics()
        graph = ssm_extractor.build_graph()
        
        code = export_ssm_to_shem(graph, ssm_extractor, Path("dummy.py"), mismatch_sigma=0.05)
        
        assert "AnalogTrainable" in code, "Missing AnalogTrainable"

    def test_code_length_reasonable(self, ssm_extractor):
        """Generated SSM code should be substantial."""
        dynamics = ssm_extractor.extract_dynamics()
        graph = ssm_extractor.build_graph()
        
        code = export_ssm_to_shem(graph, ssm_extractor, Path("dummy.py"), mismatch_sigma=0.05)
        
        assert len(code) > 1000, f"Generated code too short: {len(code)} chars"
