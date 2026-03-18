"""
Neural ODE / CDE extractor — highest analog amenability of all supported families.

Why Neural ODEs map cleanly to analog hardware:
  - The model IS an ODE: dx/dt = f_θ(x, t)  →  identical to Arco/Legno/Shem input format
  - Training uses adjoint equations            →  same math as Shem's gradient computation
  - f_θ is a small MLP (64–256 dim)           →  within demonstrated analog parameter scale
  - Adaptive solver → analog integrator       →  direct substrate mapping

The Shem export for Neural ODEs is complete (not a structural stub) — weights are
extracted from the pretrained model and the generated JAX code runs as a valid
Shem-compatible ODE specification.

Supported models:
  - torchdiffeq-based NeuralODE (Chen et al. 2018, NeurIPS)
  - torchcde-based NeuralCDE  (Kidger et al. 2020, NeurIPS)
  - Any nn.Module with a .dynamics() or .f() method returning dx/dt
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from neuro_analog.ir import (
    AnalogGraph, ArchitectureFamily, DynamicsProfile, OpType, Domain,
    AnalogNode, PrecisionSpec, make_mvm_node, make_activation_node,
    ODESystem, ParameterSpec, NoiseProfile,
)
from neuro_analog.ir.types import AnalogAmenabilityProfile
from .base import BaseExtractor

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Activation classification helpers
# ──────────────────────────────────────────────────────────────────────

_ANALOG_ACTIVATIONS = {"tanh", "sigmoid"}   # Implementable via subthreshold MOSFET diff pair
_HYBRID_ACTIVATIONS = {"relu"}              # Current mirror; accurate but limited range
_DIGITAL_ACTIVATIONS = {"gelu", "silu", "swish", "softplus", "elu", "leaky_relu"}


def _classify_activation(act_name: str) -> tuple[OpType, Domain]:
    name = act_name.lower().replace("-", "_")
    if name in _ANALOG_ACTIVATIONS:
        return OpType.ANALOG_SIGMOID if "sigmoid" in name else OpType.ANALOG_SIGMOID, Domain.ANALOG
    if name in _HYBRID_ACTIVATIONS:
        return OpType.ANALOG_RELU, Domain.ANALOG
    return OpType.SILU, Domain.DIGITAL


def _activation_name_from_module(module: nn.Module) -> str:
    return type(module).__name__.lower()


# ──────────────────────────────────────────────────────────────────────
# Stiffness estimation
# ──────────────────────────────────────────────────────────────────────

def estimate_jacobian_stiffness(
    f_theta: nn.Module,
    state_dim: int,
    num_points: int = 20,
    t_values: Optional[list[float]] = None,
    device: str = "cpu",
) -> dict:
    """Estimate ODE stiffness via Jacobian spectral radius.

    Stiffness ratio = λ_max / λ_min of the Jacobian df/dx.
    High stiffness → analog integrators need tighter time constants.
    Low stiffness → Euler with coarse steps is sufficient.

    Method: finite differences on a sample of random input points.
    Full Jacobian via autograd for small state_dim (≤ 256); FD approximation otherwise.

    Returns:
        {
            "mean_stiffness": float,
            "max_stiffness": float,
            "mean_lambda_max": float,
            "mean_lambda_min": float,
            "is_stiff": bool,  # True if ratio > 1000
        }
    """
    if t_values is None:
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    f_theta.eval()
    stiffness_ratios = []
    lambda_maxs = []
    lambda_mins = []

    with torch.no_grad():
        for _ in range(num_points):
            x = torch.randn(state_dim, device=device)
            for t_val in t_values:
                t = torch.tensor(t_val, device=device)
                try:
                    # Try autograd Jacobian (works for small dims)
                    with torch.enable_grad():
                        x_req = x.detach().requires_grad_(True)
                        t_req = t.detach().requires_grad_(False)
                        # Handle both f(x, t) and f(t, x) call signatures
                        try:
                            out = f_theta(t_req, x_req)
                        except Exception:
                            out = f_theta(x_req, t_req)
                        if out.dim() == 0:
                            continue
                        J = torch.zeros(state_dim, state_dim)
                        for i in range(min(state_dim, out.shape[0])):
                            grad = torch.autograd.grad(
                                out[i], x_req, retain_graph=True, allow_unused=True
                            )[0]
                            if grad is not None:
                                J[i] = grad.detach()
                    # Spectral analysis via singular values (proxy for eigenvalues)
                    sv = torch.linalg.svdvals(J).numpy()
                    sv_nonzero = sv[sv > 1e-10]
                    if len(sv_nonzero) >= 2:
                        ratio = float(sv_nonzero.max() / sv_nonzero.min())
                        stiffness_ratios.append(ratio)
                        lambda_maxs.append(float(sv_nonzero.max()))
                        lambda_mins.append(float(sv_nonzero.min()))
                except Exception as e:
                    log.debug(f"Jacobian computation failed at t={t_val}: {e}")
                    continue

    if not stiffness_ratios:
        return {
            "mean_stiffness": None,
            "max_stiffness": None,
            "mean_lambda_max": None,
            "mean_lambda_min": None,
            "is_stiff": None,
            "note": "Jacobian computation failed — check model call signature",
        }

    return {
        "mean_stiffness": float(np.mean(stiffness_ratios)),
        "max_stiffness": float(np.max(stiffness_ratios)),
        "mean_lambda_max": float(np.mean(lambda_maxs)),
        "mean_lambda_min": float(np.mean(lambda_mins)),
        "is_stiff": float(np.max(stiffness_ratios)) > 1000.0,
    }


# ──────────────────────────────────────────────────────────────────────
# NFE counter
# ──────────────────────────────────────────────────────────────────────

class _NFECounter:
    """Wrap f_θ to count function evaluations during an ODE solve."""

    def __init__(self, f: nn.Module):
        self.f = f
        self.count = 0

    def __call__(self, t, x):
        self.count += 1
        return self.f(t, x)

    def reset(self):
        self.count = 0


# ──────────────────────────────────────────────────────────────────────
# NeuralODEProfile
# ──────────────────────────────────────────────────────────────────────

@dataclass
class NeuralODEProfile:
    """Full characterization of a Neural ODE for analog compilation."""

    # Architecture
    state_dim: int = 0
    augmented_dim: int = 0          # ANODE extra dimensions
    hidden_dims: list[int] = field(default_factory=list)
    num_layers: int = 0
    activation_name: str = "tanh"
    activation_is_analog_native: bool = True  # tanh/sigmoid

    # ODE dynamics
    t_span: tuple[float, float] = (0.0, 1.0)
    num_function_evaluations: int = 0   # Measured from adaptive solver
    stiffness: dict = field(default_factory=dict)

    # Weight statistics (per layer)
    weight_stats: dict[str, dict] = field(default_factory=dict)

    # Analog fit
    total_params: int = 0
    fits_single_crossbar: bool = False  # True if hidden_dim ≤ 256


# ──────────────────────────────────────────────────────────────────────
# NeuralODEExtractor
# ──────────────────────────────────────────────────────────────────────

class NeuralODEExtractor(BaseExtractor):
    """Extract analog compilation parameters from Neural ODE / CDE models.

    Supports three loading modes:

    1. Direct nn.Module:
        extractor = NeuralODEExtractor.from_module(f_theta, state_dim=2, t_span=(0,1))

    2. torchdiffeq NeuralODE wrapper:
        from torchdiffeq import odeint
        extractor = NeuralODEExtractor("path/to/checkpoint.pt")

    3. Named demo model (for testing without pretrained checkpoints):
        extractor = NeuralODEExtractor.demo(state_dim=2, hidden_dim=64)

    The Shem export produced by export_neural_ode_to_shem() is complete —
    weights are extracted from the pretrained model and the generated JAX code
    is a valid Shem ODE specification supporting adjoint-based optimization.
    """

    def __init__(
        self,
        model_name: str = "neural_ode",
        device: str = "cpu",
        state_dim: int = 2,
        t_span: tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(model_name, device)
        self.state_dim = state_dim
        self.t_span = t_span
        self._node_module: Optional[nn.Module] = None
        self._profile: Optional[NeuralODEProfile] = None

    @classmethod
    def from_module(
        cls,
        f_theta: nn.Module,
        state_dim: int,
        t_span: tuple[float, float] = (0.0, 1.0),
        model_name: str = "neural_ode",
        device: str = "cpu",
    ) -> "NeuralODEExtractor":
        """Construct extractor directly from an nn.Module."""
        ext = cls(model_name=model_name, device=device, state_dim=state_dim, t_span=t_span)
        ext.model = f_theta.to(device)
        ext._node_module = f_theta
        return ext

    @classmethod
    def demo(
        cls,
        state_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = "tanh",
        t_span: tuple[float, float] = (0.0, 1.0),
    ) -> "NeuralODEExtractor":
        """Build a small demo Neural ODE for testing without a pretrained checkpoint.

        Produces a valid extractor that exercises the full pipeline including
        the Shem export, mismatch propagation, and noise budget.
        """
        act_map = {
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        ActClass = act_map.get(activation, nn.Tanh)

        layers: list[nn.Module] = []
        in_dim = state_dim + 1  # +1 for time concatenation
        for i, out_dim in enumerate([hidden_dim] * (num_layers - 1) + [state_dim]):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(ActClass())
            in_dim = out_dim

        class TimeAugmentedMLP(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, t, x):
                # Concatenate time as extra input feature
                t_vec = t.expand(*x.shape[:-1], 1) if x.dim() > 1 else t.reshape(1)
                xt = torch.cat([x, t_vec], dim=-1)
                return self.net(xt)

        f_theta = TimeAugmentedMLP(nn.Sequential(*layers))
        ext = cls.from_module(
            f_theta, state_dim=state_dim, t_span=t_span,
            model_name=f"demo_neural_ode_d{state_dim}_h{hidden_dim}",
        )
        return ext

    @property
    def family(self) -> ArchitectureFamily:
        return ArchitectureFamily.NEURAL_ODE

    def load_model(self) -> None:
        """Load from checkpoint file if model_name is a path; otherwise no-op."""
        if self._node_module is not None:
            return  # Already loaded via from_module() or demo()
        try:
            checkpoint = torch.load(self.model_name, map_location=self.device)
            # Handle common checkpoint formats
            if isinstance(checkpoint, nn.Module):
                self.model = checkpoint.to(self.device)
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                raise ValueError(
                    "Checkpoint contains state_dict but no model class. "
                    "Use NeuralODEExtractor.from_module(model) after loading manually."
                )
            else:
                self.model = checkpoint.to(self.device)
        except Exception as e:
            log.warning(f"Could not load checkpoint '{self.model_name}': {e}. "
                        "Use from_module() or demo() instead.")
            self.model = None

    def _get_f_theta(self) -> Optional[nn.Module]:
        """Unwrap the dynamics function from common wrapper patterns."""
        if self._node_module is not None:
            return self._node_module
        if self.model is not None:
            # torchdiffeq NeuralODE wraps f as .odefunc or .func
            for attr in ("odefunc", "func", "f", "dynamics"):
                if hasattr(self.model, attr):
                    return getattr(self.model, attr)
            return self.model
        return None

    def _infer_architecture(self, f_theta: nn.Module) -> NeuralODEProfile:
        """Walk f_θ's modules to infer hidden dims, activation, parameter count."""
        hidden_dims = []
        activation_name = "tanh"
        total_params = sum(p.numel() for p in f_theta.parameters())

        for name, module in f_theta.named_modules():
            if isinstance(module, nn.Linear):
                hidden_dims.append(module.out_features)
            elif type(module).__name__.lower() in {"tanh", "sigmoid", "relu", "gelu", "silu"}:
                activation_name = type(module).__name__.lower()

        # Remove final output dim from hidden_dims
        if len(hidden_dims) >= 2:
            hidden_dims = hidden_dims[:-1]

        activation_is_analog = activation_name in _ANALOG_ACTIVATIONS
        fits_crossbar = all(d <= 256 for d in hidden_dims)

        return NeuralODEProfile(
            state_dim=self.state_dim,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims) + 1,
            activation_name=activation_name,
            activation_is_analog_native=activation_is_analog,
            t_span=self.t_span,
            total_params=total_params,
            fits_single_crossbar=fits_crossbar,
        )

    def extract_dynamics(self) -> DynamicsProfile:
        """Compute ODE stiffness and NFE, producing a DynamicsProfile."""
        f_theta = self._get_f_theta()
        if f_theta is None:
            return DynamicsProfile(has_dynamics=True, dynamics_type="time_varying_ODE")

        stiffness = estimate_jacobian_stiffness(
            f_theta, self.state_dim, num_points=10, device=self.device
        )

        # Count NFE by wrapping f_theta and doing a simple Euler solve
        nfe = self._measure_nfe(f_theta)

        # Attempt torchdiffeq adaptive solve for real NFE measurement
        try:
            from torchdiffeq import odeint
            counter = _NFECounter(f_theta)
            x0 = torch.zeros(self.state_dim, device=self.device)
            t_span = torch.tensor(list(self.t_span), device=self.device)
            with torch.no_grad():
                odeint(counter, x0, t_span, method="dopri5", rtol=1e-3, atol=1e-5)
            nfe = counter.count
        except ImportError:
            log.debug("torchdiffeq not available; using Euler NFE estimate")
        except Exception as e:
            log.debug(f"Adaptive NFE measurement failed: {e}")

        return DynamicsProfile(
            has_dynamics=True,
            dynamics_type="time_varying_ODE",
            num_function_evaluations=nfe,
            stiffness_ratio=stiffness.get("mean_stiffness"),
            lipschitz_constant=stiffness.get("mean_lambda_max"),
        )

    def _measure_nfe(self, f_theta: nn.Module, n_steps: int = 100) -> int:
        """Estimate NFE for a fixed-step Euler solve as a lower bound."""
        return n_steps  # Euler uses exactly n_steps evaluations

    def extract_weight_stats(self, f_theta: nn.Module) -> dict[str, dict]:
        """Extract per-layer weight distribution statistics."""
        stats = {}
        for name, param in f_theta.named_parameters():
            data = param.detach().float()
            max_abs = float(data.abs().max())
            std = float(data.std())
            noise_floor = max(std / 1000.0, 1e-9)
            dr_db = 20.0 * math.log10(max_abs / noise_floor) if max_abs > 0 else 0.0
            bits = max(4, min(16, math.ceil(dr_db / 6.02)))
            stats[name] = {
                "shape": tuple(data.shape),
                "min": float(data.min()),
                "max": float(data.max()),
                "std": std,
                "mean": float(data.mean()),
                "required_bits": bits,
                "dynamic_range_db": dr_db,
                "sparsity": float((data.abs() < 1e-6).float().mean()),
            }
        return stats

    def build_graph(self) -> AnalogGraph:
        """Decompose f_θ into an AnalogGraph of analog/digital primitives.

        For a typical tanh-MLP Neural ODE with layers [2→64→64→2]:

            INPUT (state x, dim=2)
              ↓ [DAC boundary]
            MVM (W1: 3→64, analog crossbar — time concatenated to input)
              ↓ [no boundary — tanh is analog-native]
            tanh (ANALOG: subthreshold MOSFET diff pair)
              ↓
            MVM (W2: 64→64, analog crossbar)
              ↓
            tanh (ANALOG)
              ↓
            MVM (W3: 64→2, analog crossbar)
              ↓ [ADC boundary]
            INTEGRATION (ODE Euler step: x += dt * dx/dt, ANALOG capacitor)
              ↓ [loop back — analog feedback]
        """
        f_theta = self._get_f_theta()
        assert f_theta is not None, "No model loaded — call load_model() or use from_module()"

        profile = self._infer_architecture(f_theta)
        self._profile = profile
        weight_stats = self.extract_weight_stats(f_theta)

        total_params = sum(p.numel() for p in f_theta.parameters())
        graph = AnalogGraph(
            name=self.model_name,
            family=ArchitectureFamily.NEURAL_ODE,
            model_params=total_params,
        )

        # Walk the actual nn.Module tree to build nodes
        linear_layers = [(n, m) for n, m in f_theta.named_modules() if isinstance(m, nn.Linear)]
        act_layers = [
            (n, m) for n, m in f_theta.named_modules()
            if isinstance(m, (nn.Tanh, nn.Sigmoid, nn.ReLU, nn.GELU, nn.SiLU, nn.ELU, nn.LeakyReLU))
        ]

        prev_node_id: Optional[str] = None

        for idx, (name, linear) in enumerate(linear_layers):
            in_f, out_f = linear.in_features, linear.out_features
            ws = weight_stats.get(f"{name}.weight", {})
            bits = ws.get("required_bits", 8)
            precision = PrecisionSpec(weight_bits=bits, activation_bits=8, accumulator_bits=bits + 8,
                                      weight_min=ws.get("min", 0.0), weight_max=ws.get("max", 0.0),
                                      weight_std=ws.get("std", 0.0))

            node = make_mvm_node(f"f_theta.linear_{idx}", in_f, out_f, precision=precision)
            node.metadata["layer_name"] = name
            node.metadata["fits_single_crossbar"] = (in_f <= 256 and out_f <= 256)
            node.metadata["weight_required_bits"] = bits
            nid = graph.add_node(node)
            if prev_node_id:
                graph.add_edge(prev_node_id, nid)
            prev_node_id = nid

            # Insert activation after this linear if there's a matching one
            if idx < len(act_layers):
                act_name, act_mod = act_layers[idx]
                act_str = _activation_name_from_module(act_mod)
                op_type, domain = _classify_activation(act_str)
                act_node = AnalogNode(
                    name=f"f_theta.act_{idx}",
                    op_type=op_type,
                    domain=domain,
                    input_shape=(out_f,),
                    output_shape=(out_f,),
                    flops=out_f,
                    metadata={"activation": act_str, "analog_native": domain == Domain.ANALOG},
                )
                anid = graph.add_node(act_node)
                graph.add_edge(prev_node_id, anid)
                prev_node_id = anid

        # ODE integration node — the outer loop step x_{t+dt} = x_t + dt * f_theta(x_t, t)
        # This maps to a capacitor-based integrator: charge accumulation over dt.
        int_node = AnalogNode(
            name="ode_integrator",
            op_type=OpType.INTEGRATION,
            domain=Domain.ANALOG,
            input_shape=(self.state_dim,),
            output_shape=(self.state_dim,),
            flops=2 * self.state_dim,
            metadata={
                "description": "x_{t+dt} = x_t + dt * f_theta — capacitor charge integration",
                "t_start": self.t_span[0],
                "t_end": self.t_span[1],
                "readout_time": self.t_span[1],
            },
        )
        int_id = graph.add_node(int_node)
        if prev_node_id:
            graph.add_edge(prev_node_id, int_id)

        return graph

    def calibrate_activations(
        self,
        calibration_data: torch.Tensor | None = None,
        percentile: float = 99.9,
    ) -> dict:
        """Override: calibrate by sampling f_theta(t, z) at multiple time points.

        The dynamics network has signature ``forward(t, z)``, so a plain
        ``model(calibration_data)`` call would fail.  Instead, we register hooks
        on the leaf modules of f_theta and run it at 5 uniformly-spaced t values
        across ``t_span``, collecting the full activation distribution at each step.

        Args:
            calibration_data: Optional float tensor of shape ``(N, state_dim)``
                used as the z batch.  If None or shape-mismatched, falls back to
                ``torch.randn(32, state_dim)``.  Passing actual data samples (e.g.
                the training set) gives more representative activation statistics
                than random z.
            percentile: Same semantics as BaseExtractor — clips at the given
                upper/lower percentile to suppress outliers.
        """
        from neuro_analog.ir import PrecisionSpec
        f_theta = self._get_f_theta()
        assert f_theta is not None, "No model loaded — use from_module() or demo()"

        # Build z batch
        if (
            calibration_data is not None
            and calibration_data.is_floating_point()
            and calibration_data.shape[-1] == self.state_dim
        ):
            z_batch = calibration_data.float().to(self.device)
        else:
            z_batch = torch.randn(32, self.state_dim, device=self.device)

        # Sample t values uniformly across t_span (5 points covers trajectory well)
        t0, t1 = self.t_span
        t_samples = torch.linspace(t0, t1, 5, device=self.device)

        activation_vals: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def make_hook(name):
            def hook_fn(mod, inp, out):
                if isinstance(out, torch.Tensor):
                    activation_vals.setdefault(name, []).append(
                        out.detach().float().reshape(-1)
                    )
            return hook_fn

        for name, module in f_theta.named_modules():
            if not list(module.children()):
                hooks.append(module.register_forward_hook(make_hook(name)))

        f_theta.eval()
        with torch.no_grad():
            for t_val in t_samples:
                # _TimeAugMLP.forward(t, z): t can be scalar or (batch,)
                t_expanded = t_val.expand(z_batch.shape[0])
                try:
                    f_theta(t_expanded, z_batch)
                except TypeError:
                    f_theta(t_val, z_batch)  # fallback: scalar t

        for h in hooks:
            h.remove()

        lo_q = (100.0 - percentile) / 100.0
        hi_q = percentile / 100.0
        specs = {}
        for name, tensors in activation_vals.items():
            all_vals = torch.cat(tensors)
            act_min = float(torch.quantile(all_vals, lo_q))
            act_max = float(torch.quantile(all_vals, hi_q))
            act_std = float(all_vals.std())
            crest = act_max / max(act_std, 1e-10) if act_max > 0 else 1.0
            act_bits = max(4, min(16, math.ceil(math.log2(max(crest, 1.0))) + 4))
            specs[name] = PrecisionSpec(
                activation_min=act_min,
                activation_max=act_max,
                activation_std=act_std,
                activation_bits=act_bits,
            )
        return specs

    def run(self, calibration_data: torch.Tensor | None = None) -> AnalogAmenabilityProfile:
        """Full pipeline: load → extract dynamics → build graph → analyze.

        Args:
            calibration_data: Optional float tensor of shape ``(N, state_dim)``.
                Used to calibrate per-layer activation ranges via multi-timestep
                sampling of f_theta(t, z).  Feeds into precision_score via
                ``min_activation_precision_bits``.
        """
        print(f"[neuro-analog] Neural ODE: {self.model_name}")
        self.load_model()
        dynamics = self.extract_dynamics()
        graph = self.build_graph()
        graph.set_dynamics(dynamics)
        self._graph = graph
        if calibration_data is not None:
            print("[neuro-analog] Calibrating activations (multi-timestep)...")
            self._activation_specs = self.calibrate_activations(calibration_data)
        profile = graph.analyze()
        profile = self._apply_activation_specs(profile)
        print(f"[neuro-analog] Done. Score: {profile.overall_score:.3f}")
        return profile

    def extract_ode_system(self) -> ODESystem:
        """Extract a complete ODESystem from this Neural ODE.

        This is the primary interface between extraction and simulation/export.
        For Neural ODEs, the ODE parameters ARE the MLP weights of f_θ, so
        δ on Linear weights IS δ on ODE parameters — the same thing.

        Each weight tensor becomes a ParameterSpec with:
          - Bounds based on weight distribution (±3σ from mean as physical range)
          - analog_primitive = "crossbar_conductance" for weights
          - trainable = True (Shem should optimize all weights)

        The dynamics_module is f_θ itself, and dynamics_fn = f_θ.forward.
        """
        f_theta = self._get_f_theta()
        assert f_theta is not None, "No model loaded"

        # If we haven't run the full pipeline yet, do minimal extraction
        if self._profile is None:
            self._profile = self._infer_architecture(f_theta)

        params: dict[str, ParameterSpec] = {}
        for name, param in f_theta.named_parameters():
            data = param.detach().float()
            std = float(data.std())
            # Physical bounds: analog conductance can't be arbitrarily large.
            # Use ±4σ from mean as a conservative physical range.
            mean_val = float(data.mean())
            bound_range = max(4.0 * std, 0.5)  # at least ±0.5
            bounds = (mean_val - bound_range, mean_val + bound_range)

            primitive = "crossbar_conductance"
            if "bias" in name:
                primitive = "crossbar_bias_current"

            params[name] = ParameterSpec(
                name=name,
                value=data.clone(),
                bounds=bounds,
                mismatch_sigma=0.05,
                trainable=True,
                analog_primitive=primitive,
            )

        # Transient noise: sqrt(kT/C) for a 1pF integration cap at 300K
        # kT/C = 1.38e-23 * 300 / 1e-12 = 4.14e-12 → sqrt ≈ 2.03e-6
        # This is tiny compared to mismatch but physically correct.
        kt_over_c = 1.380649e-23 * 300.0 / 1e-12
        transient_sigma = math.sqrt(kt_over_c)

        stiffness = self._profile.stiffness if self._profile else {}

        return ODESystem(
            name=self.model_name,
            family="neural_ode",
            state_dim=self.state_dim,
            parameters=params,
            dynamics_fn=f_theta,
            dynamics_module=f_theta,
            readout_times=[self.t_span[1]],
            t_span=self.t_span,
            noise=NoiseProfile(
                sigma=transient_sigma,
                bandwidth_hz=250e3,  # typical analog bandwidth
            ),
            metadata={
                "activation": self._profile.activation_name if self._profile else "tanh",
                "hidden_dims": self._profile.hidden_dims if self._profile else [],
                "stiffness": stiffness,
                "fits_single_crossbar": self._profile.fits_single_crossbar if self._profile else False,
            },
        )


# ──────────────────────────────────────────────────────────────────────
# Shem export for Neural ODEs
# ──────────────────────────────────────────────────────────────────────

def export_neural_ode_to_shem(
    extractor,
    output_path,
    mismatch_sigma: float = 0.05,
) -> str:
    """Generate a Shem-compatible JAX class from a NeuralODEExtractor.
    Complies with BaseAnalogCkt interface with full param flattening.
    """
    import torch.nn as nn
    from pathlib import Path

    f_theta = extractor._get_f_theta()
    assert f_theta is not None, "Run extractor.load_model() or use from_module() first"

    model_name = extractor.model_name
    state_dim = extractor.state_dim
    t0, t1 = extractor.t_span

    linear_layers = [(n, m) for n, m in f_theta.named_modules() if isinstance(m, nn.Linear)]
    act_layers = [
        (n, m) for n, m in f_theta.named_modules()
        if isinstance(m, (nn.Tanh, nn.Sigmoid, nn.ReLU, nn.GELU, nn.SiLU))
    ]

    def _activation_jax(module: nn.Module) -> str:
        name = type(module).__name__.lower()
        return {
            "tanh": "jnp.tanh",
            "sigmoid": "jax.nn.sigmoid",
            "relu": "jax.nn.relu",
            "gelu": "jax.nn.gelu",
            "silu": "jax.nn.silu",
        }.get(name, "jnp.tanh")

    # Compute static per-tensor offsets at export time (shapes are known)
    import numpy as _np
    shapes = []
    for idx, (name, linear) in enumerate(linear_layers):
        shapes.append(linear.weight.detach().float().numpy().shape)
        if linear.bias is not None:
            shapes.append(linear.bias.detach().float().numpy().shape)
    offsets = []
    _off = 0
    for s in shapes:
        offsets.append(_off)
        _off += int(_np.prod(s))
    total_params = _off

    lines = [
        f'"""',
        f"Ark-compatible JAX ODE specification for {model_name}.",
        f"Proper BaseAnalogCkt subclass — compatible with OptCompiler output format.",
        f"",
        f"Usage:",
        f"    from ark.optimization.base_module import TimeInfo",
        f"    ckt = NeuralODEAnalogCkt()",
        f"    time_info = TimeInfo(t0={t0}, t1={t1}, dt0=0.01, saveat=jnp.array([{t1}]))",
        f"    result = ckt(time_info, x0, switch=jnp.array([]), args_seed=42, noise_seed=43)",
        f'"""',
        f"",
        f"import jax",
        f"import jax.numpy as jnp",
        f"import jax.random as jrandom",
        f"import diffrax",
        f"from ark.optimization.base_module import BaseAnalogCkt, TimeInfo",
        f"",
        f"class NeuralODEAnalogCkt(BaseAnalogCkt):",
        f'    """Neural ODE analog circuit (BaseAnalogCkt subclass).',
        f"",
        f"    a_trainable holds all MLP weights concatenated flat.",
        f"    make_args applies multiplicative mismatch (delta ~ N(1, sigma^2))",
        f"    and returns per-layer (W, b) tuples for ode_fn.",
        f'    """',
        f"    shapes: list",
        f"",
        f"    def __init__(self):",
        f"        arrays = []",
        f"        shapes = []",
    ]

    shapes_reset = []  # rebuild for the loop below
    for idx, (name, linear) in enumerate(linear_layers):
        W = linear.weight.detach().float().numpy()
        b = linear.bias.detach().float().numpy() if linear.bias is not None else None

        lines.append(f"        _W{idx} = jnp.array({W.tolist()})")
        lines.append(f"        arrays.append(_W{idx}.flatten())")
        lines.append(f"        shapes.append(_W{idx}.shape)")
        shapes_reset.append(W.shape)
        if b is not None:
            lines.append(f"        _b{idx} = jnp.array({b.tolist()})")
            lines.append(f"        arrays.append(_b{idx}.flatten())")
            lines.append(f"        shapes.append(_b{idx}.shape)")
            shapes_reset.append(b.shape)

    lines += [
        f"        a_trainable = jnp.concatenate(arrays)",
        f"        # Store shapes before super().__init__ freezes the module",
        f"        object.__setattr__(self, 'shapes', shapes)",
        f"        super().__init__(",
        f"            init_trainable=a_trainable,",
        f"            is_stochastic=False,",
        f"            solver=diffrax.Heun(),  # Heun provides error estimates for PIDController",
        f"        )",
        f"",
        f"    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):",
        f"        # mismatch_seed matches BaseAnalogCkt.__call__ args_seed parameter",
        f"        key = jrandom.PRNGKey(mismatch_seed)",
        f"        keys = jrandom.split(key, {len(shapes_reset)})",
        f"        sigma = {mismatch_sigma}",
    ]

    # Emit static-offset slice for each tensor — no Python loop at JAX trace time
    for i, (shape, off) in enumerate(zip(shapes_reset, offsets)):
        size = int(_np.prod(shape))
        lines.append(
            f"        _p{i} = (self.a_trainable[{off}:{off+size}]"
            f" * (1.0 + sigma * jrandom.normal(keys[{i}], ({size},)))).reshape({shape})"
        )

    param_tuple = ", ".join(f"_p{i}" for i in range(len(shapes_reset)))
    lines += [
        f"        return ({param_tuple},)",
        f"",
        f"    def ode_fn(self, t, x, args):",
    ]

    unpack_str = ", ".join(
        [f"W{idx}" + (f", b{idx}" if linear_layers[idx][1].bias is not None else "")
         for idx in range(len(linear_layers))]
    )
    if not unpack_str:
        lines.append(f"        pass  # No layers extracted")
    else:
        lines.append(f"        {unpack_str} = args")

    # f_theta takes [x, t] concatenated — prepend time feature
    lines.append(f"        xt = jnp.concatenate([x, jnp.atleast_1d(jnp.asarray(t, dtype=x.dtype))])")

    prev = "xt"
    for idx, (name, linear) in enumerate(linear_layers):
        has_bias = linear.bias is not None
        bias_str = f" + b{idx}" if has_bias else ""
        act_jax = _activation_jax(act_layers[idx][1]) if idx < len(act_layers) else None

        if act_jax:
            lines.append(f"        h{idx} = {act_jax}(W{idx} @ {prev}{bias_str})")
            prev = f"h{idx}"
        else:
            lines.append(f"        out = W{idx} @ {prev}{bias_str}")
            prev = "out"
            
    lines += [
        f"        return {prev}",
        f"",
        f"    def noise_fn(self, t, x, args):",
        f"        # Small thermal noise amplitude — only used when is_stochastic=True",
        f"        return jnp.ones_like(x) * 0.01",
        f"",
        f"    def readout(self, y):",
        f"        # y shape: (len(saveat), state_dim) — return final time point",
        f"        return y[-1]",
        f"",
        f"if __name__ == '__main__':",
        f"    ckt = NeuralODEAnalogCkt()",
        f"    time_info = TimeInfo(t0={t0}, t1={t1}, dt0=0.01, saveat=jnp.array([{t1}]))",
        f"    x0 = jnp.zeros(({state_dim},))",
        f"    switch = jnp.array([])",
        f"    result = ckt(time_info, x0, switch, args_seed=42, noise_seed=43)",
        f"    print(f'Result shape: {{result.shape}}')",
        f""
    ]

    code = "\n".join(lines) + "\n"
    Path(output_path).write_text(code, encoding="utf-8")
    try:
        from loguru import logger as log
        log.info(f"Shem export written to {output_path} ({len(code)} chars)")
    except ImportError:
        pass
    return code
