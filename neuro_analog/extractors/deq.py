"""
Deep Equilibrium Model (DEQ) analog extractor.

A DEQ finds z* satisfying z* = f_θ(z*, x) via fixed-point iteration.
In analog hardware, a feedback circuit that settles to equilibrium IS a DEQ
solver — the physics performs the root-finding without any digital iteration.

Analog DEQ mapping:
  - The f_θ block (MLP/transformer) maps to crossbar MVMs + analog activations
  - The feedback path (z → f_θ input) stays analog — no ADC/DAC needed at the loop
  - The circuit settles to z* on the RC time constant of the integrator nodes
  - D/A boundaries: only at input (x injected via DAC) and output (z* read via ADC)

Critical stability condition:
  The analog circuit converges iff the spectral radius ρ(∂f/∂z) < 1.
  Under mismatch, even a nominally stable DEQ can have ρ ≥ 1 and diverge.
  This is the DEQ's key analog insight — quantified by extract_spectral_radius().

DEQ → Ark mapping:
  z* = f_θ(z*, x) ↔ dz/dt = f_θ(z, x) - z  [gradient flow to fixed point]
  At equilibrium: dz/dt = 0 → z* = f_θ(z*, x) ✓
  This ODE form is native Arco/Legno format, and Ark can optimize θ for
  mismatch-robust convergence (pushing ρ further below 1).

Reference: Bai et al. 2019, "Deep Equilibrium Models" (NeurIPS 2019).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn

from neuro_analog.ir import (
    AnalogGraph, ArchitectureFamily, DynamicsProfile, Domain,
    AnalogNode, PrecisionSpec, make_mvm_node, make_activation_node,
)
from neuro_analog.ir.types import OpType
from .base import BaseExtractor


@dataclass
class DEQConfig:
    """Configuration for a Deep Equilibrium Model."""
    z_dim: int = 64           # Fixed-point state dimension
    x_dim: int = 64           # Input injection dimension
    hidden_dim: int = 128     # f_θ hidden layer width
    n_hidden_layers: int = 1  # Depth of f_θ MLP (1 = one hidden layer)
    activation: str = "tanh"  # Activation in f_θ
    max_iter: int = 50        # Fixed-point iteration limit
    tol: float = 1e-4         # Convergence tolerance


class DEQExtractor(BaseExtractor):
    """Extract analog-relevant parameters from Deep Equilibrium Models.

    DEQ analog mapping:
    - Fixed-point iteration z_{k+1} = f_θ(z_k, x) maps to a feedback
      analog circuit that settles to equilibrium
    - f_θ is typically a transformer/MLP block — same analog/digital
      partition as the transformer extractor
    - Convergence rate determines analog settling time (analog integration time)
    - Anderson/Broyden solver → analog circuit's natural settling IS the solver

    Key extraction targets:
    - f_θ structure (MLP layers, sizes, activations)
    - Spectral radius ρ(∂f/∂z) at equilibrium (must be < 1 for convergence)
    - Number of iterations to convergence (digital) vs settling time (analog)
    - Sensitivity of z* to weight perturbation (mismatch resilience)
    """

    def __init__(
        self,
        config: Optional[DEQConfig] = None,
        model_name: str = "theoretical_deq",
        device: str = "cpu",
    ):
        super().__init__(model_name, device)
        self.config = config or DEQConfig()

    @classmethod
    def reference(
        cls,
        z_dim: int = 64,
        x_dim: int = 64,
        hidden_dim: int = 128,
    ) -> "DEQExtractor":
        """Create a reference DEQ (no pretrained model, theoretical structure)."""
        return cls(
            DEQConfig(z_dim=z_dim, x_dim=x_dim, hidden_dim=hidden_dim),
            model_name=f"DEQ_{z_dim}d",
        )

    @property
    def family(self) -> ArchitectureFamily:
        return ArchitectureFamily.DEQ

    def load_model(self) -> None:
        """No-op for theoretical extractor."""
        self.model = self.config

    def extract_dynamics(self) -> DynamicsProfile:
        """DEQ dynamics: implicit fixed-point equilibrium.

        The circuit settles to z* via its natural RC time constant — no
        digital iteration needed. Convergence is guaranteed iff ρ(∂f/∂z) < 1.

        is_stochastic=False: unlike EBM, DEQ is deterministic given fixed θ.
        """
        return DynamicsProfile(
            has_dynamics=True,
            dynamics_type="implicit_equilibrium",
            is_stochastic=False,
            state_dimension=self.config.z_dim,
        )

    def extract_spectral_radius(
        self,
        f_theta: nn.Module,
        z_star: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        """Compute spectral radius ρ(∂f/∂z) at equilibrium.

        The spectral radius is the largest absolute eigenvalue of the Jacobian
        ∂f_θ/∂z evaluated at z = z*. It determines:
        - ρ < 1: circuit converges (analog-stable)
        - ρ = 1: marginal stability (sensitive to mismatch)
        - ρ > 1: circuit diverges (analog-unstable)

        Under mismatch σ, a nominal ρ_0 can become ρ_0 + σ * ρ_0 * C for
        some architecture-dependent constant C — pushing stable DEQs past 1.

        Computed via autograd: ∂f/∂z is a z_dim × z_dim matrix.
        """
        z_req = z_star.detach().requires_grad_(True)
        f_out = f_theta(z_req, x)

        # Build Jacobian row by row (z_dim × z_dim)
        jac_rows = []
        for i in range(self.config.z_dim):
            grad = torch.autograd.grad(
                f_out[i], z_req,
                create_graph=False, retain_graph=(i < self.config.z_dim - 1),
            )[0]
            jac_rows.append(grad.detach())
        J = torch.stack(jac_rows, dim=0)  # (z_dim, z_dim)

        # Spectral radius = max |eigenvalue|
        eigvals = torch.linalg.eigvals(J)
        return float(eigvals.abs().max().item())

    def build_graph(self) -> AnalogGraph:
        """Build AnalogGraph for the DEQ architecture.

        Graph structure:
          [INPUT x] → W_inject (MVM) → [+z feedback] → W_hidden (MVM) →
          activation → W_output (MVM) → [z* readout]

        The feedback loop (z* → input of W_inject) stays analog — no ADC/DAC.
        D/A boundaries:
          - Input: DAC converts digital x to analog voltages
          - Output: ADC converts z* to digital for downstream use
          - The fixed-point loop itself: purely analog (0 boundaries inside)
        """
        cfg = self.config

        # Total parameters in f_θ MLP
        # W_z: z_dim → hidden_dim, W_x: x_dim → hidden_dim (injection)
        # W_out: hidden_dim → z_dim
        total_params = (
            cfg.z_dim * cfg.hidden_dim      # W_z
            + cfg.x_dim * cfg.hidden_dim    # W_x
            + cfg.hidden_dim * cfg.z_dim    # W_out
            + cfg.hidden_dim + cfg.z_dim    # biases
        )

        graph = AnalogGraph(
            name=f"DEQ_{cfg.z_dim}d_{cfg.hidden_dim}h",
            family=ArchitectureFamily.DEQ,
            model_params=total_params,
        )

        # z-state injection (crossbar MVM): W_z · z_k
        graph.add_node(make_mvm_node("deq.W_z", cfg.z_dim, cfg.hidden_dim))

        # x injection (crossbar MVM): W_x · x (additive to W_z output)
        graph.add_node(make_mvm_node("deq.W_x", cfg.x_dim, cfg.hidden_dim))

        # Activation: tanh (analog diff pair)
        graph.add_node(make_activation_node("deq.act", cfg.hidden_dim, cfg.activation))

        # Output projection: W_out · h → z_{k+1}
        graph.add_node(make_mvm_node("deq.W_out", cfg.hidden_dim, cfg.z_dim))

        # Feedback node: z_{k+1} fed back to W_z input (analog feedback path)
        feedback_node = AnalogNode(
            name="deq.feedback",
            op_type=OpType.INTEGRATION,   # Models the analog state holding between iterations
            domain=Domain.ANALOG,
            input_shape=(cfg.z_dim,),
            output_shape=(cfg.z_dim,),
            flops=2 * cfg.z_dim,  # multiply + accumulate per state variable
            metadata={
                "description": "Analog state register: z held on capacitors between iterations",
                "convergence_criterion": f"||z_{{k+1}} - z_k|| < {cfg.tol}",
                "max_iterations": cfg.max_iter,
                "da_boundaries_in_loop": 0,
                "analog_settling_note": (
                    "In hardware: the loop settles to z* on the RC time constant "
                    "of the integrator node. No digital iteration needed."
                ),
            },
        )
        graph.add_node(feedback_node)

        # Wiring: W_z → act (additive sum with W_x — modeled as sequential)
        graph.add_edge("deq.W_z", "deq.act")
        # W_x also feeds into act (both sum before activation — shown sequentially)
        graph.add_edge("deq.W_x", "deq.act")
        graph.add_edge("deq.act", "deq.W_out")
        graph.add_edge("deq.W_out", "deq.feedback")
        # Feedback edge: feedback → W_z (closes the analog loop)
        # Not added as a DAG edge (would create a cycle) but noted in metadata
        graph.get_node("deq.feedback").metadata["analog_feedback_to"] = "deq.W_z"
        graph.get_node("deq.W_z").metadata["analog_feedback_from"] = "deq.feedback"

        self._graph = graph
        return graph


class DEQMLPExtractor:
    """Extractor and Ark exporter for the DEQ in experiments/cross_arch_tolerance.

    Targets _DEQClassifier: f_theta(z, x) = tanh(W_z @ z + W_x @ x + b_x)
    with spectral-normalised W_z, z_dim=64, x_dim=64.

    The gradient-flow ODE dz/dt = f_theta(z, x) - z settles to z* = f_theta(z*, x).

    Usage:
        ext = DEQMLPExtractor()
        ext.load_model()
        code = ext.export_to_ark("outputs/deq_ark.py", mismatch_sigma=0.05)
    """

    _EXP_DIR = (
        __import__("pathlib").Path(__file__).parent.parent.parent
        / "experiments" / "cross_arch_tolerance"
    )

    def __init__(self, checkpoint_path=None, z_dim: int = 64, x_dim: int = 64):
        if checkpoint_path is None:
            checkpoint_path = self._EXP_DIR / "checkpoints" / "deq.pt"
        self.checkpoint_path = __import__("pathlib").Path(checkpoint_path)
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.model = None

    def load_model(self):
        import sys
        sys.path.insert(0, str(self._EXP_DIR))
        import models.deq as deq_module
        if self.checkpoint_path.exists():
            self.model = deq_module.load_model(str(self.checkpoint_path))
        else:
            self.model = deq_module.create_model()
            deq_module.train_model(self.model, str(self.checkpoint_path))

    def export_to_ark(self, output_path, mismatch_sigma: float = 0.05) -> str:
        from neuro_analog.ark_bridge.deq_cdg import export_deq_to_ark
        assert self.model is not None, "Call load_model() first"
        return export_deq_to_ark(
            deq_model=self.model,
            output_path=output_path,
            mismatch_sigma=mismatch_sigma,
            class_name="DEQAnalogCkt",
            z_dim=self.z_dim,
            x_dim=self.x_dim,
        )
