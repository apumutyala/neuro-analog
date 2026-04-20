"""
Flow model extractor for FLUX.1 and Stable Diffusion 3.

The flow matching ODE dx/dt = v_θ(x,t) is structurally the simplest
mapping to Ark's analog ODE compiler. Key extractions:
- Velocity field Lipschitz constant (ODE stiffness)
- Flow straightness (Euler integrator sufficiency)
- Number of function evaluations (integration steps)
- MMDiT dual-stream architecture partition

Connection to Ark analog compiler:
  Flow: dx/dt = v_θ(x, t)
  Arco: rel deriv(x, t) = f(x, t)
  
  These are formally identical. The gap: v_θ is a 12B-param network,
  not a symbolic expression.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

from neuro_analog.ir import (
    AnalogGraph, ArchitectureFamily, DynamicsProfile, OpType, Domain,
    AnalogNode, make_mvm_node, make_activation_node, make_norm_node,
)
from .base import BaseExtractor


class FLUXExtractor(BaseExtractor):
    """Extract analog parameters from FLUX.1 flow models.
    
    FLUX.1 architecture:
    - 19 double-stream blocks (separate text/image streams, joint attention)
    - 38 single-stream blocks (concatenated tokens, standard transformer)
    - ~12B parameters total
    - Flow matching ODE with 4-28 integration steps
    
    Supports: FLUX.1-schnell (4 step, Apache 2.0), FLUX.1-dev (28 step)
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        img_size: int = 32,
        in_channels: int = 3,
        seq_len: int | None = None,
    ):
        super().__init__(model_name, device, seq_len=seq_len)
        self.img_size = img_size
        self.in_channels = in_channels
    
    @property
    def family(self) -> ArchitectureFamily:
        return ArchitectureFamily.FLOW
    
    def load_model(self) -> None:
        """Load FLUX transformer backbone from diffusers."""
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.float32,
        )
        self.pipe = pipe
        self.model = pipe.transformer.to(self.device)
        self.scheduler = pipe.scheduler
        self.model.eval()
    
    def extract_dynamics(self) -> DynamicsProfile:
        """Extract flow ODE characterization.
        
        For flow models, the key dynamics parameters are:
        - NFE (number of function evaluations / integration steps)
        - Flow straightness (how linear the ODE trajectories are)
        - Velocity field Lipschitz constant (stability/stiffness)
        """
        # Determine NFE from model variant
        is_schnell = "schnell" in self.model_name.lower()
        nfe = 4 if is_schnell else 28
        
        return DynamicsProfile(
            has_dynamics=True,
            dynamics_type="time_varying_ODE",
            num_function_evaluations=nfe,
            # These require calibration data to compute accurately:
            lipschitz_constant=None,    # Compute via Jacobian spectral norm
            flow_straightness=None,     # Compute via trajectory curvature
        )
    
    def measure_flow_straightness(
        self,
        num_trajectories: int = 100,
        num_timesteps: int = 50,
    ) -> dict:
        """Measure flow straightness by sampling ODE trajectories.
        
        For rectified flows, trajectories should be nearly straight lines.
        Straightness = 1 means perfect linearity (single Euler step sufficient).
        
        Straightness metric: ||x_1 - x_0||² / ∫||v_θ(x_t,t)||² dt
        For a perfectly straight flow, this ratio equals 1.
        
        Returns dict with per-trajectory straightness values and statistics.
        """
        # TODO: Implement with actual model inference
        # Requires: sample z ~ N(0,I), integrate ODE, measure curvature
        return {
            "mean_straightness": None,
            "std_straightness": None,
            "per_trajectory": [],
            "note": "Requires GPU inference — run with calibration data",
        }
    
    def estimate_lipschitz_constant(
        self,
        num_samples: int = 50,
        timesteps: list[float] | None = None,
    ) -> dict:
        """Estimate velocity field Lipschitz constant via finite differences.
        
        L = max_t sup_{x≠x'} ||v_θ(x,t) - v_θ(x',t)|| / ||x - x'||
        
        Approximated by:
        1. Sample pairs (x, x+ε) for small ε
        2. Compute ||v_θ(x+ε,t) - v_θ(x,t)|| / ||ε||
        3. Take maximum over samples and timesteps
        
        High L → stiff ODE → needs more integration steps or higher-order solver.
        Low L → easy ODE → Euler with few steps sufficient.
        """
        # TODO: Implement with actual model inference
        return {
            "estimated_lipschitz": None,
            "per_timestep_lipschitz": [],
            "note": "Requires GPU inference",
        }
    
    def _get_seq_len(self) -> int:
        """Get effective sequence length for FLOP calculations.
        
        Uses self.seq_len if provided, otherwise defaults to 1024 (FLUX standard).
        """
        if self.seq_len is not None:
            return self.seq_len
        return 1024  # FLUX.1 standard sequence length

    def build_graph(self) -> AnalogGraph:
        """Build AnalogGraph for FLUX.1 MMDiT architecture.

        Structure:
        - 19 double-stream blocks (separate text/image, joint attention)
        - 38 single-stream blocks (standard transformer)
        - ODE integration loop (4 or 28 steps)
        """
        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 12_000_000_000
        graph = AnalogGraph(
            name=self.model_name,
            family=ArchitectureFamily.FLOW,
            model_params=total_params,
        )

        # Model dimensions (FLUX.1)
        dim = 3072  # Hidden dimension
        heads = 24
        head_dim = 128
        mlp_dim = dim * 4
        seq_len = self._get_seq_len()
        seq_len_sq = seq_len * seq_len
        
        # Patch embedding
        graph.add_node(make_mvm_node("patch_embed", self.in_channels, dim))  # in_channels → dim
        
        # ── Double-stream blocks (×19) ──
        for i in range(19):
            prefix = f"double_stream.{i}"
            
            # Image stream
            graph.add_node(make_norm_node(f"{prefix}.img_norm", dim, "layer_norm"))
            graph.add_node(make_mvm_node(f"{prefix}.img_qkv", dim, 3 * dim))
            graph.add_node(make_mvm_node(f"{prefix}.img_mlp1", dim, mlp_dim))
            graph.add_node(make_activation_node(f"{prefix}.img_gelu", mlp_dim, "gelu"))
            graph.add_node(make_mvm_node(f"{prefix}.img_mlp2", mlp_dim, dim))
            
            # Text stream (same structure, different weights)
            graph.add_node(make_norm_node(f"{prefix}.txt_norm", dim, "layer_norm"))
            graph.add_node(make_mvm_node(f"{prefix}.txt_qkv", dim, 3 * dim))
            graph.add_node(make_mvm_node(f"{prefix}.txt_mlp1", dim, mlp_dim))
            graph.add_node(make_activation_node(f"{prefix}.txt_gelu", mlp_dim, "gelu"))
            graph.add_node(make_mvm_node(f"{prefix}.txt_mlp2", mlp_dim, dim))
            
            # Joint attention (concatenated Q, K, V across streams)
            graph.add_node(AnalogNode(
                name=f"{prefix}.joint_attn_score", op_type=OpType.DYNAMIC_MATMUL,
                domain=Domain.DIGITAL,
                input_shape=(heads, -1, head_dim), output_shape=(heads, -1, -1),
                flops=seq_len_sq * heads * head_dim,
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.joint_softmax", op_type=OpType.SOFTMAX,
                domain=Domain.DIGITAL,
                input_shape=(heads, -1, -1), output_shape=(heads, -1, -1),
                flops=seq_len_sq * heads,
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.joint_attn_value", op_type=OpType.DYNAMIC_MATMUL,
                domain=Domain.DIGITAL,
                input_shape=(heads, -1, -1), output_shape=(heads, -1, head_dim),
                flops=seq_len_sq * heads * head_dim,
            ))
            
            # Output projections
            graph.add_node(make_mvm_node(f"{prefix}.img_out_proj", dim, dim))
            graph.add_node(make_mvm_node(f"{prefix}.txt_out_proj", dim, dim))
            
            # AdaLN modulation (digital)
            graph.add_node(AnalogNode(
                name=f"{prefix}.adaln", op_type=OpType.ADALN,
                domain=Domain.DIGITAL,
                input_shape=(dim,), output_shape=(6 * dim,),
                flops=dim,
            ))
            
            # Residuals
            graph.add_node(AnalogNode(
                name=f"{prefix}.img_residual", op_type=OpType.SKIP_CONNECTION,
                domain=Domain.ANALOG, input_shape=(dim,), output_shape=(dim,), flops=dim,
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.txt_residual", op_type=OpType.SKIP_CONNECTION,
                domain=Domain.ANALOG, input_shape=(dim,), output_shape=(dim,), flops=dim,
            ))
        
        # ── Single-stream blocks (×38) ──
        for i in range(38):
            prefix = f"single_stream.{i}"
            graph.add_node(make_norm_node(f"{prefix}.norm", dim, "layer_norm"))
            graph.add_node(make_mvm_node(f"{prefix}.qkv", dim, 3 * dim))
            graph.add_node(AnalogNode(
                name=f"{prefix}.attn_score", op_type=OpType.DYNAMIC_MATMUL,
                domain=Domain.DIGITAL, input_shape=(heads, -1, head_dim), output_shape=(heads, -1, -1),
                flops=seq_len_sq * heads * head_dim,
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.softmax", op_type=OpType.SOFTMAX,
                domain=Domain.DIGITAL, input_shape=(heads, -1, -1), output_shape=(heads, -1, -1),
                flops=seq_len_sq * heads,
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.attn_value", op_type=OpType.DYNAMIC_MATMUL,
                domain=Domain.DIGITAL, input_shape=(heads, -1, -1), output_shape=(heads, -1, head_dim),
                flops=seq_len_sq * heads * head_dim,
            ))
            graph.add_node(make_mvm_node(f"{prefix}.out_proj", dim, dim))
            graph.add_node(make_mvm_node(f"{prefix}.mlp1", dim, mlp_dim))
            graph.add_node(make_activation_node(f"{prefix}.gelu", mlp_dim, "gelu"))
            graph.add_node(make_mvm_node(f"{prefix}.mlp2", mlp_dim, dim))
            graph.add_node(AnalogNode(
                name=f"{prefix}.residual", op_type=OpType.SKIP_CONNECTION,
                domain=Domain.ANALOG, input_shape=(dim,), output_shape=(dim,), flops=dim,
            ))
        
        # ── ODE integration step (Euler) ──
        spatial_flops = self.in_channels * self.img_size * self.img_size
        graph.add_node(AnalogNode(
            name="ode_step.scale", op_type=OpType.GAIN,
            domain=Domain.ANALOG,
            input_shape=(self.in_channels, self.img_size, self.img_size),
            output_shape=(self.in_channels, self.img_size, self.img_size),
            flops=spatial_flops,
            metadata={"description": "Δt · v_θ via programmable gain amplifier"},
        ))
        graph.add_node(AnalogNode(
            name="ode_step.accumulate", op_type=OpType.ACCUMULATION,
            domain=Domain.ANALOG,
            input_shape=(self.in_channels, self.img_size, self.img_size),
            output_shape=(self.in_channels, self.img_size, self.img_size),
            flops=spatial_flops,
            metadata={"description": "x_{t+Δt} = x_t + Δt·v_θ via capacitor integration"},
        ))

        return graph


# ──────────────────────────────────────────────────────────────────────
# FlowMLPExtractor — small flow model targeted at Ark export
# ──────────────────────────────────────────────────────────────────────

class FlowMLPExtractor:
    """Extract and export a small flow MLP to Ark's BaseAnalogCkt format.

    Targets the cross-arch experiment flow model:
        v_theta: [dim+1 → hidden → hidden → dim]  (time concatenated)
        dx/dt = v_theta(x, t)   identical to Neural ODE in Ark's ode_fn

    The Ark export is byte-for-byte identical to the Neural ODE export —
    export_neural_ode_to_ark() already emits:
        xt = jnp.concatenate([x, jnp.atleast_1d(t)])
        h0 = jnp.tanh(W0 @ xt + b0)
        ...
    which IS the flow model forward pass.

    Usage:
        ext = FlowMLPExtractor(checkpoint_path)
        ext.load_model()
        code = ext.export_to_ark("outputs/flow_ark.py")
        profile = ext.run()   # analog amenability profile
    """

    _EXP_DIR = (
        Path(__file__).parent.parent.parent
        / "experiments" / "cross_arch_tolerance"
    )

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        state_dim: int = 2,
        t_span: tuple[float, float] = (0.0, 1.0),
    ):
        if checkpoint_path is None:
            checkpoint_path = self._EXP_DIR / "checkpoints" / "flow.pt"
        self.checkpoint_path = Path(checkpoint_path)
        self.state_dim = state_dim
        self.t_span = t_span
        self._extractor = None

    def load_model(self) -> None:
        """Load the flow model checkpoint and wrap in a NeuralODEExtractor."""
        from neuro_analog.extractors.neural_ode import NeuralODEExtractor

        sys.path.insert(0, str(self._EXP_DIR))
        from models import flow as flow_module  # type: ignore

        if self.checkpoint_path.exists():
            model = flow_module.load_model(str(self.checkpoint_path))
            print(f"  Loaded flow checkpoint: {self.checkpoint_path}")
        else:
            print(f"  No checkpoint at {self.checkpoint_path} — training from scratch...")
            model = flow_module.create_model()
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            flow_module.train_model(model, str(self.checkpoint_path))
            print(f"  Saved to: {self.checkpoint_path}")

        self._extractor = NeuralODEExtractor.from_module(
            model,
            state_dim=self.state_dim,
            t_span=self.t_span,
            model_name="flow_mlp_make_moons",
        )

    def export_to_ark(
        self,
        output_path,
        mismatch_sigma: float = 0.05,
    ) -> str:
        """Generate outputs/flow_ark.py — a valid Ark BaseAnalogCkt subclass.

        Delegates entirely to export_neural_ode_to_ark() with class_name='FlowAnalogCkt'.
        The ode_fn is identical: xt = cat([x, t]); return MLP(xt).
        """
        assert self._extractor is not None, "Call load_model() first"
        from neuro_analog.extractors.neural_ode import export_neural_ode_to_ark
        return export_neural_ode_to_ark(
            self._extractor,
            output_path,
            mismatch_sigma=mismatch_sigma,
            class_name="FlowAnalogCkt",
        )

    def run(self):
        """Run the full analog amenability analysis pipeline."""
        assert self._extractor is not None, "Call load_model() first"
        return self._extractor.run()
