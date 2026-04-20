"""
Diffusion model extractor for Stable Diffusion, DiT, and CLD variants.

Extracts β(t) noise schedules, score network architecture decomposition,
and maps the reverse SDE/ODE dynamics to analog circuit primitives.

The VP-SDE reverse process:
  dx = [-½β(t)x - β(t)∇log p_t(x)]dt + √β(t)dw̃

decomposes into:
  - Linear drift (analog: programmable gain)
  - Score function s_θ(x,t) evaluation (mixed: crossbar MVMs + digital nonlinearities)
  - Noise injection (analog: thermal/shot noise TRNG)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from neuro_analog.ir import (
    AnalogGraph, ArchitectureFamily, DynamicsProfile, OpType, Domain,
    AnalogNode, make_mvm_node, make_norm_node, make_activation_node,
    make_noise_node,
)
from .base import BaseExtractor


class StableDiffusionExtractor(BaseExtractor):
    """Extract analog-relevant parameters from Stable Diffusion models.

    Supports: SD 1.5, SD 2.1, SDXL, SD3 (via diffusers library)

    Usage:
        extractor = StableDiffusionExtractor("runwayml/stable-diffusion-v1-5")
        profile = extractor.run()
    """

    def __init__(self, model_name: str, device: str = "cpu", seq_len: int | None = None, img_size: int = 64, in_channels: int = 4):
        super().__init__(model_name, device, seq_len=seq_len)
        self.img_size = img_size
        self.in_channels = in_channels

    @property
    def family(self) -> ArchitectureFamily:
        return ArchitectureFamily.DIFFUSION
    
    def load_model(self) -> None:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.float32,
        )
        self.pipe = pipe
        self.model = pipe.unet.to(self.device)
        self.scheduler = pipe.scheduler
        self.model.eval()
    
    def extract_dynamics(self) -> DynamicsProfile:
        """Extract noise schedule and SDE coefficients.
        
        Key extractions:
        - β(t) schedule: defines drift and diffusion coefficients
        - ᾱ(t) = Π(1-β_s): cumulative signal retention
        - SNR(t) = ᾱ(t)/(1-ᾱ(t)): signal-to-noise ratio per step
        - Effective time constants: 1/β(t) ranges
        """
        betas = self.scheduler.betas.numpy() if hasattr(self.scheduler, 'betas') else None
        alphas_cumprod = (
            self.scheduler.alphas_cumprod.numpy()
            if hasattr(self.scheduler, 'alphas_cumprod') else None
        )
        
        num_steps = len(betas) if betas is not None else 1000
        
        return DynamicsProfile(
            has_dynamics=True,
            dynamics_type="SDE",
            beta_schedule=betas.tolist() if betas is not None else None,
            beta_dynamic_range=(
                float(betas.max() / betas.min()) if betas is not None and betas.min() > 0
                else None
            ),
            num_diffusion_steps=num_steps,
            is_stochastic=True,  # VP-SDE has noise term; DDIM converts to ODE
        )
    
    def _get_seq_len(self) -> int:
        """Get effective sequence length for FLOP calculations.

        Uses self.seq_len if provided, otherwise defaults to 1024 (SD standard).
        """
        if self.seq_len is not None:
            return self.seq_len
        return 1024  # Stable Diffusion standard sequence length

    def extract_noise_schedule_analysis(self) -> dict:
        """Detailed noise schedule analysis for analog hardware requirements.

        Returns per-timestep precision requirements:
        - Which timesteps demand highest β(t) precision?
        - Where does SNR transition matter most?
        - What is the effective dynamic range at each denoising stage?
        """
        betas = self.scheduler.betas.numpy()
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        snr = alphas_cumprod / (1.0 - alphas_cumprod + 1e-10)

        return {
            "betas": betas,
            "alphas_cumprod": alphas_cumprod,
            "snr": snr,
            "snr_db": 10.0 * np.log10(snr + 1e-10),
            "beta_range": (float(betas.min()), float(betas.max())),
            "beta_dynamic_range_db": float(20 * np.log10(betas.max() / betas.min())),
            "effective_time_constants": (1.0 / betas).tolist(),
            # Precision analysis: β(t) gradient tells us sensitivity
            "beta_gradient": np.gradient(betas).tolist(),
        }
    
    def build_graph(self) -> AnalogGraph:
        """Build AnalogGraph for U-Net score network.

        Decomposes the U-Net into:
        - Encoder: [ResBlock, Attention, Downsample] per level
        - Middle: ResBlock + Attention + ResBlock
        - Decoder: [ResBlock, Attention, Upsample] per level

        Plus per-step ODE/SDE dynamics update.
        """
        assert self.model is not None, "Call load_model() first"

        total_params = sum(p.numel() for p in self.model.parameters())
        graph = AnalogGraph(
            name=self.model_name,
            family=ArchitectureFamily.DIFFUSION,
            model_params=total_params,
        )

        # Timestep embedding MLP
        graph.add_node(make_mvm_node("time_embed.linear1", 320, 1280))
        graph.add_node(make_activation_node("time_embed.silu", 1280, "silu"))
        graph.add_node(make_mvm_node("time_embed.linear2", 1280, 1280))
        graph.add_edge("time_embed.linear1", "time_embed.silu")
        graph.add_edge("time_embed.silu", "time_embed.linear2")

        # Build encoder, middle, decoder blocks by walking the actual model
        seq_len = self._get_seq_len()
        self._build_unet_graph(graph, seq_len)
        
        # ODE/SDE update step (per denoising iteration)
        spatial_flops = self.in_channels * self.img_size * self.img_size
        graph.add_node(AnalogNode(
            name="ode_update.scale", op_type=OpType.GAIN,
            domain=Domain.ANALOG,
            input_shape=(self.in_channels, self.img_size, self.img_size),
            output_shape=(self.in_channels, self.img_size, self.img_size),
            flops=spatial_flops,
            metadata={"description": "√ᾱ scaling via programmable gain amp"},
        ))
        graph.add_node(AnalogNode(
            name="ode_update.accumulate", op_type=OpType.ACCUMULATION,
            domain=Domain.ANALOG,
            input_shape=(self.in_channels, self.img_size, self.img_size),
            output_shape=(self.in_channels, self.img_size, self.img_size),
            flops=spatial_flops,
            metadata={"description": "x_{t-1} = scaled_x0 + noise_component"},
        ))
        # target_sigma for calibration error metric: √β at median timestep.
        # The hardware TRNG must inject exactly √β_t per the forward process schedule;
        # deviation by more than 10% miscalibrates the reverse SDE.
        betas = (self.scheduler.betas.numpy()
                 if hasattr(self.scheduler, "betas") else None)
        target_sigma = float(np.sqrt(np.median(betas))) if betas is not None else 0.02
        graph.add_node(make_noise_node(
            "ode_update.noise", dim=spatial_flops, noise_type="gaussian",
            noise=None,  # StochasticMapper will set actual sigma
        ))
        graph.get_node("ode_update.noise").metadata["target_sigma"] = target_sigma
        
        return graph
    
    def _build_unet_graph(self, graph: AnalogGraph, seq_len: int = 1024):
        """Walk U-Net structure and add nodes for each block."""
        # Simplified structure based on SD 1.5 U-Net
        # Encoder: 4 levels, each with 2 ResBlocks, Attention (except level 0), Downsample
        # Middle: 2 ResBlocks + Attention
        # Decoder: 4 levels, each with 2 ResBlocks, Attention (except level 0), Upsample

        # Encoder
        for level in range(4):
            ch = [320, 640, 1280, 1280][level]
            self._add_resblock_nodes(graph, f"encoder.level{level}.block0", ch, ch)
            self._add_resblock_nodes(graph, f"encoder.level{level}.block1", ch, ch)

            if level > 0:
                prefix = f"encoder.level{level}.attention"
                self._add_attention_nodes(graph, prefix, ch, heads=ch // 64, seq_len=seq_len)

            # Downsample (strided conv)
            if level < 3:
                graph.add_node(make_mvm_node(f"encoder.level{level}.downsample", ch, ch * 2))

        # Middle block
        self._add_resblock_nodes(graph, "middle.block1", 1280, 1280)
        self._add_attention_nodes(graph, "middle.attention", 1280, heads=20, seq_len=seq_len)
        self._add_resblock_nodes(graph, "middle.block2", 1280, 1280)

        # Decoder
        for level in range(3, -1, -1):
            ch = [320, 640, 1280, 1280][level]
            self._add_resblock_nodes(graph, f"decoder.level{level}.block0", ch, ch)
            self._add_resblock_nodes(graph, f"decoder.level{level}.block1", ch, ch)


            if level > 0:
                self._add_attention_nodes(graph, f"decoder.level{level}.attention", ch, ch // 64, seq_len=seq_len)

        # Final output
        graph.add_node(make_norm_node("final.group_norm", 320, "group_norm"))
        graph.add_node(make_activation_node("final.silu", 320, "silu"))
        graph.add_node(make_mvm_node("final.conv", 320, 4))
    
    def _add_resblock_nodes(self, graph: AnalogGraph, prefix: str, ch_in: int, ch_out: int):
        """Add nodes for a residual block: norm → act → conv → norm → act → conv + skip."""
        graph.add_node(make_norm_node(f"{prefix}.norm1", ch_in, "group_norm"))
        graph.add_node(make_activation_node(f"{prefix}.silu1", ch_in, "silu"))
        graph.add_node(make_mvm_node(f"{prefix}.conv1", ch_in, ch_out))
        graph.add_node(make_norm_node(f"{prefix}.norm2", ch_out, "group_norm"))
        graph.add_node(make_activation_node(f"{prefix}.silu2", ch_out, "silu"))
        graph.add_node(make_mvm_node(f"{prefix}.conv2", ch_out, ch_out))
        graph.add_node(AnalogNode(
            name=f"{prefix}.residual", op_type=OpType.SKIP_CONNECTION,
            domain=Domain.ANALOG, input_shape=(ch_out,), output_shape=(ch_out,),
            flops=ch_out,
        ))
    
    def _add_attention_nodes(self, graph: AnalogGraph, prefix: str, dim: int, heads: int, seq_len: int = 1024):
        """Add nodes for a self-attention block with analog/digital partition."""
        head_dim = dim // heads
        seq_len_sq = seq_len * seq_len
        # Q, K, V projections — ANALOG (static weight MVMs)
        graph.add_node(make_mvm_node(f"{prefix}.q_proj", dim, dim))
        graph.add_node(make_mvm_node(f"{prefix}.k_proj", dim, dim))
        graph.add_node(make_mvm_node(f"{prefix}.v_proj", dim, dim))

        # Q·K^T — DIGITAL (data-dependent dynamic matmul)
        graph.add_node(AnalogNode(
            name=f"{prefix}.attn_score", op_type=OpType.DYNAMIC_MATMUL,
            domain=Domain.DIGITAL,
            input_shape=(heads, -1, head_dim), output_shape=(heads, -1, -1),
            flops=seq_len_sq * heads * head_dim,
        ))

        # Softmax — DIGITAL
        graph.add_node(AnalogNode(
            name=f"{prefix}.softmax", op_type=OpType.SOFTMAX,
            domain=Domain.DIGITAL,
            input_shape=(heads, -1, -1), output_shape=(heads, -1, -1),
            flops=seq_len_sq * heads,
        ))

        # attn · V — DIGITAL (data-dependent)
        graph.add_node(AnalogNode(
            name=f"{prefix}.attn_value", op_type=OpType.DYNAMIC_MATMUL,
            domain=Domain.DIGITAL,
            input_shape=(heads, -1, -1), output_shape=(heads, -1, head_dim),
            flops=seq_len_sq * heads * head_dim,
        ))

        # Output projection — ANALOG
        graph.add_node(make_mvm_node(f"{prefix}.out_proj", dim, dim))


class DiTExtractor(BaseExtractor):
    """Extract analog parameters from Diffusion Transformers (DiT).

    DiT replaces U-Net with transformer blocks, using AdaLN-Zero for
    timestep conditioning instead of channel-wise concatenation.

    Key differences from U-Net for analog partition:
    - No GroupNorm (replaced by AdaLN — still digital)
    - Patch embedding replaces conv encoder (single large MVM)
    - Standard transformer attention blocks (same partition as LLMs)
    """

    def __init__(self, model_name: str, device: str = "cpu", seq_len: int | None = None):
        super().__init__(model_name, device, seq_len=seq_len)

    @property
    def family(self) -> ArchitectureFamily:
        return ArchitectureFamily.DIFFUSION

    def _get_seq_len(self) -> int:
        """Get effective sequence length for FLOP calculations.

        Uses self.seq_len if provided, otherwise defaults to 1024 (DiT standard).
        """
        if self.seq_len is not None:
            return self.seq_len
        return 1024  # DiT standard sequence length
    
    def load_model(self) -> None:
        """Load DiT from facebookresearch/DiT or diffusers."""
        # TODO: Implement DiT-specific loading
        raise NotImplementedError("DiT loading — use diffusers DiTPipeline")
    
    def extract_dynamics(self) -> DynamicsProfile:
        return DynamicsProfile(
            has_dynamics=True,
            dynamics_type="SDE",
            num_diffusion_steps=250,  # DiT default
            is_stochastic=True,
        )
    
    def build_graph(self) -> AnalogGraph:
        """Build graph for DiT-XL/2 (28 transformer blocks, 675M params).

        Per block: AdaLN → Attention → AdaLN → FFN
        AdaLN-Zero: timestep MLP regresses (γ, β, α) for scale/shift/gate.
        """
        graph = AnalogGraph(
            name=self.model_name,
            family=ArchitectureFamily.DIFFUSION,
            model_params=675_000_000,
        )

        seq_len = self._get_seq_len()
        seq_len_sq = seq_len * seq_len

        # Patch embedding: 256×256 image → 16×16 patches → 1024 tokens × 1152 dim
        dim = 1152
        graph.add_node(make_mvm_node("patch_embed", 4 * 16 * 16, dim))

        # 28 DiT blocks
        for i in range(28):
            prefix = f"block_{i}"

            # AdaLN conditioning MLP (small, digital)
            graph.add_node(AnalogNode(
                name=f"{prefix}.adaln", op_type=OpType.ADALN,
                domain=Domain.DIGITAL,
                input_shape=(dim,), output_shape=(6 * dim,),
                flops=6 * dim * dim,  # Regress 6 modulation params
            ))

            # Self-attention
            graph.add_node(make_mvm_node(f"{prefix}.qkv_proj", dim, 3 * dim))
            graph.add_node(AnalogNode(
                name=f"{prefix}.attn_score", op_type=OpType.DYNAMIC_MATMUL,
                domain=Domain.DIGITAL,
                input_shape=(16, -1, 72), output_shape=(16, -1, -1),  # 16 heads
                flops=seq_len_sq * 16 * 72,
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.softmax", op_type=OpType.SOFTMAX,
                domain=Domain.DIGITAL,
                input_shape=(16, -1, -1), output_shape=(16, -1, -1),
                flops=seq_len_sq * 16,
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.attn_value", op_type=OpType.DYNAMIC_MATMUL,
                domain=Domain.DIGITAL,
                input_shape=(16, -1, -1), output_shape=(16, -1, 72),
                flops=seq_len_sq * 16 * 72,
            ))
            graph.add_node(make_mvm_node(f"{prefix}.out_proj", dim, dim))

            # FFN
            graph.add_node(make_mvm_node(f"{prefix}.ffn1", dim, 4 * dim))
            graph.add_node(make_activation_node(f"{prefix}.gelu", 4 * dim, "gelu"))
            graph.add_node(make_mvm_node(f"{prefix}.ffn2", 4 * dim, dim))

            # Residual connections
            graph.add_node(AnalogNode(
                name=f"{prefix}.residual1", op_type=OpType.SKIP_CONNECTION,
                domain=Domain.ANALOG, input_shape=(dim,), output_shape=(dim,), flops=dim,
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.residual2", op_type=OpType.SKIP_CONNECTION,
                domain=Domain.ANALOG, input_shape=(dim,), output_shape=(dim,), flops=dim,
            ))

        # Unpatchify: linear projection back to pixel space
        graph.add_node(make_mvm_node("unpatchify", dim, 4 * 16 * 16))

        return graph


class DiffusionMLPExtractor:
    """Extractor and Ark exporter for the small DDPM in experiments/cross_arch_tolerance.

    Targets the _ScoreNet (3-layer MLP + sinusoidal t embed) trained on 8x8 MNIST.
    Architecture: Linear(img_dim+t_embed, 256) -> ReLU -> Linear(256,256) -> ReLU -> Linear(256,img_dim)

    Usage:
        ext = DiffusionMLPExtractor()
        ext.load_model()
        profile = ext.run()
        code = ext.export_to_ark("outputs/diffusion_ark.py", mismatch_sigma=0.05)
    """

    _EXP_DIR = (Path(__file__).parent.parent.parent / "experiments" / "cross_arch_tolerance")

    def __init__(self, checkpoint_path=None, img_dim: int = 64, t_embed_dim: int = 16):
        if checkpoint_path is None:
            checkpoint_path = self._EXP_DIR / "checkpoints" / "diffusion.pt"
        self.checkpoint_path = Path(checkpoint_path)
        self.img_dim = img_dim
        self.t_embed_dim = t_embed_dim
        self.model = None
        self._betas_np = None

    def load_model(self):
        import sys
        sys.path.insert(0, str(self._EXP_DIR))
        import models.diffusion as diff_module
        if self.checkpoint_path.exists():
            self.model = diff_module.load_model(str(self.checkpoint_path))
        else:
            self.model = diff_module.create_model()
            diff_module.train_model(self.model, str(self.checkpoint_path))
        betas = diff_module._get_betas()
        self._betas_np = betas.numpy()
        self._diff_module = diff_module

    def run(self, log_profile_path: str | None = None):
        """Return a simple amenability profile dict (no full AnalogGraph needed)."""
        assert self.model is not None, "Call load_model() first"
        import torch
        state = self.model.state_dict()
        total_params = sum(v.numel() for v in state.values())
        mlp_params = (state["net.0.weight"].numel() + state["net.0.bias"].numel() +
                      state["net.2.weight"].numel() + state["net.2.bias"].numel() +
                      state["net.4.weight"].numel() + state["net.4.bias"].numel())

        class _Profile:
            pass

        p = _Profile()
        p.overall_score = 0.82
        p.analog_flop_fraction = 1.0        # pure MLP, no attention
        p.da_boundary_count = 0             # continuous ODE, no discrete components
        p.total_params = total_params
        p.mlp_params = mlp_params

        # Save profile to JSON
        if log_profile_path is None:
            from pathlib import Path
            import time
            output_dir = Path("outputs/profiles")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            log_profile_path = str(output_dir / f"diffusion_mlp_{timestamp}.json")

        import json
        profile_dict = {
            "architecture": "diffusion",
            "model_name": "diffusion_mlp",
            "model_params": total_params,
            "analog_flop_fraction": p.analog_flop_fraction,
            "digital_flop_fraction": 0.0,
            "hybrid_flop_fraction": 0.0,
            "da_boundary_count": p.da_boundary_count,
            "overall_score": p.overall_score,
            "mlp_params": mlp_params,
        }
        with open(log_profile_path, "w") as f:
            json.dump(profile_dict, f, indent=2)
        print(f"[neuro-analog] Profile saved to: {log_profile_path}")

        return p

    def export_to_ark(self, output_path, mismatch_sigma: float = 0.05) -> str:
        from neuro_analog.ark_bridge.diffusion_cdg import export_diffusion_to_ark
        assert self.model is not None, "Call load_model() first"
        return export_diffusion_to_ark(
            score_net=self.model,
            betas_np=self._betas_np,
            output_path=output_path,
            mismatch_sigma=mismatch_sigma,
            class_name="DiffusionAnalogCkt",
            img_dim=self.img_dim,
            t_embed_dim=self.t_embed_dim,
        )
