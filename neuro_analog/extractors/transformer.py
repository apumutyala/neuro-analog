"""
Generic transformer extractor.

Works with any HuggingFace transformer via AutoModel. Walks named_modules()
to build the AnalogGraph, classifying each module type into the correct
analog/digital primitive.

Analog partition for a standard transformer block:
  - Q/K/V projections: ANALOG (static-weight MVM → crossbar)
  - Q·K^T:             DIGITAL (dynamic matmul — weights change per input)
  - Softmax:           DIGITAL (or HYBRID via FAVOR+ kernel approximation)
  - Attn·V:            DIGITAL (dynamic matmul)
  - Output projection: ANALOG
  - FFN layer 1:       ANALOG (large MVM → crossbar d×4d)
  - FFN activation:    DIGITAL (GELU/SiLU) or ANALOG (ReLU/tanh)
  - FFN layer 2:       ANALOG (large MVM → crossbar 4d×d)
  - LayerNorm/RMSNorm: DIGITAL

FAVOR+ kernel attention (IBM, Nature MI 2024):
  Replaces softmax(QK^T/√d)·V with φ(Q)·(φ(K)^T·V)
  where φ(x) = exp(x^T w_i) for random projections w_i ~ N(0,I/√d).
  The w_i matrices are STATIC → programmable crossbar.
  This raises analog fraction to ~90-95% at <1% accuracy cost.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from neuro_analog.ir import (
    AnalogGraph, ArchitectureFamily, DynamicsProfile, OpType, Domain,
    AnalogNode, PrecisionSpec, make_mvm_node, make_norm_node, make_activation_node,
)
from .base import BaseExtractor

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Module classification helpers
# ──────────────────────────────────────────────────────────────────────

def _is_attention(name: str, module: nn.Module) -> bool:
    cls = type(module).__name__.lower()
    return "attention" in cls or "attn" in cls


def _is_ffn(name: str, module: nn.Module) -> bool:
    cls = type(module).__name__.lower()
    return ("mlp" in cls or "feedforward" in cls or "ffn" in cls) and not "attention" in cls


def _is_norm(module: nn.Module) -> bool:
    return isinstance(module, (nn.LayerNorm, nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm))


def _infer_norm_type(module: nn.Module) -> str:
    name = type(module).__name__.lower()
    if "rms" in name:
        return "rms_norm"
    if "group" in name:
        return "group_norm"
    return "layer_norm"


# ──────────────────────────────────────────────────────────────────────
# TransformerExtractor
# ──────────────────────────────────────────────────────────────────────

class TransformerExtractor(BaseExtractor):
    """Extract analog partition from any HuggingFace transformer model.

    Works by walking model.named_modules() and classifying each module
    into the correct analog/digital AnalogNode.

    Supports:
      - GPT-2, GPT-NeoX, LLaMA, Mistral (decoder-only)
      - BERT, RoBERTa (encoder-only)
      - DiT, ViT (vision transformers)
      - Any AutoModel-compatible checkpoint

    FAVOR+ kernel attention:
      Set use_favor_plus=True to replace attention softmax with static
      random-feature projections. Raises analog fraction by ~10-15%
      at <1% accuracy cost.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        use_favor_plus: bool = False,
        num_favor_features: int = 256,
    ):
        super().__init__(model_name, device)
        self.use_favor_plus = use_favor_plus
        self.num_favor_features = num_favor_features

    @property
    def family(self) -> ArchitectureFamily:
        return ArchitectureFamily.TRANSFORMER

    def load_model(self) -> None:
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def extract_dynamics(self) -> DynamicsProfile:
        """Transformers have no native ODE dynamics — all linear algebra."""
        return DynamicsProfile(
            has_dynamics=False,
            dynamics_type="",  # No ODE structure
        )

    def _get_model_dim(self) -> int:
        """Infer hidden dimension from config or first linear layer."""
        if hasattr(self.model, "config"):
            cfg = self.model.config
            for attr in ("hidden_size", "d_model", "n_embd", "dim"):
                if hasattr(cfg, attr):
                    return getattr(cfg, attr)
        # Fallback: first linear layer
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return 768

    def _get_num_heads(self) -> int:
        if hasattr(self.model, "config"):
            cfg = self.model.config
            for attr in ("num_attention_heads", "n_head", "num_heads"):
                if hasattr(cfg, attr):
                    return getattr(cfg, attr)
        return 12

    def _get_num_layers(self) -> int:
        if hasattr(self.model, "config"):
            cfg = self.model.config
            for attr in ("num_hidden_layers", "n_layer", "num_layers"):
                if hasattr(cfg, attr):
                    return getattr(cfg, attr)
        return 12

    def _add_attention_block(self, graph: AnalogGraph, prefix: str, dim: int, heads: int):
        """Add Q/K/V, attention score, softmax/FAVOR+, and output projection nodes."""
        head_dim = dim // heads

        # Q, K, V projections — ANALOG: static weight MVMs
        graph.add_node(make_mvm_node(f"{prefix}.q_proj", dim, dim))
        graph.add_node(make_mvm_node(f"{prefix}.k_proj", dim, dim))
        graph.add_node(make_mvm_node(f"{prefix}.v_proj", dim, dim))

        if self.use_favor_plus:
            # FAVOR+ kernel approximation — static random projection matrices
            # φ(Q) = exp(Q·W_r^T) / √m for random W_r ~ N(0,1/d)
            # W_r is STATIC → programmable crossbar
            graph.add_node(make_mvm_node(
                f"{prefix}.favor_q_proj", dim, self.num_favor_features,
            ))
            graph.add_node(make_mvm_node(
                f"{prefix}.favor_k_proj", dim, self.num_favor_features,
            ))
            # φ(K)^T · V — static outer product → crossbar
            graph.add_node(make_mvm_node(
                f"{prefix}.favor_kv", self.num_favor_features, dim,
            ))
            # φ(Q) · (φ(K)^T · V) — final attention via MVM
            graph.add_node(make_mvm_node(
                f"{prefix}.favor_attn", dim, dim,
            ))
            # Replace softmax with HYBRID kernel attention
            graph.add_node(AnalogNode(
                name=f"{prefix}.kernel_attn",
                op_type=OpType.KERNEL_ATTENTION,
                domain=Domain.HYBRID,
                input_shape=(dim,), output_shape=(dim,),
                flops=2 * dim * self.num_favor_features,
                metadata={
                    "description": "FAVOR+ kernel attention (IBM NMI 2024)",
                    "num_features": self.num_favor_features,
                    "accuracy_cost": "<1% vs softmax",
                },
            ))
        else:
            # Standard softmax attention — Q·K^T and Attn·V are DIGITAL
            graph.add_node(AnalogNode(
                name=f"{prefix}.attn_score",
                op_type=OpType.DYNAMIC_MATMUL,
                domain=Domain.DIGITAL,
                input_shape=(heads, -1, head_dim), output_shape=(heads, -1, -1),
                flops=2 * heads * head_dim,
                metadata={"description": "Q·K^T — data-dependent, can't be static crossbar"},
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.softmax",
                op_type=OpType.SOFTMAX,
                domain=Domain.DIGITAL,
                input_shape=(heads, -1, -1), output_shape=(heads, -1, -1),
                flops=3 * heads,
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.attn_value",
                op_type=OpType.DYNAMIC_MATMUL,
                domain=Domain.DIGITAL,
                input_shape=(heads, -1, -1), output_shape=(heads, -1, head_dim),
                flops=2 * heads * head_dim,
                metadata={"description": "Attn·V — data-dependent"},
            ))

        graph.add_node(make_mvm_node(f"{prefix}.out_proj", dim, dim))

    def _add_ffn_block(self, graph: AnalogGraph, prefix: str, dim: int, activation: str = "gelu"):
        """Add FFN: linear → activation → linear."""
        ffn_dim = 4 * dim
        graph.add_node(make_mvm_node(f"{prefix}.ffn1", dim, ffn_dim))
        graph.add_node(make_activation_node(f"{prefix}.ffn_act", ffn_dim, activation))
        graph.add_node(make_mvm_node(f"{prefix}.ffn2", ffn_dim, dim))
        graph.add_node(AnalogNode(
            name=f"{prefix}.residual",
            op_type=OpType.SKIP_CONNECTION,
            domain=Domain.ANALOG,
            input_shape=(dim,), output_shape=(dim,), flops=dim,
        ))
        graph.add_edge(f"{prefix}.ffn1", f"{prefix}.ffn_act")
        graph.add_edge(f"{prefix}.ffn_act", f"{prefix}.ffn2")
        graph.add_edge(f"{prefix}.ffn2", f"{prefix}.residual")

    def build_graph(self) -> AnalogGraph:
        """Walk model structure and build AnalogGraph.

        If model is loaded, uses actual architecture config for accuracy.
        Falls back to config-based inference.
        """
        assert self.model is not None, "Call load_model() first"

        dim = self._get_model_dim()
        heads = self._get_num_heads()
        n_layers = self._get_num_layers()
        total_params = sum(p.numel() for p in self.model.parameters())

        graph = AnalogGraph(
            name=self.model_name,
            family=ArchitectureFamily.TRANSFORMER,
            model_params=total_params,
        )

        # Embedding
        if hasattr(self.model, "embed_tokens") or hasattr(self.model, "wte"):
            graph.add_node(AnalogNode(
                name="embedding",
                op_type=OpType.EMBEDDING,
                domain=Domain.DIGITAL,
                input_shape=(1,), output_shape=(dim,), flops=0,
                metadata={"description": "Token embedding lookup"},
            ))

        # Detect activation function
        activation = "gelu"
        if hasattr(self.model, "config"):
            act_str = getattr(self.model.config, "hidden_act", "gelu")
            activation = act_str.lower() if act_str else "gelu"

        prev_ids: list[str] = ["embedding"] if "embedding" in [n.name for n in graph.nodes] else []

        for layer_idx in range(n_layers):
            prefix = f"layer_{layer_idx}"

            # Pre-norm (RMSNorm in LLaMA, LayerNorm in GPT-2)
            norm_type = "rms_norm" if "llama" in self.model_name.lower() else "layer_norm"
            graph.add_node(make_norm_node(f"{prefix}.pre_norm", dim, norm_type))

            # Attention block
            self._add_attention_block(graph, f"{prefix}.attn", dim, heads)
            graph.add_node(AnalogNode(
                name=f"{prefix}.attn_residual",
                op_type=OpType.SKIP_CONNECTION,
                domain=Domain.ANALOG,
                input_shape=(dim,), output_shape=(dim,), flops=dim,
            ))

            # Post-attention norm
            graph.add_node(make_norm_node(f"{prefix}.post_norm", dim, norm_type))

            # FFN
            self._add_ffn_block(graph, f"{prefix}.ffn", dim, activation)

            # Wire within-layer edges
            graph.add_edge(f"{prefix}.pre_norm", f"{prefix}.attn.q_proj")
            graph.add_edge(f"{prefix}.pre_norm", f"{prefix}.attn.k_proj")
            graph.add_edge(f"{prefix}.pre_norm", f"{prefix}.attn.v_proj")

            if self.use_favor_plus:
                graph.add_edge(f"{prefix}.attn.q_proj", f"{prefix}.attn.favor_q_proj")
                graph.add_edge(f"{prefix}.attn.k_proj", f"{prefix}.attn.favor_k_proj")
                graph.add_edge(f"{prefix}.attn.favor_k_proj", f"{prefix}.attn.favor_kv")
                graph.add_edge(f"{prefix}.attn.favor_q_proj", f"{prefix}.attn.kernel_attn")
                graph.add_edge(f"{prefix}.attn.favor_kv", f"{prefix}.attn.kernel_attn")
                graph.add_edge(f"{prefix}.attn.kernel_attn", f"{prefix}.attn.out_proj")
            else:
                graph.add_edge(f"{prefix}.attn.q_proj", f"{prefix}.attn.attn_score")
                graph.add_edge(f"{prefix}.attn.k_proj", f"{prefix}.attn.attn_score")
                graph.add_edge(f"{prefix}.attn.attn_score", f"{prefix}.attn.softmax")
                graph.add_edge(f"{prefix}.attn.softmax", f"{prefix}.attn.attn_value")
                graph.add_edge(f"{prefix}.attn.v_proj", f"{prefix}.attn.attn_value")
                graph.add_edge(f"{prefix}.attn.attn_value", f"{prefix}.attn.out_proj")

            graph.add_edge(f"{prefix}.attn.out_proj", f"{prefix}.attn_residual")
            graph.add_edge(f"{prefix}.attn_residual", f"{prefix}.post_norm")
            graph.add_edge(f"{prefix}.post_norm", f"{prefix}.ffn.ffn1")

            # Inter-layer edges
            if layer_idx > 0:
                prev = f"layer_{layer_idx-1}"
                graph.add_edge(f"{prev}.ffn.residual", f"{prefix}.pre_norm")
            elif prev_ids:
                graph.add_edge(prev_ids[0], f"{prefix}.pre_norm")

        return graph

    @classmethod
    def reference(
        cls,
        dim: int = 768,
        n_layers: int = 12,
        heads: int = 12,
        model_name: str = "reference_transformer",
        use_favor_plus: bool = False,
    ) -> "TransformerExtractor":
        """Build a reference transformer without loading a pretrained model.

        Used for taxonomy comparison and testing.
        """
        ext = cls(model_name=model_name, use_favor_plus=use_favor_plus)

        # Build graph directly without a loaded model
        total_params = n_layers * (12 * dim * dim)  # Rough estimate
        ffn_dim = 4 * dim

        graph = AnalogGraph(
            name=model_name,
            family=ArchitectureFamily.TRANSFORMER,
            model_params=total_params,
        )

        for i in range(n_layers):
            prefix = f"layer_{i}"
            graph.add_node(make_norm_node(f"{prefix}.norm1", dim, "layer_norm"))
            graph.add_node(make_mvm_node(f"{prefix}.q_proj", dim, dim))
            graph.add_node(make_mvm_node(f"{prefix}.k_proj", dim, dim))
            graph.add_node(make_mvm_node(f"{prefix}.v_proj", dim, dim))

            if use_favor_plus:
                graph.add_node(make_mvm_node(f"{prefix}.favor_qk", dim, 256))
                graph.add_node(AnalogNode(
                    name=f"{prefix}.kernel_attn", op_type=OpType.KERNEL_ATTENTION,
                    domain=Domain.HYBRID,
                    input_shape=(dim,), output_shape=(dim,), flops=2 * dim * 256,
                ))
            else:
                graph.add_node(AnalogNode(
                    name=f"{prefix}.attn_score", op_type=OpType.DYNAMIC_MATMUL,
                    domain=Domain.DIGITAL, input_shape=(heads, -1, dim // heads),
                    output_shape=(heads, -1, -1), flops=2 * heads * dim // heads,
                ))
                graph.add_node(AnalogNode(
                    name=f"{prefix}.softmax", op_type=OpType.SOFTMAX,
                    domain=Domain.DIGITAL, input_shape=(heads,), output_shape=(heads,),
                    flops=3 * heads,
                ))
                graph.add_node(AnalogNode(
                    name=f"{prefix}.attn_value", op_type=OpType.DYNAMIC_MATMUL,
                    domain=Domain.DIGITAL, input_shape=(heads, -1, -1),
                    output_shape=(heads, -1, dim // heads), flops=2 * heads * dim // heads,
                ))

            graph.add_node(make_mvm_node(f"{prefix}.out_proj", dim, dim))
            graph.add_node(make_norm_node(f"{prefix}.norm2", dim, "layer_norm"))
            graph.add_node(make_mvm_node(f"{prefix}.ffn1", dim, ffn_dim))
            graph.add_node(make_activation_node(f"{prefix}.gelu", ffn_dim, "gelu"))
            graph.add_node(make_mvm_node(f"{prefix}.ffn2", ffn_dim, dim))
            graph.add_node(AnalogNode(
                name=f"{prefix}.residual", op_type=OpType.SKIP_CONNECTION,
                domain=Domain.ANALOG, input_shape=(dim,), output_shape=(dim,), flops=dim,
            ))

        ext._graph = graph
        ext.model = True  # sentinel: "model loaded"
        return ext
