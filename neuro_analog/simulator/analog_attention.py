"""
AnalogMultiheadAttention: nn.MultiheadAttention replacement.

What gets analogized:
  Q projection (embed_dim → embed_dim)  — static weight MVM, crossbar-native
  K projection (kdim → embed_dim)        — static weight MVM, crossbar-native
  V projection (vdim → embed_dim)        — static weight MVM, crossbar-native
  Output projection (embed_dim → embed_dim) — static weight MVM, crossbar-native

What stays digital:
  Q·K^T attention scores                 — dynamic matmul (queries and keys change
                                           per input; no static weight matrix to put
                                           on a crossbar). On real analog hardware this
                                           either uses a digital co-processor or a
                                           kernel-attention approximation (FAVOR+).
  Softmax over attention scores          — requires normalization, digital required.
  Attn·V (context vector computation)   — also dynamic; V changes per input.

NOTE on nn.MultiheadAttention internals:
  PyTorch's built-in MHA fuses Q/K/V into a single in_proj_weight tensor of shape
  (3*embed_dim, embed_dim). The recursive walk in analogize() cannot reach it because
  the weight is stored as a raw Parameter, not as an nn.Linear child. We handle MHA
  explicitly by splitting in_proj_weight into Q/K/V slices and wrapping each with
  AnalogLinear.

  When _qkv_same_embed_dim=False (kdim != embed_dim or vdim != embed_dim), PyTorch
  uses separate q_proj_weight, k_proj_weight, v_proj_weight attributes instead. Both
  cases are handled in from_module().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .analog_linear import AnalogLinear, _DEFAULT_TEMP_K, _DEFAULT_CAP_F


class AnalogMultiheadAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with analog noise on projections.

    Q·K^T and Attn·V stay digital (dynamic matmuls — inputs change every forward pass,
    no static weight matrix). Q/K/V/O linear projections are analogized with the same
    three-noise-source model as AnalogLinear.

    Interface matches nn.MultiheadAttention.forward():
        out, attn_weights = layer(query, key, value, ...)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        sigma_mismatch: float = 0.05,
        n_adc_bits: int = 8,
        temperature_K: float = _DEFAULT_TEMP_K,
        cap_F: float = _DEFAULT_CAP_F,
        v_ref: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        analog_kw = dict(
            sigma_mismatch=sigma_mismatch,
            n_adc_bits=n_adc_bits,
            temperature_K=temperature_K,
            cap_F=cap_F,
            v_ref=v_ref,
        )

        # Placeholder weights — caller must set via from_module() or manually
        _zero_qk = torch.zeros(embed_dim, embed_dim)
        _zero_v = torch.zeros(embed_dim, self.vdim)
        _zero_o = torch.zeros(embed_dim, embed_dim)
        _b = torch.zeros(embed_dim) if bias else None

        self.q_proj = AnalogLinear(embed_dim, embed_dim,
                                   weight=_zero_qk, bias=_b, **analog_kw)
        self.k_proj = AnalogLinear(self.kdim, embed_dim,
                                   weight=torch.zeros(embed_dim, self.kdim),
                                   bias=_b, **analog_kw)
        self.v_proj = AnalogLinear(self.vdim, embed_dim,
                                   weight=_zero_v.T.contiguous(),
                                   bias=_b, **analog_kw)
        self.out_proj = AnalogLinear(embed_dim, embed_dim,
                                     weight=_zero_o, bias=_b, **analog_kw)

        self.scale = self.head_dim ** -0.5

    @classmethod
    def from_module(
        cls,
        mha: nn.MultiheadAttention,
        sigma_mismatch: float = 0.05,
        n_adc_bits: int = 8,
        temperature_K: float = _DEFAULT_TEMP_K,
        cap_F: float = _DEFAULT_CAP_F,
        v_ref: float = 1.0,
    ) -> "AnalogMultiheadAttention":
        """Create AnalogMultiheadAttention from an existing nn.MultiheadAttention,
        copying all projection weights correctly."""
        E = mha.embed_dim
        has_bias = mha.in_proj_bias is not None

        analog_kw = dict(
            sigma_mismatch=sigma_mismatch, n_adc_bits=n_adc_bits,
            temperature_K=temperature_K, cap_F=cap_F, v_ref=v_ref,
        )

        obj = cls(
            embed_dim=E,
            num_heads=mha.num_heads,
            dropout=mha.dropout if isinstance(mha.dropout, float) else 0.0,
            bias=has_bias,
            kdim=mha.kdim,
            vdim=mha.vdim,
            **analog_kw,
        )

        if mha._qkv_same_embed_dim:
            # Fused in_proj_weight: shape (3E, E) — first E rows = Q, etc.
            W = mha.in_proj_weight.data
            q_w, k_w, v_w = W[:E], W[E:2*E], W[2*E:]
        else:
            q_w = mha.q_proj_weight.data
            k_w = mha.k_proj_weight.data
            v_w = mha.v_proj_weight.data

        obj.q_proj.W_nominal.copy_(q_w)
        obj.k_proj.W_nominal.copy_(k_w)
        obj.v_proj.W_nominal.copy_(v_w)
        obj.out_proj.W_nominal.copy_(mha.out_proj.weight.data)

        if has_bias:
            b = mha.in_proj_bias.data
            obj.q_proj.bias.copy_(b[:E])
            obj.k_proj.bias.copy_(b[E:2*E])
            obj.v_proj.bias.copy_(b[2*E:])
        if mha.out_proj.bias is not None:
            obj.out_proj.bias.copy_(mha.out_proj.bias.data)

        # Resample delta now that real weights are loaded
        for proj in [obj.q_proj, obj.k_proj, obj.v_proj, obj.out_proj]:
            proj._resample_delta()

        return obj

    def resample_mismatch(self, sigma: float | None = None) -> None:
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            proj.resample_mismatch(sigma)

    def set_noise_config(
        self, thermal: bool = True, quantization: bool = True, mismatch: bool = True
    ) -> None:
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            proj.set_noise_config(thermal=thermal, quantization=quantization, mismatch=mismatch)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            query: (B, T_q, E)
            key:   (B, T_k, kdim)
            value: (B, T_v, vdim)
        Returns:
            out:          (B, T_q, E)
            attn_weights: (B, T_q, T_k) if need_weights else None
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        H = self.num_heads
        D = self.head_dim

        # ── Analog projections (mismatch + thermal + ADC) ──────────────────
        Q = self.q_proj(query)              # (B, T_q, E)
        K = self.k_proj(key)                # (B, T_k, E)
        V = self.v_proj(value)              # (B, T_v, E)

        # ── Reshape for multi-head ──────────────────────────────────────────
        Q = Q.view(B, T_q, H, D).transpose(1, 2)   # (B, H, T_q, D)
        K = K.view(B, T_k, H, D).transpose(1, 2)   # (B, H, T_k, D)
        V = V.view(B, T_k, H, D).transpose(1, 2)   # (B, H, T_k, D)

        # ── Attention scores — DIGITAL ──────────────────────────────────────
        # Q·K^T is a dynamic matmul: Q and K are different every forward pass,
        # so there is no static weight matrix to put on a crossbar.
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T_q, T_k)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                attn = attn + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (B, T_k), True = ignore
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn_weights = torch.softmax(attn, dim=-1)  # (B, H, T_q, T_k)

        if self.dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Attn·V — also digital (V came from analog proj but the matmul is dynamic)
        out = torch.matmul(attn_weights, V)          # (B, H, T_q, D)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)

        # ── Analog output projection ────────────────────────────────────────
        out = self.out_proj(out)

        if need_weights:
            return out, attn_weights.mean(dim=1)    # avg over heads
        return out, None

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"σ={self.q_proj.sigma_mismatch:.3f}, bits={self.q_proj.n_adc_bits}"
        )
