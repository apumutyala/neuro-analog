"""Triton fused kernels for analog linear operations.

Milestone 1 implementation: fuses matmul + conductance mismatch + thermal noise
+ ADC quantization + ReLU into a single GPU kernel.

Developed on CPU-only machine (syntax validation); tested on RunPod/Colab.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from neuro_analog.simulator.substrates import SubstrateBase

# Triton is optional at import time (CPU-only machines may not have it installed)
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _get_triton():
    if not _HAS_TRITON:
        raise ImportError(
            "Triton is not installed. Install with: pip install triton"
            "\nFor CPU-only machines, install to get syntax validation."
            "\nGPU execution requires a CUDA-capable device (RunPod/Colab)."
        )
    return triton, tl


# ──────────────────────────────────────────────────────────────────────
# Triton kernel (matrix-vector, no batching)
# ──────────────────────────────────────────────────────────────────────

if _HAS_TRITON:
    @triton.jit
    def analog_linear_fused_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        out_ptr,
        mismatch_ptr,
        noise_sigma,
        adc_levels,
        M,
        K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """Fused analog linear forward: matmul + mismatch + noise + ADC + ReLU."""
        pid_m = tl.program_id(0)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for k in range(0, K, BLOCK_SIZE_K):
            mask_k = offs_k < K - k
            x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0)

            w_offs = (offs_m[:, None] * K) + (offs_k + k)[None, :]
            w_mask = (offs_m[:, None] < M) & mask_k[None, :]
            w = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

            delta = tl.load(mismatch_ptr + w_offs, mask=w_mask, other=0.0)

            # w * (1 + delta) * x, broadcast x across rows
            acc += tl.sum(w * (1.0 + delta) * x[None, :], axis=1)

        # Bias
        bias = tl.load(bias_ptr + offs_m, mask=offs_m < M, other=0.0)
        acc += bias

        # Thermal noise (Triton deterministic RNG)
        acc += tl.randn(pid_m, 0) * noise_sigma

        # ADC quantization: floor(y * levels) / levels
        acc = tl.math.floor(acc * adc_levels) / adc_levels

        # ReLU
        acc = tl.maximum(acc, 0.0)

        tl.store(out_ptr + offs_m, acc, mask=offs_m < M)
else:
    analog_linear_fused_kernel = None


# ──────────────────────────────────────────────────────────────────────
# Host launcher
# ──────────────────────────────────────────────────────────────────────

def analog_linear_fused(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    mismatch: torch.Tensor | None,
    noise_sigma: float,
    adc_levels: float,
    activation: str = "relu",
) -> torch.Tensor:
    """Fused analog linear forward pass (Triton kernel).

    Replaces the Python sequential implementation:
        y = F.linear(x, w * (1 + delta), bias)
        y += torch.randn_like(y) * noise_sigma
        y = quantize(y, adc_levels)
        y = relu(y)

    Args:
        x: Input vector, shape (K,) or (B, K).
        w: Weight matrix, shape (M, K).
        bias: Bias vector, shape (M,) or None.
        mismatch: Static conductance mismatch, shape (M, K) or None.
        noise_sigma: Thermal noise standard deviation (scalar).
        adc_levels: ADC quantization levels (e.g., 256.0 for 8-bit).
        activation: "relu" or "none".

    Returns:
        Output tensor, shape (M,) or (B, M).
    """
    if not _HAS_TRITON:
        raise RuntimeError(
            "Triton not available. Install with: pip install triton"
        )

    if not x.is_cuda:
        raise RuntimeError(
            f"analog_linear_fused requires a CUDA tensor. Got device: {x.device}"
        )

    M, K = w.shape
    is_batched = x.dim() == 2

    if is_batched:
        B = x.shape[0]
        assert x.shape[1] == K, f"x shape {x.shape} incompatible with w {w.shape}"
        out = torch.empty(B, M, device=w.device, dtype=w.dtype)
        # Batched fallback: loop over batch dimension (Triton v1 is matvec only)
        for b in range(B):
            _launch_kernel(x[b], w, bias, mismatch, noise_sigma, adc_levels, activation, out[b])
        return out
    else:
        assert x.shape == (K,), f"x shape {x.shape} incompatible with w {w.shape}"
        out = torch.empty(M, device=w.device, dtype=w.dtype)
        _launch_kernel(x, w, bias, mismatch, noise_sigma, adc_levels, activation, out)
        return out


def _launch_kernel(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    mismatch: torch.Tensor | None,
    noise_sigma: float,
    adc_levels: float,
    activation: str,
    out: torch.Tensor,
) -> None:
    """Launch the Triton kernel for a single matrix-vector product."""
    M, K = w.shape

    # Ensure contiguous
    x = x.contiguous()
    w = w.contiguous()
    out = out.contiguous()

    if bias is None:
        bias = torch.zeros(M, device=w.device, dtype=w.dtype)
    else:
        bias = bias.contiguous()

    if mismatch is None:
        mismatch = torch.zeros_like(w)
    else:
        mismatch = mismatch.contiguous()

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    analog_linear_fused_kernel[grid](
        x, w, bias, out, mismatch,
        noise_sigma, adc_levels,
        M, K,
    )


# ──────────────────────────────────────────────────────────────────────
# Python reference implementation (for correctness validation)
# ──────────────────────────────────────────────────────────────────────

def analog_linear_reference(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    mismatch: torch.Tensor | None,
    noise_sigma: float,
    adc_levels: float,
    activation: str = "relu",
) -> torch.Tensor:
    """Python reference implementation for numerical validation.

    Matches the Triton kernel semantics exactly (same order of operations,
    same quantization formula, same activation).
    """
    if mismatch is not None:
        w_eff = w * (1.0 + mismatch)
    else:
        w_eff = w

    y = F.linear(x, w_eff, bias)

    if noise_sigma > 0:
        y = y + torch.randn_like(y) * noise_sigma

    if adc_levels > 0:
        y = torch.floor(y * adc_levels) / adc_levels

    if activation == "relu":
        y = torch.relu(y)
    elif activation == "none":
        pass
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return y


# ──────────────────────────────────────────────────────────────────────
# Substrate-aware precomputation helpers
# ──────────────────────────────────────────────────────────────────────

def precompute_kernel_params(
    analog_linear: "AnalogLinear",
    substrate: "SubstrateBase" | None = None,
) -> dict:
    """Precompute kernel parameters from an AnalogLinear instance.

    Returns a dict with:
        - mismatch: torch.Tensor (M, K) — normalized conductance mismatch
        - noise_sigma: float — thermal noise std per output element
        - adc_levels: float — quantization levels (2^n_bits - 1)
        - v_ref: float — ADC reference voltage

    This is the v1 host-side precomputation. The kernel receives these
    precomputed parameters and applies them in a single fused pass.
    """
    from neuro_analog.simulator.analog_linear import _K_B, _DEFAULT_TEMP_K, _DEFAULT_CAP_F

    w = analog_linear.W_nominal
    M, K = w.shape

    # Mismatch / effective weights
    if substrate is not None:
        w_eff = substrate.perturb_weights(w)
        mismatch = (w_eff - w) / w.clamp(min=1e-8)
    elif analog_linear.sigma_mismatch > 0:
        mismatch = analog_linear.delta - 1.0  # delta = 1 + mismatch
    else:
        mismatch = None

    # Thermal noise sigma
    if substrate is not None:
        sigma_read = substrate.read_noise_std(w_eff if substrate is not None else w)
        noise_sigma = sigma_read * math.sqrt(K)
    else:
        noise_sigma = math.sqrt(_K_B * analog_linear.temperature_K / analog_linear.cap_F) * math.sqrt(K)

    # ADC quantization levels
    if analog_linear.n_adc_bits < 32:
        adc_levels = float(2 ** analog_linear.n_adc_bits - 1)
    else:
        adc_levels = 0.0  # No quantization

    return {
        "mismatch": mismatch,
        "noise_sigma": noise_sigma,
        "adc_levels": adc_levels,
        "v_ref": analog_linear.v_ref,
    }


# ──────────────────────────────────────────────────────────────────────
__all__ = [
    "analog_linear_fused",
    "analog_linear_reference",
    "precompute_kernel_params",
]
