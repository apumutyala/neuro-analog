"""Benchmark script: Triton fused kernel vs PyTorch baseline.

Run on GPU (RunPod/Colab):
    python benchmarks/bench_triton.py

Saves results to benchmarks/results/triton_bench.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from neuro_analog.simulator.analog_linear import AnalogLinear

from neuro_analog.kernels.triton import analog_linear_fused, analog_linear_reference


def benchmark_shape(
    M: int,
    K: int,
    B: int = 1,
    repeats: int = 100,
    warmup: int = 10,
) -> dict:
    """Benchmark fused kernel vs PyTorch baseline for one (M, K, B) shape."""
    x = torch.randn(B, K, device="cuda")
    w = torch.randn(M, K, device="cuda")
    bias = torch.randn(M, device="cuda")
    mismatch = torch.randn(M, K, device="cuda") * 0.01
    noise_sigma = 0.05
    adc_levels = 256.0

    # ── PyTorch baseline (sequential ops) ──
    torch.cuda.synchronize()
    for _ in range(warmup):
        y = F.linear(x, w, bias)
        y += torch.randn_like(y) * noise_sigma
        y = torch.floor(y * adc_levels) / adc_levels
        y = torch.relu(y)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        y = F.linear(x, w, bias)
        y += torch.randn_like(y) * noise_sigma
        y = torch.floor(y * adc_levels) / adc_levels
        y = torch.relu(y)
    torch.cuda.synchronize()
    t_baseline = (time.perf_counter() - t0) / repeats

    # ── Fused kernel ──
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = analog_linear_fused(x, w, bias, mismatch, noise_sigma, adc_levels, activation="relu")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        y_fused = analog_linear_fused(x, w, bias, mismatch, noise_sigma, adc_levels, activation="relu")
    torch.cuda.synchronize()
    t_fused = (time.perf_counter() - t0) / repeats

    speedup = t_baseline / t_fused if t_fused > 0 else float("inf")

    return {
        "M": M,
        "K": K,
        "B": B,
        "baseline_ms": t_baseline * 1000,
        "fused_ms": t_fused * 1000,
        "speedup": speedup,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires CUDA. Run on RunPod/Colab.")

    shapes = [
        (1024, 512, 1),
        (1024, 512, 8),
        (4096, 1024, 1),
        (4096, 1024, 8),
        (16384, 4096, 1),
        (16384, 4096, 8),
    ]

    results = []
    for M, K, B in shapes:
        print(f"Benchmarking M={M}, K={K}, B={B} ...")
        try:
            result = benchmark_shape(M, K, B, repeats=100, warmup=10)
            results.append(result)
            print(
                f"  Baseline: {result['baseline_ms']:.3f} ms, "
                f"Fused: {result['fused_ms']:.3f} ms, "
                f"Speedup: {result['speedup']:.2f}x"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"M": M, "K": K, "B": B, "error": str(e)})

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "triton_bench.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
