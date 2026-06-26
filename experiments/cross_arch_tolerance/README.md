# Cross-Architecture Tolerance Experiments

## Quick Start

Train all pilot models (CPU, ~15–25 min):
```bash
python experiments/cross_arch_tolerance/train_all.py
```

Run mismatch + ADC sweeps for all architectures:
```bash
python experiments/cross_arch_tolerance/sweep_all.py
```

## Fused Kernel Backend (GPU Speedup)

The sweep script supports a `--backend` flag that swaps the per-layer Python sequential implementation (matmul → mismatch → noise → ADC → ReLU) with a single fused GPU kernel.

| Backend | Build Required | Speedup | Use Case |
|---|---|---|---|
| `python` (default) | None | 1.0× | Reference / CPU-only |
| `triton` | `pip install triton` | 2–3× (A100) | **Recommended** — fastest, no nvcc build |
| `cuda` | `nvcc` + `torch.utils.cpp_extension` | 3–4× (A100) | Peak performance / correctness baseline |

### Usage

```bash
# Triton fused kernel (run on Colab / RunPod GPU)
python experiments/cross_arch_tolerance/sweep_all.py --backend triton

# CUDA extension (requires build on GPU instance)
python experiments/cross_arch_tolerance/sweep_all.py --backend cuda

# Python reference (default, no GPU needed)
python experiments/cross_arch_tolerance/sweep_all.py --backend python
```

### How It Works

`sweep_all.py` wraps `eval_fn` with a fused-backend substitution:
1. Before each trial, the analog model is deep-copied.
2. `replace_analog_layers(copy, substrate=None)` swaps every `AnalogLinear` with `AnalogLinearFused`.
3. `AnalogLinearFused` dispatches to:
   - **Triton** (`analog_linear_fused` kernel) if `NEURO_ANALOG_BACKEND=triton` and `triton` is installed.
   - **CUDA** (`neuro_analog_ops.analog_linear_fused_cuda`) if the C++ extension is built.
   - **Python fallback** (`_python_reference`) on CPU or if neither kernel is available.

The original analog model is preserved so `resample_all_mismatch()` can be called between trials without mutating the fused copy.

### Substrate-Aware Precomputation

When `physical_substrate` is passed (e.g., `--physical-substrate pcm`), the fused layer precomputes:
- `mismatch`: normalized conductance perturbation from `PCMSubstrate.perturb_weights()`.
- `noise_sigma`: thermal/read noise std from `PCMSubstrate.read_noise_std()`.
- `adc_levels`: `2**n_bits - 1`.

These are passed as kernel arguments, keeping the Triton/CUDA kernels simple (no physics in-kernel).

### Validation

Run the Colab notebook `notebooks/Fused_Kernel_Validation.ipynb` to verify:
1. Numerical match between Triton and Python reference (deterministic + seeded noise).
2. Batched and large-shape correctness.
3. Substrate mini-sweep (PCM / ReRAM / Capacitive) accuracy overlap.

### Limitations

- Batched inputs (`B, K`) currently loop over the batch dimension in the Triton launcher; a 2D-grid fused kernel is v2 work.
- FP16/BF16 Tensor Core acceleration is not yet implemented.
- CUDA extension build is best-effort on Colab T4 (may fail due to `nvcc` version mismatch).
