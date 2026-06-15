# neuro-analog Kernels

GPU-accelerated fused kernels for analog inference. Replaces the Python sequential
implementation with a single kernel that fuses matmul + mismatch + noise + ADC + ReLU.

## Quick Start

### Triton (Primary Backend)

```python
import os
os.environ["NEURO_ANALOG_BACKEND"] = "triton"

from neuro_analog.kernels.integration import replace_analog_layers

model = replace_analog_layers(model, substrate=my_substrate)
y = model(x)  # Fused kernel on GPU, Python fallback on CPU
```

### CUDA (Fallback Backend)

Build required (RunPod/Colab):
```bash
cd neuro_analog/kernels/cuda
pip install -e .
```

```python
os.environ["NEURO_ANALOG_BACKEND"] = "cuda"
# ... same as above
```

### Python Reference (Debug)

```python
os.environ["NEURO_ANALOG_BACKEND"] = "python"
```

## Backend Comparison

| Backend | Build Required | Speedup vs Python | Use Case |
|---|---|---|---|
| Triton | No (`pip install triton`) | 2-3× (A100) | **Primary**: fast iteration, no nvcc |
| CUDA | Yes (`nvcc`, `ninja`) | 3-4× (A100) | Peak performance, correctness baseline |
| Python | No | 1.0× | Debugging, CPU-only, correctness reference |

## Architecture

```
AnalogLinear (Python) → AnalogLinearFused (dispatcher)
    ├── Triton kernel (triton.analog_linear_fused)
    ├── CUDA kernel (neuro_analog_cuda.analog_linear_fused_cuda)
    └── Python reference (sequential ops)
```

## Testing

```bash
# CPU-only (import + reference tests)
pytest tests/test_triton_kernel.py -v -k "test_import or test_reference"

# GPU (full test suite)
pytest tests/test_triton_kernel.py -v
```

## Benchmarking

```bash
# Run on GPU (RunPod/Colab)
python benchmarks/bench_triton.py

# Results saved to benchmarks/results/triton_bench.json
```

## Files

```
neuro_analog/kernels/
├── __init__.py
├── README.md
├── integration.py          # AnalogLinearFused dispatcher
├── triton/
│   ├── __init__.py
│   └── analog_ops.py       # Triton kernel + Python reference
├── cuda/
│   ├── setup.py            # Build script (RunPod only)
│   ├── analog_linear_cuda.cpp  # pybind11 host dispatcher
│   └── analog_linear_cuda.cu   # CUDA kernel
```

## Known Limitations

- Batched input (B, K) falls back to sequential kernel launches (not fully fused across batch).
- FP16/BF16 not yet supported (FP32 only).
- MLIR dialect is a stretch goal (not implemented in v1).

## v2 Roadmap

- Batched fused kernel (2D grid: M × B)
- FP16/BF16 support with Tensor Core acceleration
- In-kernel substrate models (PCM drift, capacitive noise)
- Autotuning integration (`triton.autotune`)
