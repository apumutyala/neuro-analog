Technical Design Document: Fused Analog Inference Kernel
neuro-analog GPU Backend Extension

Last Updated: 2026-06-14
Status: Revised — Triton-First, RunPod-Primary Workflow

---

## 1. Executive Summary

This document designs a fused GPU kernel backend for the `neuro-analog` simulator. The goal is to replace the current Python-based, multi-op analog forward pass (matmul → conductance mismatch → thermal noise → ADC quantization → activation) with a single fused kernel that eliminates intermediate HBM round-trips.

**Key workflow decisions:**
- **Triton-first:** Triton kernels are developed first (can prototype locally on CPU, test on Colab/RunPod).
- **CUDA-second:** CUDA kernels and C++ extensions are built on a remote GPU instance (RunPod).
- **MLIR is a stretch goal:** Only if GPU phases finish early.
- **Local machine is CPU-only:** No local GPU compilation. All GPU builds happen via SSH to RunPod.

---

## 2. System Architecture

### 2.1 Layer Stack

```
┌─────────────────────────────────────────────────────────┐
│  PyTorch Frontend (analog_linear.py)                    │
│  - User calls analog_linear()                            │
│  - Splits: Python fallback vs. fused dispatch            │
├─────────────────────────────────────────────────────────┤
│  C++ Extension (neuro_analog_ops.so)                     │
│  - pybind11 dispatchers                                  │
│  - Handles CUDA device, error checking, shape validation │
│  - Single shared library (no TORCH_LIBRARY ABI issues)   │
├─────────────────────────────────────────────────────────┤
│  Triton Kernel (primary)                                 │
│  - Fused: matmul + mismatch + noise + quant + activation │
│  - Developed locally, tested on Colab/RunPod             │
│  - 2-3× speedup over PyTorch baseline on A100            │
├─────────────────────────────────────────────────────────┤
│  CUDA Kernel (fallback / reference)                      │
│  - Hand-written tiling kernel                            │
│  - Built on RunPod with nvcc                             │
│  - Baseline for Triton correctness                       │
├─────────────────────────────────────────────────────────┤
│  MLIR Dialect (stretch goal)                             │
│  - `analog` dialect with `analog.mvm` op                 │
│  - Lowers to `func.call` into C++ extension              │
│  - Only if Phase 1-2 finish early                        │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Why This Stack?

| Technology | Problem It Solves | Why Not Alternative |
|---|---|---|
| **C++ Extension** | PyTorch ↔ GPU kernel bridge; Python GIL release; dtype/device dispatch | Pure Python: GIL bound, no CUDA context |
| **Triton** | Productivity-first GPU kernel development; autotuning; no nvcc/ninja | CUDA: requires nvcc, slower iteration; best for reference/peak performance |
| **CUDA** | Peak performance; shared memory control; template specialization | Triton: abstracts too much for some tiling strategies; CUDA is the baseline |
| **MLIR** | Formal compiler IR for analog ops; hardware-aware lowering | Not needed for v1; adds weeks of build complexity; stretch goal only |

**Honest scope:** The C++ extension uses `pybind11`, not `TORCH_LIBRARY`. The latter requires ABI-stable builds and complex CMake. `pybind11` is simpler for a research project and sufficient for dispatching to custom kernels.

---

## 2.5 Why Each Technology Stack (Narrative)

This section explains the problem each technology solves and why alternatives were rejected. It is the rationale that justifies the architecture decisions in interviews.

### 2.5.1 C++ Extension (PyTorch Interop)

**Problem:** PyTorch's Python frontend cannot directly launch custom GPU kernels. Python's Global Interpreter Lock (GIL) serializes CPU work, and PyTorch's autograd/tape system expects `at::Tensor` objects with metadata (shape, stride, device, dtype) that Python alone cannot manipulate at the required speed.

**What C++ solves:**
- **GIL release:** The C++ extension releases the Python GIL before launching the kernel, allowing Python to do other work (e.g., data loading) while the GPU runs.
- **Tensor metadata validation:** C++ checks device, dtype, and shape invariants before the kernel launches, catching errors early rather than with cryptic CUDA errors.
- **Dispatcher integration:** PyTorch's dispatcher (via `torch.ops`) handles backend selection (CPU vs CUDA), dtype casting, and autograd wrapping. Even with `pybind11`, the extension registers a `torch.autograd.Function` in Python that calls the C++ forward/backward.

**Why not pure Python?**
Pure Python cannot launch CUDA kernels. `torch.cuda` provides Python bindings for cuBLAS/cuDNN, but custom kernels require either C++ (`pybind11`, `TORCH_LIBRARY`) or Triton (which JIT-compiles to PTX under the hood). The 5-op fusion in this project is not available in cuBLAS, so a custom kernel is required.

**Why `pybind11` over `TORCH_LIBRARY`?**
| Dimension | `pybind11` | `TORCH_LIBRARY` |
|---|---|---|
| Build complexity | `setup.py` + `CUDAExtension` | CMake + `torch_LIBRARY` macro + ABI matching |
| ABI stability | Not required (rebuild on PyTorch update) | Required (ABI-stable across PyTorch minor versions) |
| Dispatcher registration | Manual Python wrapper | Automatic `torch.ops.myop` registration |
| CMake required | No | Yes |
| Windows support | Works with `torch.utils.cpp_extension` | Fragile on Windows (ABI mismatch) |
| Use case | Research project, single user | Production library (PyTorch Geometric, TorchAudio) |

For a research project with a CPU-only local developer and a single GPU test environment, `pybind11` is the pragmatic choice. `TORCH_LIBRARY` is the correct choice for a library that ships to PyPI and must work across PyTorch versions without recompilation.

### 2.5.2 CUDA (Raw Performance)

**Problem:** GPU kernels are memory-bound. The analog forward pass involves 5 operations: matmul, mismatch multiplication, thermal noise addition, ADC quantization, and ReLU. In PyTorch, each operation materializes its output to High Bandwidth Memory (HBM). For a layer with M=4096, K=1024, each intermediate is a 4096-element float32 tensor (16 KB). Five operations = 5 HBM round-trips. On an A100 with 1.5 TB/s HBM bandwidth, 5 × 16 KB is negligible, but for a batch of 64 or 128, the memory traffic becomes the bottleneck.

**What CUDA solves:**
- **Fusion in a single kernel:** All 5 operations happen in registers/shared memory before a single write to HBM.
- **Shared memory tiling:** For larger K, a CUDA kernel can tile the K dimension into shared memory, achieving near-L1 bandwidth (~19 TB/s on A100) rather than HBM bandwidth.
- **Template specialization:** Compile-time constants (BLOCK_SIZE_M, BLOCK_SIZE_K) allow the compiler to unroll loops and optimize register allocation. Triton does this via JIT compilation, but CUDA templates give more control.
- **Philox RNG:** CUDA's `curand` library provides Philox4_32_10, a counter-based RNG that is deterministic and reproducible across kernel launches. Triton also provides `tl.randn`, but understanding the underlying RNG is critical for numerical validation.

**Why not just Triton?**
Triton is the right choice for rapid prototyping and matmul-like fusions where the compiler can handle tiling. For hand-tuned shared memory layouts, custom warp-shuffle reductions, or when you need to control the exact PTX instructions, CUDA is necessary. In this project, CUDA serves as the correctness baseline and performance reference for the Triton kernel.

### 2.5.3 Triton (Productivity)

**Problem:** Writing CUDA kernels requires understanding warp scheduling, shared memory bank conflicts, memory coalescing, and PTX assembly. The iteration cycle is slow: write C++ → compile with nvcc → link → test → debug. For a researcher with a CPU-only laptop, this requires SSH to a remote GPU instance for every iteration.

**What Triton solves:**
- **Python-like syntax:** Triton kernels look like NumPy vectorized code. The compiler handles tiling, shared memory, and coalescing.
- **JIT compilation:** No `nvcc` or `setup.py`. `pip install triton` and write a `@triton.jit` function. On a CPU-only machine, Triton can compile the kernel (for syntax checking) and even run small shapes on CPU.
- **Autotuning:** `triton.autotune` automatically searches hyperparameters (BLOCK_SIZE, num_stages, num_warps) across a grid of configs. In CUDA, this is manual.
- **Tensor cores:** Triton's `tl.dot` maps to Tensor Core MMA instructions on A100 (BF16/FP16). This is non-trivial to write by hand in CUDA.

**Why Triton over CUDA for Phase 1?**
This project uses a Triton-first approach because:
1. The developer machine is CPU-only. Triton can be written and partially tested locally.
2. The kernel is a matmul-like fusion, which Triton handles well.
3. The timeline is aggressive (2-3 weeks). Triton allows faster iteration.
4. The performance target is 2-3× speedup, which Triton can achieve for this workload.

### 2.5.4 MLIR (Compiler Formalization)

**Problem:** The analog operations (MVM with noise, ADC quantization) are currently defined in Python (`analog_linear.py`). There is no formal IR representation that a compiler can analyze, optimize, or lower to different backends (GPU, analog ASIC, FPGA).

**What MLIR solves:**
- **Formal IR:** An MLIR dialect defines the analog operations in a structured way that can be parsed, transformed, and optimized by passes.
- **Lowering pipeline:** The dialect can be lowered to different backends: `func.call` to the C++ extension (GPU), or future lowering to `llvm.call` (CPU), or `nvvm` (GPU PTX).
- **Analysis passes:** An MLIR pass can walk the IR and compute properties like memory access patterns, analog/digital boundary counts, or precision requirements.

**Why MLIR is a stretch goal:**
MLIR adds significant build complexity (building LLVM/MLIR from source or linking against prebuilt libraries). For v1, the value is primarily interview signaling and future extensibility, not performance. The GPU kernel phases (Triton, CUDA) must be solid before MLIR is attempted.

**Honest scope for v1:**
- Define an `analog` dialect with one operation: `analog.mvm`
- Write a conversion pattern that lowers `analog.mvm` to `func.call` into the C++ extension
- This is NOT a full GPU compiler. It does NOT emit PTX or CUDA.
- It is a proof-of-concept that the analog operation can be represented in MLIR and lowered to a callable function.

---

## 3. Triton Kernel (Phase 1 — Primary)

### 3.1 Design Goals

- **Fuse 5 operations into 1 kernel:** `matmul → mismatch → thermal noise → ADC quant → ReLU`
- **Eliminate HBM round-trips:** Intermediate tensors stay in SRAM/registers
- **Tile for A100:** BLOCK_SIZE_M/K tuned for 108 SMs, 40 GB HBM
- **Autotune:** `triton.autotune` over BLOCK_SIZE, num_stages, num_warps

### 3.2 Kernel Signature

```python
import triton
import triton.language as tl
import torch

@triton.jit
def analog_linear_fused_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    mismatch_ptr, noise_sigma, adc_levels,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused analog linear forward pass.

    Operations fused:
    1. Tiled matrix-vector multiply (w @ x)
    2. Conductance mismatch: w_eff = w * (1 + delta)
    3. Thermal noise: add Gaussian(0, noise_sigma)
    4. ADC quantization: floor(y * levels) / levels
    5. ReLU activation: max(0, y)

    All intermediates stay in registers/shared memory.
    """
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator in registers (FP32)
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Tiled matrix-vector multiply with mismatch injection
    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptr + offs_k, mask=offs_k < K - k, other=0.0)
        w = tl.load(w_ptr + offs_m[:, None] * K + (offs_k + k)[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k))
        delta = tl.load(mismatch_ptr + offs_m[:, None] * K + (offs_k + k)[None, :],
                         mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k))

        # Fused: w * (1 + delta) * x, accumulate
        acc += tl.sum(w * (1.0 + delta) * x[None, :], axis=1)

    # Add bias
    bias = tl.load(bias_ptr + offs_m, mask=offs_m < M)
    acc += bias

    # Thermal noise (Triton RNG)
    acc += tl.randn(tl.program_id(0), 0) * noise_sigma

    # ADC quantization
    acc = tl.math.floor(acc * adc_levels) / adc_levels

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store to HBM
    tl.store(out_ptr + offs_m, acc, mask=offs_m < M)
```

### 3.3 Autotuning Configuration

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def analog_linear_fused_kernel(...):
    ...
```

### 3.4 Host Launcher (Python)

```python
def analog_linear_fused(x, w, bias, mismatch, noise_sigma, adc_levels):
    M, K = w.shape
    assert x.shape == (K,)
    out = torch.empty(M, device=w.device, dtype=w.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
    analog_linear_fused_kernel[grid](
        x, w, bias, out,
        mismatch, noise_sigma, adc_levels,
        M, 1, K,
    )
    return out
```

### 3.5 Triton Internals: Block Pointers, Tensor Cores, and L2 Cache

**Block pointer arithmetic:**
Triton uses block pointers to tile the iteration space. In the kernel above, `offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)` creates a block of row indices. The load `tl.load(w_ptr + offs_m[:, None] * K + (offs_k + k)[None, :])` computes a 2D tile of the weight matrix. The `[:, None]` and `[None, :]` broadcasting create a BLOCK_SIZE_M × BLOCK_SIZE_K tile.

**`tl.dot` and Tensor Cores:**
For matrix-matrix multiplication (batch > 1), Triton's `tl.dot` maps to Tensor Core MMA instructions on A100. This requires:
- Inputs in FP16 or BF16 (A100 Tensor Cores do not accelerate FP32 matmul)
- Block sizes that are multiples of 16 (MMA instruction size)
- Accumulation in FP32 for numerical stability

For the analog linear layer (matrix-vector), Tensor Cores are not directly applicable because K is the reduction dimension and the output is a vector. However, for batched inference (multiple vectors in parallel), `tl.dot` can be used with a BLOCK_SIZE_N dimension.

**L2 cache optimization via `GROUP_SIZE_M`:**
When multiple thread blocks process the same weight matrix, Triton uses a `GROUP_SIZE_M` parameter to group blocks in the M dimension. This ensures that blocks in the same group access the same rows of `w`, which stays in L2 cache. The default `GROUP_SIZE_M=1` (no grouping) causes each block to load different rows, thrashing L2. For analog linear layers where `w` is reused across many inputs, `GROUP_SIZE_M=8` or `16` improves cache locality.

```python
# L2-optimized grid configuration
def grid(meta):
    return (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']) if N > 1 else 1,
    )
```

**Triton RNG utilities:**
Triton provides `tl.randn` and `tl.rand` for random number generation. These are deterministic given a seed and offset. For the analog kernel, thermal noise is drawn from `tl.randn(seed, offset) * noise_sigma`. The seed can be fixed for reproducibility (e.g., seed=42 for unit tests) or varied per-kernel-launch for Monte Carlo sweeps.

### 3.6 Why Triton First?

1. **Local development:** Can write and debug kernel logic on a CPU-only laptop. Triton JIT compiles and runs on CPU for small shapes (with warnings).
2. **No build system:** No `nvcc`, no `ninja`, no `setup.py`. Just `pip install triton`.
3. **Autotuning:** `triton.autotune` searches hyperparameters automatically. In CUDA, this is manual.
4. **Productivity:** A Triton kernel is ~50 lines. The equivalent CUDA kernel is ~200 lines + 50 lines of host C++.
5. **Performance:** For matmul-like fusions, Triton matches CUDA within 10-20% on A100.
6. **Tensor core access:** `tl.dot` maps to MMA instructions without writing PTX assembly.

---

## 4. CUDA Kernel (Phase 2 — Secondary / Baseline)

### 4.1 Role

The CUDA kernel serves two purposes:
1. **Correctness baseline:** Triton kernel output is compared against CUDA kernel output for numerical validation.
2. **Peak performance reference:** Hand-written CUDA can exceed Triton for specific tile sizes (e.g., asymmetric M/N/K).

### 4.2 Kernel Design

```cpp
// analog_linear_cuda.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

template <typename scalar_t, int BLOCK_SIZE_M, int BLOCK_SIZE_K>
__global__ void analog_linear_fused_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mismatch,
    float noise_sigma,
    float adc_levels,
    int M, int K
) {
    // Block index
    int bm = blockIdx.x;

    // Thread index within block
    int tid = threadIdx.x;

    // Row this thread block handles
    int row_start = bm * BLOCK_SIZE_M;
    int row = row_start + tid;

    if (row >= M) return;

    // Accumulator in register
    float acc = 0.0f;

    // Tiled dot product with mismatch
    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        int k_end = min(k + BLOCK_SIZE_K, K);
        for (int kk = k; kk < k_end; ++kk) {
            float w_val = w[row * K + kk];
            float delta = mismatch[row * K + kk];
            float x_val = x[kk];
            acc += w_val * (1.0f + delta) * x_val;
        }
    }

    // Bias
    acc += bias[row];

    // Thermal noise (Philox RNG)
    curandStatePhilox4_32_10_t state;
    curand_init(1234, row, 0, &state);
    acc += curand_normal(&state) * noise_sigma;

    // ADC quantization
    acc = floorf(acc * adc_levels) / adc_levels;

    // ReLU
    acc = fmaxf(acc, 0.0f);

    out[row] = acc;
}
```

### 4.3 Host Dispatcher (pybind11)

```cpp
// analog_linear_cuda.cpp
#include <torch/extension.h>
#include <vector>

// Forward declaration
void launch_analog_linear_fused(
    torch::Tensor x, torch::Tensor w, torch::Tensor bias,
    torch::Tensor out, torch::Tensor mismatch,
    float noise_sigma, float adc_levels
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("analog_linear_fused", &launch_analog_linear_fused,
          "Fused analog linear forward (CUDA)");
}

void launch_analog_linear_fused(
    torch::Tensor x, torch::Tensor w, torch::Tensor bias,
    torch::Tensor out, torch::Tensor mismatch,
    float noise_sigma, float adc_levels
) {
    int M = w.size(0);
    int K = w.size(1);

    const int BLOCK_SIZE_M = 64;
    const int BLOCK_SIZE_K = 32;

    dim3 blocks((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    dim3 threads(BLOCK_SIZE_M);

    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "analog_linear_fused", ([&] {
        analog_linear_fused_kernel<scalar_t, BLOCK_SIZE_M, BLOCK_SIZE_K>
            <<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                mismatch.data_ptr<scalar_t>(),
                noise_sigma, adc_levels,
                M, K
            );
    }));
}
```

### 4.4 Build (RunPod Only)

```python
# setup.py — run on RunPod, not locally
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='neuro_analog_ops',
    ext_modules=[
        CUDAExtension(
            'neuro_analog_ops',
            sources=['analog_linear_cuda.cpp', 'analog_linear_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math'],
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

**Build command on RunPod:**
```bash
pip install -e .
# Produces neuro_analog_ops.cpython-310-x86_64-linux-gnu.so
```

### 4.5 CUDA Memory Model: Coalescing, Shared Memory, and Bank Conflicts

**Global memory coalescing:**
In the simple row-wise kernel above, each thread reads its own row of `w`. Threads in a warp read consecutive elements of `w` (since each thread has a unique `row`), so `w` access is coalesced. The input vector `x` is broadcast to all threads in the warp (all threads read the same `x[kk]` address). This is not coalesced, but it is cached in the L1/constant cache because the same address is read by all threads.

**Shared memory tiling (for large K):**
For K > 1024, the row-wise kernel becomes memory-bound because each thread loads K elements from global memory. A tiled kernel uses shared memory to cache tiles of `x` and `w`:

```cpp
template <typename scalar_t, int BLOCK_SIZE_M, int BLOCK_SIZE_K>
__global__ void analog_linear_fused_tiled_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mismatch,
    float noise_sigma, float adc_levels, int M, int K
) {
    // Shared memory tiles
    __shared__ float x_smem[BLOCK_SIZE_K];
    __shared__ float w_smem[BLOCK_SIZE_M][BLOCK_SIZE_K + 1]; // +1 to avoid bank conflicts
    
    int bm = blockIdx.x;
    int tid = threadIdx.x;
    int row = bm * BLOCK_SIZE_M + tid;
    
    float acc = 0.0f;
    
    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        // Load x tile into shared memory (coalesced)
        if (tid < BLOCK_SIZE_K && k + tid < K) {
            x_smem[tid] = x[k + tid];
        }
        
        // Load w tile into shared memory (coalesced per row)
        for (int kk = tid; kk < BLOCK_SIZE_K; kk += blockDim.x) {
            if (row < M && k + kk < K) {
                w_smem[tid][kk] = w[row * K + (k + kk)];
            }
        }
        
        __syncthreads();
        
        // Compute partial dot product from shared memory
        int k_end = min(BLOCK_SIZE_K, K - k);
        for (int kk = 0; kk < k_end; ++kk) {
            float w_val = w_smem[tid][kk];
            float delta = mismatch[row * K + (k + kk)]; // Can also tile mismatch
            acc += w_val * (1.0f + delta) * x_smem[kk];
        }
        
        __syncthreads();
    }
    
    // Bias, noise, quant, ReLU (same as simple kernel)
    ...
}
```

**Shared memory bank conflicts:**
Shared memory is divided into 32 banks (on A100). A bank conflict occurs when multiple threads in a warp access the same bank simultaneously. The `w_smem` array is declared with a stride of `BLOCK_SIZE_K + 1` to ensure that consecutive rows (`w_smem[tid][kk]` and `w_smem[tid+1][kk]`) are in different banks, eliminating conflicts.

**Arithmetic intensity:**
Arithmetic intensity is the ratio of floating-point operations to bytes loaded. For the analog linear layer:
- FLOPs: 2 × M × K (multiply-add for dot product) + M (bias, noise, quant, ReLU)
- Bytes loaded: 4 × (M×K + K + M + M×K) = 4 × (2MK + K + M) for FP32
- Arithmetic intensity: ≈ 2MK / (8MK) = 0.25 FLOPs/byte

For an A100 (HBM bandwidth 1.5 TB/s, FP32 throughput 19.5 TFLOPs), the roofline model predicts:
- Memory-bound if intensity < 13 FLOPs/byte (FP32)
- The analog linear layer is deeply memory-bound
- Fusion improves arithmetic intensity by reducing HBM round-trips: 5 ops → 1 kernel, so effective intensity increases from 0.05 to 0.25 FLOPs/byte per operation, but still memory-bound.

**Template specialization for compile-time branch pruning:**
The CUDA kernel uses template parameters `BLOCK_SIZE_M` and `BLOCK_SIZE_K` to specialize at compile time. This allows the compiler to:
- Unroll the inner loop over `BLOCK_SIZE_K`
- Eliminate `if (kk < k_end)` checks when `BLOCK_SIZE_K` divides K
- Optimize register allocation for known block sizes

The `AT_DISPATCH_FLOATING_TYPES` macro in the host dispatcher generates instantiations for `float` and `double`, ensuring the kernel is compiled for both dtypes.

**Substrate-aware CUDA kernel variants:**
For v2, the CUDA kernel can be specialized per substrate:
- **PCM:** Add drift term `g(t) = g0 * (t/t0)^(-nu)` in the kernel, requiring time `t` as an additional parameter.
- **ReRAM:** The mismatch is static (fabrication variation), so precomputed mismatch is sufficient.
- **Capacitive:** Charge noise scales with temperature, requiring `T` as a parameter.

These variants are compiled as separate template instantiations:
```cpp
template <typename T, int BS_M, int BS_K, SubstrateType SUB>
__global__ void analog_linear_fused_kernel(...);
```

---

## 5. C++ Extension Host (Phase 2)

### 5.1 Architecture

The C++ extension is a thin dispatch layer:
- Validates tensor shapes, devices, dtypes
- Releases Python GIL during kernel execution
- Dispatches to CUDA kernel (or Triton kernel via Python trampoline)

### 5.2 Why pybind11, Not TORCH_LIBRARY

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| `pybind11` | Simple build; no CMake; stable across PyTorch versions | No C++ API; no dispatcher integration | **Use this** — research project, simple dispatch |
| `TORCH_LIBRARY` | Full dispatcher; custom ops appear in `torch.ops`; ABI stable | Complex CMake; must match PyTorch ABI; build fragility | Skip for v1 — overkill |

### 5.3 Extension Interface

```cpp
// neuro_analog_ops.cpp
#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Forward declarations
at::Tensor analog_linear_fused_cuda(
    at::Tensor x, at::Tensor w, at::Tensor bias,
    at::Tensor mismatch, float noise_sigma, float adc_levels
);

at::Tensor analog_linear_fused_triton(
    at::Tensor x, at::Tensor w, at::Tensor bias,
    at::Tensor mismatch, float noise_sigma, float adc_levels
);

at::Tensor analog_linear_fused(
    at::Tensor x, at::Tensor w, at::Tensor bias,
    at::Tensor mismatch, float noise_sigma, float adc_levels
) {
    // Validate inputs
    TORCH_CHECK(x.device().is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.device().is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.dtype() == w.dtype(), "dtype mismatch");
    TORCH_CHECK(w.dim() == 2, "w must be 2D");
    TORCH_CHECK(x.dim() == 1, "x must be 1D");

    // Dispatch based on env var or compile flag
    const char* use_triton = std::getenv("NEURO_ANALOG_USE_TRITON");
    if (use_triton && std::string(use_triton) == "1") {
        return analog_linear_fused_triton(x, w, bias, mismatch, noise_sigma, adc_levels);
    }
    return analog_linear_fused_cuda(x, w, bias, mismatch, noise_sigma, adc_levels);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("analog_linear_fused", &analog_linear_fused,
          "Fused analog linear forward pass");
}
```

### 5.4 Build System: `setup.py` and CMake Structure

**`setup.py` (RunPod only):**
```python
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Use TORCH_CUDA_ARCH_LIST from environment, default to A100 + RTX 3090
arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', '8.0;8.6')

setup(
    name='neuro_analog_ops',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='neuro_analog_ops',
            sources=[
                'csrc/neuro_analog_ops.cpp',
                'csrc/analog_linear_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    f'-gencode=arch=compute_80,code=sm_80',
                    f'-gencode=arch=compute_86,code=sm_86',
                ],
            },
            include_dirs=[
                os.path.join(os.path.dirname(torch.__file__), 'include'),
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
    zip_safe=False,
)
```

**CMake (alternative, for MLIR integration later):**
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(neuro_analog_ops)

set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# CUDA extension
set(CUDA_SOURCES
    csrc/analog_linear_cuda.cu
)
set(CPP_SOURCES
    csrc/neuro_analog_ops.cpp
)

cuda_add_library(neuro_analog_cuda STATIC ${CUDA_SOURCES})
target_compile_options(neuro_analog_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
)

pybind11_add_module(neuro_analog_ops ${CPP_SOURCES})
target_link_libraries(neuro_analog_ops PRIVATE
    neuro_analog_cuda
    ${TORCH_LIBRARIES}
)
```

**Why `torch.utils.cpp_extension` over raw CMake:**
PyTorch's `CUDAExtension` handles:
- Locating PyTorch headers and libraries
- Setting correct `TORCH_CUDA_ARCH_LIST` for the target GPU
- Linking against `c10`, `torch`, `torch_python` libraries
- Handling Windows-specific issues (though we build on Linux/RunPod)

For a research project, `CUDAExtension` is the standard path. CMake is useful if integrating with other C++ libraries (e.g., MLIR, LLVM) or building a standalone binary.

### 5.5 JIT vs AOT Compilation

**JIT (`torch.utils.cpp_extension.load`):**
```python
from torch.utils.cpp_extension import load

neuro_analog_ops = load(
    name='neuro_analog_ops',
    sources=['csrc/neuro_analog_ops.cpp', 'csrc/analog_linear_cuda.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=True,
)
```
- Compiles on first import, caches the `.so` in `~/.cache/torch_extensions/`
- No `setup.py` or `pip install` needed
- Useful for rapid iteration on RunPod (no need to re-install the package)
- Drawback: compilation happens on every fresh environment (Colab, new pod)

**AOT (`pip install -e .`):**
- Compiles once during `pip install`, produces `.so` in the package directory
- Required for distribution (PyPI, conda)
- Required for CI/CD (pre-built wheels)
- For this project: AOT on RunPod, but keep JIT option for rapid debugging

**Recommendation:** Use AOT (`setup.py` + `pip install -e .`) for the RunPod environment. Keep a JIT script (`jit_load.py`) for emergency debugging when the AOT build is broken.

---

## 6. MLIR Dialect (Phase 3 — Stretch Goal)

### 6.1 Honest Scope

**v1 scope (stretch goal, only if Phase 1-2 finish early):**
- Define an `analog` dialect with one op: `analog.mvm`
- Write a `ConversionPattern` that lowers `analog.mvm` to `func.call` into the C++ extension
- This is NOT a full GPU compiler. It does NOT emit PTX, CUDA, or Triton.
- It is a proof-of-concept that the analog operation can be represented in MLIR and lowered to a callable function.

**What v1 does NOT include:**
- GPU lowering via `gpu.launch` or `nvvm`
- Bufferization and memory planning
- Custom MLIR pass pipeline
- Integration with `torch-mlir` (Torch dialect)

### 6.2 Dialect Definition (ODS)

```mlir
// AnalogOps.td
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Analog_Dialect : Dialect {
  let name = "analog";
  let summary = "Analog inference operations for neuromorphic hardware";
  let description = [{
    This dialect models operations that execute on analog crossbar arrays,
    including matrix-vector multiplication with conductance noise, ADC quantization,
    and ReLU activation.
  }];
  let cppNamespace = "analog";
}

def Analog_MVMOp : Analog_Op<"mvm", [Pure]> {
  let summary = "Analog matrix-vector multiplication";
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weights,
    AnyRankedTensor:$bias,
    F32Attr:$noise_sigma,
    I32Attr:$adc_levels
  );
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = [{
    $input `,` $weights `,` $bias `,` `noise` `=` $noise_sigma `,` `adc` `=` $adc_levels attr-dict
  }];
}
```

### 6.3 Lowering Pattern

```cpp
// LowerAnalogToFunc.cpp
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

class LowerAnalogMVM : public OpConversionPattern<analog::MVMOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(analog::MVMOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Create func.call to the C++ extension
    auto funcOp = rewriter.create<func::FuncOp>(
        op.getLoc(), "analog_linear_fused",
        rewriter.getFunctionType(op.getOperandTypes(), op.getResultTypes()));

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, funcOp, adaptor.getOperands());

    return success();
  }
};
```

### 6.4 Bufferization vs. Tensor World

MLIR has two representations for tensor data:
1. **Tensor world:** `tensor<...>` values are immutable, SSA values. Operations like `analog.mvm` take and return tensors. This is the high-level representation, suitable for analysis and optimization.
2. **Buffer world:** `memref<...>` values are mutable buffers in memory. The `bufferization` pass converts tensor operations to memref operations, allocating buffers and inserting copies where necessary.

For the analog dialect, the lowering pipeline is:
```
Torch dialect (from torch-mlir)
  → Tensor world analog dialect
    → Bufferization pass
      → Memref world analog dialect
        → Lowering to func.call (C++ extension)
```

The bufferization step is critical because the C++ extension expects `at::Tensor` (which is a buffer, not a value). The MLIR `func.call` lowering must pass `memref` descriptors to the C++ function. On the C++ side, the `memref` is converted to a `torch::Tensor` via the Python C-API or PyTorch's C++ tensor bindings.

**Honest v1 scope for MLIR:**
- Define the `analog` dialect in TableGen (ODS file)
- Write a `ConversionPattern` that lowers `analog.mvm` to `func.call` into the C++ extension
- Build a toy example that parses the dialect, runs the conversion, and prints the result
- This is NOT a full GPU compiler. It does NOT emit PTX, CUDA, or Triton.
- It does NOT integrate with `torch-mlir` (Torch dialect) in v1.
- It is a proof-of-concept that the analog operation can be represented in MLIR and lowered to a callable function.

**MLIR CMake targets (RunPod):**
```cmake
# CMakeLists.txt for MLIR dialect
set(LLVM_SOURCES
    lib/AnalogDialect.cpp
    lib/AnalogOps.cpp
    lib/LowerAnalogToFunc.cpp
)

add_mlir_dialect_library(MLIRAnalog
    ${LLVM_SOURCES}
    DEPENDS
    MLIRAnalogOpsIncGen
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRTransforms
)
```

### 6.5 Why MLIR Is a Stretch Goal

1. **Time cost:** 2-4 weeks to build, test, and integrate a dialect + lowering.
2. **No immediate speedup:** MLIR lowering to `func.call` does not improve performance over direct Python dispatch.
3. **Build complexity:** Requires building MLIR/LLVM from source or linking against `libMLIR`. On RunPod, this adds 30-60 minutes to the setup.
4. **Interview value:** High, but only if the GPU kernel phases are solid. A broken MLIR pass with no working kernel is worse than no MLIR at all.
5. **Bufferization complexity:** The tensor-to-memref conversion is non-trivial for operations with in-place semantics (e.g., analog MVM that reads weights and writes to output). Understanding bufferization is a learning curve.

---

## 7. Substrate Integration (Phase 2-4)

### 7.1 How Substrate Models Map to Kernel Parameters

The `neuro-analog` simulator defines substrate models in `substrates.py`:
- **PCM:** Phase-change memory with drift and noise
- **ReRAM:** Resistive RAM with conductance mismatch
- **Capacitive:** Capacitor-based with charge noise

**v1 approach (kernel receives precomputed parameters):**
```python
# Host-side (Python/RunPod): precompute mismatch and noise parameters
w_eff, sigma_th = substrate.apply(w)  # Physics-based perturbation
mismatch = (w_eff - w) / w  # Normalized mismatch

# Kernel receives: w, mismatch, noise_sigma, adc_levels
# Kernel computes: y = (w * (1 + mismatch)) @ x + N(0, noise_sigma)
```

**Why precompute on host:**
- Substrate models involve complex physics (e.g., PCM drift is time-dependent). Computing these in-kernel requires passing extra state (time, temperature, cycle count) and increases register pressure.
- v1 keeps the kernel simple: it applies precomputed mismatch and noise, not full physics simulation.

**v2 future work:**
- In-kernel PCM drift model with Philox RNG state per device
- Temperature-dependent noise scaling (read temperature from device state)
- Cycle-dependent wear (endurance model)

### 7.2 ADC Quantization Model

```python
# Host-side: compute ADC levels from HardwareProfile
adc_levels = 2 ** profile.adc_bits  # e.g., 8 bits → 256 levels
adc_step = profile.v_range / adc_levels

# Kernel-side: quantize to discrete levels
y_quant = floor(y * adc_levels) / adc_levels
```

### 7.3 Substrate-Aware Kernel Variants (v2)

For v1, the kernel receives precomputed mismatch and noise parameters. For v2, the kernel can compute substrate-specific perturbations in-kernel:

**PCM variant:**
```cpp
template <typename T>
__device__ T pcm_drift(T g, float t, float nu, float t0) {
    return g * powf(t / t0, -nu);
}

// In kernel: w_eff = pcm_drift(w, time, 0.1f, 100.0f);
```

**ReRAM variant:**
ReRAM mismatch is static (fabrication variation), so precomputed `mismatch` is optimal. No in-kernel computation needed.

**Capacitive variant:**
```cpp
template <typename T>
__device__ T capacitive_noise(T g, float T_temp, float k_b, float C) {
    float sigma = sqrtf(k_b * T_temp / C);
    return g + curand_normal(&state) * sigma;
}
```

These variants are compiled as separate template instantiations or separate kernels. The host code selects the appropriate kernel based on the substrate type:
```cpp
if (substrate_type == SubstrateType::PCM) {
    analog_linear_fused_pcm_kernel<<<...>>>(...);
} else if (substrate_type == SubstrateType::ReRAM) {
    analog_linear_fused_reram_kernel<<<...>>>(...);
}
```

---

## 8. Build System & Remote Development

### 8.1 Local Machine (CPU-only)

**What works locally:**
- Triton kernel development (syntax, logic, small shapes)
- Python frontend code (`analog_linear.py`, `substrates.py`)
- Unit tests with mocked CUDA (use `torch.cuda.is_available()` guards)

**What does NOT work locally:**
- CUDA kernel compilation (requires `nvcc`)
- C++ extension build (requires `ninja`, `nvcc`, CUDA headers)
- Performance benchmarking (no GPU)

### 8.2 RunPod (Primary GPU Environment)

**Why RunPod over Colab for C++ builds:**
| Feature | RunPod | Colab |
|---|---|---|
| Persistent storage | Yes (SSD persists across sessions) | No (resets every 12 hours) |
| nvcc pre-installed | Yes | Yes |
| ninja pre-installed | Yes | Must `!pip install` |
| `pip install -e .` | Yes, works in terminal | Works in cell, but .so lost on reset |
| pytest in terminal | Yes | Must run in cell |
| SSH access | Yes | No (notebook only) |
| Cost | ~$0.40/hr (A100) | Free |

**Recommendation:**
- **Triton development:** Use Colab for quick testing (free, instant GPU). Or RunPod for persistent work.
- **C++ extension build:** Use RunPod ONLY. Colab is too painful for iterative C++ builds.

### 8.3 RunPod Setup Guide

**Step 1: Create Pod**
- Template: `RunPod Pytorch 2.3` (includes CUDA 12.1, PyTorch 2.3)
- GPU: RTX 3090 or A100 (cheapest for builds: RTX 3090 at ~$0.30/hr)

**Step 2: SSH Access**
```bash
# On your local Windows machine (PowerShell)
ssh -i ~/.ssh/runpod_key root@<runpod-ip>
```

**Step 3: Clone & Setup**
```bash
# On RunPod
apt-get update && apt-get install -y ninja-build
pip install triton pytest

git clone https://github.com/<user>/neuro-analog.git
cd neuro-analog
pip install -e .
```

**Step 4: Build C++ Extension**
```bash
cd neuro-analog/kernels/cuda
python setup.py install
# or: pip install -e .
```

**Step 5: Test**
```bash
pytest tests/test_fused_kernel.py -v
```

**Step 6: Download Artifacts**
```bash
# On RunPod, tar results and scp to local machine
tar czf results.tar.gz benchmarks/ tests/
# On local machine
scp -i ~/.ssh/runpod_key root@<runpod-ip>:/workspace/neuro-analog/results.tar.gz .
```

### 8.4 Windows Local Development (WSL2)

While all GPU builds happen on RunPod, the local Windows machine can still run the C++ extension build in WSL2 for local testing:

**WSL2 Ubuntu setup:**
```bash
# In WSL2
sudo apt-get update
sudo apt-get install -y build-essential ninja-build cmake
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton pytest
```

**WSL2 CUDA build:**
```bash
# WSL2 has CUDA support if the host has an NVIDIA GPU
# Since this machine is CPU-only, WSL2 CUDA is not available
# Skip WSL2 GPU builds; use RunPod for all GPU work
```

**Local Windows build (CPU only, for syntax checking):**
```powershell
# PowerShell: install Visual Studio Build Tools with C++ workload
# pip install torch (CPU version)
# The C++ extension will fail at CUDA compilation, but C++ syntax can be checked
```

### 8.5 Environment Variables

```bash
# RunPod .bashrc or pod env
export TORCH_CUDA_ARCH_LIST="8.0;8.6"  # A100=8.0, RTX 3090=8.6
export MAX_JOBS=4  # Limit parallel compilation jobs
export NEURO_ANALOG_USE_TRITON=1  # Toggle Triton vs CUDA backend
export CUDA_HOME=/usr/local/cuda  # Ensure nvcc is in PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Why `TORCH_CUDA_ARCH_LIST` matters:**
PyTorch's `CUDAExtension` uses `TORCH_CUDA_ARCH_LIST` to determine which GPU architectures to compile for. If the target GPU is an A100 (sm_80) but the list is set to `7.0` (V100), the kernel will compile but fail at runtime with "no kernel image is available." Always set this to match the RunPod GPU:
- A100: `8.0`
- RTX 3090: `8.6`
- A6000: `8.6`
- V100: `7.0`

---

## 9. Testing & Validation

### 9.1 Correctness Tests

```python
# tests/test_fused_kernel.py
import torch
import pytest

def test_fused_matches_reference():
    """Fused kernel must match Python reference implementation."""
    M, K = 256, 128
    x = torch.randn(K, device='cuda')
    w = torch.randn(M, K, device='cuda')
    bias = torch.randn(M, device='cuda')
    mismatch = torch.randn(M, K, device='cuda') * 0.01
    noise_sigma = 0.05
    adc_levels = 256.0

    # Python reference
    y_ref = torch.nn.functional.linear(x, w * (1 + mismatch), bias)
    y_ref += torch.randn_like(y_ref) * noise_sigma
    y_ref = torch.floor(y_ref * adc_levels) / adc_levels
    y_ref = torch.relu(y_ref)

    # Fused kernel (Triton)
    y_fused = analog_linear_fused_triton(x, w, bias, mismatch, noise_sigma, adc_levels)

    # Allow small numerical differences (RNG, order of operations)
    torch.testing.assert_close(y_ref, y_fused, atol=1e-3, rtol=1e-3)

def test_fused_matches_cuda():
    """Triton and CUDA kernels must match."""
    # Same setup, compare Triton vs CUDA
    ...
```

### 9.2 Performance Benchmarks

```python
# benchmarks/bench_fused_kernel.py
import torch
import time

def benchmark():
    shapes = [(1024, 512), (4096, 1024), (16384, 4096)]
    for M, K in shapes:
        x = torch.randn(K, device='cuda')
        w = torch.randn(M, K, device='cuda')
        bias = torch.randn(M, device='cuda')
        mismatch = torch.randn(M, K, device='cuda') * 0.01

        # PyTorch baseline (5 separate ops)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            y = torch.nn.functional.linear(x, w, bias)
            y += torch.randn_like(y) * 0.05
            y = torch.floor(y * 256) / 256
            y = torch.relu(y)
        torch.cuda.synchronize()
        t_baseline = (time.time() - t0) / 100

        # Fused kernel
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            y = analog_linear_fused_triton(x, w, bias, mismatch, 0.05, 256)
        torch.cuda.synchronize()
        t_fused = (time.time() - t0) / 100

        print(f"M={M}, K={K}: Baseline={t_baseline*1000:.2f}ms, Fused={t_fused*1000:.2f}ms, Speedup={t_baseline/t_fused:.2f}x")
```

**Target performance:**
- 2× speedup for M=1024, K=512
- 3× speedup for M=4096, K=1024
- 3× speedup for M=16384, K=4096

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Triton kernel slower than PyTorch** | Medium | Low | Interview value is in code + understanding, not raw speed. Even 1.2× with correct explanation is valuable. |
| **C++ extension build fails on RunPod** | Medium | Medium | Use `torch.utils.cpp_extension.load()` for JIT compilation (no setup.py). Fallback to Colab if persistent. |
| **MLIR dialect takes too long** | High | Medium | MLIR is stretch goal. Drop it if Phase 1-2 run over. |
| **Numerical mismatch between fused and reference** | Medium | High | Add deterministic RNG (fixed seed), compare element-wise, debug each op separately. |
| **RunPod cost overruns** | Low | Low | Use RTX 3090 for builds (~$0.30/hr), A100 only for benchmarking. Limit sessions to 2-4 hours. |
| **Windows local environment issues** | Medium | Low | All GPU work is on RunPod. Local machine only for Python code editing. |

---

## 11. Timeline & Milestones

### Milestone 0: Environment Setup (Day 0-1)
- [ ] Set up RunPod account with SSH key
- [ ] Launch test pod, verify `nvcc --version`, `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Clone neuro-analog repo on RunPod
- [ ] **Go/No-Go:** Can SSH into RunPod and PyTorch sees CUDA. If not, use Colab fallback.

### Milestone 1: Triton Kernel (Day 1-4)
- [ ] Write `analog_linear_fused_kernel` in Triton
- [ ] Test on Colab with small shapes (M=256, K=128)
- [ ] Test on RunPod with larger shapes (M=4096, K=1024)
- [ ] Write benchmark script, compare to PyTorch baseline
- [ ] **Go/No-Go:** Kernel runs without errors, produces output within 2× of PyTorch baseline. If not, debug tiling and autotune configs.

### Milestone 2: CUDA Baseline Kernel (Day 5-7)
- [ ] Write `analog_linear_fused_kernel` in CUDA
- [ ] Write `setup.py` and build on RunPod
- [ ] Validate against Triton kernel (numerical match)
- [ ] Benchmark CUDA vs Triton vs PyTorch
- [ ] **Go/No-Go:** Build succeeds, CUDA output matches Triton. If not, fix `nvcc` flags or build environment.

### Milestone 3: C++ Extension & PyTorch Integration (Day 8-10)
- [ ] Write `neuro_analog_ops.cpp` with pybind11 dispatch
- [ ] Integrate with `analog_linear.py` (fallback logic: if fused available, use it)
- [ ] Write unit tests for correctness and shape handling
- [ ] Run full test suite on RunPod
- [ ] **Go/No-Go:** `import neuro_analog_ops` works, `analog_linear` dispatches correctly. If not, fix ABI or path issues.

### Milestone 4: Substrate Integration & Benchmarks (Day 11-14)
- [ ] Hook `substrates.py` output into kernel parameters (precomputed mismatch)
- [ ] Run end-to-end: train small model → analogize → fused forward pass
- [ ] Generate benchmark plots (speedup vs. shape, vs. baseline)
- [ ] Write README section on fused kernel
- [ ] **Go/No-Go:** End-to-end pipeline works, benchmarks show 2-3× speedup. If not, profile with Nsight Compute on RunPod.

### Milestone 5: MLIR Stretch Goal (Day 15-20, if time)
- [ ] Define `analog` dialect with `analog.mvm` op in TableGen (ODS)
- [ ] Write `ConversionPattern` that lowers `analog.mvm` to `func.call` into C++ extension
- [ ] Build on RunPod with MLIR/LLVM (install `mlir` via pip or build from source)
- [ ] Run toy example: `mlir-opt` → lowered module → print
- [ ] Document bufferization vs tensor world in README
- [ ] **Go/No-Go:** If Milestone 4 completed by Day 14, proceed. Otherwise, defer MLIR to post-application.

**Timeline summary:**
| Phase | Days | Deliverables | Go/No-Go |
|---|---|---|---|
| Milestone 0 | 0-1 | RunPod SSH, PyTorch CUDA works | Can run GPU code on RunPod |
| Milestone 1 | 1-4 | Triton kernel, Colab/RunPod tests, benchmark | Kernel runs, ≤2× baseline |
| Milestone 2 | 5-7 | CUDA kernel, build on RunPod, numerical match | CUDA output matches Triton |
| Milestone 3 | 8-10 | C++ extension, pybind11 dispatch, Python integration | `import` works, dispatch correct |
| Milestone 4 | 11-14 | Substrate hooks, end-to-end, benchmarks, README | 2-3× speedup, docs complete |
| Milestone 5 | 15-20 | MLIR dialect, lowering, toy example | Only if M4 done early |

---

## 12. References

1. Triton Documentation: https://triton-lang.org/main/getting-started/tutorials/index.html
2. PyTorch C++ Extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html
3. CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
4. MLIR Dialect Tutorial: https://mlir.llvm.org/docs/Tutorials/Toy/
5. RunPod Documentation: https://docs.runpod.io/
6. neuro-analog source: `analog_linear.py`, `substrates.py`, `hardware_profile.py`

---

## Appendix A: RunPod Quick Reference

**SSH into pod:**
```powershell
# Windows PowerShell
ssh -i $env:USERPROFILE\.ssh\runpod_key root@<ip>
```

**Copy files to/from pod:**
```powershell
# To pod
scp -i $env:USERPROFILE\.ssh\runpod_key .\file.py root@<ip>:/workspace/

# From pod
scp -i $env:USERPROFILE\.ssh\runpod_key root@<ip>:/workspace/results.tar.gz .
```

**Stop pod (save money):**
```bash
# On RunPod dashboard, click "Stop" on the pod
# Storage persists; you resume by clicking "Start"
```
