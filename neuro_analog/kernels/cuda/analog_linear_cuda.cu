/**
 * analog_linear_cuda.cu
 * CUDA kernel: fused analog linear forward pass.
 *
 * Fuses: matmul + mismatch + thermal noise + ADC quantization + ReLU.
 * Built on RunPod with nvcc; compiled via setup.py.
 */
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

// Forward declaration of host dispatcher (in analog_linear_cuda.cpp)
extern "C" void launch_analog_linear_fused(
    const float* x, const float* w, const float* bias,
    float* out, const float* mismatch,
    float noise_sigma, float adc_levels,
    int M, int K, int B
);

/**
 * Simple row-wise kernel (no shared memory tiling).
 * Each thread block processes one row of the output (BLOCK_SIZE_M threads).
 * Optimal for K <= 1024 (all data fits in registers + L1).
 */
template <int BLOCK_SIZE_M, int BLOCK_SIZE_K>
__global__ void analog_linear_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const float* __restrict__ mismatch,
    float noise_sigma,
    float adc_levels,
    int M, int K
) {
    int bm = blockIdx.x;
    int tid = threadIdx.x;

    int row = bm * BLOCK_SIZE_M + tid;
    if (row >= M) return;

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

    // Thermal noise (Philox RNG — deterministic, reproducible)
    curandStatePhilox4_32_10_t state;
    curand_init(42, row, 0, &state);
    acc += curand_normal(&state) * noise_sigma;

    // ADC quantization
    acc = floorf(acc * adc_levels) / adc_levels;

    // ReLU
    acc = fmaxf(acc, 0.0f);

    out[row] = acc;
}

/**
 * Tiled kernel with shared memory (for large K > 1024).
 * Tiles the K dimension into shared memory to reduce global memory traffic.
 * Bank-conflict-free via +1 padding on w_smem.
 */
template <int BLOCK_SIZE_M, int BLOCK_SIZE_K>
__global__ void analog_linear_fused_tiled_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const float* __restrict__ mismatch,
    float noise_sigma,
    float adc_levels,
    int M, int K
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
            float delta = mismatch[row * K + (k + kk)];
            acc += w_val * (1.0f + delta) * x_smem[kk];
        }

        __syncthreads();
    }

    // Bias, noise, quant, ReLU (same as simple kernel)
    acc += bias[row];

    curandStatePhilox4_32_10_t state;
    curand_init(42, row, 0, &state);
    acc += curand_normal(&state) * noise_sigma;

    acc = floorf(acc * adc_levels) / adc_levels;
    acc = fmaxf(acc, 0.0f);

    out[row] = acc;
}

/**
 * Host dispatcher: selects kernel variant based on shape.
 * Called from analog_linear_cuda.cpp via extern "C" linkage.
 */
extern "C" void launch_analog_linear_fused(
    const float* x, const float* w, const float* bias,
    float* out, const float* mismatch,
    float noise_sigma, float adc_levels,
    int M, int K, int B
) {
    // Kernel launch configuration
    const int BLOCK_SIZE_M = 64;

    // Select simple or tiled kernel based on K
    if (K <= 1024) {
        const int BLOCK_SIZE_K = 32;
        dim3 blocks((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
        dim3 threads(BLOCK_SIZE_M);

        for (int b = 0; b < B; ++b) {
            analog_linear_fused_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K>
                <<<blocks, threads>>>(
                    x + b * K, w, bias, out + b * M, mismatch,
                    noise_sigma, adc_levels, M, K
                );
        }
    } else {
        const int BLOCK_SIZE_K = 64;
        dim3 blocks((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
        dim3 threads(BLOCK_SIZE_M);

        for (int b = 0; b < B; ++b) {
            analog_linear_fused_tiled_kernel<BLOCK_SIZE_M, BLOCK_SIZE_K>
                <<<blocks, threads>>>(
                    x + b * K, w, bias, out + b * M, mismatch,
                    noise_sigma, adc_levels, M, K
                );
        }
    }
}

// Explicit template instantiations for common shapes (optional, for build speed)
template __global__ void analog_linear_fused_kernel<64, 32>(
    const float*, const float*, const float*, float*, const float*,
    float, float, int, int
);
template __global__ void analog_linear_fused_tiled_kernel<64, 64>(
    const float*, const float*, const float*, float*, const float*,
    float, float, int, int
);
