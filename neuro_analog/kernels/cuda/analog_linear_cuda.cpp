/**
 * analog_linear_cuda.cpp
 * C++ host dispatcher: pybind11 module wrapping CUDA kernels.
 *
 * Validates inputs, releases GIL, and launches the fused kernel.
 * Built on RunPod with nvcc via setup.py.
 */
#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Forward declaration from .cu file
extern "C" void launch_analog_linear_fused(
    const float* x, const float* w, const float* bias,
    float* out, const float* mismatch,
    float noise_sigma, float adc_levels,
    int M, int K, int B
);

at::Tensor analog_linear_fused_cuda(
    at::Tensor x,
    at::Tensor w,
    c10::optional<at::Tensor> bias,
    c10::optional<at::Tensor> mismatch,
    float noise_sigma,
    float adc_levels
) {
    // ── Input validation ──
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "w must be float32");
    TORCH_CHECK(w.dim() == 2, "w must be 2D (M, K)");
    TORCH_CHECK(x.dim() == 1 || x.dim() == 2, "x must be 1D (K,) or 2D (B, K)");

    int M = w.size(0);
    int K = w.size(1);

    if (x.dim() == 1) {
        TORCH_CHECK(x.size(0) == K,
            "x size mismatch: expected ", K, " but got ", x.size(0));
    } else {
        TORCH_CHECK(x.size(1) == K,
            "x size mismatch: expected ", K, " in dim 1 but got ", x.size(1));
    }

    // ── Allocate output ──
    at::Tensor out;
    if (x.dim() == 1) {
        out = torch::empty({M}, x.options());
    } else {
        int B = x.size(0);
        out = torch::empty({B, M}, x.options());
    }

    // ── Prepare optional tensors ──
    at::Tensor bias_t;
    if (bias.has_value()) {
        bias_t = bias.value().to(x.device()).contiguous();
        TORCH_CHECK(bias_t.size(0) == M, "bias size mismatch");
    } else {
        bias_t = torch::zeros({M}, x.options());
    }

    at::Tensor mismatch_t;
    if (mismatch.has_value()) {
        mismatch_t = mismatch.value().to(x.device()).contiguous();
        TORCH_CHECK(mismatch_t.sizes() == w.sizes(), "mismatch shape mismatch");
    } else {
        mismatch_t = torch::zeros_like(w);
    }

    // ── Make contiguous ──
    x = x.contiguous();
    w = w.contiguous();
    out = out.contiguous();

    // ── Release GIL and launch kernel ──
    int B = (x.dim() == 2) ? x.size(0) : 1;

    // PyGILState_Release is called implicitly by py::gil_scoped_release
    {
        py::gil_scoped_release release;
        launch_analog_linear_fused(
            x.data_ptr<float>(),
            w.data_ptr<float>(),
            bias_t.data_ptr<float>(),
            out.data_ptr<float>(),
            mismatch_t.data_ptr<float>(),
            noise_sigma,
            adc_levels,
            M, K, B
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "analog_linear_fused_cuda",
        &analog_linear_fused_cuda,
        "Fused analog linear forward pass (CUDA kernel)",
        py::arg("x"),
        py::arg("w"),
        py::arg("bias") = c10::nullopt,
        py::arg("mismatch") = c10::nullopt,
        py::arg("noise_sigma") = 0.0f,
        py::arg("adc_levels") = 0.0f
    );
}
