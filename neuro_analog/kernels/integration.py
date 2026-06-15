"""Integration wrapper: AnalogLinearFused module.

Dispatches between:
  1. Triton fused kernel (primary, NEURO_ANALOG_BACKEND=triton)
  2. CUDA fused kernel (fallback, NEURO_ANALOG_BACKEND=cuda)
  3. Python reference (debug, NEURO_ANALOG_BACKEND=python)

Usage:
    from neuro_analog.kernels.integration import AnalogLinearFused

    layer = AnalogLinearFused.from_analog_linear(analog_layer)
    y = layer(x)  # Dispatches based on backend + device
"""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from neuro_analog.simulator.analog_linear import AnalogLinear
    from neuro_analog.simulator.substrates import SubstrateBase

# Backend selection via environment variable
_BACKEND = os.environ.get("NEURO_ANALOG_BACKEND", "triton").lower()

# Lazy imports — don't fail if Triton/CUDA not available
try:
    from .triton import analog_linear_fused as _triton_fused
    from .triton import precompute_kernel_params as _precompute_params
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    _triton_fused = None
    _precompute_params = None

try:
    import neuro_analog_cuda
    _HAS_CUDA_EXT = True
except ImportError:
    _HAS_CUDA_EXT = False
    neuro_analog_cuda = None


class AnalogLinearFused(nn.Module):
    """Fused analog linear layer that dispatches to the appropriate backend.

    Parameters are extracted from an existing AnalogLinear instance,
    so no retraining or weight copying is needed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        mismatch: torch.Tensor | None = None,
        noise_sigma: float = 0.0,
        adc_levels: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # Buffers (non-trainable)
        self.register_buffer("weight", weight.clone().detach().float())
        self.register_buffer("bias", bias.clone().detach().float() if bias is not None else None)
        if mismatch is not None:
            self.register_buffer("mismatch", mismatch.clone().detach().float())
        else:
            self.register_buffer("mismatch", None)

        self.noise_sigma = noise_sigma
        self.adc_levels = adc_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dispatch(x)

    def _dispatch(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to the appropriate backend based on config + device."""
        x = x.float()

        # CPU: always use Python reference
        if not x.is_cuda:
            return self._python_reference(x)

        # GPU: select backend
        if _BACKEND == "triton" and _HAS_TRITON:
            return _triton_fused(
                x, self.weight, self.bias, self.mismatch,
                self.noise_sigma, self.adc_levels,
                activation=self.activation,
            )
        elif _BACKEND == "cuda" and _HAS_CUDA_EXT:
            return neuro_analog_cuda.analog_linear_fused_cuda(
                x, self.weight, self.bias, self.mismatch,
                self.noise_sigma, self.adc_levels,
            )
        else:
            # Fallback to Python reference
            return self._python_reference(x)

    def _python_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Python sequential implementation (correctness baseline)."""
        w_eff = self.weight * (1.0 + self.mismatch) if self.mismatch is not None else self.weight
        y = F.linear(x, w_eff, self.bias)

        if self.noise_sigma > 0:
            y = y + torch.randn_like(y) * self.noise_sigma

        if self.adc_levels > 0:
            y = torch.floor(y * self.adc_levels) / self.adc_levels

        if self.activation == "relu":
            y = torch.relu(y)
        elif self.activation == "none":
            pass
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        return y

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"backend={_BACKEND}, activation={self.activation}"
        )

    @classmethod
    def from_analog_linear(
        cls,
        analog_linear: "AnalogLinear",
        substrate: "SubstrateBase" | None = None,
    ) -> "AnalogLinearFused":
        """Create a fused layer from an existing AnalogLinear instance.

        Extracts precomputed mismatch, noise sigma, and ADC levels
        from the source layer.
        """
        if _precompute_params is not None:
            params = _precompute_params(analog_linear, substrate)
        else:
            # Fallback: compute params manually without substrate
            w = analog_linear.W_nominal
            M, K = w.shape

            if analog_linear.sigma_mismatch > 0:
                mismatch = analog_linear.delta - 1.0
            else:
                mismatch = None

            from neuro_analog.simulator.analog_linear import _K_B
            noise_sigma = math.sqrt(_K_B * analog_linear.temperature_K / analog_linear.cap_F) * math.sqrt(K)

            if analog_linear.n_adc_bits < 32:
                adc_levels = float(2 ** analog_linear.n_adc_bits - 1)
            else:
                adc_levels = 0.0

            params = {
                "mismatch": mismatch,
                "noise_sigma": noise_sigma,
                "adc_levels": adc_levels,
            }

        return cls(
            in_features=analog_linear.in_features,
            out_features=analog_linear.out_features,
            weight=analog_linear.W_nominal,
            bias=analog_linear.bias,
            mismatch=params.get("mismatch"),
            noise_sigma=params.get("noise_sigma", 0.0),
            adc_levels=params.get("adc_levels", 0.0),
            activation="relu",
        )


def replace_analog_layers(
    model: nn.Module,
    substrate: "SubstrateBase" | None = None,
) -> nn.Module:
    """Replace all AnalogLinear modules in a model with AnalogLinearFused.

    In-place replacement. The original modules are destroyed.

    Example:
        model = MyModel()
        model = replace_analog_layers(model, substrate=my_substrate)
        y = model(x)  # Uses fused kernels on GPU
    """
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "AnalogLinear":
            fused = AnalogLinearFused.from_analog_linear(
                module, substrate or getattr(module, "substrate", None)
            )
            # Replace in parent
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, fused)
    return model
