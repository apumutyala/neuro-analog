"""
AnalogConv1d / AnalogConv2d / AnalogConv3d: convolutional layer replacements.

Physical model — identical to AnalogLinear, applied to the convolution weight tensor:

0. INPUT DAC QUANTIZATION
   Same uniform quantizer applied to the input tensor before convolution.
   Gated by _is_readout (same semantics as AnalogLinear).

1. CONDUCTANCE MISMATCH (static, per device)
   W_device = W_nominal * δ_W,  δ_W ~ N(1, σ²)  — same shape as weight tensor
   b_device = b_nominal * δ_b,  δ_b ~ N(1, σ²)  — per output channel
   In analog hardware, a convolution is a tiled MVM: the input is unfolded
   (im2col) and the kernel weights form one row of the crossbar per output channel.
   Each weight cell has its own fabricated conductance with independent mismatch.
   Bias current sources follow the same mismatch model.

2. THERMAL READ NOISE
   σ_thermal = sqrt(kT/C) * sqrt(N_in)
   N_in = in_channels * kH * kW / groups  (receptive field size — the number of
   input cells accumulated into one output node on the crossbar).

3. ADC QUANTIZATION
   Same uniform quantizer as AnalogLinear, applied to the convolution output.

References:
  Shem §4.1-4.3 (mismatch, thermal, quantization).
  Legno §4 (sqrt(N) thermal noise model for summed column currents).
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_K_B: float = 1.380649e-23
_DEFAULT_TEMP_K: float = 300.0
_DEFAULT_CAP_F: float = 1e-12


def _make_analog_conv(
    conv_cls,           # F.conv1d / F.conv2d / F.conv3d
    spatial_dims: int,  # 1, 2, or 3
):
    """Factory that returns an AnalogConvNd class for a given spatial dimension."""

    class AnalogConvNd(nn.Module):
        __doc__ = f"""Analog crossbar simulation of nn.Conv{spatial_dims}d.

        Same three noise sources as AnalogLinear (mismatch, thermal, quantization)
        applied to the convolution weight tensor. N_in for thermal noise is the
        receptive field size: in_channels * {'*'.join(['k'] * spatial_dims)} / groups.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple],
            stride: Union[int, tuple] = 1,
            padding: Union[int, tuple, str] = 0,
            dilation: Union[int, tuple] = 1,
            groups: int = 1,
            weight: torch.Tensor = None,
            bias_data: torch.Tensor | None = None,
            sigma_mismatch: float = 0.05,
            n_adc_bits: int = 8,
            temperature_K: float = _DEFAULT_TEMP_K,
            cap_F: float = _DEFAULT_CAP_F,
            v_ref: float = 1.0,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.sigma_mismatch = sigma_mismatch
            self.n_adc_bits = n_adc_bits
            self.temperature_K = temperature_K
            self.cap_F = cap_F
            self.v_ref = v_ref
            self._use_mismatch = True
            self._use_thermal = True
            self._use_quantization = True
            self._is_readout = True
            self._conv_fn = conv_cls

            # Normalise kernel_size to tuple
            if isinstance(kernel_size, int):
                self.kernel_size = (kernel_size,) * spatial_dims
            else:
                self.kernel_size = tuple(kernel_size)

            # N_in: receptive field size — number of input values summed at one output node
            self._n_in = (in_channels // groups) * math.prod(self.kernel_size)

            if weight is not None:
                self.register_buffer("W_nominal", weight.clone().detach().float())
            else:
                W = torch.empty(out_channels, in_channels // groups, *self.kernel_size)
                nn.init.kaiming_uniform_(W, a=math.sqrt(5))
                self.register_buffer("W_nominal", W)

            if bias_data is not None:
                self.register_buffer("bias", bias_data.clone().detach().float())
            else:
                self.register_buffer("bias", None)

            # δ_W has the same shape as the weight tensor; δ_b is per output channel
            self.register_buffer("delta", torch.ones_like(self.W_nominal))
            if bias_data is not None:
                self.register_buffer("delta_bias", torch.ones(out_channels))
            else:
                self.register_buffer("delta_bias", None)
            self._resample_delta()

        def resample_mismatch(self, sigma: float | None = None) -> None:
            if sigma is not None:
                self.sigma_mismatch = sigma
            self._resample_delta()

        def set_noise_config(
            self, thermal: bool = True, quantization: bool = True, mismatch: bool = True
        ) -> None:
            self._use_mismatch = mismatch
            self._use_thermal = thermal
            self._use_quantization = quantization

        def calibrate(self, x: torch.Tensor) -> None:
            """Set V_ref from a noiseless forward pass."""
            with torch.no_grad():
                y = self._conv_fn(
                    x.float(), self.W_nominal, self.bias,
                    self.stride, self.padding, self.dilation, self.groups,
                )
                peak = float(y.abs().max().item())
                self.v_ref = peak * 1.1 if peak > 0 else 1.0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.float()

            # Step 0 — Input DAC quantization
            if self._use_quantization and self._is_readout and self.n_adc_bits < 32:
                n_levels = 2 ** self.n_adc_bits - 1
                scale = n_levels / (2.0 * self.v_ref)
                x = torch.clamp(x, -self.v_ref, self.v_ref)
                x = torch.round(x * scale) / scale

            # Step 1 — conductance mismatch
            if self._use_mismatch and self.sigma_mismatch > 0:
                W_eff = self.W_nominal * self.delta
                bias_eff = self.bias * self.delta_bias if self.delta_bias is not None else self.bias
            else:
                W_eff = self.W_nominal
                bias_eff = self.bias

            y = self._conv_fn(
                x, W_eff, bias_eff,
                self.stride, self.padding, self.dilation, self.groups,
            )

            # Step 2 — thermal read noise: σ = sqrt(kT/C) * sqrt(N_in)
            if self._use_thermal:
                sigma_th = (
                    math.sqrt(_K_B * self.temperature_K / self.cap_F)
                    * math.sqrt(self._n_in)
                )
                y = y + torch.randn_like(y) * sigma_th

            # Step 3 — ADC quantization
            if self._use_quantization and self._is_readout and self.n_adc_bits < 32:
                n_levels = 2 ** self.n_adc_bits - 1
                scale = n_levels / (2.0 * self.v_ref)
                y = torch.clamp(y, -self.v_ref, self.v_ref)
                y = torch.round(y * scale) / scale

            return y

        def _resample_delta(self) -> None:
            device = self.W_nominal.device
            if self.sigma_mismatch > 0:
                self.delta = (
                    torch.ones_like(self.W_nominal, device=device)
                    + self.sigma_mismatch * torch.randn_like(self.W_nominal, device=device)
                )
                if self.delta_bias is not None:
                    self.delta_bias = 1.0 + self.sigma_mismatch * torch.randn(
                        self.out_channels, device=device
                    )
            else:
                self.delta = torch.ones_like(self.W_nominal, device=device)
                if self.delta_bias is not None:
                    self.delta_bias = torch.ones(self.out_channels, device=device)

        def extra_repr(self) -> str:
            return (
                f"in={self.in_channels}, out={self.out_channels}, "
                f"kernel={self.kernel_size}, σ={self.sigma_mismatch:.3f}, bits={self.n_adc_bits}"
            )

    AnalogConvNd.__name__ = f"AnalogConv{spatial_dims}d"
    AnalogConvNd.__qualname__ = f"AnalogConv{spatial_dims}d"
    return AnalogConvNd


AnalogConv1d = _make_analog_conv(F.conv1d, 1)
AnalogConv2d = _make_analog_conv(F.conv2d, 2)
AnalogConv3d = _make_analog_conv(F.conv3d, 3)


def analog_conv_from_module(
    conv: nn.Module,
    sigma_mismatch: float = 0.05,
    n_adc_bits: int = 8,
    temperature_K: float = _DEFAULT_TEMP_K,
    cap_F: float = _DEFAULT_CAP_F,
    v_ref: float = 1.0,
) -> nn.Module:
    """Build the appropriate AnalogConvNd from an existing nn.Conv{1,2,3}d module."""
    dim_map = {nn.Conv1d: AnalogConv1d, nn.Conv2d: AnalogConv2d, nn.Conv3d: AnalogConv3d}
    cls = dim_map.get(type(conv))
    if cls is None:
        raise TypeError(f"Unsupported conv type: {type(conv)}")

    return cls(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        weight=conv.weight.data,
        bias_data=conv.bias.data if conv.bias is not None else None,
        sigma_mismatch=sigma_mismatch,
        n_adc_bits=n_adc_bits,
        temperature_K=temperature_K,
        cap_F=cap_F,
        v_ref=v_ref,
    )
