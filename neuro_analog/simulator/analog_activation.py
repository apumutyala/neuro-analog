"""
Analog activation function replacements.

ANALOG-NATIVE (Tanh, Sigmoid, ReLU):
  These can be implemented directly in CMOS circuits.
  Tanh: subthreshold MOSFET differential pair — the I-V characteristic IS tanh.
  Sigmoid: same diff pair with single-ended output.
  ReLU: diode-connected transistor.

  Nonidealities modeled:
  1. Gain mismatch: α ~ N(1, σ²)  — W/L ratio variance between M1/M2 in the pair
  2. Offset mismatch: β ~ N(0, (0.5σ)²) — V_th (threshold voltage) mismatch
  3. Output swing saturation: real diff pairs don't reach the ±1 mathematical limit

  Resulting transfer functions:
    tanh_analog(x)    = clip(tanh(α·x + β), -0.95, +0.95)
    sigmoid_analog(x) = clip(σ(α·x + β), 0.025, 0.975)
    relu_analog(x)    = clip(max(0, α·x + β), 0, rail)

  Source: Murmann & Meindl (2020), subthreshold MOSFET mismatch models;
  Chicca et al. (2014), neuromorphic analog building blocks.

DIGITAL-REQUIRED (GELU, SiLU):
  No efficient analog implementation exists. These require polynomial
  approximation circuits or digital co-processors.

  Modeled as: digital computation + ADC→DAC round-trip penalty
    gelu_analog(x) = quantize(gelu(x), n_bits) + ε_thermal

  This captures the domain crossing cost: the output must go through
  an ADC to run the digital GELU, then a DAC to re-enter analog.

  DOUBT NOTED: The directive gives n_bits as a parameter but does not
  specify whether it's the same n_bits as the linear layer's ADC or a
  separate specification. We use the same n_bits for consistency (one
  converter spec per datapath). If the hardware has separate specs for
  activation converters, this can be trivially extended.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_K_B: float = 1.380649e-23
_DEFAULT_TEMP_K: float = 300.0
_DEFAULT_CAP_F: float = 1e-12


class _BaseAnalogActivation(nn.Module):
    """Shared mismatch/noise infrastructure for all analog activations."""

    def __init__(self, sigma_mismatch: float = 0.05):
        super().__init__()
        self.sigma_mismatch = sigma_mismatch
        self._use_mismatch = True
        self._use_thermal = True
        # α ~ N(1, σ²),  β ~ N(0, (0.5σ)²)  — scalar per activation unit
        self.register_buffer("alpha", torch.tensor(1.0))
        self.register_buffer("beta", torch.tensor(0.0))
        self._resample_params()

    def resample_mismatch(self, sigma: float | None = None) -> None:
        if sigma is not None:
            self.sigma_mismatch = sigma
        self._resample_params()

    def set_noise_config(self, thermal: bool = True, quantization: bool = True, mismatch: bool = True) -> None:
        self._use_mismatch = mismatch
        self._use_thermal = thermal

    def _resample_params(self) -> None:
        device = self.alpha.device
        if self.sigma_mismatch > 0:
            self.alpha = torch.tensor(
                1.0 + self.sigma_mismatch * torch.randn(1).item(), device=device
            )
            self.beta = torch.tensor(
                0.5 * self.sigma_mismatch * torch.randn(1).item(), device=device
            )
        else:
            self.alpha = torch.tensor(1.0, device=device)
            self.beta = torch.tensor(0.0, device=device)

    def _apply_mismatch(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_mismatch and self.sigma_mismatch > 0:
            return self.alpha * x + self.beta
        return x


class AnalogTanh(_BaseAnalogActivation):
    """tanh via MOSFET differential pair with gain/offset mismatch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(self._apply_mismatch(x.float()))
        return torch.clamp(y, -0.95, 0.95)


class AnalogSigmoid(_BaseAnalogActivation):
    """sigmoid via single-ended differential pair."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.sigmoid(self._apply_mismatch(x.float()))
        return torch.clamp(y, 0.025, 0.975)


class AnalogReLU(_BaseAnalogActivation):
    """ReLU via diode-connected transistor. β creates an offset threshold."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_mismatch and self.sigma_mismatch > 0:
            y = F.relu(self.alpha * x.float() + self.beta)
        else:
            y = F.relu(x.float())
        return y


class _DigitalWithCrossing(nn.Module):
    """
    Digital activation + ADC→DAC domain crossing penalty.

    The exact digital function is computed (deterministic), then a
    quantization pass + small thermal noise represents the converter penalty.

    DOUBT NOTED: The directive models this as quantize(f(x), n_bits) + ε_thermal.
    The thermal noise here is from the DAC output buffer, not the ADC input
    (which is modeled in AnalogLinear). We use σ = sqrt(kT/C) without the
    sqrt(N) factor since this is a single-output DAC, not a summed crossbar.
    """

    def __init__(
        self,
        fn,
        sigma_mismatch: float = 0.05,
        n_bits: int = 8,
        v_ref: float = 1.0,
        temperature_K: float = _DEFAULT_TEMP_K,
        cap_F: float = _DEFAULT_CAP_F,
    ):
        super().__init__()
        self.fn = fn
        self.sigma_mismatch = sigma_mismatch
        self.n_bits = n_bits
        self.v_ref = v_ref
        self.temperature_K = temperature_K
        self.cap_F = cap_F
        self._use_mismatch = True
        self._use_thermal = True
        self._use_quantization = True

    def resample_mismatch(self, sigma: float | None = None) -> None:
        if sigma is not None:
            self.sigma_mismatch = sigma

    def set_noise_config(self, thermal: bool = True, quantization: bool = True, mismatch: bool = True) -> None:
        self._use_mismatch = mismatch
        self._use_thermal = thermal
        self._use_quantization = quantization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fn(x.float())

        # ADC→DAC quantization
        if self._use_quantization and self.n_bits < 32:
            n_levels = 2 ** self.n_bits - 1
            scale = n_levels / (2.0 * self.v_ref)
            y = torch.clamp(y, -self.v_ref, self.v_ref)
            y = torch.round(y * scale) / scale

        # DAC output thermal noise: bare kT/C (no sqrt(N) factor).
        # This models a single DAC output stage, not a crossbar column summation.
        # Contrast with AnalogLinear which uses sqrt(kT/C * in_features): that
        # sqrt(N) factor comes from N independent column currents summing on one
        # sense capacitor. Here there is only one signal source (the DAC output),
        # so the noise is just kT/C. Both models are correct for their respective
        # hardware components; the different formulas are intentional.
        if self._use_thermal:
            sigma_th = math.sqrt(_K_B * self.temperature_K / self.cap_F)
            y = y + torch.randn_like(y) * sigma_th

        return y


class AnalogGELU(_DigitalWithCrossing):
    def __init__(self, **kwargs):
        super().__init__(fn=F.gelu, **kwargs)


class AnalogSiLU(_DigitalWithCrossing):
    def __init__(self, **kwargs):
        super().__init__(fn=F.silu, **kwargs)


# ── Additional analog-native activations ────────────────────────────────────

class AnalogELU(_BaseAnalogActivation):
    """ELU via MOSFET circuit with exponential sub-threshold region.

    The positive half (x > 0) is a linear pass-through (same as diode-connected
    transistor above threshold). The negative half (x <= 0) uses the exponential
    sub-threshold characteristic, approximated here by the mathematical ELU.
    Same α/β gain+offset mismatch as AnalogTanh; no hard output clipping
    (ELU is unbounded above, bounded at -alpha below).
    """

    def __init__(self, alpha: float = 1.0, sigma_mismatch: float = 0.05):
        super().__init__(sigma_mismatch=sigma_mismatch)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self._apply_mismatch(x.float()), alpha=self.alpha)


class AnalogLeakyReLU(_BaseAnalogActivation):
    """LeakyReLU via diode with bleed resistor.

    The negative-slope region is implemented by a parallel resistor that lets
    a fraction (negative_slope) of the current through. Both the positive and
    negative slopes get gain mismatch α; the threshold gets offset mismatch β.
    """

    def __init__(self, negative_slope: float = 0.01, sigma_mismatch: float = 0.05):
        super().__init__(sigma_mismatch=sigma_mismatch)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self._apply_mismatch(x.float()), negative_slope=self.negative_slope)


# ── Digital-required activations (ADC→DAC crossing) ─────────────────────────

class AnalogHardswish(_DigitalWithCrossing):
    """Hardswish: piecewise linear approximation of SiLU.

    No efficient pure-analog circuit exists for the piecewise-linear conditional.
    Modeled as digital computation + ADC→DAC domain crossing, identical to GELU/SiLU.
    """
    def __init__(self, **kwargs):
        super().__init__(fn=F.hardswish, **kwargs)


class AnalogMish(_DigitalWithCrossing):
    """Mish = x * tanh(softplus(x)).

    The softplus requires a logarithm which has no simple analog circuit.
    Modeled as digital computation + ADC→DAC domain crossing.
    """
    def __init__(self, **kwargs):
        super().__init__(fn=F.mish, **kwargs)
