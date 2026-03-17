"""
AnalogLinear: nn.Linear replacement modeling a crossbar MVM array.

Physical model (three noise sources applied in order):

1. CONDUCTANCE MISMATCH (static, sampled once per device instance)
   W_device = W_nominal * δ,   δ ~ N(1, σ_mismatch²)
   Same δ persists across all forward passes — it is baked into the
   fabricated conductance values of the RRAM/PCM cells. This is the
   Shem §4.1 formulation: θ' = δ ◦ θ, δ ~ N(1, σ²·I).

2. THERMAL READ NOISE (dynamic, drawn fresh each forward pass)
   y = W_device @ x + ε,   ε ~ N(0, σ_thermal² · I)
   σ_thermal = sqrt(kT/C) * sqrt(in_features)

   The sqrt(in_features) factor comes from current accumulation:
   The output node is a sense capacitor C driven by N = in_features
   column currents. If each column wire contributes independent
   Johnson-Nyquist noise with variance kT/C (one capacitor per wire),
   the total output noise variance is N · kT/C, giving σ = sqrt(N·kT/C).
   Reference: Legno §4 noise model; Shem §4.2 SDE diffusion term g(x,θ,t).

   DOUBT NOTED: The directive's sqrt(in_features) factor assumes independent
   per-column noise sources. An alternative model puts all thermal noise
   at the single sense amplifier output: σ = sqrt(kT/C) (no N scaling).
   The directive explicitly states the sqrt(N) model — we follow it as a
   conservative upper bound. Real hardware falls between these two limits.

3. ADC QUANTIZATION (deterministic)
   scale = (2^n_bits - 1) / (2 * V_ref)
   y_q = clamp(y, -V_ref, V_ref)
   y_q = round(y_q * scale) / scale

   V_ref is per-layer from calibration (max absolute activation value).
   At n_bits >= 32, this is numerically identical to the unquantized output.
   Source: Shem §4.3 models DAC levels via Gumbel-Softmax for training;
   we use hard quantization since we simulate inference, not training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Physical constants (same as nonidealities/noise.py for consistency)
_K_B: float = 1.380649e-23   # Boltzmann constant [J/K]
_DEFAULT_TEMP_K: float = 300.0
_DEFAULT_CAP_F: float = 1e-12  # 1 pF — HCDCv2 integration capacitor


class AnalogLinear(nn.Module):
    """Crossbar MVM analog implementation of nn.Linear.

    Weights are stored as non-trainable buffers (inference simulation only).
    Static mismatch δ is sampled in __init__ and reused across forward calls.
    Call resample_mismatch() between Monte Carlo trials.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        sigma_mismatch: float = 0.05,
        n_adc_bits: int = 8,
        temperature_K: float = _DEFAULT_TEMP_K,
        cap_F: float = _DEFAULT_CAP_F,
        v_ref: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_mismatch = sigma_mismatch
        self.n_adc_bits = n_adc_bits
        self.temperature_K = temperature_K
        self.cap_F = cap_F
        self.v_ref = v_ref

        # Noise source toggles (for ablation experiments)
        self._use_mismatch = True
        self._use_thermal = True
        self._use_quantization = True

        # Profile toggle: True = this layer is a readout (ADC boundary).
        # In 'conservative' profile all layers are readouts (default).
        # In 'full_analog' profile only the final output layer is a readout.
        self._is_readout = True

        # Weights stored as buffers — not trained, never accumulate gradients
        self.register_buffer("W_nominal", weight.clone().detach().float())
        if bias is not None:
            self.register_buffer("bias", bias.clone().detach().float())
        else:
            self.register_buffer("bias", None)

        # Static mismatch δ ~ N(1, σ²): sampled once, held fixed per device
        self.register_buffer("delta", torch.ones(out_features, in_features))
        self._resample_delta()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def resample_mismatch(self, sigma: float | None = None) -> None:
        """Re-roll δ. Call between Monte Carlo trials, not between batches."""
        if sigma is not None:
            self.sigma_mismatch = sigma
        self._resample_delta()

    def set_noise_config(
        self,
        thermal: bool = True,
        quantization: bool = True,
        mismatch: bool = True,
    ) -> None:
        """Toggle individual noise sources for ablation experiments."""
        self._use_mismatch = mismatch
        self._use_thermal = thermal
        self._use_quantization = quantization

    def calibrate(self, x: torch.Tensor) -> None:
        """Set V_ref from a calibration forward pass (noiseless).

        Call this once with representative input data before running sweeps
        to avoid excessive clipping in the ADC stage.
        """
        with torch.no_grad():
            y = F.linear(x.float(), self.W_nominal, self.bias)
            peak = float(y.abs().max().item())
            self.v_ref = peak * 1.1 if peak > 0 else 1.0  # 10% headroom

    # ──────────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()

        # Step 1 — Conductance mismatch (static per device)
        if self._use_mismatch and self.sigma_mismatch > 0:
            W_eff = self.W_nominal * self.delta
        else:
            W_eff = self.W_nominal

        y = F.linear(x, W_eff, self.bias)

        # Step 2 — Thermal read noise: σ = sqrt(kT/C) * sqrt(in_features)
        if self._use_thermal:
            sigma_th = math.sqrt(_K_B * self.temperature_K / self.cap_F) * math.sqrt(self.in_features)
            y = y + torch.randn_like(y) * sigma_th

        # Step 3 — ADC quantization over [-V_ref, V_ref]
        # Gated by _is_readout: in 'full_analog' profile, intermediate layers
        # skip quantization (signal stays in continuous analog domain).
        if self._use_quantization and self._is_readout and self.n_adc_bits < 32:
            n_levels = 2 ** self.n_adc_bits - 1
            scale = n_levels / (2.0 * self.v_ref)
            y = torch.clamp(y, -self.v_ref, self.v_ref)
            y = torch.round(y * scale) / scale

        return y

    # ──────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────

    def _resample_delta(self) -> None:
        device = self.W_nominal.device
        if self.sigma_mismatch > 0:
            self.delta = torch.ones(self.out_features, self.in_features, device=device) + \
                         self.sigma_mismatch * torch.randn(self.out_features, self.in_features, device=device)
        else:
            self.delta = torch.ones(self.out_features, self.in_features, device=device)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"σ={self.sigma_mismatch:.3f}, bits={self.n_adc_bits}"
        )
