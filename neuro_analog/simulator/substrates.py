"""
Physically distinct analog substrate noise models for crossbar inference.

Three substrate classes grounded in published silicon data:

1. PCMSubstrate — IBM AIHWKit statistical model (Nature Communications 2020, 2023)
   Programming noise + temporal drift + 1/f read noise.
   Calibrated on 1 million PCM devices at IBM.

2. ReRAMSubstrate — Asymmetric SET/RESET switching variability
   No temporal drift, asymmetric conductance-dependent σ,
   log-normal-like tail behavior at intermediate states.
   Sources: Ambrogio et al. IEEE TED 2014; Wu et al. APL 2020; MemSim+ 2024.

3. CapacitiveSubstrate — Switched-capacitor / SRAM-IMC thermal noise only
   Pure Johnson-Nyquist kT/C noise, no conductance drift,
   weights are stored as charge on capacitors (not programmable conductances).
   Source: Khaddam-Aljameh et al. IEEE TVLSI 2021.

All substrates map normalized weights w ∈ [-1, 1] to a perturbed effective weight.
The perturbation is a multiplicative factor applied to the nominal weight tensor.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


# ── Physical constants ───────────────────────────────────────────────────
_K_B: float = 1.380649e-23   # Boltzmann constant [J/K]
_T_REF_K: float = 300.0      # Reference temperature [K]


# ── Base class ───────────────────────────────────────────────────────────

class SubstrateBase(ABC, nn.Module):
    """Abstract base for analog substrate noise models."""

    def __init__(self, temperature_K: float = _T_REF_K):
        super().__init__()
        self.temperature_K = temperature_K

    @abstractmethod
    def perturb_weights(self, W: torch.Tensor, t_inference_s: float = 3600.0) -> torch.Tensor:
        """Apply static weight perturbation (programming + drift) and return effective weights."""
        ...

    @abstractmethod
    def read_noise_std(self, W_eff: torch.Tensor, t_inference_s: float = 3600.0) -> float:
        """Return per-element read noise standard deviation for dynamic injection."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...


# ── PCM Substrate (IBM AIHWKit calibrated model) ─────────────────────────

@dataclass
class PCMSubstrateConfig:
    """Configuration for PCM statistical noise model.

    Default values match IBM AIHWKit PCMLikeNoiseModel calibrated on
    1M PCM devices (Joshi et al. Nature Communications 2020).
    """
    # Programming noise polynomial: σ_prog = max(c2·g² + c1·g + c0, 0)
    # where g = |w| (normalized conductance in [0,1])
    prog_c2: float = -1.1731   # coefficient of g²
    prog_c1: float = 1.9650    # coefficient of g
    prog_c0: float = 0.2635    # constant term

    # Drift exponent sampling
    # μ_ν = min(max(-0.0155·log(g) + 0.0244, 0.049), 0.1)
    # σ_ν = min(max(-0.0125·log(g) - 0.0059, 0.008), 0.045)
    drift_nu_log_coeff: float = -0.0155
    drift_nu_log_offset: float = 0.0244
    drift_nu_min: float = 0.049
    drift_nu_max: float = 0.10
    drift_nu_std_log_coeff: float = -0.0125
    drift_nu_std_log_offset: float = -0.0059
    drift_nu_std_min: float = 0.008
    drift_nu_std_max: float = 0.045

    # Read noise (1/f integrated)
    # σ_nG = g_drift · Q_s · sqrt(log((t + t_read)/(2·t_read)))
    # Q_s = min(0.0088 / g^0.65, 0.2)
    read_noise_Q_coeff: float = 0.0088
    read_noise_Q_exponent: float = 0.65
    read_noise_Q_cap: float = 0.2
    t_read_s: float = 2.5e-7   # 250 ns pulse width
    t_0_s: float = 20.0        # reference time for drift (programming completion)

    # Drift compensation (global scaling factor)
    enable_drift_compensation: bool = True

    # Differential pair mapping: how many conductance devices per weight
    # IBM HERMES uses 4 PCM devices per weight (2 pairs)
    n_devices_per_weight: int = 4


class PCMSubstrate(SubstrateBase):
    """IBM-calibrated PCM statistical noise model for inference.

    Implements the full three-stage noise model from AIHWKit:
      1. Programming noise (static, sampled once per device)
      2. Temporal drift (static, depends on inference time)
      3. Read noise (dynamic, fresh per forward pass)

    Weight-to-conductance mapping:
      Each weight w ∈ [-1, 1] is mapped to differential conductances:
        G+ = max(w, 0) · g_max   (positive device)
        G- = max(-w, 0) · g_max  (negative device)
      Programming noise is applied independently to G+ and G-.
      Drift scales both conductances by (t/t_0)^(-ν).
      The effective weight is (G+_drift - G-_drift) / g_max.

    Reference:
      Joshi et al., "Accurate deep neural network inference using
      computational phase-change memory", Nature Communications 11, 2473 (2020).
      https://aihwkit.readthedocs.io/en/latest/pcm_inference.html
    """

    def __init__(self, config: PCMSubstrateConfig | None = None, temperature_K: float = _T_REF_K):
        super().__init__(temperature_K)
        self.cfg = config or PCMSubstrateConfig()

    def name(self) -> str:
        return "PCM"

    def _normalized_conductance(self, W: torch.Tensor) -> torch.Tensor:
        """Map weights to [0, 1] conductance range (absolute value)."""
        return W.abs().clamp(min=1e-6, max=1.0)

    def _programming_noise_std(self, g: torch.Tensor) -> torch.Tensor:
        """σ_prog = max(c2·g² + c1·g + c0, 0)"""
        c = self.cfg
        sigma = c.prog_c2 * (g ** 2) + c.prog_c1 * g + c.prog_c0
        return sigma.clamp(min=0.0)

    def _apply_programming_noise(self, G: torch.Tensor) -> torch.Tensor:
        """Add programming noise to target conductances."""
        g_norm = self._normalized_conductance(G)
        sigma_prog = self._programming_noise_std(g_norm)
        noise = torch.randn_like(G) * sigma_prog
        return (G + noise).clamp(min=0.0)

    def _drift_exponent(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample drift exponent ν ~ N(μ_ν, σ_ν²) per device."""
        c = self.cfg
        log_g = g.clamp(min=1e-6).log()
        mu_nu = (c.drift_nu_log_coeff * log_g + c.drift_nu_log_offset).clamp(
            min=c.drift_nu_min, max=c.drift_nu_max
        )
        sigma_nu = (c.drift_nu_std_log_coeff * log_g + c.drift_nu_std_log_offset).clamp(
            min=c.drift_nu_std_min, max=c.drift_nu_std_max
        )
        nu = mu_nu + sigma_nu * torch.randn_like(g)
        return nu, mu_nu  # return mu for compensation

    def _apply_drift(self, G_prog: torch.Tensor, t_inference_s: float) -> torch.Tensor:
        """Apply temporal drift: G(t) = G_prog · (t/t_0)^(-ν)."""
        c = self.cfg
        g_norm = self._normalized_conductance(G_prog)
        nu, mu_nu = self._drift_exponent(g_norm)
        drift_factor = (t_inference_s / c.t_0_s) ** (-nu)
        G_drift = G_prog * drift_factor

        if c.enable_drift_compensation:
            # Global scaling compensation: α̂ = G(t)/G(t_0) averaged
            # Simplified: apply inverse of mean drift factor
            mean_drift = (t_inference_s / c.t_0_s) ** (-mu_nu.mean())
            G_drift = G_drift / mean_drift.clamp(min=1e-6)

        return G_drift

    def _read_noise_std(self, G_drift: torch.Tensor, t_inference_s: float) -> torch.Tensor:
        """Per-element read noise std: σ_nG = g_drift · Q_s · sqrt(log((t+t_read)/(2·t_read)))"""
        c = self.cfg
        g_norm = self._normalized_conductance(G_drift)
        Q_s = (c.read_noise_Q_coeff / (g_norm ** c.read_noise_Q_exponent)).clamp(max=c.read_noise_Q_cap)
        bandwidth_factor = torch.tensor((t_inference_s + c.t_read_s) / (2.0 * c.t_read_s), dtype=G_drift.dtype, device=G_drift.device)
        sigma_nG = G_drift * Q_s * (bandwidth_factor.clamp(min=1.0).log()).sqrt()
        return sigma_nG

    def perturb_weights(self, W: torch.Tensor, t_inference_s: float = 3600.0) -> torch.Tensor:
        """Apply PCM programming noise + drift to weights.

        For differential mapping:
          W_eff = (G+_drift - G-_drift) / g_max
        where G+ = max(W, 0), G- = max(-W, 0).
        """
        device = W.device
        dtype = W.dtype

        # Differential conductance mapping (positive and negative parts)
        G_pos = W.clamp(min=0.0)
        G_neg = (-W).clamp(min=0.0)

        # Apply programming noise
        G_pos_prog = self._apply_programming_noise(G_pos)
        G_neg_prog = self._apply_programming_noise(G_neg)

        # Apply temporal drift
        G_pos_drift = self._apply_drift(G_pos_prog, t_inference_s)
        G_neg_drift = self._apply_drift(G_neg_prog, t_inference_s)

        # Reconstruct effective weight
        W_eff = G_pos_drift - G_neg_drift
        return W_eff

    def read_noise_std(self, W_eff: torch.Tensor, t_inference_s: float = 3600.0) -> float:
        """Return scalar read noise std (mean over all elements for simplicity)."""
        # Compute per-element std, return mean as a representative value
        # The actual read noise is sampled in the forward pass using this std
        g_pos = W_eff.clamp(min=0.0)
        g_neg = (-W_eff).clamp(min=0.0)
        sigma_pos = self._read_noise_std(g_pos, t_inference_s)
        sigma_neg = self._read_noise_std(g_neg, t_inference_s)
        mean_sigma = (sigma_pos.mean() + sigma_neg.mean()) / 2.0
        return float(mean_sigma.item())


# ── ReRAM Substrate ──────────────────────────────────────────────────────

@dataclass
class ReRAMSubstrateConfig:
    """Configuration for ReRAM asymmetric switching variability model.

    ReRAM key differences from PCM:
      - Negligible temporal drift (filamentary conduction is stable)
      - Asymmetric SET/RESET: positive and negative weights may have different σ
      - Higher variability at intermediate states (log-normal-like tails)
      - Cycle-to-cycle variability (modeled as additional random offset)

    Sources:
      - Ambrogio et al., "Statistical Fluctuations in HfOx RRAM",
        IEEE TED 61(8), 2912-2919 (2014).
      - Wu et al., "Maximum extreme-value distribution model for switching
        conductance of oxide-RRAM", APL 116, 082901 (2020).
      - Nature Communications Materials 2024 (MemSim+).
    """
    # Base mismatch sigma (fractional conductance variation)
    sigma_mismatch: float = 0.05

    # Asymmetry factor: SET (positive weights) vs RESET (negative weights)
    # sigma_SET = sigma_mismatch * asymmetry_factor
    # sigma_RESET = sigma_mismatch / asymmetry_factor
    asymmetry_factor: float = 1.3   # SET is ~30% more variable than RESET

    # Intermediate-state enhancement: weights near 0 have higher relative noise
    # This models the log-normal tail behavior of filamentary switching
    intermediate_enhancement: float = 2.0  # σ multiplied by this factor for |w| < 0.3

    # Cycle-to-cycle variability (additional independent random offset per read)
    c2c_sigma: float = 0.01   # fraction of nominal weight

    # No drift for ReRAM (filament stability)
    enable_drift: bool = False


class ReRAMSubstrate(SubstrateBase):
    """ReRAM asymmetric switching substrate model.

    Models the key physical distinction of ReRAM: bipolar filamentary switching
    has asymmetric SET/RESET variability and no temporal drift.

    For inference, the weight perturbation captures:
      1. Device-to-device (D2D) variability: static per-device σ
      2. Directional asymmetry: positive vs negative weights have different σ
      3. Intermediate-state enhancement: small weights are noisier (filament
         partially formed → higher relative variability)
    """

    def __init__(self, config: ReRAMSubstrateConfig | None = None, temperature_K: float = _T_REF_K):
        super().__init__(temperature_K)
        self.cfg = config or ReRAMSubstrateConfig()

    def name(self) -> str:
        return "ReRAM"

    def _sigma_per_weight(self, W: torch.Tensor) -> torch.Tensor:
        """Compute per-element sigma based on sign and magnitude."""
        c = self.cfg
        base_sigma = torch.full_like(W, c.sigma_mismatch)

        # Asymmetry: positive weights (SET-dominated) vs negative (RESET-dominated)
        is_positive = W >= 0
        base_sigma = torch.where(
            is_positive,
            base_sigma * c.asymmetry_factor,
            base_sigma / max(c.asymmetry_factor, 1e-6)
        )

        # Intermediate-state enhancement: small |w| → higher relative σ
        w_abs = W.abs()
        near_zero = w_abs < 0.3
        enhancement = torch.where(
            near_zero,
            1.0 + (c.intermediate_enhancement - 1.0) * (1.0 - w_abs / 0.3),
            torch.ones_like(W)
        )
        return base_sigma * enhancement

    def perturb_weights(self, W: torch.Tensor, t_inference_s: float = 3600.0) -> torch.Tensor:
        """Apply ReRAM D2D variability to weights.

        No temporal drift — ReRAM filament conductance is stable over time.
        """
        sigma = self._sigma_per_weight(W)
        # Multiplicative noise: W_eff = W · (1 + ξ), ξ ~ N(0, σ²)
        noise = torch.randn_like(W) * sigma
        W_eff = W * (1.0 + noise)
        return W_eff

    def read_noise_std(self, W_eff: torch.Tensor, t_inference_s: float = 3600.0) -> float:
        """ReRAM read noise is small (filament conduction is deterministic).

        Main noise source is the D2D variability already captured in perturb_weights.
        Cycle-to-cycle variability is modeled as a small dynamic component.
        """
        return self.cfg.c2c_sigma * float(W_eff.abs().mean().item())


# ── Capacitive Substrate ─────────────────────────────────────────────────

@dataclass
class CapacitiveSubstrateConfig:
    """Configuration for capacitive/switched-capacitor substrate.

    Key physical characteristics:
      - Weights stored as charge on capacitors (not programmable conductances)
      - No programming noise (charge is transferred precisely via switches)
      - Pure Johnson-Nyquist thermal noise: σ = sqrt(kT/C)
      - No temporal drift (charge leaks slowly, refreshed periodically)
      - Excellent matching (metal capacitors have <1% mismatch)

    Source: Khaddam-Aljameh et al., "SRAM-Based Multibit In-Memory
    Matrix-Vector Multiplier", IEEE TVLSI 29(2), 372-383 (2021).
    """
    # Thermal noise: σ_th = sqrt(kT/C) per capacitor
    # For a 1 pF capacitor at 300K: σ_th ≈ 64 µV
    cap_F: float = 1e-12

    # Capacitor matching: metal capacitors have very low mismatch
    mismatch_sigma: float = 0.005   # 0.5% matching (much better than RRAM/PCM)

    # Charge injection noise from switch transistors
    charge_injection_sigma: float = 0.001   # fraction of nominal weight

    # No drift (charge is refreshed)
    refresh_interval_s: float = 1e-3   # 1 ms refresh (irrelevant for inference)


class CapacitiveSubstrate(SubstrateBase):
    """Capacitive / switched-capacitor substrate model.

    Models SRAM-IMC and switched-capacitor analog MAC arrays where:
      - Weights are stored as digital values (SRAM cells) or charge (capacitors)
      - Computation uses charge redistribution (no conductance-based MVM)
      - Thermal noise comes from sampling capacitors and switch resistance
      - Matching is excellent (<1% vs 5-10% for RRAM/PCM)

    The forward pass applies minimal static mismatch (good matching) but
    thermal noise is the dominant nonideality.
    """

    def __init__(self, config: CapacitiveSubstrateConfig | None = None, temperature_K: float = _T_REF_K):
        super().__init__(temperature_K)
        self.cfg = config or CapacitiveSubstrateConfig()

    def name(self) -> str:
        return "Capacitive"

    def _thermal_noise_std(self) -> float:
        """Johnson-Nyquist noise: σ = sqrt(kT/C) [V]"""
        return math.sqrt(_K_B * self.temperature_K / self.cfg.cap_F)

    def perturb_weights(self, W: torch.Tensor, t_inference_s: float = 3600.0) -> torch.Tensor:
        """Apply minimal static mismatch (capacitor matching is excellent).

        No temporal drift — weights are digital or refreshed charge.
        """
        c = self.cfg
        noise = torch.randn_like(W) * c.mismatch_sigma
        W_eff = W * (1.0 + noise)
        return W_eff

    def read_noise_std(self, W_eff: torch.Tensor, t_inference_s: float = 3600.0) -> float:
        """Return thermal noise std in units of weight.

        The thermal noise voltage is scaled to the weight range [0, 1].
        For a capacitor array with V_ref = 1V, σ_weight = σ_voltage / V_ref.
        """
        sigma_v = self._thermal_noise_std()
        # Normalize to weight range (assuming 1V full scale)
        return sigma_v  # ~6.4e-5 for 1pF at 300K


# ── Registry ─────────────────────────────────────────────────────────────

SUBSTRATE_REGISTRY: dict[str, type[SubstrateBase]] = {
    "pcm": PCMSubstrate,
    "reram": ReRAMSubstrate,
    "capacitive": CapacitiveSubstrate,
}


def get_substrate(name: str, **kwargs) -> SubstrateBase:
    """Factory: instantiate a substrate by name.

    Args:
        name: One of "pcm", "reram", "capacitive" (case-insensitive).
        **kwargs: Passed to substrate constructor (e.g., temperature_K=350).

    Returns:
        Instantiated SubstrateBase subclass.
    """
    key = name.lower()
    if key not in SUBSTRATE_REGISTRY:
        raise ValueError(
            f"Unknown substrate '{name}'. Available: {list(SUBSTRATE_REGISTRY.keys())}"
        )
    return SUBSTRATE_REGISTRY[key](**kwargs)


__all__ = [
    "SubstrateBase",
    "PCMSubstrate",
    "PCMSubstrateConfig",
    "ReRAMSubstrate",
    "ReRAMSubstrateConfig",
    "CapacitiveSubstrate",
    "CapacitiveSubstrateConfig",
    "get_substrate",
    "SUBSTRATE_REGISTRY",
]
