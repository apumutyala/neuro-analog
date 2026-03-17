"""
Monte Carlo mismatch sweep harness.

Runs quality metrics across a grid of mismatch levels (sigma_values) and
ADC bit widths, collecting per-trial results for statistical analysis.

Protocol:
  For each sigma in sigma_values:
    For each trial in range(n_trials):
      analog_model = analogize(model, sigma, n_adc_bits)
      metric = eval_fn(analog_model)
    Record mean, std, all trials

DOUBT NOTED: The directive says "analog_model = analogize(model, sigma, ...)"
inside the inner loop — this creates a NEW analog model each trial (new δ).
An equivalent and faster approach: analogize once, then call resample_all_mismatch()
between trials. We use the resample approach since it avoids repeated deepcopy
overhead. Result is statistically identical.

DOUBT NOTED: The sweep eval_fn receives the analog model as its only argument.
This means the eval_fn must carry its own test data (via closure or module-level
state). This is by design — the sweep doesn't know about dataset specifics.
Each model file's evaluate() function uses module-level test data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .analog_model import analogize, resample_all_mismatch, set_all_noise, calibrate_analog_model, configure_analog_profile

_DEFAULT_SIGMA_VALUES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
_DEFAULT_BIT_VALUES = [2, 4, 6, 8, 10, 12, 16]


@dataclass
class SweepResult:
    """Results of a Monte Carlo mismatch or ADC precision sweep.

    per_trial: shape (n_sigma, n_trials) — raw metric values per trial
    digital_baseline: metric at sigma=0 (or highest bits), averaged over n_trials
    """
    sigma_values: list[float]
    metric_name: str
    per_trial: np.ndarray           # shape: (n_sigma, n_trials)
    digital_baseline: float         # metric at sigma=0, avg over n_trials

    @property
    def mean(self) -> np.ndarray:
        """Mean quality across trials, shape (n_sigma,)."""
        return self.per_trial.mean(axis=1)

    @property
    def std(self) -> np.ndarray:
        """Std dev across trials, shape (n_sigma,)."""
        return self.per_trial.std(axis=1)

    @property
    def normalized_mean(self) -> np.ndarray:
        """Quality normalized so digital_baseline = 1.0, degraded = < 1.0.

        Works for both positive and negative metrics (higher = better in both cases):
            normalized = 1 - (baseline - mean) / |baseline|
        At sigma=0: normalized = 1.0.
        10% degradation: normalized = 0.9.
        Using mean/abs(baseline) instead would give negative values for negative
        metrics, making degradation_threshold always return 0.0.
        """
        if self.digital_baseline == 0:
            return self.mean
        return 1.0 + (self.mean - self.digital_baseline) / abs(self.digital_baseline)

    @property
    def normalized_std(self) -> np.ndarray:
        """Normalized std dev."""
        if self.digital_baseline == 0:
            return self.std
        return self.std / abs(self.digital_baseline)

    def degradation_threshold(self, max_relative_loss: float = 0.10) -> float:
        """Return the largest sigma where mean quality is within max_relative_loss of baseline.

        E.g., max_relative_loss=0.10 → returns the sigma where quality drops to 90%.
        Returns 0.0 if even sigma=0 fails the threshold (shouldn't happen).
        Returns sigma_values[-1] if the model never degrades past the threshold.
        """
        norm = self.normalized_mean
        threshold = 1.0 - max_relative_loss
        # Find the largest sigma where quality >= threshold
        # Walk from high sigma to low sigma, find first that passes
        last_passing = 0.0
        for i, (sigma, q) in enumerate(zip(self.sigma_values, norm)):
            if q >= threshold:
                last_passing = sigma
        return last_passing

    def to_dict(self) -> dict:
        return {
            "sigma_values": self.sigma_values,
            "metric_name": self.metric_name,
            "per_trial": self.per_trial.tolist(),
            "digital_baseline": self.digital_baseline,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "normalized_mean": self.normalized_mean.tolist(),
            "normalized_std": self.normalized_std.tolist(),
            "degradation_threshold_10pct": self.degradation_threshold(0.10),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SweepResult":
        with open(path) as f:
            d = json.load(f)
        return cls(
            sigma_values=d["sigma_values"],
            metric_name=d["metric_name"],
            per_trial=np.array(d["per_trial"]),
            digital_baseline=d["digital_baseline"],
        )


def mismatch_sweep(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    sigma_values: list[float] | None = None,
    n_trials: int = 50,
    n_adc_bits: int = 8,
    calibration_data: torch.Tensor | None = None,
    analog_domain: str = "conservative",
    **analog_kwargs,
) -> SweepResult:
    """Monte Carlo mismatch sweep.

    For each sigma in sigma_values:
      - Create analogized model with that sigma
      - For each trial: resample mismatch, evaluate, record metric
    
    Args:
        calibration_data: Sample input for V_ref calibration. If None, uses v_ref=1.0.
    """
    if sigma_values is None:
        sigma_values = _DEFAULT_SIGMA_VALUES

    per_trial = np.zeros((len(sigma_values), n_trials), dtype=np.float64)

    # Create analog model once, resample between trials
    analog_model = analogize(model, sigma_mismatch=0.0, n_adc_bits=n_adc_bits, **analog_kwargs)
    configure_analog_profile(analog_model, analog_domain)

    # Calibrate V_ref if data provided
    if calibration_data is not None:
        calibrate_analog_model(analog_model, calibration_data)

    # Digital baseline: sigma=0, all noise off, multiple trials (should be deterministic
    # except for thermal noise which is also zero at sigma=0... but thermal is always on
    # by default. For true digital baseline we disable all noise.)
    set_all_noise(analog_model, thermal=False, quantization=False, mismatch=False)
    baseline_trials = [eval_fn(analog_model) for _ in range(n_trials)]
    digital_baseline = float(np.mean(baseline_trials))
    set_all_noise(analog_model, thermal=True, quantization=True, mismatch=True)

    for i, sigma in enumerate(sigma_values):
        resample_all_mismatch(analog_model, sigma=sigma)
        for j in range(n_trials):
            resample_all_mismatch(analog_model)  # new δ each trial
            per_trial[i, j] = eval_fn(analog_model)

        if i % 3 == 0:
            print(f"  sigma={sigma:.3f}: mean={per_trial[i].mean():.4f} ± {per_trial[i].std():.4f}")

    return SweepResult(
        sigma_values=sigma_values,
        metric_name="quality",
        per_trial=per_trial,
        digital_baseline=digital_baseline,
    )


def adc_sweep(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    bit_values: list[int] | None = None,
    sigma_mismatch: float = 0.05,
    n_trials: int = 50,
    calibration_data: torch.Tensor | None = None,
    analog_domain: str = "conservative",
    **analog_kwargs,
) -> SweepResult:
    """Sweep ADC precision at fixed mismatch level.

    Uses bit_values as the "sigma_values" axis in SweepResult for uniform
    interface, though the axis represents ADC bits.

    DOUBT NOTED: The SweepResult dataclass has sigma_values field. For the
    ADC sweep we reuse it for bit values. This is a typing awkwardness but
    avoids duplicating the entire dataclass. The metric_name distinguishes them.
    """
    if bit_values is None:
        bit_values = _DEFAULT_BIT_VALUES

    per_trial = np.zeros((len(bit_values), n_trials), dtype=np.float64)

    # Digital baseline: max bits, sigma=0, no noise
    analog_model = analogize(model, sigma_mismatch=0.0, n_adc_bits=max(bit_values), **analog_kwargs)
    configure_analog_profile(analog_model, analog_domain)
    if calibration_data is not None:
        calibrate_analog_model(analog_model, calibration_data)
    set_all_noise(analog_model, thermal=False, quantization=False, mismatch=False)
    baseline_trials = [eval_fn(analog_model) for _ in range(n_trials)]
    digital_baseline = float(np.mean(baseline_trials))

    for i, bits in enumerate(bit_values):
        analog_model = analogize(
            model, sigma_mismatch=sigma_mismatch, n_adc_bits=bits, **analog_kwargs
        )
        configure_analog_profile(analog_model, analog_domain)
        if calibration_data is not None:
            calibrate_analog_model(analog_model, calibration_data)
        for j in range(n_trials):
            resample_all_mismatch(analog_model)
            per_trial[i, j] = eval_fn(analog_model)

        print(f"  bits={bits}: mean={per_trial[i].mean():.4f} ± {per_trial[i].std():.4f}")

    return SweepResult(
        sigma_values=[float(b) for b in bit_values],
        metric_name=f"quality_vs_adc_bits_sigma{sigma_mismatch:.3f}",
        per_trial=per_trial,
        digital_baseline=digital_baseline,
    )


def ablation_sweep(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    sigma_values: list[float] | None = None,
    n_trials: int = 50,
    calibration_data: torch.Tensor | None = None,
    analog_domain: str = "conservative",
    **analog_kwargs,
) -> dict[str, SweepResult]:
    """Run three sweeps isolating each noise source independently.

    Returns dict with keys: 'mismatch', 'thermal', 'quantization'.

    For each sweep, the isolated noise source is ON and the others are OFF.
    This identifies which nonideality dominates per architecture.
    """
    if sigma_values is None:
        sigma_values = _DEFAULT_SIGMA_VALUES

    results = {}

    noise_configs = {
        "mismatch":     dict(mismatch=True,  thermal=False, quantization=False),
        "thermal":      dict(mismatch=False, thermal=True,  quantization=False),
        "quantization": dict(mismatch=False, thermal=False, quantization=True),
    }

    for name, noise_cfg in noise_configs.items():
        print(f"\n  Ablation: {name}-only sweep...")
        per_trial = np.zeros((len(sigma_values), n_trials), dtype=np.float64)

        analog_model = analogize(model, sigma_mismatch=0.0, **analog_kwargs)
        configure_analog_profile(analog_model, analog_domain)
        if calibration_data is not None:
            calibrate_analog_model(analog_model, calibration_data)
        set_all_noise(analog_model, thermal=False, quantization=False, mismatch=False)
        baseline_trials = [eval_fn(analog_model) for _ in range(n_trials)]
        digital_baseline = float(np.mean(baseline_trials))
        set_all_noise(analog_model, **noise_cfg)

        for i, sigma in enumerate(sigma_values):
            if name in ("mismatch",):
                # For mismatch-only: sigma is the mismatch level
                resample_all_mismatch(analog_model, sigma=sigma)
            # For thermal/quantization: sigma is ignored (they have fixed physical params)
            # but we still vary sigma for the x-axis to match the figure

            for j in range(n_trials):
                if name == "mismatch":
                    resample_all_mismatch(analog_model)
                per_trial[i, j] = eval_fn(analog_model)

        results[name] = SweepResult(
            sigma_values=sigma_values,
            metric_name=f"quality_{name}_only",
            per_trial=per_trial,
            digital_baseline=digital_baseline,
        )

    return results
