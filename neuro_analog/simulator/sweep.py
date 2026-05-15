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

Note: analogize() is called inside the inner loop, creating a fresh analog model
(new δ draw) each trial. An equivalent and faster approach: analogize once, then
call resample_all_mismatch() between trials. We use the resample approach since it
avoids repeated deepcopy overhead. Result is statistically identical.

The sweep eval_fn receives the analog model as its only argument.
This means the eval_fn must carry its own test data (via closure or module-level
state). This is by design — the sweep doesn't know about dataset specifics.
Each model file's evaluate() function uses module-level test data.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .analog_model import analogize, resample_all_mismatch, set_all_noise, calibrate_analog_model, configure_analog_profile
from ..ir.energy_model import HardwareProfile

_DEFAULT_SIGMA_VALUES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
_DEFAULT_BIT_VALUES = [2, 4, 6, 8, 10, 12, 16]

_SEED_STREAMS = {
    "digital_baseline": 101,
    "ideal_analog_baseline": 202,
    "mismatch": 303,
    "adc": 404,
    "ablation_mismatch": 505,
    "ablation_thermal": 606,
    "ablation_quantization": 707,
}


def _trial_seed(base_seed: int | None, stream: str, sweep_index: int, trial_index: int) -> int | None:
    if base_seed is None:
        return None
    stream_id = _SEED_STREAMS[stream]
    return int((base_seed + stream_id * 1_000_003 + sweep_index * 10_007 + trial_index) % (2**31 - 1))


def _set_eval_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_seeded(eval_fn: Callable[[nn.Module], float], model: nn.Module, seed: int | None) -> float:
    _set_eval_seed(seed)
    return float(eval_fn(model))


def _baseline_trials(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    n_trials: int,
    base_seed: int | None,
    stream: str,
) -> tuple[float, list[int | None]]:
    seeds = [_trial_seed(base_seed, stream, 0, j) for j in range(n_trials)]
    values = [_evaluate_seeded(eval_fn, model, seed) for seed in seeds]
    return float(np.mean(values)), seeds


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
    ideal_analog_baseline: float | None = None  # analogized, noise-off diagnostic
    trial_seeds: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Energy/latency metrics (optional, computed if hardware_profile provided)
    analog_energy_pJ: float | None = None
    digital_energy_pJ: float | None = None
    analog_latency_ns: float | None = None
    digital_latency_ns: float | None = None
    energy_saving_vs_digital: float | None = None  # 1 - analog/digital
    speedup_vs_digital: float | None = None  # digital/analog

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

    @property
    def normalized_mean_vs_ideal_analog(self) -> np.ndarray | None:
        """Quality normalized against the noiseless analog wrapper diagnostic."""
        if self.ideal_analog_baseline is None:
            return None
        if self.ideal_analog_baseline == 0:
            return self.mean
        return 1.0 + (self.mean - self.ideal_analog_baseline) / abs(self.ideal_analog_baseline)

    @property
    def normalized_std_vs_ideal_analog(self) -> np.ndarray | None:
        """Std dev normalized against the noiseless analog wrapper diagnostic."""
        if self.ideal_analog_baseline is None:
            return None
        if self.ideal_analog_baseline == 0:
            return self.std
        return self.std / abs(self.ideal_analog_baseline)

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

    def degradation_threshold_vs_ideal_analog(self, max_relative_loss: float = 0.10) -> float | None:
        """Return degradation threshold normalized to the ideal analog baseline."""
        norm = self.normalized_mean_vs_ideal_analog
        if norm is None:
            return None
        threshold = 1.0 - max_relative_loss
        last_passing = 0.0
        for sigma, q in zip(self.sigma_values, norm):
            if q >= threshold:
                last_passing = sigma
        return last_passing

    def to_dict(self) -> dict:
        d = {
            "sigma_values": self.sigma_values,
            "metric_name": self.metric_name,
            "per_trial": self.per_trial.tolist(),
            "digital_baseline": self.digital_baseline,
            "ideal_analog_baseline": self.ideal_analog_baseline,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "normalized_mean": self.normalized_mean.tolist(),
            "normalized_std": self.normalized_std.tolist(),
            "degradation_threshold_10pct": self.degradation_threshold(0.10),
            "trial_seeds": self.trial_seeds,
            "metadata": self.metadata,
        }
        if self.normalized_mean_vs_ideal_analog is not None:
            d["normalized_mean_vs_ideal_analog"] = self.normalized_mean_vs_ideal_analog.tolist()
            d["normalized_std_vs_ideal_analog"] = self.normalized_std_vs_ideal_analog.tolist()
            d["degradation_threshold_10pct_vs_ideal_analog"] = self.degradation_threshold_vs_ideal_analog(0.10)
        # Add energy/latency metrics if available
        if self.analog_energy_pJ is not None:
            d["analog_energy_pJ"] = self.analog_energy_pJ
            d["digital_energy_pJ"] = self.digital_energy_pJ
            d["analog_latency_ns"] = self.analog_latency_ns
            d["digital_latency_ns"] = self.digital_latency_ns
            d["energy_saving_vs_digital"] = self.energy_saving_vs_digital
            d["speedup_vs_digital"] = self.speedup_vs_digital
        return d

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
            ideal_analog_baseline=d.get("ideal_analog_baseline"),
            trial_seeds=d.get("trial_seeds", {}),
            metadata=d.get("metadata", {}),
            analog_energy_pJ=d.get("analog_energy_pJ"),
            digital_energy_pJ=d.get("digital_energy_pJ"),
            analog_latency_ns=d.get("analog_latency_ns"),
            digital_latency_ns=d.get("digital_latency_ns"),
            energy_saving_vs_digital=d.get("energy_saving_vs_digital"),
            speedup_vs_digital=d.get("speedup_vs_digital"),
        )


def mismatch_sweep(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    sigma_values: list[float] | None = None,
    n_trials: int = 50,
    n_adc_bits: int = 8,
    calibration_data: torch.Tensor | None = None,
    calibration_runner: Callable[[nn.Module], object] | None = None,
    analog_domain: str = "conservative",
    hardware_profile: HardwareProfile | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
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

    # Calibrate V_ref if a representative input or custom runner is provided.
    if calibration_data is not None or calibration_runner is not None:
        calibrate_analog_model(analog_model, calibration_data, calibration_runner=calibration_runner)

    # Primary baseline: untouched digital model. This avoids contaminating the
    # reference with analog wrapper transfer functions such as clipped tanh/ReLU.
    digital_baseline, digital_baseline_seeds = _baseline_trials(
        model, eval_fn, n_trials, seed, "digital_baseline"
    )

    # Diagnostic baseline: analogized model with all nonidealities disabled.
    set_all_noise(analog_model, thermal=False, quantization=False, mismatch=False)
    ideal_analog_baseline, ideal_analog_baseline_seeds = _baseline_trials(
        analog_model, eval_fn, n_trials, seed, "ideal_analog_baseline"
    )
    set_all_noise(analog_model, thermal=True, quantization=True, mismatch=True)

    per_trial_seeds: list[list[int | None]] = []
    for i, sigma in enumerate(sigma_values):
        row_seeds = []
        for j in range(n_trials):
            trial_seed = _trial_seed(seed, "mismatch", i, j)
            row_seeds.append(trial_seed)
            _set_eval_seed(trial_seed)
            resample_all_mismatch(analog_model, sigma=sigma)
            per_trial[i, j] = eval_fn(analog_model)
        per_trial_seeds.append(row_seeds)

    if i % 3 == 0:
            print(f"  sigma={sigma:.3f}: mean={per_trial[i].mean():.4f} ± {per_trial[i].std():.4f}")

    # Hardware metrics are computed via AnalogGraph.analyze(), not here.
    # The SweepResult fields are left as None; they are populated after
    # graph analysis by the caller (sweep_all.py).

    return SweepResult(
        sigma_values=sigma_values,
        metric_name="quality",
        per_trial=per_trial,
        digital_baseline=digital_baseline,
        ideal_analog_baseline=ideal_analog_baseline,
        trial_seeds={
            "digital_baseline": digital_baseline_seeds,
            "ideal_analog_baseline": ideal_analog_baseline_seeds,
            "per_trial": per_trial_seeds,
        },
        metadata=metadata or {},
    )


def adc_sweep(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    bit_values: list[int] | None = None,
    sigma_mismatch: float = 0.05,
    n_trials: int = 50,
    calibration_data: torch.Tensor | None = None,
    calibration_runner: Callable[[nn.Module], object] | None = None,
    analog_domain: str = "conservative",
    hardware_profile: HardwareProfile | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
    **analog_kwargs,
) -> SweepResult:
    """Sweep ADC precision at fixed mismatch level.

    Uses bit_values as the "sigma_values" axis in SweepResult for uniform
    interface, though the axis represents ADC bits.

    The SweepResult dataclass has sigma_values field. For the
    ADC sweep we reuse it for bit values. This is a typing awkwardness but
    avoids duplicating the entire dataclass. The metric_name distinguishes them.
    """
    if bit_values is None:
        bit_values = _DEFAULT_BIT_VALUES

    per_trial = np.zeros((len(bit_values), n_trials), dtype=np.float64)

    # Primary baseline: untouched digital model.
    digital_baseline, digital_baseline_seeds = _baseline_trials(
        model, eval_fn, n_trials, seed, "digital_baseline"
    )

    # Diagnostic baseline: max-bit analog wrapper with all nonidealities disabled.
    analog_model = analogize(model, sigma_mismatch=0.0, n_adc_bits=max(bit_values), **analog_kwargs)
    configure_analog_profile(analog_model, analog_domain)
    if calibration_data is not None or calibration_runner is not None:
        calibrate_analog_model(analog_model, calibration_data, calibration_runner=calibration_runner)
    set_all_noise(analog_model, thermal=False, quantization=False, mismatch=False)
    ideal_analog_baseline, ideal_analog_baseline_seeds = _baseline_trials(
        analog_model, eval_fn, n_trials, seed, "ideal_analog_baseline"
    )

    per_trial_seeds: list[list[int | None]] = []
    for i, bits in enumerate(bit_values):
        analog_model = analogize(
            model, sigma_mismatch=sigma_mismatch, n_adc_bits=bits, **analog_kwargs
        )
        configure_analog_profile(analog_model, analog_domain)
        if calibration_data is not None or calibration_runner is not None:
            calibrate_analog_model(analog_model, calibration_data, calibration_runner=calibration_runner)
        row_seeds = []
        for j in range(n_trials):
            trial_seed = _trial_seed(seed, "adc", i, j)
            row_seeds.append(trial_seed)
            _set_eval_seed(trial_seed)
            resample_all_mismatch(analog_model)
            per_trial[i, j] = eval_fn(analog_model)
        per_trial_seeds.append(row_seeds)

        print(f"  bits={bits}: mean={per_trial[i].mean():.4f} ± {per_trial[i].std():.4f}")

    # Hardware metrics are computed via AnalogGraph.analyze(), not here.

    return SweepResult(
        sigma_values=[float(b) for b in bit_values],
        metric_name=f"quality_vs_adc_bits_sigma{sigma_mismatch:.3f}",
        per_trial=per_trial,
        digital_baseline=digital_baseline,
        ideal_analog_baseline=ideal_analog_baseline,
        trial_seeds={
            "digital_baseline": digital_baseline_seeds,
            "ideal_analog_baseline": ideal_analog_baseline_seeds,
            "per_trial": per_trial_seeds,
        },
        metadata=metadata or {},
    )


def ablation_sweep(
    model: nn.Module,
    eval_fn: Callable[[nn.Module], float],
    sigma_values: list[float] | None = None,
    n_trials: int = 50,
    calibration_data: torch.Tensor | None = None,
    calibration_runner: Callable[[nn.Module], object] | None = None,
    analog_domain: str = "conservative",
    hardware_profile: HardwareProfile | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
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

        kwargs = {k: v for k, v in analog_kwargs.items() if k != "sigma_mismatch"}
        analog_model = analogize(model, sigma_mismatch=0.0, **kwargs)
        configure_analog_profile(analog_model, analog_domain)
        if calibration_data is not None or calibration_runner is not None:
            calibrate_analog_model(analog_model, calibration_data, calibration_runner=calibration_runner)
        digital_baseline, digital_baseline_seeds = _baseline_trials(
            model, eval_fn, n_trials, seed, "digital_baseline"
        )
        set_all_noise(analog_model, thermal=False, quantization=False, mismatch=False)
        ideal_analog_baseline, ideal_analog_baseline_seeds = _baseline_trials(
            analog_model, eval_fn, n_trials, seed, "ideal_analog_baseline"
        )
        set_all_noise(analog_model, **noise_cfg)

        per_trial_seeds: list[list[int | None]] = []
        for i, sigma in enumerate(sigma_values):
            row_seeds = []
            if name in ("mismatch",):
                # For mismatch-only: sigma is the mismatch level
                resample_all_mismatch(analog_model, sigma=sigma)
            # For thermal/quantization: sigma is ignored (they have fixed physical params)
            # but we still vary sigma for the x-axis to match the figure

            for j in range(n_trials):
                trial_seed = _trial_seed(seed, f"ablation_{name}", i, j)
                row_seeds.append(trial_seed)
                _set_eval_seed(trial_seed)
                if name == "mismatch":
                    resample_all_mismatch(analog_model, sigma=sigma)
                per_trial[i, j] = eval_fn(analog_model)
        per_trial_seeds.append(row_seeds)

        # Hardware metrics are computed via AnalogGraph.analyze(), not here.

        results[name] = SweepResult(
            sigma_values=sigma_values,
            metric_name=f"quality_{name}_only",
            per_trial=per_trial,
            digital_baseline=digital_baseline,
            ideal_analog_baseline=ideal_analog_baseline,
            trial_seeds={
                "digital_baseline": digital_baseline_seeds,
                "ideal_analog_baseline": ideal_analog_baseline_seeds,
                "per_trial": per_trial_seeds,
            },
            metadata={**(metadata or {}), "ablation_noise": name},
        )

    return results
