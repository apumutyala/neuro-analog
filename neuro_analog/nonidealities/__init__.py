"""Nonideality modeling layer — five physical imperfections modeled by analog compilation pipelines."""

from .mismatch import propagate_mismatch, MismatchReport
from .noise import compute_noise_budget, NoiseBudget
from .quantization import compute_precision_requirements, QuantizationReport
from .signal_scaling import analyze_signal_ranges, ScalingReport

__all__ = [
    "propagate_mismatch", "MismatchReport",
    "compute_noise_budget", "NoiseBudget",
    "compute_precision_requirements", "QuantizationReport",
    "analyze_signal_ranges", "ScalingReport",
]
