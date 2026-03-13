"""
Noise budget stacked bar chart.

Shows per-layer noise contributions:
  - Thermal noise (kT/C from integrators)
  - Shot noise (current-mode circuits)
  - Quantization noise (ADC/DAC)
  - Mismatch-induced error variance
  - Total SNR vs. digital 32-bit baseline
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuro_analog.ir import AnalogGraph
from neuro_analog.nonidealities.noise import NoiseBudget, compute_noise_budget
from neuro_analog.nonidealities.mismatch import MismatchReport


_NOISE_COLORS = {
    "thermal": "#3498db",      # Blue
    "shot": "#9b59b6",          # Purple
    "quantization": "#e67e22",  # Orange
    "mismatch": "#e74c3c",      # Red
}


def plot_noise_budget(
    graph: AnalogGraph,
    noise_budgets: Optional[dict[str, NoiseBudget]] = None,
    mismatch_reports: Optional[dict[str, MismatchReport]] = None,
    output_path: Optional[str | Path] = None,
    max_nodes: int = 40,
    figsize: tuple[int, int] = (16, 6),
    target_snr_db: float = 40.0,
) -> plt.Figure:
    """Generate stacked bar chart of noise contributions per node.

    Args:
        graph: AnalogGraph from any extractor.
        noise_budgets: Output of compute_noise_budget(). Auto-computed if None.
        mismatch_reports: Output of propagate_mismatch(). Optional.
        output_path: Save PNG to this path if provided.
        max_nodes: Truncate to first N analog nodes.
        figsize: Figure dimensions.
        target_snr_db: SNR target line to draw.

    Returns:
        matplotlib Figure.
    """
    # Auto-compute if not provided
    if noise_budgets is None:
        noise_budgets = compute_noise_budget(graph)

    # Filter to analog nodes only, truncate
    analog_budgets = [b for b in noise_budgets.values() if b.domain == "ANALOG"][:max_nodes]
    if not analog_budgets:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No analog nodes in graph", ha="center", va="center")
        return fig

    names = [b.node_name.split(".")[-1] for b in analog_budgets]  # Short names
    n = len(analog_budgets)
    x = np.arange(n)

    # Noise variances per category (in log scale for visualization)
    thermal = np.array([b.thermal_noise_variance for b in analog_budgets])
    shot = np.array([b.shot_noise_variance for b in analog_budgets])
    quant = np.array([b.quantization_noise_variance for b in analog_budgets])

    # Mismatch variance (if available)
    mismatch_var = np.zeros(n)
    if mismatch_reports:
        for i, b in enumerate(analog_budgets):
            r = mismatch_reports.get(b.node_name)
            if r:
                mismatch_var[i] = r.mean_relative_error ** 2

    # Convert to dB for display
    def to_db(var):
        return np.where(var > 0, 10 * np.log10(var + 1e-30), -100.0)

    snr_values = np.array([b.snr_db if b.snr_db != float("inf") else 80.0 for b in analog_budgets])
    meets_target = np.array([b.meets_target_snr for b in analog_budgets])

    fig, (ax_noise, ax_snr) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax_noise, ax_snr):
        ax.set_facecolor("#16213e")

    bar_w = 0.8
    bottom = np.zeros(n)
    for label, values, color in [
        ("Thermal (kT/C)", thermal, _NOISE_COLORS["thermal"]),
        ("Shot", shot, _NOISE_COLORS["shot"]),
        ("Quantization", quant, _NOISE_COLORS["quantization"]),
        ("Mismatch", mismatch_var, _NOISE_COLORS["mismatch"]),
    ]:
        ax_noise.bar(x, values, width=bar_w, bottom=bottom, color=color, label=label, alpha=0.85)
        bottom += values

    ax_noise.set_ylabel("Noise variance", color="white", fontsize=9)
    ax_noise.tick_params(colors="white")
    ax_noise.spines[:].set_color("#444")
    ax_noise.set_title(f"Analog Noise Budget — {graph.name}", color="white", fontsize=11, fontweight="bold")
    ax_noise.legend(facecolor="#16213e", labelcolor="white", fontsize=8, loc="upper right")
    ax_noise.set_yscale("log")
    ax_noise.set_ylim(bottom=1e-25)

    # SNR panel
    bar_colors = ["#2ecc71" if ok else "#e74c3c" for ok in meets_target]
    ax_snr.bar(x, snr_values, color=bar_colors, width=bar_w, alpha=0.85)
    ax_snr.axhline(y=target_snr_db, color="#f1c40f", linewidth=1.5, linestyle="--",
                   label=f"Target {target_snr_db:.0f} dB")
    ax_snr.set_ylabel("SNR (dB)", color="white", fontsize=9)
    ax_snr.set_xticks(x)
    ax_snr.set_xticklabels(names, rotation=60, ha="right", color="white", fontsize=7)
    ax_snr.tick_params(colors="white")
    ax_snr.spines[:].set_color("#444")
    ax_snr.legend(facecolor="#16213e", labelcolor="white", fontsize=8)

    violations = int((~meets_target).sum())
    fig.text(
        0.5, 0.01,
        f"SNR violations: {violations}/{n} nodes  |  Green = meets target  |  Red = needs improvement",
        ha="center", color="#aaa", fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig
