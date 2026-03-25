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

    # Stochastic-native nodes: SNR is undefined — noise is the intended signal.
    #   GIBBS_STEP (gold //): thermodynamic, calibration-free by physics
    #   SAMPLE / NOISE_INJECTION (gold \\): calibration error metric applies instead
    is_gibbs      = np.array([b.op_type == "GIBBS_STEP" for b in analog_budgets])
    is_calibration = np.array([b.op_type in ("NOISE_INJECTION", "SAMPLE") for b in analog_budgets])
    is_snr_exempt  = is_gibbs | is_calibration
    snr_values = np.array([b.snr_db if b.snr_db != float("inf") else 80.0 for b in analog_budgets])
    meets_target = np.array([b.meets_target_snr for b in analog_budgets])
    meets_cal = np.array([b.meets_calibration for b in analog_budgets])

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
    #   Green  : meets SNR target
    #   Red    : fails SNR target
    #   Gold // : GIBBS_STEP — thermodynamic, SNR undefined
    #   Gold \\ : SAMPLE / NOISE_INJECTION — calibration metric applies, not SNR
    bar_colors = []
    for gibbs, cal, snr_ok, cal_ok in zip(is_gibbs, is_calibration, meets_target, meets_cal):
        if gibbs:
            bar_colors.append("#f39c12")
        elif cal:
            bar_colors.append("#f39c12" if cal_ok else "#e67e22")  # dim orange = cal violation
        else:
            bar_colors.append("#2ecc71" if snr_ok else "#e74c3c")

    bars = ax_snr.bar(x, snr_values, color=bar_colors, width=bar_w, alpha=0.85)
    for bar, gibbs, cal in zip(bars, is_gibbs, is_calibration):
        if gibbs:
            bar.set_hatch("//")
            bar.set_edgecolor("#f1c40f")
        elif cal:
            bar.set_hatch("\\\\")
            bar.set_edgecolor("#f1c40f")

    # Annotate calibration error % on SAMPLE / NOISE_INJECTION bars
    for i, (b, cal) in enumerate(zip(analog_budgets, is_calibration)):
        if cal and not np.isnan(b.calibration_error):
            ax_snr.text(
                x[i], snr_values[i] + 1.0,
                f"{b.calibration_error * 100:.0f}%",
                ha="center", va="bottom", color="white", fontsize=6,
            )

    ax_snr.axhline(y=target_snr_db, color="#f1c40f", linewidth=1.5, linestyle="--",
                   label=f"Target {target_snr_db:.0f} dB")
    ax_snr.set_ylabel("SNR (dB)", color="white", fontsize=9)
    ax_snr.set_xticks(x)
    ax_snr.set_xticklabels(names, rotation=60, ha="right", color="white", fontsize=7)
    ax_snr.tick_params(colors="white")
    ax_snr.spines[:].set_color("#444")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color="#f1c40f", linestyle="--", label=f"Target {target_snr_db:.0f} dB"),
        Patch(facecolor="#2ecc71", label="Meets SNR target"),
        Patch(facecolor="#e74c3c", label="SNR violation"),
    ]
    if is_gibbs.any():
        legend_handles.append(
            Patch(facecolor="#f39c12", hatch="//", edgecolor="#f1c40f", label="GIBBS_STEP (SNR N/A)")
        )
    if is_calibration.any():
        legend_handles.append(
            Patch(facecolor="#f39c12", hatch="\\\\", edgecolor="#f1c40f", label="SAMPLE/NOISE (cal. error %)")
        )
    ax_snr.legend(handles=legend_handles, facecolor="#16213e", labelcolor="white", fontsize=7)

    # Violation counts: exclude SNR-exempt nodes from SNR count
    snr_violations = int((~meets_target & ~is_snr_exempt).sum())
    cal_violations = int((~meets_cal & is_calibration).sum())
    n_snr_nodes = int((~is_snr_exempt).sum())
    n_cal_nodes = int(is_calibration.sum())

    parts = [f"SNR violations: {snr_violations}/{n_snr_nodes}"]
    if n_cal_nodes:
        parts.append(f"Cal. violations: {cal_violations}/{n_cal_nodes}")
    if is_gibbs.any():
        parts.append("Gold // = thermodynamic (SNR N/A)")
    fig.text(0.5, 0.01, "  |  ".join(parts), ha="center", color="#aaa", fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig
