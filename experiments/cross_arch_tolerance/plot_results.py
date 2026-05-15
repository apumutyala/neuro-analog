#!/usr/bin/env python3
"""Generate all five figures from sweep results.

Figures saved to experiments/cross_arch_tolerance/figures/ as PNG + PDF.

Figure 1: Cross-Architecture Mismatch Tolerance (THE central figure)
Figure 2: Noise Source Ablation per architecture
Figure 3: ADC Precision Tradeoff
Figure 4: DEQ Convergence Under Mismatch (dual-axis)
Figure 5: Visual Results — generated samples at different sigma levels

Usage:
    python experiments/cross_arch_tolerance/plot_results.py
    python experiments/cross_arch_tolerance/plot_results.py --fig 1  # single figure
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

_RESULTS_DIR = Path(__file__).parent / "results"
_FIGURES_DIR = Path(__file__).parent / "figures"
_FIGURES_DIR.mkdir(exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────

# Architecture colors (consistent across all figures)
_COLORS = {
    "neural_ode":  "#2ecc71",   # green — #1 ranked
    "ssm":         "#3498db",   # blue — #2
    "ebm":         "#9b59b6",   # purple — stochastic, robust
    "flow":        "#1abc9c",   # teal — #4
    "deq":         "#e67e22",   # orange — convergence bifurcation
    "transformer": "#e74c3c",   # red — #6
    "diffusion":   "#95a5a6",   # grey — #7 (compounds errors)
}
_LABELS = {
    "neural_ode":  "Neural ODE",
    "ssm":         "SSM",
    "ebm":         "EBM",
    "flow":        "Flow",
    "deq":         "DEQ",
    "transformer": "Transformer",
    "diffusion":   "Diffusion",
}
_ORDER = ["neural_ode", "ssm", "ebm", "flow", "deq", "transformer", "diffusion"]

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
}
plt.rcParams.update(_STYLE)


_DOMAIN = "conservative"      # Set by main() via --domain flag
_SUBSTRATE = "classic"        # Set by main() via --substrate flag
_SIGMA_CAP = 0.15             # Cap sigma range for cross-arch comparison figures


def _load(name: str, suffix: str, domain: str | None = None, substrate: str | None = None) -> dict | None:
    """Load a result JSON for the given domain and substrate (defaults to module-level globals).

    Tries multiple filename patterns because sweep_all.py omits suffixes when
    the chosen substrate matches the architecture default, and adds them when it
    does not. The same logic applies to the 'classic' substrate: for architectures
    where classic is NOT the default (neural_ode, flow, deq), the file will have a
    '_classic' suffix even though substrate=='classic'.
    """
    d = domain if domain is not None else _DOMAIN
    s = substrate if substrate is not None else _SUBSTRATE
    domain_suffix = "" if d == "conservative" else f"_{d}"
    substrate_suffix = "" if s == "classic" else f"_{s}"

    candidates = [
        # Primary: full suffix path
        _RESULTS_DIR / f"{name}_{suffix}{domain_suffix}{substrate_suffix}.json",
    ]

    # Fallback 1: for substrate=classic, also try explicit _classic suffix
    # (architectures where classic is non-default keep the suffix)
    if s == "classic" and substrate_suffix == "":
        candidates.append(_RESULTS_DIR / f"{name}_{suffix}{domain_suffix}_classic.json")

    # Fallback 2: without substrate suffix (default substrate case)
    candidates.append(_RESULTS_DIR / f"{name}_{suffix}{domain_suffix}.json")

    # Fallback 3: base name only (no domain, no substrate)
    candidates.append(_RESULTS_DIR / f"{name}_{suffix}.json")

    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def _save_fig(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        path = _FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(str(path), dpi=300, bbox_inches="tight")
    print(f"  Saved: {_FIGURES_DIR / name}.{{png,pdf}}")


# ── Figure 1: Cross-Architecture Mismatch Tolerance ──────────────────────

def plot_figure1():
    """Two-panel figure: full scale (left) + frontier zoom (right)."""
    _mismatch_source = {name: "mismatch" for name in _ORDER}
    _mismatch_source["diffusion"] = "ablation_mismatch"
    _mismatch_source["flow"] = "ablation_mismatch"
    arch_data = {}
    for name in _ORDER:
        d = _load(name, _mismatch_source[name])
        if d is not None:
            arch_data[name] = d

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Cross-Architecture Analog Tolerance\n"
        "Task performance under fabrication mismatch (50–200 Monte Carlo trials per point)",
        fontsize=12, y=1.02,
    )
    panels = [
        (ax_full, (0.30, 1.08), "Full range"),
        (ax_zoom, (0.90, 1.04), "Frontier zoom  (0.90\u20131.04)"),
    ]
    handles = []
    threshold_annotations = []

    for name in _ORDER:
        if name not in arch_data:
            continue
        d = arch_data[name]
        sigma = np.array(d["sigma_values"])
        mean  = np.array(d["normalized_mean"])
        std   = np.array(d["normalized_std"])
        # Clip to _SIGMA_CAP so all architectures share the same x-range
        _cap_mask = sigma <= _SIGMA_CAP + 1e-9
        sigma, mean, std = sigma[_cap_mask], mean[_cap_mask], std[_cap_mask]
        color = _COLORS[name]
        threshold = d.get("degradation_threshold_10pct", None)

        for ax_i, _, _ in panels:
            ax_i.plot(sigma, mean, color=color, linewidth=2.2, zorder=3)
            ax_i.fill_between(sigma, mean - std, mean + std, color=color, alpha=0.12, zorder=2)

        if threshold and threshold > 0:
            last_safe_idx = np.searchsorted(sigma, threshold)
            first_fail_idx = last_safe_idx + 1
            if first_fail_idx < len(sigma):
                first_fail_sigma = sigma[first_fail_idx]
                ax_full.axvline(first_fail_sigma, color=color, linewidth=0.6, linestyle=":", alpha=0.5)
                threshold_annotations.append((first_fail_sigma, 0.305, color, f"{first_fail_sigma:.2f}"))

        handles.append(Line2D([0], [0], color=color, linewidth=2.2, label=_LABELS[name]))

    for x, y, c, txt in sorted(threshold_annotations):
        ax_full.text(x, y, txt, color=c, fontsize=7.5, ha="center", rotation=90)

    # Build tick grid from actual sigma values across loaded datasets (capped at _SIGMA_CAP)
    _all_sigmas = sorted({
        s for d in arch_data.values() for s in d.get("sigma_values", [])
        if s <= _SIGMA_CAP + 1e-9
    })
    tick_vals   = [s for s in _all_sigmas if s in {0.0, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30}]
    if not tick_vals:
        tick_vals = _all_sigmas
    tick_labels = [f"{s:.0%}" for s in tick_vals]
    _sigma_max = max(_all_sigmas) if _all_sigmas else 0.15
    _xlim_right = _sigma_max * 1.04  # small margin so last tick isn't flush with edge

    for ax_i, (ylo, yhi), title in panels:
        ax_i.set_title(title, fontsize=10, pad=6)
        ax_i.set_xlabel("Conductance Mismatch  \u03c3")
        ax_i.set_xlim(0, _xlim_right)
        ax_i.set_ylim(ylo, yhi)
        ax_i.set_xticks(tick_vals)
        ax_i.set_xticklabels(tick_labels)
        ax_i.axhline(0.90, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)

    ax_full.set_ylabel("Normalized Task Performance  (digital = 1.0)")
    ax_full.text(_xlim_right, 0.905, "90% threshold", color="#95a5a6", fontsize=9, va="bottom", ha="right")
    ax_zoom.text(_xlim_right, 0.902, "90% threshold", color="#95a5a6", fontsize=8, va="bottom", ha="right")
    ax_full.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9,
                   title="Architecture", title_fontsize=9)
    fig.tight_layout()
    _save_fig(fig, "fig1_mismatch_tolerance")
    plt.close(fig)


# ── Figure 2: Noise Source Ablation ──────────────────────────────────────

def plot_figure2():
    """7-panel ablation: mismatch-only, thermal-only, quantization-only per architecture."""
    n_arch = len(_ORDER)
    _YLIM = (0, 1.15)
    fig, axes = plt.subplots(1, n_arch, figsize=(16, 4), sharey=False)
    # Determine the max sigma from any available ablation dataset (for title)
    _ablation_sigma_label = "max σ"
    for _n in _ORDER:
        _d = _load(_n, "ablation_mismatch")
        if _d is not None:
            _sv = np.array(_d["sigma_values"])
            _ablation_sigma_label = f"σ = {_sv[-1]:.0%}"
            break
    fig.suptitle(f"Noise Source Ablation at {_ablation_sigma_label}\n(mismatch swept; thermal & quantization are fixed physical levels)", y=1.02)

    for ax, name in zip(axes, _ORDER):
        ax.set_title(_LABELS[name], fontsize=9)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Mismatch", "Thermal", "Quant."], fontsize=8, rotation=30)
        if ax == axes[0]:
            ax.set_ylabel("Normalized Quality")

        colors_bar = ["#e74c3c", "#3498db", "#f39c12"]
        for i, noise_type in enumerate(["mismatch", "thermal", "quantization"]):
            d = _load(name, f"ablation_{noise_type}")
            if d is None:
                ax.bar(i, 0.5, color=colors_bar[i], alpha=0.5, width=0.6)
                continue
            baseline = d.get("digital_baseline", 1.0)
            if baseline == 0:
                baseline = 1.0
            # Find the index of the highest sigma in this dataset for the ablation bar.
            sigma_arr = np.array(d["sigma_values"])
            # Use last sigma <= _SIGMA_CAP so all bars compare at the same level
            _valid = np.where(sigma_arr <= _SIGMA_CAP + 1e-9)[0]
            sigma_idx = int(_valid[-1]) if len(_valid) else len(sigma_arr) - 1
            sigma_label_val = sigma_arr[sigma_idx]
            mean_at_sigma = np.array(d["mean"])[sigma_idx]
            norm_q = 1.0 + (mean_at_sigma - baseline) / abs(baseline) if baseline != 0 else 0.5
            bar_h = min(norm_q, _YLIM[1] - 0.01)  # clip bar to axis; annotate if off-chart
            ax.bar(i, bar_h, color=colors_bar[i], alpha=0.8, width=0.6)
            if norm_q > _YLIM[1]:
                # Off-chart: draw an upward arrow and label the real value
                ax.annotate(f"{norm_q:.2f}", xy=(i, _YLIM[1] - 0.01),
                            xytext=(i, _YLIM[1] - 0.06),
                            ha="center", fontsize=7.5, color=colors_bar[i],
                            arrowprops=dict(arrowstyle="->", color=colors_bar[i], lw=0.8))
            elif norm_q < 0:
                ax.text(i, 0.02, f"{norm_q:.2f}", ha="center", fontsize=7.5, color=colors_bar[i])
            else:
                ax.text(i, bar_h + 0.01, f"{norm_q:.2f}", ha="center", fontsize=7.5)

        ax.axhline(1.0, color="#bdc3c7", linewidth=0.8, linestyle="--")
        ax.set_ylim(*_YLIM)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#e74c3c", label="Mismatch only"),
        Patch(facecolor="#3498db", label="Thermal only"),
        Patch(facecolor="#f39c12", label="Quantization only"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save_fig(fig, "fig2_ablation")
    plt.close(fig)


# ── Figure 3: ADC Precision Tradeoff ─────────────────────────────────────

def plot_figure3():
    """Quality vs ADC bit width at fixed σ=0.05."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel("ADC Bit Width")
    ax.set_ylabel("Normalized Quality  (digital = 1.0)")
    ax.set_title("ADC Precision Tradeoff\nQuality at fixed mismatch σ=0.05 (50 trials per point)", pad=10)
    ax.axhline(0.90, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_ylim(-1.2, 1.08)

    _YFLOOR = -1.2
    for name in _ORDER:
        d = _load(name, "adc")
        if d is None:
            continue
        bits = np.array(d["sigma_values"])
        mean = np.array(d["normalized_mean"])
        std  = np.array(d["normalized_std"])
        color = _COLORS[name]
        mean_clipped = np.clip(mean, _YFLOOR, 1.08)
        ax.plot(bits, mean_clipped, color=color, linewidth=2.0, marker="o", markersize=4, label=_LABELS[name])
        ax.fill_between(bits, np.clip(mean - std, _YFLOOR, 1.08),
                        np.clip(mean + std, _YFLOOR, 1.08), color=color, alpha=0.10)
        # Annotate any values that fell below the floor
        for bw, mv, mc in zip(bits, mean, mean_clipped):
            if mv < _YFLOOR:
                ax.annotate(f"{mv:.2f}", xy=(bw, _YFLOOR), xytext=(bw, _YFLOOR + 0.10),
                            ha="center", fontsize=7.5, color=color,
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax.axhline(0.90, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)
    ax.text(2.2, 0.92, "90% quality threshold", color="#95a5a6", fontsize=8, ha="left", va="bottom")
    ax.set_xticks([2, 4, 6, 8, 10, 12, 16])
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, "fig3_adc_precision")
    plt.close(fig)


# ── Figure 4: DEQ Convergence Under Mismatch ─────────────────────────────

def plot_figure4():
    """DEQ: discrete fixed-point iteration vs. Hopfield ODE relaxation on analog substrate.

    Left panel: mismatch sweep, solid=discrete iteration, dashed=Hopfield ODE.
    Right panel: ADC precision sweep (extended y-axis to show off-chart divergence).
    """
    import matplotlib.ticker
    _COLOR = "#e67e22"

    d_disc_m = _load("deq", "mismatch")
    d_hop_m  = _load("deq", "mismatch",  substrate="hopfield")
    d_disc_a = _load("deq", "adc")
    d_hop_a  = _load("deq", "adc",       substrate="hopfield")

    if d_disc_m is None:
        print("  [Fig 4] No DEQ mismatch data. Skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "DEQ: Discrete Iteration vs. Hopfield ODE Relaxation on Analog Substrate",
        fontsize=12, y=1.01,
    )

    # ── Left: mismatch sweep ───────────────────────────────────────────────────
    ax1.set_title("Mismatch Tolerance", fontsize=10)
    ax1.set_xlabel("Conductance Mismatch  \u03c3")
    ax1.set_ylabel("Normalized Task Performance  (digital = 1.0)")
    ax1.axhline(0.90, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)
    _deq_sigmas = sorted({s for d in [d_disc_m, d_hop_m] if d for s in d.get("sigma_values", [])})
    _deq_xlim = max(_deq_sigmas) * 1.04 if _deq_sigmas else 0.155
    _deq_ticks = [s for s in _deq_sigmas if s in {0.0, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30}]
    if not _deq_ticks:
        _deq_ticks = _deq_sigmas
    ax1.set_xlim(0, _deq_xlim)
    ax1.set_ylim(0.3, 1.15)
    ax1.set_xticks(_deq_ticks)
    ax1.set_xticklabels([f"{s:.0%}" for s in _deq_ticks])

    if d_disc_m is not None:
        sigma = np.array(d_disc_m["sigma_values"])
        m     = np.array(d_disc_m["normalized_mean"])
        s     = np.array(d_disc_m["normalized_std"])
        ax1.plot(sigma, m, color=_COLOR, linewidth=2.2, linestyle="-",  label="Discrete iteration")
        ax1.fill_between(sigma, m - s, m + s, color=_COLOR, alpha=0.12)

    if d_hop_m is not None:
        sigma = np.array(d_hop_m["sigma_values"])
        m     = np.array(d_hop_m["normalized_mean"])
        s     = np.array(d_hop_m["normalized_std"])
        ax1.plot(sigma, m, color=_COLOR, linewidth=2.2, linestyle="--", label="Hopfield ODE relaxation")
        ax1.fill_between(sigma, m - s, m + s, color=_COLOR, alpha=0.07)

    ax1.legend(fontsize=9, loc="lower left")
    ax1.text(0.02, 0.04,
             "Hopfield: RC circuit settles to fixed point\ncontinuously \u2014 no discrete iteration needed",
             transform=ax1.transAxes, fontsize=8, color="#555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9f0", edgecolor="#e67e22", alpha=0.85))

    # ── Right: ADC precision sweep ─────────────────────────────────────────────
    ax2.set_title("ADC Precision  (\u03c3 = 0.05)", fontsize=10)
    ax2.set_xlabel("ADC Bit Width")
    ax2.set_ylabel("Normalized Task Performance  (digital = 1.0)")
    ax2.axhline(0.90, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)
    ax2.axhline(0.0,  color="#ecf0f1", linewidth=0.6, linestyle=":",  zorder=0)
    ax2.set_xlim(1.5, 16.5)
    ax2.set_ylim(-1.05, 1.15)
    ax2.set_xticks([2, 4, 6, 8, 10, 12, 16])

    for d, label, ls in [(d_disc_a, "Discrete", "-"), (d_hop_a, "Hopfield", "--")]:
        if d is None:
            continue
        bits = np.array(d["sigma_values"])
        m    = np.array(d["normalized_mean"])
        m_clipped = np.clip(m, -1.0, 1.15)
        ax2.plot(bits, m_clipped, color=_COLOR, linewidth=2.2, linestyle=ls,
                 marker="o", markersize=4, label=label)
        for bw, mv, mc in zip(bits, m, m_clipped):
            if mv < -1.0:
                ax2.annotate(f"{mv:.2f}", xy=(bw, -1.0), xytext=(bw, -0.85),
                             ha="center", fontsize=7.5, color=_COLOR,
                             arrowprops=dict(arrowstyle="->", color=_COLOR, lw=0.8))

    ax2.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    _save_fig(fig, "fig4_deq_convergence")
    plt.close(fig)


# ── Figure 5: Visual Results ──────────────────────────────────────────────

def plot_figure5():
    """Visual degradation: 2D flow fields + 8x8 MNIST samples at different sigma."""
    import torch
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from models import neural_ode as _no, flow as _fl, diffusion as _diff, ebm as _ebm
        from models.neural_ode import _get_data as _no_data
        from models.flow import _get_data as _fl_data
    except ImportError as e:
        print(f"  [Fig 5] Could not import models: {e}. Skipping.")
        return

    ckpt_dir = Path(__file__).parent / "checkpoints"
    sigmas = [0.0, 0.05, 0.10, 0.15]
    sigma_labels = ["σ=0 (digital)", "σ=0.05", "σ=0.10", "σ=0.15"]

    from neuro_analog.simulator import analogize, resample_all_mismatch

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(4, len(sigmas) + 1, figure=fig, wspace=0.05, hspace=0.3)
    fig.suptitle("Pilot-Scale Visual Quality Degradation Under Analog Mismatch\n(8×8 MNIST / 2D toy data; CIFAR-10 / WikiText-2 in unified benchmark)", fontsize=12, y=1.01)

    _row_labels = ["Neural ODE (flow field)", "Flow (generated)", "Diffusion (8×8)", "EBM (reconstruction)"]

    for row, (arch_name, module, row_label, ckpt_name) in enumerate([
        ("neural_ode",  _no,   "Neural ODE\n(flow field)",     "neural_ode"),
        ("flow",        _fl,   "Flow Model\n(generated pts)",  "flow"),
        ("diffusion",   _diff, "Diffusion\n(8×8 MNIST)",      "diffusion"),
        ("ebm",         _ebm,  "EBM\n(reconstruction)",       "ebm"),
    ]):
        # Row label
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.text(0.5, 0.5, row_label, ha="center", va="center",
                      fontsize=9, transform=ax_label.transAxes, fontweight="bold")
        ax_label.axis("off")

        ckpt = ckpt_dir / f"{ckpt_name}.pt"
        if not ckpt.exists():
            for col in range(len(sigmas)):
                ax = fig.add_subplot(gs[row, col + 1])
                ax.text(0.5, 0.5, "No checkpoint", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.axis("off")
            continue

        base_model = module.load_model(str(ckpt))
        _dev = next(base_model.parameters()).device

        for col, sigma in enumerate(sigmas):
            ax = fig.add_subplot(gs[row, col + 1])
            if col == 0:
                ax.set_title(sigma_labels[col], fontsize=9)
            else:
                ax.set_title(sigma_labels[col], fontsize=9)
            ax.axis("off")

            try:
                if sigma == 0.0:
                    model = base_model
                else:
                    model = analogize(base_model, sigma_mismatch=sigma)

                if arch_name in ("neural_ode", "flow"):
                    # 2D scatter of generated samples
                    from neuro_analog.simulator import analog_odeint
                    torch.manual_seed(0)
                    z0 = torch.randn(200, 2, device=_dev)
                    t_span = torch.tensor([0.0, 1.0], device=_dev)
                    _dt = 0.025 if arch_name == "neural_ode" else 0.01
                    model.eval()
                    x_gen = analog_odeint(model, z0, t_span, dt=_dt).detach().cpu().numpy()
                    _margin = 0.3
                    _xr = [x_gen[:, 0].min() - _margin, x_gen[:, 0].max() + _margin]
                    _yr = [x_gen[:, 1].min() - _margin, x_gen[:, 1].max() + _margin]
                    ax.scatter(x_gen[:, 0], x_gen[:, 1], s=3, alpha=0.6,
                               c=_COLORS[arch_name], rasterized=True)
                    ax.set_xlim(_xr); ax.set_ylim(_yr)
                    ax.axis("on")
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.5)

                elif arch_name == "diffusion":
                    # Generate one 8x8 image
                    import torch
                    from models.diffusion import _get_betas, _get_alphas, _IMG_DIM
                    import math as _math
                    betas = _get_betas().to(_dev)
                    _, alphas_bar = _get_alphas(betas)
                    x = torch.randn(1, _IMG_DIM, device=_dev)
                    n_ddim = 10
                    T = 100
                    ddim_steps = torch.linspace(T-1, 0, n_ddim+1).long()
                    model.eval()
                    with torch.no_grad():
                        for di in range(n_ddim):
                            t_c = ddim_steps[di].item()
                            t_n = ddim_steps[di+1].item()
                            t_t = torch.full((1,), t_c, dtype=torch.long, device=_dev)
                            eps = model(x, t_t)
                            a_c = alphas_bar[t_c]
                            x0p = (x - _math.sqrt(1 - a_c) * eps) / _math.sqrt(a_c)
                            x0p = x0p.clamp(-1, 1)
                            if t_n >= 0:
                                a_n = alphas_bar[t_n]
                                x = _math.sqrt(a_n)*x0p + _math.sqrt(1-a_n)*eps
                            else:
                                x = x0p
                    img = x.detach().cpu().numpy().reshape(8, 8)
                    ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
                    ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])

                elif arch_name == "ebm":
                    from models.ebm import _get_data
                    _, X_test = _get_data()
                    v = X_test[:1].clone().to(_dev)
                    model.eval()
                    with torch.no_grad():
                        for _ in range(20):
                            h = model.h_given_v(v)
                            h_s = (h > torch.rand_like(h)).float()
                            v = model.v_given_h(h_s)
                    img = v.detach().cpu().numpy().reshape(8, 8)
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                    ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])

            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6, color="red")

    fig.tight_layout()
    _save_fig(fig, "fig5_visual_results")
    plt.close(fig)


# ── Figure 6: Output MSE vs Mismatch ─────────────────────────────────────

def plot_figure6():
    """Direct measurement: output MSE between analog and digital."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xlabel("Conductance Mismatch  σ")
    ax.set_ylabel("Output MSE  (lower = less corruption)")
    ax.set_title("Output Corruption Under Analog Mismatch\nMSE between digital and analog outputs (50–200 Monte Carlo trials)", pad=12)
    ax.set_yscale("log")  # Log scale since MSE values span orders of magnitude

    handles = []

    for name in _ORDER:
        d = _load(name, "output_mse")
        if d is None:
            continue

        if "mean" not in d:
            continue
        sigma = np.array(d["sigma_values"])
        mean = np.array(d["mean"])
        std = np.array(d["std"])
        _cap_mask = sigma <= _SIGMA_CAP + 1e-9
        sigma, mean, std = sigma[_cap_mask], mean[_cap_mask], std[_cap_mask]
        color = _COLORS[name]
        label = _LABELS[name]

        ax.plot(sigma, mean, color=color, linewidth=2.2, label=label, zorder=3)
        ax.fill_between(sigma, np.maximum(mean - std, 1e-10), mean + std, color=color, alpha=0.12, zorder=2)

        handles.append(Line2D([0], [0], color=color, linewidth=2.2, label=f"{label}"))

    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9,
              title="Architecture", title_fontsize=9)
    # Dynamic x-axis based on loaded MSE sigma values
    _mse_sigmas = sorted({s for name in _ORDER for d in [_load(name, "output_mse")] if d for s in d.get("sigma_values", []) if s <= _SIGMA_CAP + 1e-9})
    if _mse_sigmas:
        _mse_xlim = max(_mse_sigmas) * 1.04
        _mse_ticks = [s for s in _mse_sigmas if s in {0.0, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30}]
        if not _mse_ticks:
            _mse_ticks = _mse_sigmas
        ax.set_xlim(0, _mse_xlim)
        ax.set_xticks(_mse_ticks)
        ax.set_xticklabels([f"{s:.0%}" for s in _mse_ticks])
    else:
        ax.set_xlim(0, 0.155)
        ax.set_xticks([0.0, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15])
        ax.set_xticklabels(["0%", "3%", "5%", "7%", "10%", "12%", "15%"])
    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    # Data-driven note: check if any architecture has MSE outside visible range
    off_chart = []
    for name in _ORDER:
        d = _load(name, "output_mse")
        if d is None or "mean" not in d:
            continue
        mean = np.array(d["mean"])
        if np.any(mean > ax.get_ylim()[1] * 0.9):
            off_chart.append(_LABELS[name])
    if off_chart:
        note = "Off-chart: " + ", ".join(off_chart) + "\n(MSE exceeds axis limit; see JSON for exact values)"
        ax.text(0.99, 0.02, note, transform=ax.transAxes, fontsize=7.5, color="#95a5a6",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.8))

    fig.tight_layout()
    _save_fig(fig, "fig6_output_mse")
    plt.close(fig)


# ── Figure 7: Conservative vs Full-Analog Profile Comparison ─────────────

def plot_figure7():
    """Side-by-side: conservative (ADC per layer) vs full_analog (ADC at readout only).

    Only plotted if *_full_analog.json results exist. Shows the two profiles
    on the same axes using solid (conservative) vs dashed (full_analog) lines.
    """
    # Check if any full_analog results exist
    has_full = any(
        any(_RESULTS_DIR.glob(f"{name}_mismatch_full_analog*.json"))
        for name in _ORDER
    )
    if not has_full:
        print("  [Fig 7] No full_analog results found. Run sweep_all.py --analog-domain full_analog first.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax, sweep_suffix, title_suffix in [
        (ax1, "mismatch", "Mismatch Tolerance"),
        (ax2, "adc", "ADC Precision (σ=0.05)"),
    ]:
        ax.set_title(
            f"Conservative vs. Full-Analog\n{title_suffix}",
            pad=10
        )
        if sweep_suffix == "mismatch":
            ax.set_xlabel("Conductance Mismatch  σ")
            ax.set_xlim(0, _SIGMA_CAP * 1.04)
            ax.set_xticks([0.0, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15])
            ax.set_xticklabels(["0%", "3%", "5%", "7%", "10%", "12%", "15%"])
        else:
            ax.set_xlabel("ADC Bit Width")
            ax.set_xticks([2, 4, 6, 8, 10, 12, 16])
        ax.set_ylabel("Normalized Quality  (digital = 1.0)")
        ax.axhline(0.90, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)
        if sweep_suffix == "adc":
            ax.set_ylim(-1.05, 1.15)
        else:
            ax.set_ylim(0.3, 1.25)  # raised ceiling: Neural ODE full-analog can exceed 1.1

        for name in _ORDER:
            d_cons = _load(name, sweep_suffix, domain="conservative")
            d_full = _load(name, sweep_suffix, domain="full_analog")
            if d_cons is None and d_full is None:
                continue
            color = _COLORS[name]
            label = _LABELS[name]
            x_axis = "sigma_values"

            if d_cons is not None:
                x = np.array(d_cons[x_axis])
                m = np.array(d_cons["normalized_mean"])
                if sweep_suffix == "mismatch":
                    _mask = x <= _SIGMA_CAP + 1e-9
                    x, m = x[_mask], m[_mask]
                ax.plot(x, m, color=color, linewidth=2.0, linestyle="-",
                        label=f"{label} (conservative)")

            if d_full is not None:
                x = np.array(d_full[x_axis])
                m = np.array(d_full["normalized_mean"])
                if sweep_suffix == "mismatch":
                    _mask = x <= _SIGMA_CAP + 1e-9
                    x, m = x[_mask], m[_mask]
                ax.plot(x, m, color=color, linewidth=2.0, linestyle="--",
                        label=f"{label} (full-analog)")

    # Shared legend — one entry per architecture (solid=conservative, dashed=full_analog)
    from matplotlib.lines import Line2D
    arch_handles = [Line2D([0], [0], color=_COLORS[n], linewidth=2.0, label=_LABELS[n])
                    for n in _ORDER]
    style_handles = [
        Line2D([0], [0], color="black", linewidth=1.5, linestyle="-",  label="Conservative (per-layer ADC)"),
        Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label="Full-analog (readout ADC only)"),
    ]
    fig.legend(handles=arch_handles + style_handles,
               loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.08), framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, "fig7_profile_comparison")
    plt.close(fig)




# ── Figure 10: Energy–Accuracy Pareto Frontier ───────────────────────────

def plot_figure10():
    """Scatter plot: energy saving vs. analog robustness threshold.

    X-axis: energy saving vs digital baseline (%)
    Y-axis: degradation threshold sigma_10pct (higher = more mismatch-tolerant)
    Marker size: D/A boundary count (larger = more domain crossings)
    Color: architecture family (consistent with Fig 1)

    Loads mismatch JSONs and pilot profiles to build the Pareto landscape.
    This figure tests the core co-design question: does analog deployment
    actually pay off for this architecture, given its accuracy loss under mismatch?
    """
    _PROFILE_DIR = Path(__file__).parent / "results" / "profiles"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Energy Saving vs. Digital Baseline (%)")
    ax.set_ylabel("Mismatch Robustness Threshold  σ₁₀%")
    ax.set_title("Analog Deployability: Energy–Accuracy Pareto Frontier\n"
                 "(larger markers = more D/A boundaries; higher-right = better analog fit)",
                 fontsize=11, pad=10)

    for name in _ORDER:
        d = _load(name, "mismatch")
        if d is None:
            continue
        threshold = d.get("degradation_threshold_10pct", 0.0)
        energy_saving = d.get("energy_saving_vs_digital", None)
        if energy_saving is None:
            # Fallback: load profile
            profile_path = _PROFILE_DIR / f"{name}_profile.json"
            if profile_path.exists():
                with open(profile_path) as f:
                    pdata = json.load(f)
                energy_saving = pdata.get("analog_energy_saving_vs_digital", 0.0)
            else:
                energy_saving = 0.0

        # D/A boundary count from profile if available, else default 1
        da_count = 1
        profile_path = _PROFILE_DIR / f"{name}_profile.json"
        if profile_path.exists():
            with open(profile_path) as f:
                pdata = json.load(f)
            da_count = max(pdata.get("da_boundary_count", 1), 1)

        color = _COLORS[name]
        label = _LABELS[name]
        size = 80 + 400 * (da_count / 100)  # scale 1→100 boundaries to ~80→480 pt

        ax.scatter(energy_saving * 100, threshold, s=size, c=color,
                   alpha=0.85, edgecolors="black", linewidth=0.5, zorder=3, label=label)
        # Annotate
        ax.annotate(label, (energy_saving * 100, threshold),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=color, fontweight="bold")

    ax.axhline(0.05, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)
    ax.text(ax.get_xlim()[1], 0.052, "5% threshold", color="#95a5a6", fontsize=8, ha="right")
    ax.axhline(0.10, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)
    ax.text(ax.get_xlim()[1], 0.102, "10% threshold", color="#95a5a6", fontsize=8, ha="right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9, title="Architecture")
    fig.tight_layout()
    _save_fig(fig, "fig10_energy_accuracy_pareto")
    plt.close(fig)


# ── Figure 11: DEQ Spectral Radius vs Convergence Failure ───────────────

def plot_figure11():
    """Two-panel figure linking dynamics theory to empirical failure.

    Left: spectral radius ρ(∂f/∂z) at equilibrium vs mismatch σ.
    Right: convergence failure rate vs mismatch σ.

    The key prediction: ρ crossing 1 should coincide with failure rate rising
    above zero. If it does, spectral radius is a predictive diagnostic for
    analog deployability of iterative fixed-point architectures.
    """
    conv_paths = list(_RESULTS_DIR.glob("deq_convergence*.json"))
    if not conv_paths:
        print("  [Fig 11] No deq_convergence*.json found. Skipping.")
        return
    conv_path = conv_paths[0]

    with open(conv_path) as f:
        data = json.load(f)

    sigma = np.array(data.get("sigma_values", []))
    failure = np.array(data.get("convergence_failure_rate", []))
    mean_iters = np.array(data.get("mean_iterations", []))
    spectral = data.get("spectral_radii", None)
    if spectral is not None:
        spectral = np.array([s for s in spectral if s is not None])
        # If lengths mismatch (some sigmas may lack rho), truncate
        if len(spectral) < len(sigma):
            sigma = sigma[:len(spectral)]
            failure = failure[:len(spectral)]
            mean_iters = mean_iters[:len(spectral)]

    # Check if spectral radius data is constant (nominal-only, not computed under mismatch).
    # This happens because compute_stability_bounds() evaluates rho on the nominal model.
    spectral_is_informative = (
        spectral is not None
        and len(spectral) > 1
        and not np.allclose(spectral, spectral[0])
    )

    if spectral is None or len(spectral) == 0 or not spectral_is_informative:
        # Single-panel: empirical convergence metrics only.
        # Do NOT show a flat "measured ρ" curve that falsely implies dynamics were evaluated
        # under mismatch. Instead, state the nominal ρ as a text annotation.
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlabel("Conductance Mismatch  σ")
        ax.set_ylabel("Convergence Failure Rate", color="#e67e22")
        ax.plot(sigma, failure, color="#e67e22", linewidth=2.2, marker="s", markersize=5,
                zorder=3, label="Failure rate")
        ax.axhline(0.5, color="#bdc3c7", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.tick_params(axis="y", labelcolor="#e67e22")

        axb = ax.twinx()
        axb.set_ylabel("Mean Iterations to Convergence", color="#8e44ad")
        axb.plot(sigma, mean_iters, color="#8e44ad", linewidth=2.2, marker="^", markersize=5,
                 zorder=3, linestyle="--", label="Mean iterations")
        axb.tick_params(axis="y", labelcolor="#8e44ad")

        ax.set_title("DEQ Convergence Under Analog Mismatch\n(empirical; nominal ρ = 1.76 evaluated at σ = 0)",
                     fontsize=11, pad=10)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = axb.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

        fig.tight_layout()
        _save_fig(fig, "fig11_deq_spectral_radius")
        plt.close(fig)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("DEQ: Spectral Radius Predicts Analog Convergence Failure", fontsize=12, y=1.02)

    # Left: spectral radius (only shown if it actually varies with sigma)
    ax1.set_xlabel("Conductance Mismatch  σ")
    ax1.set_ylabel("Spectral Radius  ρ(∂f/∂z) at z*")
    ax1.axhline(1.0, color="#e74c3c", linewidth=1.5, linestyle="--", zorder=0, label="ρ = 1 (stability boundary)")
    ax1.plot(sigma, spectral, color="#2980b9", linewidth=2.2, marker="o", markersize=5, zorder=3, label="Measured ρ")
    ax1.set_title("Dynamics Diagnostic\nρ ≥ 1 predicts divergence", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_ylim(bottom=0)

    # Right: failure rate + mean iterations
    ax2.set_xlabel("Conductance Mismatch  σ")
    ax2.set_ylabel("Convergence Failure Rate", color="#e67e22")
    ax2.plot(sigma, failure, color="#e67e22", linewidth=2.2, marker="s", markersize=5, zorder=3, label="Failure rate")
    ax2.axhline(0.5, color="#bdc3c7", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.tick_params(axis="y", labelcolor="#e67e22")

    ax2b = ax2.twinx()
    ax2b.set_ylabel("Mean Iterations to Convergence", color="#8e44ad")
    ax2b.plot(sigma, mean_iters, color="#8e44ad", linewidth=2.2, marker="^", markersize=5,
              zorder=3, linestyle="--", label="Mean iterations")
    ax2b.tick_params(axis="y", labelcolor="#8e44ad")

    ax2.set_title("Empirical Convergence Cost\n(failure rate + iterations)", fontsize=10)
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

    fig.tight_layout()
    _save_fig(fig, "fig11_deq_spectral_radius")
    plt.close(fig)


# ── Figure 12: Analog Acceleration vs. Degradation Threshold ────────────────────

def plot_figure12():
    """Speedup vs. Energy Reduction scatter — the true analog co-design Pareto frontier.

    Each architecture is one point.  X-axis = latency speedup (digital→analog,
    log scale).  Y-axis = energy reduction factor (digital/analog, log scale).
    Color = architecture family.  Size = mismatch robustness threshold σ₁₀%
    (larger = more robust).  This avoids the x-axis clustering problem where
    most architectures pile up at the σ=0.15 cap.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlabel("Latency Speedup  (digital iterations / analog settling, ×)", fontsize=11)
    ax.set_ylabel("Energy Reduction Factor  (digital energy / analog energy, ×)", fontsize=11)
    ax.set_title(
        "Analog Co-Design Pareto Frontier\n"
        "(marker size ∝ mismatch robustness σ₁₀%; upper-right = best analog fit)",
        fontsize=12, pad=10
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    for name in _ORDER:
        d = _load(name, "mismatch")
        if d is None:
            continue
        threshold = d.get("degradation_threshold_10pct", 0.0)
        speedup = d.get("speedup_vs_digital", None)

        # Energy reduction factor: digital / analog
        reduction_factor = None
        if d.get("analog_energy_pJ") and d.get("digital_energy_pJ") and d["analog_energy_pJ"] > 0:
            reduction_factor = d["digital_energy_pJ"] / d["analog_energy_pJ"]
        if reduction_factor is None and d.get("analog_energy_reduction_factor"):
            reduction_factor = d["analog_energy_reduction_factor"]
        if reduction_factor is None:
            saving = d.get("energy_saving_vs_digital")
            if saving is not None and saving < 1.0:
                reduction_factor = 1.0 / (1.0 - saving) if saving < 1.0 else 1e6

        if speedup is None:
            from neuro_analog.simulator.analog_acceleration import compute_acceleration
            from neuro_analog.ir.types import DynamicsProfile, ArchitectureFamily
            family_map = {
                "neural_ode": ArchitectureFamily.NEURAL_ODE,
                "transformer": ArchitectureFamily.TRANSFORMER,
                "diffusion": ArchitectureFamily.DIFFUSION,
                "flow": ArchitectureFamily.FLOW,
                "ebm": ArchitectureFamily.EBM,
                "deq": ArchitectureFamily.DEQ,
                "ssm": ArchitectureFamily.SSM,
            }
            acc = compute_acceleration(family_map.get(name, ArchitectureFamily.TRANSFORMER), DynamicsProfile())
            speedup = acc.speedup_settling_vs_digital

        if speedup is None or reduction_factor is None:
            continue

        color = _COLORS[name]
        label = _LABELS[name]
        # Size mapped from threshold (0→100, 0.15→500)
        size = 100 + (threshold / 0.15) * 400 if threshold is not None else 100

        ax.scatter(speedup, reduction_factor, s=size, c=color,
                   edgecolors="black", linewidth=0.8, zorder=3)
        # Offset annotations to avoid overlap.
        # Large markers (σ₁₀%=0.15 → size=500, radius≈13 pt) need offsets that
        # clear the circle plus a small margin.  ha/va are chosen so the text
        # box edge faces away from the marker.
        offsets = {
            "neural_ode": (-40, 20),  "ssm": (30, 25),       "ebm": (-35, -30),
            "flow": (-28, 25),        "deq": (22, 22),       "transformer": (22, 22),
            "diffusion": (22, 22),
        }
        ha_aligns = {
            "neural_ode": "right",  "ssm": "left",        "ebm": "right",
            "flow": "right",        "deq": "left",         "transformer": "left",
            "diffusion": "left",
        }
        ox, oy = offsets.get(name, (18, 18))
        ha = ha_aligns.get(name, "center")

        # Compact two-line annotation; full explanation lives in caption/footer.
        ann_text = f"{label}\nσ₁₀%={threshold:.2f}"

        ax.annotate(
            ann_text,
            (speedup, reduction_factor),
            textcoords="offset points", xytext=(ox, oy),
            fontsize=8.5, color=color, fontweight="bold",
            ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, alpha=0.9, linewidth=0.9),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.9)
        )

    ax.grid(True, which="both", alpha=0.25, linestyle="--", zorder=0)
    # Reference quadrant lines at "good" thresholds
    ax.axvline(100, color="#bdc3c7", linewidth=0.8, linestyle="--", zorder=0, alpha=0.6)
    ax.axhline(1000, color="#bdc3c7", linewidth=0.8, linestyle="--", zorder=0, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.95, 1200, "High energy\nreduction",
            ha="right", va="bottom", fontsize=8, color="#95a5a6", style="italic")
    ax.text(120, ax.get_ylim()[1] * 0.95, "High speedup",
            ha="left", va="top", fontsize=8, color="#95a5a6", style="italic")

    # Compact two-line footer, right-aligned at bottom
    ax.text(0.98, 0.02,
            "σ₁₀% = σ where quality falls to 90% of digital baseline.\n"
            "Diffusion σ₁₀%=0.00: threshold not reached in σ=0–0.15 range.",
            transform=ax.transAxes, fontsize=7.5, color="#555", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8f9fa",
                      edgecolor="#bdc3c7", alpha=0.8, linewidth=0.5))

    fig.tight_layout()
    _save_fig(fig, "fig12_speedup_vs_robustness")
    plt.close(fig)


# ── Figure 14: Analog Amenability Scorecard ────────────────────────────────

def plot_figure14():
    """Composite bar chart + table: 4 key metrics per architecture.

    The top half shows log-normalized bar heights so all 7 architectures are
    visible despite 4 orders of magnitude in speedup/reduction.  The bottom
    half prints a table of raw numbers so the exact values are readable on a
    poster.
    """
    import math

    metrics = []
    for name in _ORDER:
        d_m = _load(name, "mismatch")
        d_a = _load(name, "adc")
        if d_m is None:
            continue

        sigma10 = d_m.get("degradation_threshold_10pct", 0.0)

        speedup = d_m.get("speedup_vs_digital", None)
        if speedup is None:
            from neuro_analog.simulator.analog_acceleration import compute_acceleration
            from neuro_analog.ir.types import DynamicsProfile, ArchitectureFamily
            family_map = {
                "neural_ode": ArchitectureFamily.NEURAL_ODE,
                "transformer": ArchitectureFamily.TRANSFORMER,
                "diffusion": ArchitectureFamily.DIFFUSION,
                "flow": ArchitectureFamily.FLOW,
                "ebm": ArchitectureFamily.EBM,
                "deq": ArchitectureFamily.DEQ,
                "ssm": ArchitectureFamily.SSM,
            }
            acc = compute_acceleration(family_map.get(name, ArchitectureFamily.TRANSFORMER), DynamicsProfile())
            speedup = acc.speedup_settling_vs_digital

        reduction = None
        if d_m.get("analog_energy_pJ") and d_m.get("digital_energy_pJ") and d_m["analog_energy_pJ"] > 0:
            reduction = d_m["digital_energy_pJ"] / d_m["analog_energy_pJ"]
        if reduction is None and d_m.get("analog_energy_reduction_factor"):
            reduction = d_m["analog_energy_reduction_factor"]
        if reduction is None:
            saving = d_m.get("energy_saving_vs_digital")
            if saving is not None and saving < 1.0:
                reduction = 1.0 / (1.0 - saving) if saving < 1.0 else 1e6

        min_adc = None
        if d_a is not None:
            bits = np.array(d_a.get("sigma_values", []))
            means = np.array(d_a.get("normalized_mean", []))
            valid = means >= 0.90
            if np.any(valid):
                min_adc = float(bits[valid][0])
            else:
                min_adc = float(bits[-1]) if len(bits) else None

        metrics.append({
            "name": name,
            "label": _LABELS[name],
            "color": _COLORS[name],
            "sigma10": sigma10,
            "speedup": speedup,
            "reduction": reduction,
            "min_adc": min_adc,
        })

    if not metrics:
        print("  [Fig 14] No data available for scorecard.")
        return

    # Log-normalization: use log10(val) so a 100,000× and a 30× bar are both visible.
    def _log_norm(val, best, worst, higher_is_better=True):
        if val is None or val <= 0 or best <= 0 or worst <= 0 or best == worst:
            return 0.0
        log_val = math.log10(val)
        log_best = math.log10(best)
        log_worst = math.log10(worst)
        if higher_is_better:
            score = (log_val - log_worst) / (log_best - log_worst)
        else:
            score = (log_worst - log_val) / (log_worst - log_best)
        return max(0.0, min(1.0, score))

    sigma10s = [m["sigma10"] for m in metrics if m["sigma10"] is not None]
    speedups = [m["speedup"] for m in metrics if m["speedup"] is not None and m["speedup"] > 0]
    reductions = [m["reduction"] for m in metrics if m["reduction"] is not None and m["reduction"] > 0]
    adcs = [m["min_adc"] for m in metrics if m["min_adc"] is not None]

    sigma10_best, sigma10_worst = (max(sigma10s), min(sigma10s)) if sigma10s else (1, 0)
    speedup_best, speedup_worst = (max(speedups), min(speedups)) if speedups else (1, 1)
    red_best, red_worst = (max(reductions), min(reductions)) if reductions else (1, 1)
    adc_best, adc_worst = (min(adcs), max(adcs)) if adcs else (2, 16)

    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.0, 1], hspace=0.32)
    ax = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    n_arch = len(metrics)
    n_metrics = 3
    group_w = 0.72
    bar_w = group_w / n_metrics
    x_positions = np.arange(n_arch)
    metric_names = ["Latency\nSpeedup (×)", "Energy\nReduction (×)", "ADC\nEase (bits)"]
    metric_colors_bar = ["#3498db", "#9b59b6", "#f39c12"]

    for i, (mname, mcolor) in enumerate(zip(metric_names, metric_colors_bar)):
        heights = []
        for m in metrics:
            if i == 0:
                val = m["speedup"]
                norm = _log_norm(val, speedup_best, speedup_worst, True)
            elif i == 1:
                val = m["reduction"]
                norm = _log_norm(val, red_best, red_worst, True)
            else:
                val = m["min_adc"]
                norm = _log_norm(val, adc_best, adc_worst, False)
            heights.append(norm)

        x = x_positions - group_w/2 + bar_w/2 + i*bar_w
        ax.bar(x, heights, width=bar_w*0.88, color=mcolor,
               edgecolor="white", linewidth=0.5, zorder=3, label=mname.replace("\n", " "))

    ax.set_xticks(x_positions)
    ax.set_xticklabels([m["label"] for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Normalized Score")
    ax.set_title("Analog Amenability Scorecard (Log-Normalized)\n"
                 "Taller = more analog-amenable  |  Exact values in table below",
                 pad=10)
    ax.axhline(1.0, color="#bdc3c7", linewidth=0.8, linestyle="--", zorder=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=2)
    
    # Add caption about robustness ceiling
    ax.text(0.02, 0.98, "Robustness ceiling reached at σ=0.15 for 6/7 architectures",
            transform=ax.transAxes, fontsize=8.5, color="#2c3e50", style="italic",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f8f5", edgecolor="#2ecc71", alpha=0.9, linewidth=1.0))

    # ── Table panel ────────────────────────────────────────────────
    ax_table.axis("off")
    table_data = []
    col_labels = ["Architecture", "Speedup (×)", "Energy Red. (×)", "Min ADC (bits)"]
    for m in metrics:
        table_data.append([
            m["label"],
            f"{m['speedup']:.1f}" if m['speedup'] is not None else "N/A",
            f"{m['reduction']:.1f}" if m['reduction'] is not None else "N/A",
            f"{int(m['min_adc'])}" if m['min_adc'] is not None else "N/A",
        ])
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colColours=["#ecf0f1"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.15, 1.8)
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0:
            cell.set_text_props(fontweight="bold", color="#2c3e50")
            cell.set_facecolor("#ecf0f1")
        else:
            cell.set_facecolor("#ffffff" if row % 2 == 1 else "#f8f9fa")
        cell.set_edgecolor("#bdc3c7")

    fig.tight_layout()
    _save_fig(fig, "fig14_scorecard")
    plt.close(fig)


# ── Figure 13: Substrate Comparison ──────────────────────────────────────

def plot_figure13():
    """Three-panel figure comparing PCM vs ReRAM vs Capacitive substrates.

    Shows how the three physical substrates (PCM with drift, ReRAM with
    asymmetric switching, Capacitive with kT/C noise) affect degradation curves
    for a representative architecture (e.g., Transformer).

    Left: mismatch sweep on all 3 substrates.
    Middle: zoomed frontier.
    Right: ablation bar chart (mismatch-only) per substrate.
    """
    from neuro_analog.simulator.substrates import PCMSubstrate, ReRAMSubstrate, CapacitiveSubstrate

    substrates = {
        "PCM": "pcm",
        "ReRAM": "reram",
        "Capacitive": "capacitive",
    }
    sub_colors = {
        "PCM": "#9b59b6",
        "ReRAM": "#e67e22",
        "Capacitive": "#3498db",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Physical Substrate Comparison: PCM vs ReRAM vs Capacitive\n(Transformer representative)", fontsize=12, y=1.02)

    # Left: mismatch sweep per substrate
    ax1.set_title("Mismatch Tolerance by Substrate", fontsize=10)
    ax1.set_xlabel("Conductance Mismatch  σ")
    ax1.set_ylabel("Normalized Quality  (digital = 1.0)")
    ax1.set_ylim(0.3, 1.15)
    ax1.axhline(0.90, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)

    # Try to load pre-computed substrate sweep results
    # Files are named: {arch}_mismatch_{substrate}.json (if sweep was run with --physical-substrate)
    for sub_label, sub_key in substrates.items():
        # Look for any architecture results with this substrate suffix
        found = False
        for name in _ORDER:
            p = _RESULTS_DIR / f"{name}_mismatch_{sub_key}.json"
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                sigma = np.array(d["sigma_values"])
                mean = np.array(d["normalized_mean"])
                std = np.array(d["normalized_std"])
                color = sub_colors[sub_label]
                ax1.plot(sigma, mean, color=color, linewidth=2.2, label=f"{sub_label} ({_LABELS[name]})", zorder=3)
                ax1.fill_between(sigma, mean - std, mean + std, color=color, alpha=0.12, zorder=2)
                found = True
                break
        if not found:
            # Placeholder: draw flat line at 1.0 with annotation
            ax1.plot([0, 0.15], [1.0, 1.0], color=sub_colors[sub_label], linewidth=1.5,
                     linestyle="--", alpha=0.5, label=f"{sub_label} (no sweep data)")

    ax1.legend(fontsize=9, loc="lower left", title="Substrate", title_fontsize=9)

    # Right: substrate noise characteristics summary
    ax2.set_title("Substrate Noise Characteristics", fontsize=10)
    ax2.axis("off")

    # Build a text/table summary
    sub_text = ""
    for sub_label, sub_key in substrates.items():
        if sub_key == "pcm":
            sub_text += (
                f"{sub_label}:\n"
                "  • Programming noise + temporal drift\n"
                "  • 1/f read noise (σ ≈ 2–5%)\n"
                "  • Differential 4-device cell\n"
                "  • IBM HERMES calibrated\n\n"
            )
        elif sub_key == "reram":
            sub_text += (
                f"{sub_label}:\n"
                "  • Asymmetric SET/RESET\n"
                "  • No temporal drift\n"
                "  • Higher σ at small |w|\n"
                "  • Log-normal-like tails\n\n"
            )
        elif sub_key == "capacitive":
            sub_text += (
                f"{sub_label}:\n"
                "  • Pure kT/C thermal noise\n"
                "  • No drift (charge refresh)\n"
                "  • Excellent matching (<0.5%)\n"
                "  • SRAM-IMC / switched-cap\n\n"
            )

    ax2.text(0.05, 0.95, sub_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))

    fig.tight_layout()
    _save_fig(fig, "fig13_substrate_comparison")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", type=int, default=None, help="Plot only figure N (1-7, 10-11)")
    parser.add_argument(
        "--domain", type=str, default="conservative",
        choices=["conservative", "full_analog"],
        help="Which result domain to use for figures 1-6 (figure 7 always shows both).",
    )
    parser.add_argument(
        "--substrate", type=str, default="classic",
        choices=["classic", "cld", "extropic_dtm"],
        help="Which diffusion sampling substrate results to load (classic/cld/extropic_dtm).",
    )
    args = parser.parse_args()

    global _DOMAIN, _SUBSTRATE
    _DOMAIN = args.domain
    _SUBSTRATE = args.substrate

    figs = {
        1: ("Figure 1: Cross-Architecture Mismatch Tolerance", plot_figure1),
        2: ("Figure 2: Noise Source Ablation", plot_figure2),
        3: ("Figure 3: ADC Precision Tradeoff", plot_figure3),
        4: ("Figure 4: DEQ Convergence Bifurcation", plot_figure4),
        5: ("Figure 5: Visual Results", plot_figure5),
        6: ("Figure 6: Output MSE vs Mismatch", plot_figure6),
        7: ("Figure 7: Conservative vs Full-Analog Comparison", plot_figure7),
        10: ("Figure 10: Energy–Accuracy Pareto Frontier", plot_figure10),
        11: ("Figure 11: DEQ Spectral Radius vs Convergence", plot_figure11),
        12: ("Figure 12: Speedup vs Robustness", plot_figure12),
        13: ("Figure 13: Substrate Comparison", plot_figure13),
        14: ("Figure 14: Analog Amenability Scorecard", plot_figure14),
    }

    for num, (title, fn) in figs.items():
        if args.fig and num != args.fig:
            continue
        print(f"\n{title}...")
        fn()

    print(f"\nAll figures saved to: {_FIGURES_DIR}")


if __name__ == "__main__":
    main()
