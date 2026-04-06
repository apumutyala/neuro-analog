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
    "ssm":         "SSM (S4D)",
    "ebm":         "EBM (RBM)",
    "flow":        "Flow (Rectified)",
    "deq":         "DEQ (Implicit)",
    "transformer": "Transformer",
    "diffusion":   "Diffusion (DDPM)",
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
    """Load a result JSON for the given domain and substrate (defaults to module-level globals)."""
    d = domain if domain is not None else _DOMAIN
    s = substrate if substrate is not None else _SUBSTRATE
    domain_suffix = "" if d == "conservative" else f"_{d}"
    substrate_suffix = "" if s == "classic" else f"_{s}"
    p = _RESULTS_DIR / f"{name}_{suffix}{domain_suffix}{substrate_suffix}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


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
        (ax_zoom, (0.87, 1.06), "Frontier zoom  (0.87\u20131.06)"),
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
    fig.suptitle(f"Noise Source Ablation at {_ablation_sigma_label}\n(which nonideality dominates per architecture at high noise?)", y=1.02)

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
    ax.text(16.2, 0.905, "90%", color="#95a5a6", fontsize=8)
    ax.set_ylim(0.3, 1.08)

    _YFLOOR = 0.3
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
                ax.annotate(f"{mv:.2f}", xy=(bw, _YFLOOR), xytext=(bw, _YFLOOR + 0.07),
                            ha="center", fontsize=7.5, color=color,
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

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
    fig.suptitle("Visual Quality Degradation Under Analog Mismatch", fontsize=13, y=1.01)

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
        mean = -np.array(d["mean"])  # Negate back to positive MSE
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
    ax.text(0.99, 0.02,
            "Diffusion not shown: MSE ≈ 0.69 (off-chart)\n"
            "ADC quantization floor dominates, flat at all σ",
            transform=ax.transAxes, fontsize=7.5, color="#95a5a6",
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
        (_RESULTS_DIR / f"{name}_mismatch_full_analog.json").exists()
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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", type=int, default=None, help="Plot only figure N (1-7)")
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
    }

    for num, (title, fn) in figs.items():
        if args.fig and num != args.fig:
            continue
        print(f"\n{title}...")
        fn()

    print(f"\nAll figures saved to: {_FIGURES_DIR}")


if __name__ == "__main__":
    main()
