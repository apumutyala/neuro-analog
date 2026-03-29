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
    """THE central figure: 7 quality-vs-sigma curves."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xlabel("Conductance Mismatch  σ")
    ax.set_ylabel("Normalized Quality  (digital = 1.0)")
    ax.set_title("Cross-Architecture Analog Tolerance\nQuality under fabrication mismatch (50 Monte Carlo trials per point)", pad=12)
    ax.axhline(0.90, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)
    ax.text(0.155, 0.905, "90% threshold", color="#95a5a6", fontsize=9, va="bottom", ha="right")
    ax.set_xlim(0, 0.155)
    ax.set_ylim(0.3, 1.08)

    handles = []
    threshold_annotations = []

    # Diffusion and Flow use ablation_mismatch: their mismatch.json baselines are
    # unreliable (Diffusion: conservative ADC floor contaminates σ=0 baseline;
    # Flow: z0 sampling seed inflates σ=0 to 1.38). ablation_mismatch uses an
    # internally consistent baseline and correctly reflects mismatch sensitivity.
    _mismatch_source = {name: "mismatch" for name in _ORDER}
    _mismatch_source["diffusion"] = "ablation_mismatch"
    _mismatch_source["flow"] = "ablation_mismatch"

    for name in _ORDER:
        d = _load(name, _mismatch_source[name])
        if d is None:
            continue

        sigma = np.array(d["sigma_values"])
        mean = np.array(d["normalized_mean"])
        std = np.array(d["normalized_std"])
        color = _COLORS[name]
        label = _LABELS[name]
        threshold = d.get("degradation_threshold_10pct", None)

        ax.plot(sigma, mean, color=color, linewidth=2.2, label=label, zorder=3)
        ax.fill_between(sigma, mean - std, mean + std, color=color, alpha=0.12, zorder=2)

        # Mark threshold crossing: annotate the FIRST FAILURE sigma (quality < 0.9),
        # which is one step beyond the stored "last safe" degradation_threshold_10pct.
        if threshold and threshold > 0:
            last_safe_idx = np.searchsorted(sigma, threshold)
            first_fail_idx = last_safe_idx + 1
            if first_fail_idx < len(sigma):
                first_fail_sigma = sigma[first_fail_idx]
            else:
                first_fail_sigma = threshold  # already at last point
            ax.axvline(first_fail_sigma, color=color, linewidth=0.6, linestyle=":", alpha=0.5)
            threshold_annotations.append((first_fail_sigma, 0.305, color, f"{first_fail_sigma:.2f}"))

        handles.append(Line2D([0], [0], color=color, linewidth=2.2, label=f"{label}"))

    # Threshold markers on x-axis
    for x, y, c, txt in sorted(threshold_annotations):
        ax.text(x, y, txt, color=c, fontsize=7.5, ha="center", rotation=90)

    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9,
              title="Architecture", title_fontsize=9)
    ax.set_xticks([0.0, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15])
    ax.set_xticklabels(["0%", "3%", "5%", "7%", "10%", "12%", "15%"])

    fig.tight_layout()
    _save_fig(fig, "fig1_mismatch_tolerance")
    plt.close(fig)


# ── Figure 2: Noise Source Ablation ──────────────────────────────────────

def plot_figure2():
    """7-panel ablation: mismatch-only, thermal-only, quantization-only per architecture."""
    n_arch = len(_ORDER)
    fig, axes = plt.subplots(1, n_arch, figsize=(16, 4), sharey=True)
    fig.suptitle("Noise Source Ablation at σ = 0.05\n(which nonideality dominates per architecture?)", y=1.02)

    sigma_idx = 4  # sigma=0.05 is index 4 in default sigma_values

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
            mean_at_sigma = np.array(d["mean"])[sigma_idx]
            norm_q = 1.0 + (mean_at_sigma - baseline) / abs(baseline) if baseline != 0 else 0.5
            ax.bar(i, norm_q, color=colors_bar[i], alpha=0.8, width=0.6)
            ax.text(i, norm_q + 0.01, f"{norm_q:.2f}", ha="center", fontsize=7.5)

        ax.axhline(1.0, color="#bdc3c7", linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 1.15)

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

    for name in _ORDER:
        d = _load(name, "adc")
        if d is None:
            continue
        bits = d["sigma_values"]   # reused field holds bit values
        mean = np.array(d["normalized_mean"])
        std = np.array(d["normalized_std"])
        color = _COLORS[name]
        ax.plot(bits, mean, color=color, linewidth=2.0, marker="o", markersize=4, label=_LABELS[name])
        ax.fill_between(bits, mean - std, mean + std, color=color, alpha=0.10)

    ax.set_xticks([2, 4, 6, 8, 10, 12, 16])
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, "fig3_adc_precision")
    plt.close(fig)


# ── Figure 4: DEQ Convergence Under Mismatch ─────────────────────────────

def plot_figure4():
    """DEQ: accuracy AND convergence failure rate on dual y-axes."""
    d_acc = _load("deq", "mismatch")
    conv_path = _RESULTS_DIR / "deq_convergence.json"

    if d_acc is None:
        print("  [Fig 4] No DEQ mismatch data. Skipping.")
        return

    sigma = np.array(d_acc["sigma_values"])
    acc = np.array(d_acc["normalized_mean"])
    acc_std = np.array(d_acc["normalized_std"])

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    ax1.set_xlabel("Conductance Mismatch  σ")
    ax1.set_ylabel("Classification Accuracy (normalized)", color="#e67e22")
    ax2.set_ylabel("Convergence Failure Rate", color="#c0392b")
    ax1.set_title("DEQ Convergence Bifurcation Under Mismatch\nFailure rate = 100% at all σ — metric artifact, not true divergence (see §3.4)", pad=10)

    l1 = ax1.plot(sigma, acc, color="#e67e22", linewidth=2.2, label="Accuracy", zorder=3)
    ax1.fill_between(sigma, acc - acc_std, acc + acc_std, color="#e67e22", alpha=0.15)
    ax1.axhline(0.90, color="#bdc3c7", linewidth=1.0, linestyle="--")
    ax1.tick_params(axis="y", labelcolor="#e67e22")
    ax1.set_ylim(0, 1.1)

    if conv_path.exists():
        with open(conv_path) as f:
            conv_data = json.load(f)
        fail_rates = np.array(conv_data["convergence_failure_rate"])
        l2 = ax2.plot(sigma, fail_rates, color="#c0392b", linewidth=2.0,
                      linestyle="--", marker="s", markersize=4, label="Failure Rate")
        ax2.tick_params(axis="y", labelcolor="#c0392b")
        ax2.set_ylim(0, 1.0)
        ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

        # Annotate bifurcation point (where failure rate first exceeds 5%)
        for i, r in enumerate(fail_rates):
            if r > 0.05:
                ax1.axvline(sigma[i], color="#c0392b", linewidth=1.0, linestyle=":", alpha=0.6)
                ax1.text(sigma[i] + 0.003, 0.05, f"ρ > 1\n(σ={sigma[i]:.2f})",
                         color="#c0392b", fontsize=8)
                break

    ax1.set_xticks([0.0, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15])
    ax1.set_xticklabels(["0%", "3%", "5%", "7%", "10%", "12%", "15%"])
    lines = [l for l in ax1.get_lines() + ax2.get_lines() if not l.get_label().startswith("_")]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=9)

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
                    z0 = torch.randn(200, 2)
                    t_span = torch.tensor([0.0, 1.0])
                    model.eval()
                    x_gen = analog_odeint(model, z0, t_span, dt=0.25).detach().cpu().numpy()
                    ax.scatter(x_gen[:, 0], x_gen[:, 1], s=3, alpha=0.6,
                               c=_COLORS[arch_name], rasterized=True)
                    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
                    ax.axis("on")
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.5)

                elif arch_name == "diffusion":
                    # Generate one 8x8 image
                    import torch
                    from models.diffusion import _get_betas, _get_alphas, _IMG_DIM
                    import math as _math
                    betas = _get_betas()
                    _, alphas_bar = _get_alphas(betas)
                    x = torch.randn(1, _IMG_DIM)
                    n_ddim = 10
                    T = 100
                    ddim_steps = torch.linspace(T-1, 0, n_ddim+1).long()
                    model.eval()
                    with torch.no_grad():
                        for di in range(n_ddim):
                            t_c = ddim_steps[di].item()
                            t_n = ddim_steps[di+1].item()
                            t_t = torch.full((1,), t_c, dtype=torch.long)
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
                    v = X_test[:1].clone()
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
    ax.set_title("Output Corruption Under Analog Mismatch\nMSE between digital and analog outputs (50 Monte Carlo trials)", pad=12)
    ax.set_xlim(0, 0.155)
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
        color = _COLORS[name]
        label = _LABELS[name]

        ax.plot(sigma, mean, color=color, linewidth=2.2, label=label, zorder=3)
        ax.fill_between(sigma, np.maximum(mean - std, 1e-10), mean + std, color=color, alpha=0.12, zorder=2)

        handles.append(Line2D([0], [0], color=color, linewidth=2.2, label=f"{label}"))

    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9,
              title="Architecture", title_fontsize=9)
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
            ax.set_xticks([0.0, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15])
            ax.set_xticklabels(["0%", "3%", "5%", "7%", "10%", "12%", "15%"])
        else:
            ax.set_xlabel("ADC Bit Width")
            ax.set_xticks([2, 4, 6, 8, 10, 12, 16])
        ax.set_ylabel("Normalized Quality  (digital = 1.0)")
        ax.axhline(0.90, color="#bdc3c7", linewidth=1.0, linestyle="--", zorder=0)
        ax.set_ylim(0.3, 1.15)

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
                ax.plot(x, m, color=color, linewidth=2.0, linestyle="-",
                        label=f"{label} (conservative)")

            if d_full is not None:
                x = np.array(d_full[x_axis])
                m = np.array(d_full["normalized_mean"])
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
