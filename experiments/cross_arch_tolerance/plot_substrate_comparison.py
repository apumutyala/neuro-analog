#!/usr/bin/env python3
"""Figure 9: Cross-substrate mismatch tolerance comparison.

For each substrate-aware architecture, overlays mismatch degradation curves
across all analog integration substrates tested. Shows how substrate choice
changes noise resilience — the most novel cross-architecture claim in this study.

Substrate-aware architectures and their substrates:
    neural_ode: euler (noiseless integrator) | rc_integrator (Johnson-Nyquist)
    flow:       euler                        | rc_integrator
    deq:        discrete (fixed-point iter)  | hopfield (damped ODE relaxation)
    diffusion:  classic (DDIM)              | cld (RLC/Langevin)

Non-substrate-aware (transformer, ebm, ssm) are shown for reference with
their single result labeled as "standard".

Uses existing results JSONs — no new sweep needed.

Output:
    figures/fig9_substrate_comparison.{png,pdf}

Usage:
    python experiments/cross_arch_tolerance/plot_substrate_comparison.py
    python experiments/cross_arch_tolerance/plot_substrate_comparison.py --domain full_analog
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT = Path(__file__).parent.parent.parent
_EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

_RESULTS_DIR = _EXP_DIR / "results"
_FIGURES_DIR = _EXP_DIR / "figures"
_FIGURES_DIR.mkdir(exist_ok=True)

# Substrate-aware architectures with their substrates (default first)
_SUBSTRATE_CONFIGS = {
    "neural_ode": {
        "substrates": ["euler", "rc_integrator"],
        "labels":     ["Euler (noiseless integrator)", "RC integrator (Johnson-Nyquist)"],
        "styles":     ["-",  "--"],
        "file_suffixes": ["", "_rc_integrator"],  # "" = default (euler)
    },
    "flow": {
        "substrates": ["euler", "rc_integrator"],
        "labels":     ["Euler (noiseless integrator)", "RC integrator (Johnson-Nyquist)"],
        "styles":     ["-",  "--"],
        "file_suffixes": ["", "_rc_integrator"],
    },
    "deq": {
        "substrates": ["discrete", "hopfield"],
        "labels":     ["Discrete (fixed-point iter.)", "Hopfield (damped ODE)"],
        "styles":     ["-",  "--"],
        "file_suffixes": ["", "_hopfield"],
    },
    "diffusion": {
        "substrates": ["classic", "cld"],
        "labels":     ["Classic DDIM", "CLD (RLC/Langevin)"],
        "styles":     ["-",  "--"],
        "file_suffixes": ["", "_cld"],
    },
}

# Non-substrate-aware (single result each)
_SINGLE_SUBSTRATE = {
    "transformer": "standard",
    "ebm":         "standard",
    "ssm":         "standard",
}

_ARCH_COLORS = {
    "neural_ode":  "#2ecc71",
    "ssm":         "#3498db",
    "ebm":         "#9b59b6",
    "flow":        "#1abc9c",
    "deq":         "#e67e22",
    "transformer": "#e74c3c",
    "diffusion":   "#95a5a6",
}
_ARCH_LABELS = {
    "neural_ode":  "Neural ODE",
    "ssm":         "SSM (S4D)",
    "ebm":         "EBM (RBM)",
    "flow":        "Flow",
    "deq":         "DEQ",
    "transformer": "Transformer",
    "diffusion":   "Diffusion",
}


def _load_mismatch(name: str, domain: str, substrate_suffix: str) -> dict | None:
    """Load mismatch JSON for given arch, domain, and substrate file suffix."""
    domain_suffix = "" if domain == "conservative" else f"_{domain}"
    # Diffusion and Flow: use ablation_mismatch for cleaner baseline
    src = "ablation_mismatch" if name in ("diffusion", "flow") else "mismatch"
    p = _RESULTS_DIR / f"{name}_{src}{domain_suffix}{substrate_suffix}.json"
    if not p.exists():
        # Try plain mismatch as fallback
        p2 = _RESULTS_DIR / f"{name}_mismatch{domain_suffix}{substrate_suffix}.json"
        if not p2.exists():
            return None
        p = p2
    with open(p) as f:
        return json.load(f)


def plot_substrate_comparison(domain: str = "conservative"):
    """Generate Figure 9: substrate comparison across all architectures."""

    # Layout: 2 rows × 4 cols
    # Row 1: substrate-aware arches (neural_ode, flow, deq, diffusion)
    # Row 2: reference arches (transformer, ebm, ssm) + legend panel
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Analog Substrate × Architecture Mismatch Tolerance\n"
        f"Domain: {domain}  |  Normalized quality vs conductance mismatch σ",
        fontsize=12, fontweight="bold",
    )

    substrate_aware_order = ["neural_ode", "flow", "deq", "diffusion"]
    reference_order = ["transformer", "ebm", "ssm"]

    def _plot_arch(ax, name, substrate_configs=None):
        color = _ARCH_COLORS[name]
        ax.set_title(_ARCH_LABELS[name], fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("Mismatch σ", fontsize=9)
        ax.set_ylabel("Normalized Quality", fontsize=9)
        ax.axhline(0.90, color="#bdc3c7", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xlim(0, 0.155)
        ax.set_ylim(0.3, 1.10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if substrate_configs is None:
            # Single substrate
            d = _load_mismatch(name, domain, "")
            if d is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="#aaa")
                return []
            sigma = np.array(d["sigma_values"])
            mean = np.array(d["normalized_mean"])
            std = np.array(d["normalized_std"])
            ax.plot(sigma, mean, color=color, linewidth=2.0, linestyle="-", label="standard")
            ax.fill_between(sigma, mean - std, mean + std, color=color, alpha=0.15)
            ax.text(0.97, 0.97, "substrate-agnostic", transform=ax.transAxes,
                    fontsize=7, ha="right", va="top", color="#888", style="italic")
            return [Line2D([0], [0], color=color, linewidth=2, linestyle="-", label="standard")]

        handles = []
        for suffix, label, style in zip(
            substrate_configs["file_suffixes"],
            substrate_configs["labels"],
            substrate_configs["styles"],
        ):
            d = _load_mismatch(name, domain, suffix)
            if d is None:
                print(f"  [{name}] No data for suffix='{suffix}'")
                continue
            sigma = np.array(d["sigma_values"])
            mean = np.array(d["normalized_mean"])
            std = np.array(d["normalized_std"])
            ax.plot(sigma, mean, color=color, linewidth=2.0, linestyle=style, label=label)
            ax.fill_between(sigma, mean - std, mean + std, color=color, alpha=0.10)
            handles.append(Line2D([0], [0], color=color, linewidth=2, linestyle=style, label=label))

        if handles:
            ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.8)
        return handles

    # Row 0: substrate-aware architectures
    for col, name in enumerate(substrate_aware_order):
        ax = axes[0, col]
        _plot_arch(ax, name, _SUBSTRATE_CONFIGS[name])

    # Row 1: reference architectures
    for col, name in enumerate(reference_order):
        ax = axes[1, col]
        _plot_arch(ax, name, None)

    # Last cell (row 1, col 3): summary panel
    ax_summary = axes[1, 3]
    ax_summary.axis("off")
    ax_summary.set_title("Substrate Key", fontsize=10, fontweight="bold")

    summary_lines = [
        ("Neural ODE / Flow", "━━  Euler (noiseless)\n- -  RC integrator (+thermal)"),
        ("DEQ",               "━━  Discrete (fixed-point)\n- -  Hopfield (ODE relax.)"),
        ("Diffusion",         "━━  Classic DDIM\n- -  CLD (RLC/Langevin)"),
        ("Transformer/EBM/SSM", "━━  Substrate-agnostic"),
    ]
    y = 0.90
    for arch, desc in summary_lines:
        ax_summary.text(0.05, y, arch, fontsize=9, fontweight="bold",
                        transform=ax_summary.transAxes)
        ax_summary.text(0.05, y - 0.08, desc, fontsize=8, color="#555",
                        transform=ax_summary.transAxes, family="monospace")
        y -= 0.22

    ax_summary.text(0.05, 0.05,
                    "Dashed line = physically richer substrate\n(adds thermal noise from hardware)",
                    fontsize=7, color="#888", transform=ax_summary.transAxes, style="italic")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = _FIGURES_DIR / f"fig9_substrate_comparison.{ext}"
        fig.savefig(str(p), dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)
    print("Figure 9 done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["conservative", "full_analog"],
                        default="conservative")
    args = parser.parse_args()
    plot_substrate_comparison(args.domain)


if __name__ == "__main__":
    main()
