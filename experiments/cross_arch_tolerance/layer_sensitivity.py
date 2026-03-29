#!/usr/bin/env python3
"""Per-layer mismatch sensitivity analysis across all 7 architectures.

For each architecture, applies mismatch (sigma=0.05) to ONE analog layer at
a time while keeping all other layers noiseless. Measures the resulting
performance degradation to identify which specific layers drive analog failure.

This answers: WHERE in each architecture does mismatch matter most?

Complements sweep_all.py (which measures total mismatch tolerance) by
attributing degradation to specific layers rather than the full model.

Output:
    results/{arch}_layer_sensitivity.json   — per-layer degradation scores
    figures/fig8_layer_sensitivity.{png,pdf} — bar charts per architecture

Usage:
    python experiments/cross_arch_tolerance/layer_sensitivity.py
    python experiments/cross_arch_tolerance/layer_sensitivity.py --only neural_ode
    python experiments/cross_arch_tolerance/layer_sensitivity.py --sigma 0.07
    python experiments/cross_arch_tolerance/layer_sensitivity.py --n-trials 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent.parent
_EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EXP_DIR))

_CKPT_DIR = _EXP_DIR / "checkpoints"
_RESULTS_DIR = _EXP_DIR / "results"
_FIGURES_DIR = _EXP_DIR / "figures"
_RESULTS_DIR.mkdir(exist_ok=True)
_FIGURES_DIR.mkdir(exist_ok=True)

import torch
from models import neural_ode, transformer, diffusion, flow, ebm, deq, ssm
from neuro_analog.simulator import (
    analogize, resample_all_mismatch, set_all_noise,
    calibrate_analog_model, configure_analog_profile, AnalogLinear,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODELS = [
    ("neural_ode",  neural_ode),
    ("transformer", transformer),
    ("diffusion",   diffusion),
    ("flow",        flow),
    ("ebm",         ebm),
    ("deq",         deq),
    ("ssm",         ssm),
]

_SUBSTRATE_AWARE = {"diffusion", "neural_ode", "flow", "deq"}
_GAUSSIAN_INPUT_MODELS = {"neural_ode", "diffusion", "flow"}

_COLORS = {
    "neural_ode":  "#2ecc71",
    "ssm":         "#3498db",
    "ebm":         "#9b59b6",
    "flow":        "#1abc9c",
    "deq":         "#e67e22",
    "transformer": "#e74c3c",
    "diffusion":   "#95a5a6",
}
_LABELS = {
    "neural_ode":  "Neural ODE",
    "ssm":         "SSM (S4D)",
    "ebm":         "EBM (RBM)",
    "flow":        "Flow",
    "deq":         "DEQ",
    "transformer": "Transformer",
    "diffusion":   "Diffusion",
}


def _get_analog_layers(model):
    """Return (name, module) for all analog layers that support resample_mismatch."""
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if hasattr(mod, "resample_mismatch") and name != ""
    ]


def run_layer_sensitivity(name, module, sigma=0.05, n_trials=20, force=False):
    ckpt_path = str(_CKPT_DIR / f"{name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[{name}] No checkpoint. Run train_all.py first.")
        return None

    out_path = _RESULTS_DIR / f"{name}_layer_sensitivity.json"
    if out_path.exists() and not force:
        print(f"[{name}] Layer sensitivity exists, skipping. (--force to re-run)")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n{'='*50}")
    print(f"Layer sensitivity: {name}  sigma={sigma}  n_trials={n_trials}")
    print(f"{'='*50}")

    model = module.load_model(ckpt_path).to(_DEVICE)

    gaussian_kwargs = {"v_ref_input": 3.3} if name in _GAUSSIAN_INPUT_MODELS else {}

    # Analogize at sigma=0 (noiseless structure, correct layer types)
    analog_model = analogize(model, sigma_mismatch=0.0, n_adc_bits=8, **gaussian_kwargs)
    configure_analog_profile(analog_model, "conservative")

    # Calibrate v_ref from real data if possible
    if hasattr(module, "_get_data") and name not in _GAUSSIAN_INPUT_MODELS:
        data = module._get_data()
        if len(data) == 4:
            calib = data[0][:32].to(_DEVICE)
        elif isinstance(data[0], tuple):
            calib = data[0][0][:32].to(_DEVICE)
        else:
            calib = data[0][:32].to(_DEVICE) if isinstance(data[0], torch.Tensor) else data[0][0][:32].to(_DEVICE)
        calibrate_analog_model(analog_model, calib)

    # Build eval_fn
    if name in _SUBSTRATE_AWARE:
        eval_fn = lambda m: module.evaluate(m, analog_substrate="euler" if name in ("neural_ode", "flow") else "discrete" if name == "deq" else "classic")
    else:
        eval_fn = module.evaluate

    # Disable all noise, resample to sigma=0 → clean baseline
    set_all_noise(analog_model, thermal=False, quantization=False, mismatch=True)
    resample_all_mismatch(analog_model, sigma=0.0)
    baseline = eval_fn(analog_model)
    print(f"  Noiseless baseline: {baseline:.4f}")

    # Collect analog layers
    analog_layers = _get_analog_layers(analog_model)
    print(f"  Analog layers ({len(analog_layers)}): {[n for n,_ in analog_layers]}")

    results = {"sigma": sigma, "n_trials": n_trials, "baseline": float(baseline), "layers": {}}

    t0 = time.time()
    for layer_name, layer_mod in analog_layers:
        trial_scores = []
        for _ in range(n_trials):
            # Reset all layers to sigma=0
            resample_all_mismatch(analog_model, sigma=0.0)
            # Apply sigma to this layer only
            layer_mod.resample_mismatch(sigma=sigma)
            score = eval_fn(analog_model)
            trial_scores.append(float(score))

        mean_score = float(np.mean(trial_scores))
        std_score = float(np.std(trial_scores))
        # Normalized: 1.0 = no degradation, <1.0 = degraded
        norm = mean_score / baseline if abs(baseline) > 1e-9 else 1.0
        degradation = 1.0 - norm  # positive = degraded

        layer_type = type(layer_mod).__name__
        results["layers"][layer_name] = {
            "type": layer_type,
            "mean": mean_score,
            "std": std_score,
            "normalized_mean": float(norm),
            "degradation": float(degradation),
        }
        print(f"  {layer_name:40s} ({layer_type:20s})  norm={norm:.4f}  degradation={degradation:+.4f}")

    print(f"  Done in {time.time()-t0:.0f}s")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    return results


def plot_layer_sensitivity(all_results: dict):
    """Figure 8: per-layer sensitivity bar charts for all architectures."""
    order = ["neural_ode", "ssm", "ebm", "flow", "deq", "transformer", "diffusion"]
    present = [name for name in order if name in all_results and all_results[name]]
    n = len(present)
    if n == 0:
        print("  No results to plot.")
        return

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Per-Layer Mismatch Sensitivity  (σ = {list(all_results.values())[0]['sigma']:.2f})\n"
        "Degradation when only that layer has mismatch — all others noiseless",
        fontsize=11, y=1.02,
    )

    for ax, name in zip(axes, present):
        res = all_results[name]
        layers = res["layers"]
        color = _COLORS.get(name, "#7f8c8d")

        names_list = list(layers.keys())
        degradations = [layers[n]["degradation"] for n in names_list]

        # Short display names: strip common prefixes
        short_names = []
        for ln in names_list:
            parts = ln.split(".")
            # Keep last 2 parts for readability
            short = ".".join(parts[-2:]) if len(parts) >= 2 else ln
            short_names.append(short)

        x = np.arange(len(names_list))
        bars = ax.bar(x, degradations, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

        # Color bars: red tint for high degradation
        max_deg = max(abs(d) for d in degradations) if degradations else 1
        for bar, deg in zip(bars, degradations):
            intensity = min(1.0, abs(deg) / max(max_deg, 0.01))
            if deg > 0:
                bar.set_facecolor((0.8 * (1 - intensity) + 0.9 * intensity,
                                   0.2 * (1 - intensity),
                                   0.2 * (1 - intensity), 0.85))
            else:
                bar.set_facecolor((0.2 * (1 - intensity),
                                   0.6 * (1 - intensity) + 0.4,
                                   0.2 * (1 - intensity), 0.85))

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.axhline(0.10, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(_LABELS.get(name, name), fontsize=10, fontweight="bold")
        ax.set_ylabel("Degradation (1 − norm. quality)" if ax == axes[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate baseline
        ax.text(0.98, 0.97, f"baseline={res['baseline']:.3f}",
                transform=ax.transAxes, fontsize=7, ha="right", va="top", color="#555")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = _FIGURES_DIR / f"fig8_layer_sensitivity.{ext}"
        fig.savefig(str(p), dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_results = {}
    for name, module in _MODELS:
        if args.only and name != args.only:
            continue
        res = run_layer_sensitivity(name, module, sigma=args.sigma,
                                    n_trials=args.n_trials, force=args.force)
        if res:
            all_results[name] = res

    print("\nGenerating Figure 8: Layer Sensitivity...")
    plot_layer_sensitivity(all_results)
    print(f"\nDone. Results: {_RESULTS_DIR}")
    print(f"Figure: {_FIGURES_DIR / 'fig8_layer_sensitivity.png'}")


if __name__ == "__main__":
    main()
