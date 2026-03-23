#!/usr/bin/env python3
"""
Seven-architecture cross-architecture analog tolerance study.

Loads sweep results from experiments/cross_arch_tolerance/ and prints
a ranked summary of analog tolerance across all 7 neural network families.

Architecture families (ordered by analog amenability):
    1. Neural ODE   — IS an ODE; direct Ark (Arco) input format
    2. SSM/Mamba    — diagonal A → independent RC integrators
    3. EBM          — Boltzmann sampling → p-bit / sMTJ arrays
    4. Flow         — clean ODE but large vector field (MMDiT)
    5. Transformer  — MVM-heavy, no dynamics, softmax bottleneck
    6. Diffusion    — high D/A boundary count per denoising step
    7. DEQ          — implicit fixed-point; circuit settles naturally

Substrate options per architecture:
    neural_ode:  "euler" (default) | "rc_integrator"
    flow:        "euler" (default) | "rc_integrator"
    diffusion:   "classic" (default) | "cld" | "extropic_dtm"
    deq:         "discrete" (default) | "hopfield"

Usage:
    # One-time: train all 7 models (~20 min)
    python experiments/cross_arch_tolerance/train_all.py

    # Full sweep (~1-2 hours, 50 trials per architecture)
    python experiments/cross_arch_tolerance/sweep_all.py

    # Faster development sweep (5 trials)
    python experiments/cross_arch_tolerance/sweep_all.py --n-trials 5

    # Single architecture
    python experiments/cross_arch_tolerance/sweep_all.py --only neural_ode

    # Alternate substrate
    python experiments/cross_arch_tolerance/sweep_all.py --only diffusion --analog-substrate cld

    # Generate figures
    python experiments/cross_arch_tolerance/plot_results.py

    # Summarize existing results (this script)
    python examples/02_seven_arch_sweep.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_EXP_DIR = _ROOT / "experiments" / "cross_arch_tolerance"
_RESULTS_DIR = _EXP_DIR / "results"

sys.path.insert(0, str(_ROOT))

# Analog amenability order: most → least analog-native
_ARCH_ORDER = [
    ("neural_ode",  "Neural ODE",  "dx/dt = f_θ(x,t) — IS Ark's input format"),
    ("ssm",         "SSM/Mamba",   "diagonal A → independent RC integrators"),
    ("ebm",         "EBM",         "Boltzmann sampling → p-bit / sMTJ arrays"),
    ("flow",        "Flow",        "clean ODE, large v_θ vector field"),
    ("transformer", "Transformer", "MVM-heavy, no dynamics, softmax bottleneck"),
    ("diffusion",   "Diffusion",   "high D/A boundary count per denoising step"),
    ("deq",         "DEQ",         "implicit fixed-point; circuit settles naturally"),
]


def load_result(name: str) -> dict | None:
    path = _RESULTS_DIR / f"{name}_mismatch.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def degradation_threshold(result: dict, max_relative_loss: float = 0.05) -> float:
    """Return largest σ where quality stays within max_relative_loss of digital baseline."""
    baseline = result["digital_baseline"]
    threshold = 1.0 - max_relative_loss
    last_passing = 0.0
    for sigma, mean in zip(result["sigma_values"], result["mean"]):
        if baseline != 0:
            retained = 1.0 + (mean - baseline) / abs(baseline)
            if retained >= threshold:
                last_passing = sigma
    return last_passing


def print_arch_table(label: str, result: dict) -> None:
    baseline = result["digital_baseline"]
    metric = result.get("metric_name", "quality")
    print(f"\n  {label}  [{metric}]")
    print(f"  {'σ':>6}  {'mean':>8}  {'±':>6}  {'retained':>9}")
    for sigma, mean, std in zip(result["sigma_values"], result["mean"], result["std"]):
        retained = 1.0 + (mean - baseline) / abs(baseline) if baseline != 0 else float("nan")
        print(f"  {sigma:6.2f}  {mean:8.4f}  {std:5.4f}  {retained:8.1%}")


def main():
    print("neuro-analog: 7-architecture cross-architecture sweep\n")

    available, missing = [], []
    for name, label, _ in _ARCH_ORDER:
        if load_result(name) is not None:
            available.append(name)
        else:
            missing.append(name)

    if missing:
        print("Missing sweep results for:", ", ".join(missing))
        print()
        print("Run:")
        print("  python experiments/cross_arch_tolerance/train_all.py")
        print("  python experiments/cross_arch_tolerance/sweep_all.py")
        if not available:
            return

    print(f"Results available for: {', '.join(available)}\n")

    # Per-architecture detail tables
    for name, label, _ in _ARCH_ORDER:
        result = load_result(name)
        if result is not None:
            print_arch_table(label, result)

    # Ranking by 5% degradation threshold
    rankings = []
    for name, label, description in _ARCH_ORDER:
        result = load_result(name)
        if result is None:
            continue
        threshold = degradation_threshold(result, max_relative_loss=0.05)
        rankings.append((label, threshold, description))

    rankings.sort(key=lambda x: -x[1])

    print("\n" + "=" * 65)
    print("  ANALOG TOLERANCE RANKING (σ threshold for 5% quality loss)")
    print("=" * 65)
    for i, (label, threshold, description) in enumerate(rankings, 1):
        bar = "█" * int(threshold * 100)
        print(f"  {i}. {label:<14}  σ≥{threshold:.2f}  {bar:<12}  {description}")

    print()
    print("To regenerate results:")
    print("  python experiments/cross_arch_tolerance/sweep_all.py")
    print()
    print("To plot figures:")
    print("  python experiments/cross_arch_tolerance/plot_results.py")


if __name__ == "__main__":
    main()
