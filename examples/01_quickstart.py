#!/usr/bin/env python3
"""
Quick start: analogize a PyTorch model and run a mismatch sweep.

Shows the neuro-analog core API on a tiny MLP — no checkpoints needed.

    analogize()        replace Linear layers with physics-grounded AnalogLinear
    mismatch_sweep()   Monte Carlo over σ ∈ [0, 0.15] (RRAM conductance variance)
    adc_sweep()        quality vs. ADC bit width
    ablation_sweep()   which noise source (mismatch / thermal / quantization) matters

Runtime: ~2 minutes on CPU.

Usage:
    python examples/01_quickstart.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from neuro_analog.simulator import (
    analogize,
    mismatch_sweep,
    adc_sweep,
    ablation_sweep,
    count_analog_vs_digital,
)


# ── Model ─────────────────────────────────────────────────────────────────────

class TinyMLP(nn.Module):
    """3-layer MLP with tanh activations (analog-native)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ── Dataset ───────────────────────────────────────────────────────────────────

def get_data():
    X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    return (
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
        torch.tensor(y_te, dtype=torch.long),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, X_tr, y_tr, epochs=150):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        ce(model(X_tr), y_tr).backward()
        opt.step()


# ── Eval closure (sweep harness calls this with the analog model) ─────────────

def make_eval_fn(X_te, y_te):
    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            pred = model(X_te).argmax(dim=1)
        return float((pred == y_te).float().mean())
    return eval_fn


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("neuro-analog quick start\n")

    # 1. Train
    X_tr, X_te, y_tr, y_te = get_data()
    model = TinyMLP()
    print("Training TinyMLP (150 epochs)...")
    train(model, X_tr, y_tr)
    eval_fn = make_eval_fn(X_te, y_te)
    digital_acc = eval_fn(model)
    print(f"Digital accuracy: {digital_acc*100:.1f}%\n")

    # 2. Inspect analog composition
    analog = analogize(model, sigma_mismatch=0.05, n_adc_bits=8)
    counts = count_analog_vs_digital(analog)
    print("Analog model composition:")
    print(f"  Analog layers:   {counts['analog_layers']}  "
          f"({counts['analog_params']:,} params, {counts['coverage_pct']:.0f}% coverage)")
    print(f"  Digital layers:  {counts['digital_layers']}  "
          f"({counts['digital_params']:,} params)\n")

    # 3. Mismatch sweep (σ = 0.0 → 0.15)
    print("Running mismatch sweep (σ = 0.0 → 0.15, 30 trials)...")
    sweep = mismatch_sweep(
        model, eval_fn,
        sigma_values=[0.0, 0.02, 0.05, 0.07, 0.10, 0.12, 0.15],
        n_trials=30,
        n_adc_bits=8,
    )
    print(f"\n  {'σ':>6}  {'accuracy':>9}  {'±':>6}  {'retained':>9}")
    for i, s in enumerate(sweep.sigma_values):
        print(f"  {s:6.2f}  {sweep.mean[i]*100:8.1f}%  {sweep.std[i]*100:5.1f}%  "
              f"{sweep.normalized_mean[i]:8.1%}")
    threshold = sweep.degradation_threshold(max_relative_loss=0.05)
    print(f"\n  5% degradation threshold: σ ≈ {threshold:.3f}")

    # 4. ADC precision sweep
    print("\nRunning ADC precision sweep (bits = 2 → 12, 20 trials)...")
    adc = adc_sweep(
        model, eval_fn,
        bit_values=[2, 4, 6, 8, 10, 12],
        n_trials=20,
        sigma_mismatch=0.05,
    )
    print(f"\n  {'bits':>6}  {'accuracy':>9}  {'±':>6}  {'retained':>9}")
    for i, b in enumerate(adc.sigma_values):  # sigma_values reused for bit axis
        print(f"  {int(b):6d}  {adc.mean[i]*100:8.1f}%  {adc.std[i]*100:5.1f}%  "
              f"{adc.normalized_mean[i]:8.1%}")

    # 5. Noise attribution ablation
    print("\nRunning noise attribution ablation (30 trials each)...")
    ablation = ablation_sweep(
        model, eval_fn,
        sigma_values=[0.0, 0.05, 0.10],
        n_trials=30,
        n_adc_bits=8,
        sigma_mismatch=0.05,
    )
    print("\n  Noise source attribution at σ=0.10 (degradation from digital baseline):")
    # Show quality retained at the last sigma for each noise source
    for src, result in ablation.items():
        q = result.normalized_mean[-1]  # quality at highest sigma
        bar = "█" * max(0, int((1.0 - q) * 40))
        print(f"    {src:<16}  {bar:<20}  retained={q:.1%}")

    print(f"\nDigital baseline: {digital_acc*100:.1f}%")
    print("Done.")


if __name__ == "__main__":
    main()
