#!/usr/bin/env python3
"""
Deep Equilibrium Model (DEQ) — analog degradation sweep + Ark export.

Demonstrates the gradient-flow ODE bridge:
  1. Load / train DEQ (_DEQClassifier, 8x8 MNIST, z_dim=64)
  2. Analog degradation sweep (mismatch vs accuracy + convergence failure rate)
  3. Ark export: DEQAnalogCkt subclass with real trained weights

ODE form in Ark:
    dz/dt = tanh(W_z @ z + W_x @ x_input + b_x) - z
    State augmented: y = [z, x_input], dx_input/dt = 0
    Settles to z* = f_theta(z*, x_input) at t ~ 5.0

Architecture: W_z (64x64, spectral norm), W_x (64x64 + bias), tanh
Task: 8x8 MNIST 10-class (sklearn digits fallback), z_dim=64

Usage:
    python examples/10_deq_ark.py
    python examples/10_deq_ark.py --sigma 0.05 --n-trials 10

Output:
    outputs/deq_ark.py  -- valid Ark DEQAnalogCkt subclass with trained weights
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_EXP_DIR = _ROOT / "experiments" / "cross_arch_tolerance"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EXP_DIR))

from neuro_analog.simulator import mismatch_sweep
from neuro_analog.extractors.deq import DEQMLPExtractor

_OUT = _ROOT / "outputs"
_CKPT = _EXP_DIR / "checkpoints" / "deq.pt"


def sep(title=""):
    w = 62
    print(f"\n{'='*w}")
    if title:
        print(f"  {title}")
        print(f"{'='*w}")


def main(sigma: float = 0.05, n_adc_bits: int = 8, n_trials: int = 10):
    _OUT.mkdir(exist_ok=True)

    sys.path.insert(0, str(_EXP_DIR))
    import models.deq as deq_module

    # -- Step 1: Load / train model -------------------------------------------
    sep("STEP 1 / 3  Load DEQ model")
    ext = DEQMLPExtractor(checkpoint_path=_CKPT)
    ext.load_model()

    digital_acc = deq_module.evaluate(ext.model)
    conv_fail   = deq_module.evaluate_convergence_failure(ext.model)
    n_params    = sum(p.numel() for p in ext.model.parameters())
    print(f"  Parameters:          {n_params:,}")
    print(f"  Digital accuracy:    {-digital_acc:.4f} (neg CE loss, higher=better)")
    print(f"  Convergence failure: {conv_fail:.2%} of inputs diverge")
    print(f"  f_theta: tanh(W_z @ z + W_x @ x + b_x)")
    print(f"  W_z spectral norm enforced — rho(df/dz) < 1 guaranteed")

    # -- Step 2: Analog degradation sweep ------------------------------------
    sep("STEP 2 / 3  Analog Degradation")
    print(f"  sigma={sigma:.0%} mismatch, {n_adc_bits}-bit ADC, {n_trials} trials\n")

    sweep = mismatch_sweep(
        ext.model,
        lambda m: deq_module.evaluate(m),
        sigma_values=[0.0, 0.02, 0.05, 0.07, 0.10, 0.15, 0.20],
        n_trials=n_trials,
        n_adc_bits=n_adc_bits,
    )

    print(f"  {'sigma':>6}  {'score':>8}  {'sd':>6}  {'retained':>9}")
    for i, s in enumerate(sweep.sigma_values):
        print(f"  {s:6.2f}  {sweep.mean[i]:8.4f}  {sweep.std[i]:5.4f}  "
              f"{sweep.normalized_mean[i]:8.1%}")

    threshold = sweep.degradation_threshold(max_relative_loss=0.05)
    print(f"\n  5% degradation threshold:  sigma ~ {threshold:.3f}")

    # -- Step 3: Ark export --------------------------------------------------
    sep("STEP 3 / 3  Ark Export")
    ark_path = _OUT / "deq_ark.py"
    code = ext.export_to_ark(ark_path, mismatch_sigma=sigma)
    n_lines = len(code.splitlines())
    print(f"  Written:             {ark_path}")
    print(f"  Lines:               {n_lines}")
    print(f"  Mismatch sigma:      {sigma}")
    print(f"  a_trainable params:  {ext.z_dim**2 + ext.z_dim*ext.x_dim + ext.z_dim}")
    print()

    # Show ode_fn
    in_fn, shown = False, 0
    for line in code.splitlines():
        if "def ode_fn" in line:
            in_fn = True
        if in_fn:
            print(f"  {line}")
            shown += 1
            if shown > 9:
                break

    sep()
    print("  SUMMARY")
    print(f"    Digital score:          {-digital_acc:.4f} (neg CE)")
    print(f"    Convergence failure:    {conv_fail:.2%}")
    print(f"    5% tolerance threshold: sigma ~ {threshold:.3f}")
    print(f"    Ark export:             {ark_path}")
    print()
    print("  VERIFY -- run the exported file against Ark:")
    print(f"    python {ark_path}")
    print()
    print("  DEQ ODE form:")
    print("    dz/dt = tanh(W_z @ z + W_x @ x_input + b_x) - z")
    print("    Augmented state y = [z, x_input], dx/dt = 0")
    print("    Settles to z* = f_theta(z*, x) at t = 5.0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--n-adc-bits", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()
    main(args.sigma, args.n_adc_bits, args.n_trials)
