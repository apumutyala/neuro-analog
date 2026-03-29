#!/usr/bin/env python3
"""
Diffusion model (DDPM) — analog degradation sweep + Ark export.

Demonstrates the VP-SDE probability flow ODE bridge:
  1. Load / train small DDPM (3-layer MLP score network, 8x8 MNIST)
  2. Analog degradation sweep (mismatch vs generation quality)
  3. Analog amenability profile
  4. Ark export: DiffusionAnalogCkt subclass

ODE form in Ark:
    dx/dt = (T-1) * [-beta(k)/2 * x + beta(k)/(2*sqrt(1-ab_k)) * eps_theta(x,k)]
    t in [0,1]:  t=0 = noisy (k=T-1),  t=1 = clean (k=0)

Architecture: Linear(80, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 64)
Task: 8x8 MNIST (falls back to sklearn digits), T=100 linear beta schedule

Usage:
    python examples/08_diffusion_ark.py
    python examples/08_diffusion_ark.py --sigma 0.05 --n-trials 10

Output:
    outputs/diffusion_ark.py  -- valid Ark DiffusionAnalogCkt subclass
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_EXP_DIR = _ROOT / "experiments" / "cross_arch_tolerance"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EXP_DIR))

from neuro_analog.simulator import mismatch_sweep
from neuro_analog.extractors.diffusion import DiffusionMLPExtractor

_OUT = _ROOT / "outputs"
_CKPT = _EXP_DIR / "checkpoints" / "diffusion.pt"


def sep(title=""):
    w = 62
    print(f"\n{'='*w}")
    if title:
        print(f"  {title}")
        print(f"{'='*w}")


def main(sigma: float = 0.05, n_adc_bits: int = 8, n_trials: int = 10):
    _OUT.mkdir(exist_ok=True)

    sys.path.insert(0, str(_EXP_DIR))
    import models.diffusion as diff_module

    # -- Step 1: Load / train model -----------------------------------------------
    sep("STEP 1 / 4  Load Diffusion model")
    ext = DiffusionMLPExtractor(checkpoint_path=_CKPT)
    ext.load_model()

    digital_score = diff_module.evaluate(ext.model)
    n_params = sum(p.numel() for p in ext.model.parameters())
    print(f"  Parameters:        {n_params:,}")
    print(f"  Digital score:     {digital_score:.4f}  (neg. nearest-neighbor distance)")

    # -- Step 2: Analog degradation sweep ----------------------------------------
    sep("STEP 2 / 4  Analog Degradation")
    print(f"  sigma={sigma:.0%} mismatch, {n_adc_bits}-bit ADC, {n_trials} trials\n")

    def eval_fn(m):
        return diff_module.evaluate(m)

    sweep = mismatch_sweep(
        ext.model, eval_fn,
        sigma_values=[0.0, 0.02, 0.05, 0.07, 0.10, 0.12, 0.15],
        n_trials=n_trials,
        n_adc_bits=n_adc_bits,
    )

    print(f"  {'s':>6}  {'score':>8}  {'sd':>6}  {'retained':>9}")
    for i, s in enumerate(sweep.sigma_values):
        print(f"  {s:6.2f}  {sweep.mean[i]:8.4f}  {sweep.std[i]:5.4f}  "
              f"{sweep.normalized_mean[i]:8.1%}")

    threshold = sweep.degradation_threshold(max_relative_loss=0.05)
    print(f"\n  5% degradation threshold:  sigma ~ {threshold:.3f}")

    # -- Step 3: Analog amenability profile --------------------------------------
    sep("STEP 3 / 4  Analog Amenability Profile")
    profile = ext.run()

    print(f"  Amenability score:  {profile.overall_score:.3f}")
    print(f"  Analog FLOP share:  {profile.analog_flop_fraction*100:.0f}%")
    print(f"  D/A boundaries:     {profile.da_boundary_count}")
    print(f"  Ark compiler fit:   VP-SDE PF-ODE via beta schedule + sinusoidal embed")
    print(f"  Total params:       {profile.total_params:,}")
    print(f"  MLP params:         {profile.mlp_params:,}")

    # -- Step 4: Ark export -------------------------------------------------------
    sep("STEP 4 / 4  Ark Export")
    ark_path = _OUT / "diffusion_ark.py"
    code = ext.export_to_ark(ark_path, mismatch_sigma=sigma)
    n_lines = len(code.splitlines())
    n_params_exported = sum(1 for line in code.splitlines() if "_p" in line and "reshape" in line)
    print(f"  Written:            {ark_path}")
    print(f"  Lines:              {n_lines}")
    print(f"  Param tensors:      {n_params_exported}")
    print(f"  Mismatch sigma:     {sigma}")
    print()

    # Show ode_fn
    in_fn, shown = False, 0
    for line in code.splitlines():
        if "def ode_fn" in line:
            in_fn = True
        if in_fn:
            print(f"  {line}")
            shown += 1
            if shown > 12:
                break

    sep()
    print("  SUMMARY")
    print(f"    Digital score:          {digital_score:.4f}")
    print(f"    5% tolerance threshold: sigma ~ {threshold:.3f}")
    print(f"    Ark export:             {ark_path}")
    print()
    print("  VERIFY -- run the exported file against Ark:")
    print(f"    python {ark_path}")
    print()
    print("  Diffusion ODE form:")
    print("    dx/dt = (T-1)*[-beta(k)/2*x + beta(k)/(2*sqrt(1-ab_k))*eps_theta(x,k)]")
    print("    VP-SDE probability flow ODE (DDIM deterministic reverse)")
    print("    t=0: noisy (k=T-1),  t=1: clean (k=0)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--n-adc-bits", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()
    main(args.sigma, args.n_adc_bits, args.n_trials)
