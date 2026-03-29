#!/usr/bin/env python3
"""
Flow model (rectified flow) — analog degradation sweep + Ark export.

The flow ODE dx/dt = v_theta(x, t) is structurally identical to Neural ODE.
The Ark export is literally the same code path — export_neural_ode_to_ark()
with class_name='FlowAnalogCkt'. The ode_fn already handles time concatenation:
    xt = cat([x, t]);  return MLP(xt)

Architecture: v_theta = MLP([2+1 → 64 → 64 → 2], tanh)
Task: make_moons 2D distribution, 4 Euler integration steps (like FLUX-schnell)

Usage:
    python examples/07_flow_ark.py
    python examples/07_flow_ark.py --sigma 0.07 --n-trials 15

Output:
    outputs/flow_ark.py  — valid Ark FlowAnalogCkt subclass
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_EXP_DIR = _ROOT / "experiments" / "cross_arch_tolerance"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EXP_DIR))

from neuro_analog.simulator import analogize, mismatch_sweep, ablation_sweep
from neuro_analog.extractors.flow import FlowMLPExtractor

_OUT = _ROOT / "outputs"
_CKPT = _EXP_DIR / "checkpoints" / "flow.pt"


def sep(title=""):
    w = 62
    print(f"\n{'='*w}")
    if title:
        print(f"  {title}")
        print(f"{'='*w}")


def main(sigma: float = 0.10, n_adc_bits: int = 8, n_trials: int = 20):
    _OUT.mkdir(exist_ok=True)

    # Need flow module on path for evaluate()
    sys.path.insert(0, str(_EXP_DIR))
    from models import flow as flow_module

    # ── Step 1: Load / train model ─────────────────────────────────────────────
    sep("STEP 1 / 4  Load Flow model")
    ext = FlowMLPExtractor(checkpoint_path=_CKPT)
    ext.load_model()

    digital_score = flow_module.evaluate(ext._extractor.model)
    n_params = sum(p.numel() for p in ext._extractor.model.parameters())
    print(f"  Parameters:        {n_params:,}")
    print(f"  Digital score:     {digital_score:.4f}  (neg. sliced Wasserstein)")

    # ── Step 2: Analog degradation sweep ──────────────────────────────────────
    sep("STEP 2 / 4  Analog Degradation")
    print(f"  sigma={sigma:.0%} mismatch, {n_adc_bits}-bit ADC, {n_trials} trials\n")

    def eval_fn(m):
        return flow_module.evaluate(m)

    sweep = mismatch_sweep(
        ext._extractor.model, eval_fn,
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

    # ── Step 3: Analog amenability profile ─────────────────────────────────────
    sep("STEP 3 / 4  Analog Amenability Profile")
    profile = ext.run()

    print(f"  Amenability score:  {profile.overall_score:.3f}")
    print(f"  Analog FLOP share:  {profile.analog_flop_fraction*100:.0f}%")
    print(f"  D/A boundaries:     {profile.da_boundary_count}")
    print(f"  Ark compiler fit:   PERFECT — dx/dt = v_theta(x,t) IS Ark's ode_fn")
    print(f"  Key difference from Neural ODE: None (same code path)")

    # ── Step 4: Ark export ─────────────────────────────────────────────────────
    sep("STEP 4 / 4  Ark Export")
    ark_path = _OUT / "flow_ark.py"
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
            if shown > 8:
                break

    sep()
    print("  SUMMARY")
    print(f"    Digital score:          {digital_score:.4f}")
    print(f"    5% tolerance threshold: sigma ~ {threshold:.3f}")
    print(f"    Ark export:             {ark_path}")
    print()
    print("  VERIFY — run the exported file against Ark:")
    print(f"    python {ark_path}")
    print()
    print("  Flow vs Neural ODE:")
    print("    Flow:       dx/dt = v_theta(x, t)  [generation ODE]")
    print("    Neural ODE: dx/dt = f_theta(x, t)  [dynamics ODE]")
    print("    In Ark:     identical — both are ode_fn(t, x, args) with cat([x,t])")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--n-adc-bits", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()
    main(args.sigma, args.n_adc_bits, args.n_trials)
