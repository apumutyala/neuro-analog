#!/usr/bin/env python3
"""
S4D SSM — analog degradation sweep + Ark export.

Demonstrates the continuous-time SSM dynamics kernel bridge:
  1. Load / train S4D SSM (_SSMClassifier, synthetic bigram task, d_model=16)
  2. Analog degradation sweep (mismatch vs accuracy)
  3. Ark export: SSMAnalogCkt subclass (single _S4DLayer, real/imag split ODE)

ODE form in Ark (real/imag split of complex diagonal SSM, autonomous u=0):
    dh_re/dt = A_re * h_re - A_im * h_im
    dh_im/dt = A_im * h_re + A_re * h_im

where A_re = -exp(log_A_real)  (RC decay rates, all negative)
      A_im = log_A_imag        (oscillation frequencies)

State: h = [h_re, h_im]  (dim = 2 * d_state = 16 real values)

Architecture: d_model=16, d_state=8, 2 layers, bilinear S4D, vocab=32
Task: detect adjacent bigram [3,5] in 64-token sequence

Usage:
    python examples/11_ssm_ark.py
    python examples/11_ssm_ark.py --sigma 0.05 --n-trials 10 --layer-idx 0

Output:
    outputs/ssm_ark.py  -- valid Ark SSMAnalogCkt subclass (layer 0 dynamics)
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_EXP_DIR = _ROOT / "experiments" / "cross_arch_tolerance"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EXP_DIR))

from neuro_analog.simulator import mismatch_sweep
from neuro_analog.extractors.ssm import S4DMLPExtractor

_OUT = _ROOT / "outputs"
_CKPT = _EXP_DIR / "checkpoints" / "ssm.pt"


def sep(title=""):
    w = 62
    print(f"\n{'='*w}")
    if title:
        print(f"  {title}")
        print(f"{'='*w}")


def main(sigma: float = 0.05, n_adc_bits: int = 8, n_trials: int = 10, layer_idx: int = 0):
    _OUT.mkdir(exist_ok=True)

    sys.path.insert(0, str(_EXP_DIR))
    import models.ssm as ssm_module

    # -- Step 1: Load / train model -------------------------------------------
    sep("STEP 1 / 3  Load S4D SSM model")
    ext = S4DMLPExtractor(checkpoint_path=_CKPT, layer_idx=layer_idx)
    ext.load_model()

    digital_acc = ssm_module.evaluate(ext.model)
    n_params    = sum(p.numel() for p in ext.model.parameters())
    layer       = ext.model.layers[layer_idx]

    import torch, numpy as np
    A_re = -torch.exp(layer.log_A_real).detach().cpu().numpy()
    A_im = layer.log_A_imag.detach().cpu().numpy()
    time_constants = -1.0 / A_re  # tau_i = 1/|A_re_i|

    print(f"  Parameters (full model): {n_params:,}")
    print(f"  Digital score:           {digital_acc:.4f} (neg CE, higher=better)")
    print(f"  Exporting:               layer {layer_idx} of {len(ext.model.layers)}")
    print(f"  d_model={ext.d_model}, d_state={ext.d_state} (complex) = {2*ext.d_state} real")
    print(f"  A_re range: [{A_re.min():.4f}, {A_re.max():.4f}]  (all negative)")
    print(f"  A_im range: [{A_im.min():.4f}, {A_im.max():.4f}]")
    print(f"  Time constants tau_i = 1/|A_re|: [{time_constants.min():.3f}, {time_constants.max():.3f}]")

    # -- Step 2: Analog degradation sweep ------------------------------------
    sep("STEP 2 / 3  Analog Degradation")
    print(f"  sigma={sigma:.0%} mismatch, {n_adc_bits}-bit ADC, {n_trials} trials\n")

    sweep = mismatch_sweep(
        ext.model,
        lambda m: ssm_module.evaluate(m),
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
    ark_path = _OUT / "ssm_ark.py"
    code = ext.export_to_ark(ark_path, mismatch_sigma=sigma)
    n_lines = len(code.splitlines())
    state_dim = 2 * ext.d_state
    trainable_count = ext.d_state + ext.d_state + state_dim * ext.d_model  # A_re + A_im + B
    print(f"  Written:             {ark_path}")
    print(f"  Lines:               {n_lines}")
    print(f"  Mismatch sigma:      {sigma}")
    print(f"  a_trainable params:  {trainable_count}  (A_re:{ext.d_state}, A_im:{ext.d_state}, B:{state_dim*ext.d_model})")
    print(f"  ODE state dim:       {state_dim}  (h_re and h_im, each d_state={ext.d_state})")
    print()

    # Show ode_fn
    in_fn, shown = False, 0
    for line in code.splitlines():
        if "def ode_fn" in line:
            in_fn = True
        if in_fn:
            print(f"  {line}")
            shown += 1
            if shown > 11:
                break

    sep()
    print("  SUMMARY")
    print(f"    Digital score:          {digital_acc:.4f} (neg CE)")
    print(f"    5% tolerance threshold: sigma ~ {threshold:.3f}")
    print(f"    Ark export:             {ark_path}  (layer {layer_idx})")
    print()
    print("  VERIFY -- run the exported file against Ark:")
    print(f"    python {ark_path}")
    print()
    print("  SSM ODE form (real/imag split, autonomous u=0):")
    print("    dh_re/dt = A_re * h_re - A_im * h_im")
    print("    dh_im/dt = A_im * h_re + A_re * h_im")
    print("    A_re = -exp(log_A_real)  (RC decay, all negative)")
    print("    A_im = log_A_imag        (oscillation frequency)")
    print("  Mismatch model:")
    print("    A: complex polar — |A_c'| = |A_c|*(1+sigma*eps_mag),")
    print("                        angle' = angle + sigma*eps_phase")
    print("    B: multiplicative delta ~ N(1, sigma^2) per element")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--n-adc-bits", type=int, default=8)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--layer-idx", type=int, default=0)
    args = parser.parse_args()
    main(args.sigma, args.n_adc_bits, args.n_trials, args.layer_idx)
