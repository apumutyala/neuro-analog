#!/usr/bin/env python3
"""
EBM (Hopfield network) — Langevin ODE reformulation + Ark export.

Demonstrates the EBM → Ark bridge:
  1. Build a theoretical Hopfield/RBM via EBMExtractor
  2. Show the Langevin ODE reformulation: dx/dt = -x + tanh(W_sym @ x + b)
  3. Export to a valid Ark BaseAnalogCkt subclass

Why Hopfield maps cleanly to Ark:
  - Gibbs sampling equilibrium ≡ Boltzmann distribution P(x) ∝ exp(-E(x)/T)
  - Mean-field (T→0) converges to same energy minima via ODE, not sampling
  - ODE form is IDENTICAL to the Neural ODE CANN paradigm already in Ark
  - Zero new CDGSpec code — compile_hopfield_cdg() wraps compile_neural_ode_cdg()

Usage:
    python examples/06_ebm_ark.py                      # 4-neuron Hopfield (default)
    python examples/06_ebm_ark.py --n 16               # 16-neuron Hopfield
    python examples/06_ebm_ark.py --model-type rbm     # RBM → augmented Hopfield
    python examples/06_ebm_ark.py --n-visible 8 --n-hidden 8 --model-type rbm

Output:
    outputs/ebm_ark.py   — valid Ark HopfieldAnalogCkt subclass
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

from neuro_analog.extractors.ebm import EBMExtractor, EBMConfig
from neuro_analog.ark_bridge.ebm_cdg import (
    compile_hopfield_cdg,
    make_rbm_hopfield_weights,
    export_hopfield_to_ark,
)

_OUT = _ROOT / "outputs"


def sep(title=""):
    w = 62
    print(f"\n{'='*w}")
    if title:
        print(f"  {title}")
        print(f"{'='*w}")


def main(
    n: int = 4,
    n_visible: int = 8,
    n_hidden: int = 8,
    model_type: str = "hopfield",
    mismatch_sigma: float = 0.05,
    seed: int = 42,
):
    _OUT.mkdir(exist_ok=True)

    # ── Step 1: Build EBM IR ───────────────────────────────────────────────────
    sep("STEP 1 / 3  Build EBM IR")
    if model_type == "hopfield":
        cfg = EBMConfig(num_visible=n, model_type="hopfield")
        print(f"  Hopfield network: n={n}")
    elif model_type == "rbm":
        cfg = EBMConfig(num_visible=n_visible, num_hidden=n_hidden, model_type="rbm")
        print(f"  RBM: n_visible={n_visible}, n_hidden={n_hidden}")
        print(f"  Augmented state: z=[v; h], n_aug={n_visible + n_hidden}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    extractor = EBMExtractor(config=cfg)
    extractor.load_model()
    profile = extractor.run()

    print(f"\n  Analog amenability: {profile.overall_score:.3f}")
    print(f"  D/A boundaries:     {profile.da_boundary_count}")
    print(f"  Analog FLOP share:  {profile.analog_flop_fraction*100:.0f}%")

    # ── Step 2: Show ODE reformulation ────────────────────────────────────────
    sep("STEP 2 / 3  Langevin ODE Reformulation")
    print("  Gibbs sampling (discrete, stochastic):")
    print("    h_i ~ Bernoulli(s(Wv + b_h))   [p-bit hardware]")
    print("    v_i ~ Bernoulli(s(W^T h + b_v)) [p-bit hardware]")
    print()
    print("  Langevin mean-field ODE (continuous, deterministic):")
    print("    dx/dt = -x + tanh(W_sym @ x + b)")
    print()
    print("  Why they're equivalent:")
    print("    Both converge to energy minima of E(x) = -1/2 x^T W x - b^T tanh(x)")
    print("    ODE form is the CANN paradigm — already in Ark's neural_ode_cdg.py")
    print("    compile_hopfield_cdg() = compile_neural_ode_cdg(J=W_sym, b=b, K=None)")

    # ── Step 3: Ark export ────────────────────────────────────────────────────
    sep("STEP 3 / 3  Ark Export")
    ark_path = _OUT / "ebm_ark.py"
    code = extractor.export_to_ark(ark_path, mismatch_sigma=mismatch_sigma, seed=seed)
    n_lines = len(code.splitlines())
    print(f"  Written:           {ark_path}")
    print(f"  Lines:             {n_lines}")
    print(f"  Mismatch sigma:    {mismatch_sigma}")

    # Show the ode_fn
    print()
    for line in code.splitlines():
        if "def ode_fn" in line or "def make_args" in line or "return -x" in line:
            print(f"  {line}")

    sep()
    print("  SUMMARY")
    print(f"    EBM analog score:  {profile.overall_score:.3f}")
    print(f"    ODE form:          dx/dt = -x + tanh(W_sym @ x + b)")
    print(f"    CDGSpec:           Neural ODE CANN (zero new spec code)")
    print(f"    Ark export:        {ark_path}")
    print()
    print("  VERIFY — run the exported file against Ark:")
    print(f"    python {ark_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4,
                        help="Hopfield network size (default: 4)")
    parser.add_argument("--n-visible", type=int, default=8,
                        help="RBM visible units (default: 8)")
    parser.add_argument("--n-hidden", type=int, default=8,
                        help="RBM hidden units (default: 8)")
    parser.add_argument("--model-type", choices=["hopfield", "rbm"], default="hopfield",
                        help="EBM variant (default: hopfield)")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Mismatch sigma (default: 0.05)")
    args = parser.parse_args()
    main(
        n=args.n,
        n_visible=args.n_visible,
        n_hidden=args.n_hidden,
        model_type=args.model_type,
        mismatch_sigma=args.sigma,
    )
