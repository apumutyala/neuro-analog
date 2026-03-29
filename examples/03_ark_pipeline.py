#!/usr/bin/env python3
"""
Neural ODE — analog degradation sweep + Ark export.

Demonstrates the complete neuro-analog workflow for the Neural ODE family:
  1. Load (or train) the Neural ODE from the cross-arch experiment
  2. Measure analog degradation via mismatch sweep and noise attribution
  3. Extract the IR: amenability score, D/A boundaries, Ark compiler fit
  4. Export to a valid Ark BaseAnalogCkt subclass

neuro-analog's role in the Ark ecosystem:
  MEASUREMENT   analogize()                 — quantifies the analog gap
  TRANSLATION   export_neural_ode_to_ark()  — generates Ark-compatible BaseAnalogCkt
  VALIDATION    run outputs/neural_ode_ark.py — verifies the export against Ark

The Neural ODE (dx/dt = f_θ(x,t)) is the most analog-native architecture:
  - The model IS an ODE — identical to Arco/Legno/Ark input format
  - Only 2 D/A boundaries (1 DAC input + 1 ADC output)
  - f_θ is a small MLP → fits within demonstrated 128x128 crossbar scale
  - Adjoint training uses the same math as Ark's gradient computation

Usage:
    python examples/03_ark_pipeline.py
    python examples/03_ark_pipeline.py --sigma 0.07 --n-trials 15
    python examples/03_ark_pipeline.py --n-adc-bits 6

Output:
    outputs/neural_ode_ark.py  — valid Ark BaseAnalogCkt subclass
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_EXP_DIR = _ROOT / "experiments" / "cross_arch_tolerance"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EXP_DIR))

import torch

from neuro_analog.simulator import (
    analogize,
    mismatch_sweep,
    ablation_sweep,
    count_analog_vs_digital,
)
from neuro_analog.extractors.neural_ode import NeuralODEExtractor, export_neural_ode_to_ark
from neuro_analog.analysis.taxonomy import AnalogTaxonomy
from neuro_analog.mappers.crossbar import CrossbarMapper
from neuro_analog.mappers.integrator import IntegratorMapper
from neuro_analog.visualization.comparison_radar import plot_radar_from_taxonomy
from models import neural_ode as neural_ode_module

_CKPT = _EXP_DIR / "checkpoints" / "neural_ode.pt"
_OUT = _ROOT / "outputs"


def sep(title=""):
    w = 62
    print(f"\n{'='*w}")
    if title:
        print(f"  {title}")
        print(f"{'='*w}")


def main(sigma: float = 0.10, n_adc_bits: int = 8, n_trials: int = 20):
    _OUT.mkdir(exist_ok=True)

    # ── Step 1: Load checkpoint ────────────────────────────────────────────────
    sep("STEP 1 / 4  Load Neural ODE checkpoint")
    if _CKPT.exists():
        model = neural_ode_module.load_model(str(_CKPT))
        print(f"  Loaded checkpoint: {_CKPT}")
    else:
        print(f"  No checkpoint at {_CKPT}")
        print("  Training from scratch (~2 min)...")
        model = neural_ode_module.create_model()
        neural_ode_module.train_model(model, str(_CKPT))
        print(f"  Saved to: {_CKPT}")

    digital_ll = neural_ode_module.evaluate(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:       {n_params:,}")
    print(f"  Log-likelihood:   {digital_ll:.4f} nats  (digital baseline)")

    # ── Step 2: Analog degradation ─────────────────────────────────────────────
    sep("STEP 2 / 4  Analog Degradation")
    print(f"  σ={sigma:.0%} mismatch, {n_adc_bits}-bit ADC, {n_trials} trials\n")

    def eval_fn(m):
        return neural_ode_module.evaluate(m)

    sweep = mismatch_sweep(
        model, eval_fn,
        sigma_values=[0.0, 0.02, 0.05, 0.07, 0.10, 0.12, 0.15],
        n_trials=n_trials,
        n_adc_bits=n_adc_bits,
    )

    print(f"  {'σ':>6}  {'log-lik':>8}  {'±':>6}  {'retained':>9}")
    for i, s in enumerate(sweep.sigma_values):
        print(f"  {s:6.2f}  {sweep.mean[i]:8.4f}  {sweep.std[i]:5.4f}  "
              f"{sweep.normalized_mean[i]:8.1%}")

    threshold = sweep.degradation_threshold(max_relative_loss=0.05)
    print(f"\n  5% degradation threshold:  σ ≈ {threshold:.3f}")

    # Noise attribution
    print("\n  Noise attribution ablation (10 trials each)...")
    ablation = ablation_sweep(
        model, eval_fn,
        sigma_values=[0.0, 0.05, 0.10],
        n_trials=10,
        n_adc_bits=n_adc_bits,
    )
    print("  Degradation at σ=0.10 per noise source (quality retained):")
    for src, result in sorted(ablation.items(), key=lambda kv: kv[1].normalized_mean[-1]):
        q = result.normalized_mean[-1]
        bar = "█" * max(0, int((1.0 - q) * 40))
        print(f"    {src:<16}  {bar:<20}  retained={q:.1%}")

    # ── Step 3: Analog amenability profile ─────────────────────────────────────
    sep("STEP 3 / 4  Analog Amenability Profile")

    extractor = NeuralODEExtractor.from_module(
        model, state_dim=2, t_span=(0.0, 1.0),
        model_name="neural_ode_make_circles",
    )
    extractor.load_model()
    profile = extractor.run()

    # Annotate graph with hardware noise specs (CrossbarSpec defaults: 8-bit RRAM,
    # IntegratorSpec defaults: 1 ms RC, 250 kHz BW, kT/C thermal model).
    graph = extractor.graph
    CrossbarMapper().annotate_graph(graph)
    IntegratorMapper().annotate_graph(graph)

    taxonomy = AnalogTaxonomy()
    taxonomy.add_profile(
        profile,
        has_native_dynamics=True,
        dynamics_description="dx/dt = f_θ(x, t)  [time-augmented MLP vector field]",
        analog_circuit_primitive="Crossbar MVM + tanh diff pair + RC integrator",
        key_digital_bottleneck="Adaptive step-size controller (digital bookkeeping only)",
        analog_compiler_fit="Perfect — dx/dt = f_θ(x,t) IS Ark's input format",
    )
    # Populate reference profiles for all other architectures so the radar
    # shows the full cross-architecture comparison, not just Neural ODE.
    taxonomy.add_reference_profiles()

    counts = count_analog_vs_digital(analogize(model, sigma_mismatch=0.0, n_adc_bits=n_adc_bits))
    print(f"  Amenability score:  {profile.overall_score:.3f}")
    print(f"  Analog FLOP share:  {profile.analog_flop_fraction*100:.0f}%")
    print(f"  D/A boundaries:     {profile.da_boundary_count}")
    print(f"  Analog layers:      {counts['analog_layers']}  "
          f"({counts['coverage_pct']:.0f}% param coverage)")
    print(f"  Digital layers:     {counts['digital_layers']}")
    print(f"  Ark compiler fit:   PERFECT")

    # Hardware noise annotations on the IR graph
    print()
    print("  Hardware noise model (HCDCv2 defaults):")
    from neuro_analog.ir.types import OpType
    for node in graph.nodes:
        if node.noise is not None:
            tag = f"{node.op_type.name:<16}"
            print(f"    {node.name:<30} {tag}  σ={node.noise.sigma:.3e}"
                  + (f"  BW={node.noise.bandwidth_hz/1e3:.0f} kHz" if node.noise.bandwidth_hz else ""))

    # Cross-architecture comparison table
    print()
    print(taxonomy.comparison_table())

    # Radar chart — all 6 architectures, all 6 axes including Ark compatibility
    radar_path = _OUT / "analog_amenability_radar.png"
    plot_radar_from_taxonomy(taxonomy, output_path=radar_path)
    print(f"\n  Radar chart:        {radar_path}")

    # ── Step 4: Ark export ─────────────────────────────────────────────────────
    sep("STEP 4 / 4  Ark Export")

    ark_path = _OUT / "neural_ode_ark.py"
    code = export_neural_ode_to_ark(extractor, ark_path, mismatch_sigma=sigma)
    n_lines = len(code.splitlines())
    n_params = sum(1 for line in code.splitlines() if "a_trainable[" in line)
    print(f"  Written:            {ark_path}")
    print(f"  Lines:              {n_lines}")
    print(f"  Trainable slices:   {n_params}  (mismatch sigma={sigma})")
    print()

    # Print the ode_fn method
    in_fn, shown = False, 0
    for line in code.splitlines():
        if "def ode_fn" in line:
            in_fn = True
        if in_fn:
            print(f"  {line}")
            shown += 1
            if shown > 15:
                print("  ...")
                break

    # ── Summary ────────────────────────────────────────────────────────────────
    sep()
    print("  SUMMARY")
    print(f"    Digital baseline:       {digital_ll:.4f} nats")
    print(f"    5% tolerance threshold: σ ≈ {threshold:.3f}")
    print(f"    Ark export:             {ark_path}")
    print()
    print("  VERIFY — run the exported file against Ark:")
    print(f"    python {ark_path}")
    print()
    print("  The generated NeuralODEAnalogCkt is a BaseAnalogCkt subclass.")
    print("  Ark can compile it to an analog circuit and optimize weights")
    print("  for mismatch robustness via adjoint-based gradient computation.")
    print()
    print(f"  Neural ODE has only {profile.da_boundary_count} D/A boundaries vs CNN 6-8")
    print("  — fewer domain crossings means less quantization loss and better analog fit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=0.10,
                        help="Mismatch level to report (default: 0.10)")
    parser.add_argument("--n-adc-bits", type=int, default=8,
                        help="ADC bit width (default: 8)")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="MC trials per sigma level (default: 20)")
    args = parser.parse_args()
    main(args.sigma, args.n_adc_bits, args.n_trials)
