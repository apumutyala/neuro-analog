#!/usr/bin/env python3
"""
Example: Analyze Mamba-370M for analog hardware feasibility.

Extracts A_log eigenvalue spectra, decomposes the model into
analog/digital primitives, and generates a feasibility report.

Usage:
    python examples/01_analyze_mamba.py [--model state-spaces/mamba-370m]
"""

import argparse
import json
from pathlib import Path

from neuro_analog.extractors.ssm import MambaExtractor
from neuro_analog.ir.shem_export import export_ssm_to_shem


def main(model_name: str = "state-spaces/mamba-370m"):
    print(f"\n{'='*60}")
    print(f"NEURO-ANALOG: Mamba SSM Analysis")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    # 1. Extract and analyze
    extractor = MambaExtractor(model_name)
    profile = extractor.run()
    
    # 2. Print IR graph summary
    print("\n" + extractor.graph.summary_table())
    
    # 3. Extract detailed A_log spectra
    print("\n--- A_log Eigenvalue Spectra ---")
    spectra = extractor.extract_a_log_spectra()
    for name, values in list(spectra.items())[:3]:
        print(f"  {name}: shape={values.shape}, "
              f"range=[{values.min():.4f}, {values.max():.4f}]")
    
    # 4. Selective mechanism statistics
    print("\n--- Selective Mechanism ---")
    sel_stats = extractor.extract_selective_mechanism_stats()
    for name, stats in list(sel_stats.items())[:4]:
        print(f"  {name}: shape={stats['shape']}, "
              f"std={stats['std']:.6f}, sparsity={stats['sparsity']:.1%}")
    
    # 5. Export to Shem-compatible format
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    shem_code = export_ssm_to_shem(extractor.graph, output_dir / "mamba_shem_ode.py")
    print(f"\n--- Shem Export ---")
    print(f"Generated {len(shem_code)} chars → outputs/mamba_shem_ode.py")
    
    # 6. Noise Budget
    from neuro_analog.analysis.precision import flag_snr_violations
    from neuro_analog.mappers.crossbar import CrossbarMapper
    from neuro_analog.mappers.integrator import IntegratorMapper
    from neuro_analog.mappers.stochastic import StochasticMapper
    
    # Apply mappers to explicitly set noise specs
    CrossbarMapper().annotate_graph(extractor.graph)
    IntegratorMapper().annotate_graph(extractor.graph)
    StochasticMapper().annotate_graph(extractor.graph)
    
    violations = flag_snr_violations(extractor.graph, signal_rms=1.0, target_snr_db=30.0)
    print(f"\n--- Noise Budget (target SNR ≥ 30 dB) ---")
    print("(No model download needed for this analysis — runs on the extracted IR graph)")
    if violations:
        for v in violations:
            print(f"  ⚠  {v['name']}: SNR={v['snr_db']:.1f} dB (margin={v['margin_db']:.1f} dB)")
    else:
        print("  ✓ All analog nodes meet SNR target.")
    
    # 7. Summary
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Analog FLOP fraction:  {profile.analog_flop_fraction:.1%}")
    print(f"  D/A boundaries:        {profile.da_boundary_count}")
    print(f"  Overall score:          {profile.overall_score:.3f}")
    print(f"  Dynamics type:          {profile.dynamics.dynamics_type}")
    if profile.dynamics.time_constant_spread:
        print(f"  Time constant spread:   {profile.dynamics.time_constant_spread:.1f}x")
    print(f"{'='*60}\n")
    
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="state-spaces/mamba-370m")
    args = parser.parse_args()
    main(args.model)
