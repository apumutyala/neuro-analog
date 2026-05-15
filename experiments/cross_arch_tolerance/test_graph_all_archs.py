#!/usr/bin/env python3
"""Architecture test: load checkpoints, build AnalogGraph, validate, analyze.

For each architecture with a checkpoint present:
  1. Load model from checkpoint
  2. Analogize (sigma=0, max ADC bits) to get the analogized module graph
  3. Build pilot graph via sweep_all._build_pilot_graph
  4. Validate graph (no None seq_len, no negative FLOPs)
  5. Analyze with default HardwareProfile
  6. Assert physically-correct metric bounds

Usage:
    python experiments/cross_arch_tolerance/test_graph_all_archs.py
    python experiments/cross_arch_tolerance/test_graph_all_archs.py --only transformer
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import argparse
import torch

from experiments.cross_arch_tolerance import sweep_all as sw
from neuro_analog.ir.energy_model import HardwareProfile
from neuro_analog.simulator import analogize, configure_analog_profile


_ARCH_NAMES = [name for name, _ in sw._MODELS]
_CKPT_DIR = Path(__file__).parent / "checkpoints"


def _test_one(name: str, analog_domain: str = "conservative") -> dict:
    ckpt_path = _CKPT_DIR / f"{name}.pt"
    if not ckpt_path.exists():
        return {"status": "SKIP", "reason": f"No checkpoint at {ckpt_path}"}

    module = dict(sw._MODELS)[name]
    model = module.load_model(str(ckpt_path))
    device = next(iter(model.parameters())).device
    print(f"\n[{name}] device={device}")

    # Build graph from nominal analog model
    analog_model = analogize(model, sigma_mismatch=0.0, n_adc_bits=32)
    configure_analog_profile(analog_model, analog_domain)

    profile = HardwareProfile()
    try:
        result = sw._build_pilot_graph(name, analog_model, analog_domain, profile)
    except Exception as e:
        return {"status": "FAIL", "reason": f"_build_pilot_graph raised {type(e).__name__}: {e}"}

    # Validate graph indirectly: _build_pilot_graph calls graph.validate() inside analyze,
    # but we also check the returned profile for physically-sensible bounds.
    issues = []
    if result.analog_energy_pJ < 0:
        issues.append(f"analog_energy_pJ={result.analog_energy_pJ} < 0")
    if result.digital_energy_pJ < 0:
        issues.append(f"digital_energy_pJ={result.digital_energy_pJ} < 0")
    if result.analog_latency_ns < 0:
        issues.append(f"analog_latency_ns={result.analog_latency_ns} < 0")
    if result.digital_latency_ns < 0:
        issues.append(f"digital_latency_ns={result.digital_latency_ns} < 0")

    es = result.analog_energy_saving_vs_digital
    sp = result.analog_speedup_vs_digital

    if es is not None and not (0.0 <= es <= 1.0):
        issues.append(f"energy_saving={es} not in [0,1]")
    if sp is not None and not (sp > 0):
        issues.append(f"speedup={sp} not > 0")

    # Architecture-specific iteration-aware expectations
    expected = {
        "neural_ode": {"speedup_min": 2.0,  "energy_saving_min": 0.50},
        "deq":      {"speedup_min": 2.0,  "energy_saving_min": 0.50},
        "diffusion": {"speedup_min": 10.0, "energy_saving_min": 0.80},
        "flow":     {"speedup_min": 5.0,  "energy_saving_min": 0.70},
        "ebm":      {"speedup_min": 10.0, "energy_saving_min": 0.80},
        "ssm":      {"speedup_min": 0.5,  "energy_saving_min": 0.30},
        "transformer": {"speedup_min": 0.5, "energy_saving_min": 0.30},
    }
    if name in expected:
        exp = expected[name]
        if sp is not None and sp < exp["speedup_min"]:
            issues.append(
                f"speedup={sp:.1f}x < expected_min={exp['speedup_min']}x for {name}"
            )
        if es is not None and es < exp["energy_saving_min"]:
            issues.append(
                f"energy_saving={es*100:.1f}% < expected_min={exp['energy_saving_min']*100:.0f}% for {name}"
            )

    if issues:
        return {"status": "FAIL", "reason": "; ".join(issues)}

    return {
        "status": "PASS",
        "analog_energy_pJ": result.analog_energy_pJ,
        "digital_energy_pJ": result.digital_energy_pJ,
        "analog_latency_ns": result.analog_latency_ns,
        "digital_latency_ns": result.digital_latency_ns,
        "energy_saving": es,
        "speedup": sp,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--analog-domain", type=str, default="conservative")
    args = parser.parse_args()

    names = [args.only] if args.only else _ARCH_NAMES
    passed = 0
    failed = 0
    skipped = 0

    for name in names:
        res = _test_one(name, analog_domain=args.analog_domain)
        st = res["status"]
        if st == "PASS":
            passed += 1
            print(
                f"  PASS  energy={res['energy_saving']*100:.1f}%  speedup={res['speedup']:.1f}x  "
                f"(A={res['analog_energy_pJ']:.2e}pJ  D={res['digital_energy_pJ']:.2e}pJ)"
            )
        elif st == "SKIP":
            skipped += 1
            print(f"  SKIP  {res['reason']}")
        else:
            failed += 1
            print(f"  FAIL  {res['reason']}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
