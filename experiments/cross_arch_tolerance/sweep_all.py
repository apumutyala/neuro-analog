#!/usr/bin/env python3
"""Run mismatch sweeps across all 7 architecture families.

For each family:
  1. Load pretrained model
  2. Run mismatch_sweep (sigma: 0.0 → 0.15, 50 trials)
  3. Run ablation_sweep (mismatch-only, thermal-only, quantization-only)
  4. Run adc_sweep (bits: 2 → 16 at sigma=0.05, 50 trials)
  5. For DEQ: also track convergence failure rate vs sigma

Results saved as JSON to results/ directory.

Substrate-aware architectures and their supported substrates:
  diffusion:  classic (DDIM) | cld (RLC/Langevin) | extropic_dtm (Extropic arXiv:2510.23972)
  neural_ode: euler (noiseless) | rc_integrator (Johnson-Nyquist capacitor noise)
  flow:       euler (noiseless) | rc_integrator (Johnson-Nyquist capacitor noise)
  deq:        discrete (fixed-point iter) | hopfield (damped continuous-time relaxation)

Usage:
    python experiments/cross_arch_tolerance/sweep_all.py
    python experiments/cross_arch_tolerance/sweep_all.py --only neural_ode
    python experiments/cross_arch_tolerance/sweep_all.py --n-trials 20  # faster
    python experiments/cross_arch_tolerance/sweep_all.py --only diffusion --analog-substrate cld
    python experiments/cross_arch_tolerance/sweep_all.py --only diffusion --analog-substrate extropic_dtm
    python experiments/cross_arch_tolerance/sweep_all.py --analog-substrate all  # all substrates for all arches
    python experiments/cross_arch_tolerance/sweep_all.py --analog-domain both  # conservative + full_analog
    python experiments/cross_arch_tolerance/sweep_all.py --sigma-values "0.0,0.05,0.10,0.15,0.20,0.25,0.30"
    python experiments/cross_arch_tolerance/sweep_all.py --adc-only --adc-sigma 0.0  # isolate pure quantization
    python experiments/cross_arch_tolerance/sweep_all.py --adc-only --adc-bits "2,4,6,8,10,12,16,20,32"
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

_CKPT_DIR = Path(__file__).parent / "checkpoints"
_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

import torch

from models import neural_ode, transformer, diffusion, flow, ebm, deq, ssm
from neuro_analog.simulator import mismatch_sweep, adc_sweep, ablation_sweep, resample_all_mismatch, set_all_noise, analogize, configure_analog_profile
from neuro_analog.ir.energy_model import HardwareProfile

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODELS = [
    ("neural_ode",  neural_ode),
    ("transformer", transformer),
    ("diffusion",   diffusion),
    ("flow",        flow),
    ("ebm",         ebm),
    ("deq",         deq),
    ("ssm",         ssm),
]

_SIGMA_VALUES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
_BIT_VALUES = [2, 4, 6, 8, 10, 12, 16]

# Architectures whose evaluate() accepts an analog_substrate kwarg.
_SUBSTRATE_AWARE = {"diffusion", "neural_ode", "flow", "deq"}

# All supported substrates per architecture (first = default, adds no filename suffix).
# Circuit modes are now default (first in list) to measure true analog hyperefficiency.
_ALL_SUBSTRATES_BY_NAME = {
    "diffusion":  ["classic", "cld"],  # extropic_dtm excluded from 'all' — substrate broken, re-enable explicitly
    "neural_ode": ["rc_integrator", "euler"],  # CHANGED: circuit mode first
    "flow":       ["rc_integrator", "euler"],  # CHANGED: circuit mode first
    "deq":        ["hopfield", "discrete"],    # CHANGED: circuit mode first
}
# Default substrate per architecture (used for filename suffix logic).
_DEFAULT_SUBSTRATE = {name: subs[0] for name, subs in _ALL_SUBSTRATES_BY_NAME.items()}


def sweep_one(
    name: str,
    module,
    n_trials: int = 50,
    force: bool = False,
    analog_substrate: str = "classic",
    analog_domain: str = "conservative",
    sigma_values: list | None = None,
    adc_sigma: float = 0.05,
    adc_bits: list | None = None,
    adc_only: bool = False,
) -> None:
    sigma_values = sigma_values if sigma_values is not None else _SIGMA_VALUES
    adc_bits = adc_bits if adc_bits is not None else _BIT_VALUES

    ckpt_path = str(_CKPT_DIR / f"{name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[{name}] No checkpoint found. Run train_all.py first.")
        return

    # Suffix for result filenames:
    #   domain: "" for conservative (default), "_full_analog" for full_analog
    #   substrate: "" for the default substrate of this architecture, else "_{substrate}"
    domain_suffix = "" if analog_domain == "conservative" else f"_{analog_domain}"
    default_sub = _DEFAULT_SUBSTRATE.get(name, "classic")
    substrate_suffix = "" if analog_substrate == default_sub else f"_{analog_substrate}"
    suffix = domain_suffix + substrate_suffix

    result_path = _RESULTS_DIR / f"{name}_mismatch{suffix}.json"
    if not adc_only and result_path.exists() and not force:
        print(f"[{name}] Results exist ({analog_domain}/{analog_substrate}), skipping. (--force to re-run)")
        return

    print(f"\n{'='*50}")
    print(f"Sweeping {module.get_family_name()} ({name}), n_trials={n_trials}, domain={analog_domain}, substrate={analog_substrate}")
    if adc_only:
        print(f"  ADC-only mode: adc_sigma={adc_sigma}, bits={adc_bits}")
    print(f"{'='*50}")

    model = module.load_model(ckpt_path)
    model = model.to(_DEVICE)
    print(f"  Device: {next(iter(model.parameters())).device}")
    if name in _SUBSTRATE_AWARE:
        eval_fn = lambda m, _s=analog_substrate: module.evaluate(m, analog_substrate=_s)
    else:
        eval_fn = module.evaluate

    # Get calibration data (small batch of inputs)
    if hasattr(module, '_get_data'):
        data = module._get_data()
        # Handle different return formats
        if len(data) == 4:  # X_train, y_train, X_test, y_test
            calib_data = data[0][:32]
        elif len(data) == 2 and isinstance(data[0], tuple):
            calib_data = data[0][0][:32]
        else:  # (X_train, X_test) or similar
            calib_data = data[0][:32] if isinstance(data[0], torch.Tensor) else data[0][0][:32]
    else:
        calib_data = None
        print(f"  Warning: No _get_data() found, skipping V_ref calibration")

    # Time-dependent models need (t, x) call signature — skip output V_ref calibration.
    # Their first-layer input is z0 ~ N(0, I); set v_ref_input = 3.3 (3-sigma + 10%
    # headroom) so the input DAC doesn't clip valid Gaussian starting points.
    _GAUSSIAN_INPUT_MODELS = {"neural_ode", "diffusion", "flow"}
    calib_data_to_use = None if name in _GAUSSIAN_INPUT_MODELS else calib_data
    if calib_data_to_use is not None:
        calib_data_to_use = calib_data_to_use.to(_DEVICE)

    # Models whose first-layer input is z0 ~ N(0, I): set v_ref_input = 3.3
    # (3-sigma of standard normal + 10% headroom) so the input DAC doesn't clip
    # valid Gaussian starting points. Calibration is skipped for these models
    # (time-dependent call signature), so without this the default v_ref=1.0
    # would clip ~32% of samples and corrupt the generation baseline.
    gaussian_kwargs = {"v_ref_input": 3.3} if name in _GAUSSIAN_INPUT_MODELS else {}

    # Hardware profile for energy/latency estimation
    profile = HardwareProfile()

    if not adc_only:
        # 1. Mismatch sweep
        print(f"\n[{name}] Mismatch sweep ({analog_domain})...")
        t0 = time.time()
        result = mismatch_sweep(model, eval_fn, sigma_values=sigma_values, n_trials=n_trials,
                                calibration_data=calib_data_to_use, analog_domain=analog_domain,
                                hardware_profile=profile, **gaussian_kwargs)
        result.save(str(_RESULTS_DIR / f"{name}_mismatch{suffix}.json"))
        print(f"  Done in {time.time()-t0:.0f}s. Threshold@10%: {result.degradation_threshold():.3f}")
        if result.energy_saving_vs_digital is not None:
            print(f"  Energy saving vs digital: {result.energy_saving_vs_digital*100:.1f}%")
        if result.speedup_vs_digital is not None:
            print(f"  Speedup vs digital: {result.speedup_vs_digital:.1f}x")

        # 2. Ablation sweep
        print(f"\n[{name}] Ablation sweep ({analog_domain})...")
        t0 = time.time()
        ablation = ablation_sweep(model, eval_fn, sigma_values=sigma_values,
                                  n_trials=max(10, n_trials//5), calibration_data=calib_data_to_use,
                                  analog_domain=analog_domain, hardware_profile=profile, **gaussian_kwargs)
        for noise_type, res in ablation.items():
            res.save(str(_RESULTS_DIR / f"{name}_ablation_{noise_type}{suffix}.json"))
        print(f"  Done in {time.time()-t0:.0f}s")

    # 3. ADC sweep
    print(f"\n[{name}] ADC precision sweep ({analog_domain}, sigma_mismatch={adc_sigma})...")
    t0 = time.time()
    adc_result = adc_sweep(model, eval_fn, bit_values=adc_bits, sigma_mismatch=adc_sigma,
                           n_trials=max(20, n_trials//2), calibration_data=calib_data_to_use,
                           analog_domain=analog_domain, hardware_profile=profile, **gaussian_kwargs)
    adc_result.save(str(_RESULTS_DIR / f"{name}_adc{suffix}.json"))
    print(f"  Done in {time.time()-t0:.0f}s")
    if adc_result.energy_saving_vs_digital is not None:
        print(f"  Energy saving vs digital: {adc_result.energy_saving_vs_digital*100:.1f}%")
    if adc_result.speedup_vs_digital is not None:
        print(f"  Speedup vs digital: {adc_result.speedup_vs_digital:.1f}x")

    # 4. Output MSE sweep (direct corruption measurement)
    # Only run for conservative domain (MSE is domain-agnostic: it measures
    # output divergence regardless of quantization profile)
    if not adc_only and analog_domain == "conservative":
        print(f"\n[{name}] Output MSE sweep...")
        t0 = time.time()
        import numpy as np

        # Load digital baseline
        digital_model = module.load_model(ckpt_path)

        # MSE sweep: compare analog vs digital outputs at each sigma
        per_trial_mse = np.zeros((len(sigma_values), n_trials), dtype=np.float64)

        for i, sigma in enumerate(sigma_values):
            for j in range(n_trials):
                analog_model = analogize(digital_model, sigma_mismatch=sigma, n_adc_bits=8,
                                         **gaussian_kwargs)
                configure_analog_profile(analog_model, analog_domain)
                if calib_data_to_use is not None:
                    from neuro_analog.simulator import calibrate_analog_model
                    calibrate_analog_model(analog_model, calib_data_to_use)
                resample_all_mismatch(analog_model, sigma=sigma)

                # Call evaluate_output_mse
                if name in _SUBSTRATE_AWARE:
                    mse_val = module.evaluate_output_mse(analog_model, digital_model, analog_substrate=analog_substrate)
                else:
                    mse_val = module.evaluate_output_mse(analog_model, digital_model)
                per_trial_mse[i, j] = mse_val

            if i % 3 == 0:
                print(f"  sigma={sigma:.3f}: MSE={per_trial_mse[i].mean():.6f} ± {per_trial_mse[i].std():.6f}")

        # Save MSE results
        from neuro_analog.simulator import SweepResult
        mse_result = SweepResult(
            sigma_values=sigma_values,
            metric_name="output_mse",
            per_trial=per_trial_mse,
            digital_baseline=0.0,  # MSE at sigma=0 is baseline
        )
        mse_result.save(str(_RESULTS_DIR / f"{name}_output_mse.json"))
        print(f"  Done in {time.time()-t0:.0f}s")

    # 5. DEQ-specific: convergence failure rate
    if not adc_only and name == "deq" and analog_domain == "conservative" and hasattr(module, "evaluate_convergence_failure"):
        print(f"\n[DEQ] Convergence failure rate sweep...")
        failure_rates = []
        from models.deq import _get_data
        (_, _), (X_test, _) = _get_data()
        for sigma in sigma_values:
            rates = []
            for _ in range(max(10, n_trials // 5)):
                analog_model = analogize(model, sigma_mismatch=sigma)
                configure_analog_profile(analog_model, analog_domain)
                # DEQ model retains convergence_failure_rate if it's a _DEQClassifier
                # wrapped by analogize, it loses the method. Check inner model.
                rate = module.evaluate_convergence_failure(analog_model)
                rates.append(rate)
            failure_rates.append(float(sum(rates) / len(rates)))
            print(f"  sigma={sigma:.3f}: failure_rate={failure_rates[-1]:.3f}")

        deq_extra = {
            "sigma_values": sigma_values,
            "convergence_failure_rate": failure_rates,
        }
        with open(_RESULTS_DIR / "deq_convergence.json", "w") as f:
            json.dump(deq_extra, f, indent=2)
        print(f"  DEQ convergence data saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--analog-substrate", type=str, default="classic",
        choices=["classic", "euler", "discrete", "cld", "extropic_dtm", "rc_integrator", "hopfield", "all"],
        help=(
            "Analog integration substrate. 'all' expands per-architecture to all supported substrates. "
            "diffusion: classic|cld|extropic_dtm. "
            "neural_ode/flow: euler|rc_integrator. "
            "deq: discrete|hopfield. "
            "transformer/ebm/ssm: substrate-agnostic (always run once under their default)."
        ),
    )
    parser.add_argument(
        "--analog-domain", type=str, default="conservative",
        choices=["conservative", "full_analog", "both"],
        help=(
            "conservative: ADC at every layer (upper bound on quantization sensitivity). "
            "full_analog: ADC only at final readout layer (lower bound). "
            "both: run both profiles and save separate result files."
        ),
    )
    parser.add_argument(
        "--sigma-values", type=str, default=None,
        help=(
            "Comma-separated list of sigma values to sweep, e.g. '0.0,0.05,0.10,0.20,0.30'. "
            "Default: 0.0,0.01,0.02,0.03,0.05,0.07,0.10,0.12,0.15"
        ),
    )
    parser.add_argument(
        "--adc-sigma", type=float, default=0.05,
        help="Mismatch sigma used during ADC precision sweep. Set to 0.0 for pure quantization isolation.",
    )
    parser.add_argument(
        "--adc-bits", type=str, default=None,
        help=(
            "Comma-separated list of ADC bit widths to sweep, e.g. '2,4,6,8,10,12,16,20,32'. "
            "Default: 2,4,6,8,10,12,16"
        ),
    )
    parser.add_argument(
        "--adc-only", action="store_true",
        help="Skip mismatch/ablation/MSE sweeps — run only the ADC precision sweep.",
    )
    args = parser.parse_args()

    domains = ["conservative", "full_analog"] if args.analog_domain == "both" else [args.analog_domain]

    sigma_values = (
        [float(x) for x in args.sigma_values.split(",")]
        if args.sigma_values else None
    )
    adc_bits = (
        [int(x) for x in args.adc_bits.split(",")]
        if args.adc_bits else None
    )

    # "all" expands per-architecture to all supported substrates.
    # Substrate-agnostic architectures (transformer, ebm, ssm) always run once under their default.
    def get_substrates(arch_name: str) -> list[str]:
        if args.analog_substrate == "all":
            return _ALL_SUBSTRATES_BY_NAME.get(arch_name, [_DEFAULT_SUBSTRATE.get(arch_name, "classic")])
        return [args.analog_substrate]

    total_t0 = time.time()
    for domain in domains:
        print(f"\n{'#'*60}")
        print(f"# Analog domain: {domain}")
        print(f"{'#'*60}")
        for name, module in _MODELS:
            if args.only and name != args.only:
                continue
            substrates = get_substrates(name)
            for substrate in substrates:
                sweep_one(name, module, n_trials=args.n_trials, force=args.force,
                          analog_substrate=substrate, analog_domain=domain,
                          sigma_values=sigma_values, adc_sigma=args.adc_sigma,
                          adc_bits=adc_bits, adc_only=args.adc_only)

    print(f"\nAll sweeps done in {time.time()-total_t0:.0f}s")
    print(f"Results in: {_RESULTS_DIR}")


if __name__ == "__main__":
    main()
