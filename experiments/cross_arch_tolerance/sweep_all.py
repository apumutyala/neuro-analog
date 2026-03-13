#!/usr/bin/env python3
"""Run mismatch sweeps across all 7 architecture families.

For each family:
  1. Load pretrained model
  2. Run mismatch_sweep (sigma: 0.0 → 0.15, 50 trials)
  3. Run ablation_sweep (mismatch-only, thermal-only, quantization-only)
  4. Run adc_sweep (bits: 2 → 16 at sigma=0.05, 50 trials)
  5. For DEQ: also track convergence failure rate vs sigma

Results saved as JSON to results/ directory.

Usage:
    python experiments/cross_arch_tolerance/sweep_all.py
    python experiments/cross_arch_tolerance/sweep_all.py --only neural_ode
    python experiments/cross_arch_tolerance/sweep_all.py --n-trials 20  # faster
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

from models import neural_ode, transformer, diffusion, flow, ebm, deq, ssm
from neuro_analog.simulator import mismatch_sweep, adc_sweep, ablation_sweep, resample_all_mismatch, set_all_noise, analogize

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


def sweep_one(name: str, module, n_trials: int = 50, force: bool = False, analog_substrate: str = "classic") -> None:
    ckpt_path = str(_CKPT_DIR / f"{name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[{name}] No checkpoint found. Run train_all.py first.")
        return

    result_path = _RESULTS_DIR / f"{name}_mismatch.json"
    if result_path.exists() and not force:
        print(f"[{name}] Results exist, skipping. (--force to re-run)")
        return

    print(f"\n{'='*50}")
    print(f"Sweeping {module.get_family_name()} ({name}), n_trials={n_trials}")
    print(f"{'='*50}")

    model = module.load_model(ckpt_path)
    if name == "diffusion":
        eval_fn = lambda m: module.evaluate(m, analog_substrate=analog_substrate)
    else:
        eval_fn = module.evaluate

    # Get calibration data (small batch of inputs)
    import torch
    if hasattr(module, '_get_data'):
        data = module._get_data()
        # Handle different return formats
        if len(data) == 4:  # X_train, y_train, X_test, y_test
            calib_data = data[0][:32]  # First 32 training samples
        elif len(data) == 2 and isinstance(data[0], tuple):  # ((X_train, y_train), (X_test, y_test))
            calib_data = data[0][0][:32]
        else:  # (X_train, X_test) or similar
            calib_data = data[0][:32] if isinstance(data[0], torch.Tensor) else data[0][0][:32]
    else:
        calib_data = None
        print(f"  Warning: No _get_data() found, skipping V_ref calibration")

    # Time-dependent models (Neural ODE, Diffusion, Flow) need (t, x) signature
    # Skip calibration for them (V_ref=1.0 default is fine for normalized data)
    calib_data_to_use = None if name in ["neural_ode", "diffusion", "flow"] else calib_data
    
    # 1. Mismatch sweep
    print(f"\n[{name}] Mismatch sweep...")
    t0 = time.time()
    result = mismatch_sweep(model, eval_fn, sigma_values=_SIGMA_VALUES, n_trials=n_trials, calibration_data=calib_data_to_use)
    result.save(str(_RESULTS_DIR / f"{name}_mismatch.json"))
    print(f"  Done in {time.time()-t0:.0f}s. Threshold@10%: {result.degradation_threshold():.3f}")

    # 2. Ablation sweep
    print(f"\n[{name}] Ablation sweep (3 noise sources)...")
    t0 = time.time()
    ablation = ablation_sweep(model, eval_fn, sigma_values=_SIGMA_VALUES, n_trials=max(10, n_trials//5), calibration_data=calib_data_to_use)
    for noise_type, res in ablation.items():
        res.save(str(_RESULTS_DIR / f"{name}_ablation_{noise_type}.json"))
    print(f"  Done in {time.time()-t0:.0f}s")

    # 3. ADC sweep
    print(f"\n[{name}] ADC precision sweep...")
    t0 = time.time()
    adc_result = adc_sweep(model, eval_fn, bit_values=_BIT_VALUES, sigma_mismatch=0.05, n_trials=max(20, n_trials//2), calibration_data=calib_data_to_use)
    adc_result.save(str(_RESULTS_DIR / f"{name}_adc.json"))
    print(f"  Done in {time.time()-t0:.0f}s")

    # 4. Output MSE sweep (direct corruption measurement)
    print(f"\n[{name}] Output MSE sweep...")
    t0 = time.time()
    import torch
    import numpy as np
    
    # Load digital baseline
    digital_model = module.load_model(ckpt_path)
    
    # MSE sweep: compare analog vs digital outputs at each sigma
    per_trial_mse = np.zeros((len(_SIGMA_VALUES), n_trials), dtype=np.float64)
    
    for i, sigma in enumerate(_SIGMA_VALUES):
        for j in range(n_trials):
            analog_model = analogize(digital_model, sigma_mismatch=sigma, n_adc_bits=8)
            if calib_data_to_use is not None:
                from neuro_analog.simulator import calibrate_analog_model
                calibrate_analog_model(analog_model, calib_data_to_use)
            resample_all_mismatch(analog_model, sigma=sigma)
            
            # Call evaluate_output_mse
            if name == "diffusion":
                mse_val = module.evaluate_output_mse(analog_model, digital_model, analog_substrate=analog_substrate)
            else:
                mse_val = module.evaluate_output_mse(analog_model, digital_model)
            per_trial_mse[i, j] = mse_val
        
        if i % 3 == 0:
            print(f"  sigma={sigma:.3f}: MSE={per_trial_mse[i].mean():.6f} ± {per_trial_mse[i].std():.6f}")
    
    # Save MSE results
    from neuro_analog.simulator import SweepResult
    mse_result = SweepResult(
        sigma_values=_SIGMA_VALUES,
        metric_name="output_mse",
        per_trial=per_trial_mse,
        digital_baseline=0.0,  # MSE at sigma=0 is baseline
    )
    mse_result.save(str(_RESULTS_DIR / f"{name}_output_mse.json"))
    print(f"  Done in {time.time()-t0:.0f}s")
    
    # 5. DEQ-specific: convergence failure rate
    if name == "deq" and hasattr(module, "evaluate_convergence_failure"):
        print(f"\n[DEQ] Convergence failure rate sweep...")
        failure_rates = []
        from models.deq import _get_data
        (_, _), (X_test, _) = _get_data()
        for sigma in _SIGMA_VALUES:
            rates = []
            for _ in range(max(10, n_trials // 5)):
                analog_model = analogize(model, sigma_mismatch=sigma)
                # DEQ model retains convergence_failure_rate if it's a _DEQClassifier
                # wrapped by analogize, it loses the method. Check inner model.
                rate = module.evaluate_convergence_failure(analog_model)
                rates.append(rate)
            failure_rates.append(float(sum(rates) / len(rates)))
            print(f"  sigma={sigma:.3f}: failure_rate={failure_rates[-1]:.3f}")

        deq_extra = {
            "sigma_values": _SIGMA_VALUES,
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
    parser.add_argument("--analog-substrate", type=str, default="classic", choices=["classic", "cld", "extropic_dtm"])
    args = parser.parse_args()

    total_t0 = time.time()
    for name, module in _MODELS:
        if args.only and name != args.only:
            continue
        sweep_one(name, module, n_trials=args.n_trials, force=args.force, analog_substrate=args.analog_substrate)

    print(f"\nAll sweeps done in {time.time()-total_t0:.0f}s")
    print(f"Results in: {_RESULTS_DIR}")


if __name__ == "__main__":
    main()
