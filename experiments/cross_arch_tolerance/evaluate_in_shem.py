#!/usr/bin/env python3
"""
Evaluate continuous models (SSM, Neural ODE) natively in JAX/Diffrax.

Implements the end-to-end Shem-compatible evaluation workflow:
1. Extract the PyTorch model into an ODESystem (neuro-analog IR).
2. Export the ODE system to a Shem-compatible JAX file (shem_export.py).
3. Evaluate the generated JAX file natively, as it would run under
   the Shem compiler or on analog hardware.

Usage:
    python experiments/cross_arch_tolerance/evaluate_in_shem.py --model ssm
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import torch

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from models import neural_ode, ssm
from neuro_analog.ir.shem_export import export_ssm_to_shem, export_neural_ode_to_shem


def load_generated_module(module_name: str, file_path: Path):
    """Dynamically load the generated JAX Python file."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_shem_evaluation(model_type: str, n_trials: int = 5):
    print(f"\n{'='*50}")
    print(f"Executing Shem-Native Evaluation for: {model_type.upper()}")
    print(f"{'='*50}")
    
    ckpt_dir = _ROOT / "experiments" / "cross_arch_tolerance" / "checkpoints"
    ckpt_path = ckpt_dir / f"{model_type}.pt"
    
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found. Run train_all.py first.")
        return

    # 1. Load PyTorch Digital Model & Extract IR
    print("[1/4] Loading PyTorch model and extracting IR...")
    if model_type == "ssm":
        model_module = ssm
        model = model_module.load_model(str(ckpt_path))
        from neuro_analog.extractors.ssm import MambaExtractor
        extractor = MambaExtractor.from_module(model, state_dim=16)
        extractor.run()
        export_fn = export_ssm_to_shem
    elif model_type == "neural_ode":
        model_module = neural_ode
        model = model_module.load_model(str(ckpt_path))
        from neuro_analog.extractors.neural_ode import NeuralODEExtractor
        extractor = NeuralODEExtractor.from_module(model, state_dim=2)
        extractor.run()
        export_fn = export_neural_ode_to_shem
    else:
        raise ValueError(f"Unsupported continuous model: {model_type}")

    # 2. Export to Shem JAX Format
    print("[2/4] Exporting to Shem (JAX/Diffrax) format...")
    export_dir = _ROOT / "experiments" / "cross_arch_tolerance" / "shem_exports"
    export_dir.mkdir(exist_ok=True)
    
    export_path = export_dir / f"{model_type}_shem_ckt.py"
    try:
        export_fn(extractor.graph, extractor=extractor, output_path=export_path, mismatch_sigma=0.05)
    except TypeError:
        # Neural ODE export signature is slightly different
        export_fn(extractor=extractor, output_path=export_path, mismatch_sigma=0.05)
        
    print(f"      -> Generated {export_path}")

    # 3. Load Generated JAX Code
    print("[3/4] Compiling JAX Simulation Environment...")
    shem_module = load_generated_module(f"{model_type}_shem", export_path)
    
    # Instantiate the circuit class
    if model_type == "ssm":
        ckt = shem_module.SSMAnalogCkt()
        state_dim = extractor.graph._dynamics.state_dimension or 16
    elif model_type == "neural_ode":
        ckt = shem_module.NeuralODEAnalogCkt()
        state_dim = extractor.state_dim

    # 4. Run Evaluation Sweep in JAX (Mimicking Shem Mismatch)
    print(f"[4/4] Running {n_trials} mismatch trials natively in JAX...")
    
    # We use a dummy input for demonstration (e.g. zeros, or data from test set)
    # In a real pipeline, we'd feed the actual test set through.
    x0 = jnp.zeros((state_dim,))
    
    # JIT compile the solve method for speed
    solve_jit = jax.jit(ckt.solve, static_argnums=(1,))
    
    # Warmup
    _ = solve_jit(x0, seed=0)
    
    t0 = time.time()
    for trial in range(n_trials):
        # We vary the seed, which changes the parameter mismatch sampling
        # inside the generated make_args() function.
        out = solve_jit(x0, seed=trial)
        print(f"      Trial {trial+1:02d} | Seed {trial} | Output Norm: {jnp.linalg.norm(out):.6f}")

    print(f"      -> {n_trials} trials completed in {time.time() - t0:.3f}s")
    print("\nSUCCESS: Model successfully bypassed PyTorch discrete simulation and ran purely via JAX/Diffrax!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["ssm", "neural_ode", "all"], default="all")
    parser.add_argument("--n-trials", type=int, default=5)
    args = parser.parse_args()

    models_to_run = ["ssm", "neural_ode"] if args.model == "all" else [args.model]
    
    for m in models_to_run:
        run_shem_evaluation(m, args.n_trials)
