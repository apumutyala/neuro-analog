"""
Unified execution pipeline for cross-architecture analog tolerance profiling.

Extracts hardware-specific ODE configurations (for Neural ODE/SSMs) or graph-level constraints
(for others), runs mismatch and ablation sweeps, and emits Shem-compatible JAX configurations.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional
import torch

from neuro_analog.ir import AnalogGraph, ODESystem
from neuro_analog.simulator import mismatch_sweep, adc_sweep, analogize, resample_all_mismatch

@dataclass
class PipelineResult:
    model_name: str
    family: str
    graph: Optional[AnalogGraph]
    ode_system: Optional[ODESystem]
    sweep_results: dict[str, Any]
    shem_export_path: Optional[str]


def run_pipeline(
    model,
    family: str,
    extractor_class,
    eval_fn,
    calibration_data=None,
    model_name: str = "model",
    device: str = "cpu",
    output_dir: str = "outputs",
    n_trials: int = 50,
) -> PipelineResult:
    """Run extraction, simulation sweeps, and export for a model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\\n--- PIPELINE START: {model_name} ({family}) ---")
    
    # 1. Extraction
    print("[1/3] Extracting Intermediate Representation...")
    extractor = extractor_class.from_module(model, state_dim=getattr(model, 'state_dim', 2)) if hasattr(extractor_class, 'from_module') else extractor_class(model_name)
    if not hasattr(extractor_class, 'from_module'):
        extractor.model = model
        
    profile = extractor.run()
    graph = extractor.graph
    
    ode_system = None
    if family in ["neural_ode", "ssm"]:
        ode_system = extractor.extract_ode_system()
        print(f"      Extracted ODESystem with {ode_system.parameter_count} parameters.")

    # 2. Sweeps
    print(f"[2/3] Simulating Mismatch (n={n_trials})...")
    sigma_values = [0.0, 0.05, 0.10, 0.15]
    
    # For ODE/SSM, sweep logic uses ODESystem resampling internally 
    # (The sweep.py will be modified to check for this or we do it here)
    mismatch_res = mismatch_sweep(
        model, eval_fn, sigma_values=sigma_values, n_trials=n_trials, 
        calibration_data=calibration_data
    )
    
    sweep_results = {"mismatch": mismatch_res.to_dict()}
    
    # 3. Shem Export
    print("[3/3] Exporting to Shem JAX representation...")
    shem_path = None
    if family == "neural_ode":
        from neuro_analog.ir.shem_export import export_neural_ode_to_shem
        shem_path = str(output_path / f"{model_name}_shem.py")
        export_neural_ode_to_shem(extractor, shem_path, mismatch_sigma=0.05)
    elif family == "ssm":
        from neuro_analog.ir.shem_export import export_ssm_to_shem
        shem_path = str(output_path / f"{model_name}_shem.py")
        export_ssm_to_shem(graph, extractor, shem_path, mismatch_sigma=0.05)

    print(f"--- PIPELINE DONE ---")
    return PipelineResult(
        model_name=model_name,
        family=family,
        graph=graph,
        ode_system=ode_system,
        sweep_results=sweep_results,
        shem_export_path=shem_path
    )
