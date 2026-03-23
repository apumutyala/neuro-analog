#!/usr/bin/env python3
"""
Export a trained PyTorch model to Ark's BaseAnalogCkt format.

Ark (github.com/WangYuNeng/Ark) is a compiler for analog compute circuits.
This script shows the minimal path from a trained Neural ODE or SSM to a
running BaseAnalogCkt subclass that Ark can compile.

Unlike 03_ark_pipeline.py, this script has no dependency on the cross-arch
experiment directory -- it builds a small demo model from scratch and exports it.
Follow the same pattern with your own trained model.

Prerequisites:
    git clone https://github.com/WangYuNeng/Ark
    pip install -e ./Ark
    pip install jax diffrax equinox lineax

Usage:
    python examples/04_ark_export.py               # Neural ODE (default)
    python examples/04_ark_export.py --arch ssm    # SSM
    python examples/04_ark_export.py --arch deq    # DEQ

Output:
    outputs/neural_ode_ark.py  -- BaseAnalogCkt subclass, run directly to verify
    outputs/ssm_ark.py
    outputs/deq_ark.py
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

_OUT = _ROOT / "outputs"


def sep(title=""):
    w = 62
    print(f"\n{'='*w}")
    if title:
        print(f"  {title}")
        print(f"{'='*w}")


# ── Check Ark is installed ─────────────────────────────────────────────────────

def check_ark():
    try:
        from ark.optimization.base_module import BaseAnalogCkt, TimeInfo
        return BaseAnalogCkt, TimeInfo
    except ImportError:
        print("ERROR: Ark is not installed.")
        print()
        print("  git clone https://github.com/WangYuNeng/Ark")
        print("  pip install -e ./Ark")
        print("  pip install jax diffrax equinox lineax")
        sys.exit(1)


# ── Neural ODE export ──────────────────────────────────────────────────────────

def export_neural_ode(sigma: float, out_path: Path) -> None:
    from neuro_analog.extractors.neural_ode import NeuralODEExtractor, export_neural_ode_to_ark

    sep("Neural ODE -> Ark BaseAnalogCkt")

    # Build a demo extractor (2D state, time-augmented 3-layer tanh MLP).
    # For your own model: use NeuralODEExtractor.from_module(f_theta, state_dim=...)
    extractor = NeuralODEExtractor.demo(state_dim=2, hidden_dim=20, num_layers=3)
    extractor.load_model()
    profile = extractor.run()

    print(f"  Architecture:       Neural ODE  dx/dt = f_theta(x, t)")
    print(f"  State dim:          {profile.dynamics.state_dimension}")
    print(f"  Analog FLOP share:  {profile.analog_flop_fraction*100:.0f}%")
    print(f"  D/A boundaries:     {profile.da_boundary_count}  (DAC in, ADC out)")
    print(f"  Mismatch sigma:     {sigma}")
    print()

    # Generate the BaseAnalogCkt subclass
    code = export_neural_ode_to_ark(extractor, out_path, mismatch_sigma=sigma)
    print(f"  Written: {out_path}  ({len(code.splitlines())} lines)")

    # Verify it runs against Ark
    _verify(out_path, "NeuralODEAnalogCkt", state_dim=2)


# ── SSM export ─────────────────────────────────────────────────────────────────

class _TinySSM(nn.Module):
    """Minimal Mamba-style SSM for demo export (no diffusers/mamba_ssm dependency)."""
    def __init__(self, d_model: int = 32, d_state: int = 8, expand: int = 1):
        super().__init__()
        d_inner = d_model * expand
        self.A_log   = nn.Parameter(torch.randn(d_inner, d_state))
        self.x_proj  = nn.Linear(d_inner, 2 * d_state)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.d_model  = d_model
        self.d_state  = d_state


def export_ssm(sigma: float, out_path: Path) -> None:
    from neuro_analog.extractors.ssm import MambaExtractor
    from neuro_analog.ir.ark_export import export_ssm_to_ark

    sep("SSM -> Ark BaseAnalogCkt")

    d_state = 8
    model = _TinySSM(d_model=32, d_state=d_state)
    model.eval()

    extractor = MambaExtractor(model_name="demo_ssm", device="cpu")
    extractor.model = model
    dyn = extractor.extract_dynamics()
    graph = extractor.build_graph()
    graph.set_dynamics(dyn)   # build_graph() doesn't set _dynamics — wire it manually
    # Skip extractor.run() — MambaExtractor.load_model() always hits HuggingFace.
    # Print known demo dimensions directly.
    print(f"  Architecture:       Diagonal S4D SSM  dx/dt = A*x + B*u")
    print(f"  State dim:          {d_state}  (demo: d_state={d_state})")
    print(f"  Mismatch sigma:     {sigma}")
    print()

    code = export_ssm_to_ark(graph, extractor, out_path, mismatch_sigma=sigma)
    print(f"  Written: {out_path}  ({len(code.splitlines())} lines)")

    _verify(out_path, "SSMAnalogCkt", state_dim=8)


# ── DEQ export ─────────────────────────────────────────────────────────────────

def export_deq(sigma: float, out_path: Path) -> None:
    from neuro_analog.extractors.deq import DEQExtractor
    from neuro_analog.ir.ark_export import export_deq_to_ark

    sep("DEQ -> Ark BaseAnalogCkt")

    extractor = DEQExtractor.reference(z_dim=64, x_dim=64, hidden_dim=128)
    extractor.load_model()
    graph = extractor.build_graph()
    profile = extractor.run()

    print(f"  Architecture:       DEQ  dz/dt = f_theta(z, x) - z")
    print(f"  State dim:          {profile.dynamics.state_dimension}  (augmented z+x = {profile.dynamics.state_dimension + 64})")
    print(f"  Analog FLOP share:  {profile.analog_flop_fraction*100:.0f}%")
    print(f"  D/A boundaries:     {profile.da_boundary_count}")
    print(f"  Mismatch sigma:     {sigma}")
    print()

    code = export_deq_to_ark(graph, out_path, sigma=sigma)
    print(f"  Written: {out_path}  ({len(code.splitlines())} lines)")

    # DEQ uses augmented state y = concat([z0, x_input])
    _verify(out_path, "DEQAnalogCkt", state_dim=64 + 64, readout_dim=64)


# ── Ark verification ───────────────────────────────────────────────────────────

def _verify(path: Path, class_name: str, state_dim: int, readout_dim: int = None) -> None:
    """Load the generated file and run a forward pass through Ark's __call__."""
    import importlib.util
    import jax.numpy as jnp
    import diffrax
    from ark.optimization.base_module import BaseAnalogCkt, TimeInfo

    if readout_dim is None:
        readout_dim = state_dim

    spec = importlib.util.spec_from_file_location(class_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ckt = getattr(mod, class_name)()

    assert issubclass(type(ckt), BaseAnalogCkt), f"{class_name} is not a BaseAnalogCkt subclass"

    print(f"  BaseAnalogCkt:      {issubclass(type(ckt), BaseAnalogCkt)}")
    print(f"  a_trainable shape:  {ckt.a_trainable.shape}")
    print(f"  is_stochastic:      {ckt.is_stochastic}")
    print(f"  solver:             {type(ckt.solver).__name__}")

    # Forward pass: TimeInfo.saveat takes a bare jnp.array of time points,
    # NOT a diffrax.SaveAt object. switch must be jnp.array([]), not False.
    y0 = jnp.zeros(state_dim)
    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.1, saveat=jnp.array([1.0]))
    switch = jnp.array([])

    result = ckt(time_info, y0, switch=switch, args_seed=42, noise_seed=43)
    assert result.shape == (readout_dim,), f"Expected ({readout_dim},), got {result.shape}"

    print(f"  Output shape:       {result.shape}  OK")
    print(f"\n  Ark export verified: {path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(arch: str = "neural_ode", sigma: float = 0.05) -> None:
    _OUT.mkdir(exist_ok=True)
    check_ark()

    if arch == "neural_ode":
        export_neural_ode(sigma, _OUT / "neural_ode_ark.py")
    elif arch == "ssm":
        export_ssm(sigma, _OUT / "ssm_ark.py")
    elif arch == "deq":
        export_deq(sigma, _OUT / "deq_ark.py")
    else:
        print(f"Unknown arch: {arch}. Choose from: neural_ode, ssm, deq")
        sys.exit(1)

    sep()
    print("  Using your own trained model:")
    print()
    print("  Neural ODE:")
    print("    from neuro_analog.extractors.neural_ode import NeuralODEExtractor, export_neural_ode_to_ark")
    print("    extractor = NeuralODEExtractor.from_module(your_f_theta, state_dim=N)")
    print("    extractor.load_model()")
    print("    export_neural_ode_to_ark(extractor, 'my_model_ark.py', mismatch_sigma=0.05)")
    print()
    print("  SSM:")
    print("    from neuro_analog.extractors.ssm import MambaExtractor")
    print("    from neuro_analog.ir.ark_export import export_ssm_to_ark")
    print("    extractor = MambaExtractor(model_name='my_ssm', device='cpu')")
    print("    extractor.model = your_ssm_module")
    print("    extractor.extract_dynamics()")
    print("    graph = extractor.build_graph()")
    print("    export_ssm_to_ark(graph, extractor, 'my_ssm_ark.py', mismatch_sigma=0.05)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="neural_ode",
                        choices=["neural_ode", "ssm", "deq"],
                        help="Architecture to export (default: neural_ode)")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Mismatch sigma for the export (default: 0.05)")
    args = parser.parse_args()
    main(args.arch, args.sigma)
