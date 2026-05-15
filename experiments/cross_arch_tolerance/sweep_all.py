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
import random
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

_CKPT_DIR = Path(__file__).parent / "checkpoints"
_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_DIR.mkdir(exist_ok=True)
_PROFILE_DIR = _RESULTS_DIR / "profiles"
_PROFILE_DIR.mkdir(exist_ok=True)

import torch
import torch.nn as nn
import numpy as np

from models import neural_ode, transformer, diffusion, flow, ebm, deq, ssm
from neuro_analog.simulator import (
    mismatch_sweep, adc_sweep, ablation_sweep, resample_all_mismatch,
    set_all_noise, analogize, configure_analog_profile, calibrate_analog_model,
    count_analog_vs_digital,
)
from neuro_analog.ir.energy_model import HardwareProfile
from neuro_analog.ir.types import AnalogAmenabilityProfile, ArchitectureFamily, DynamicsProfile
from neuro_analog.ir import AnalogGraph, AnalogNode, OpType, Domain
from neuro_analog.simulator.analog_linear import AnalogLinear
from neuro_analog.simulator.analog_activation import AnalogTanh, AnalogSigmoid, AnalogReLU, AnalogGELU, AnalogSiLU

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

_ARCH_FAMILY = {
    "neural_ode": ArchitectureFamily.NEURAL_ODE,
    "transformer": ArchitectureFamily.TRANSFORMER,
    "diffusion": ArchitectureFamily.DIFFUSION,
    "flow": ArchitectureFamily.FLOW,
    "ebm": ArchitectureFamily.EBM,
    "deq": ArchitectureFamily.DEQ,
    "ssm": ArchitectureFamily.SSM,
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _arch_seed(base_seed: int, name: str, offset: int = 0) -> int:
    arch_index = next((i for i, (arch, _) in enumerate(_MODELS) if arch == name), 0)
    return int(base_seed + arch_index * 10_000 + offset)


def _build_calibration_runner(name: str, module, calib_data, device):
    """Return model-specific calibration data/runner for the pilot models."""
    if calib_data is not None:
        calib_data = calib_data.to(device)

    if name == "neural_ode":
        X_train, _ = module._get_data()
        x_data = X_train[:32].to(device)
        x_noise = torch.randn_like(x_data)

        def runner(m):
            for t_val, x in ((0.0, x_noise), (0.5, x_data), (1.0, x_data)):
                t = torch.full((x.shape[0],), t_val, device=device)
                m(t, x)

        return None, runner, {"calibration": "runner: neural_ode(t,x), data+standard-normal states"}

    if name == "flow":
        X_train, _ = module._get_data()
        x1 = X_train[:32].to(device)
        x0 = torch.randn_like(x1)

        def runner(m):
            for t_val in (0.0, 0.5, 1.0):
                t = torch.full((x1.shape[0],), t_val, device=device)
                x_t = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
                m(t, x_t)

        return None, runner, {"calibration": "runner: flow(t,x_t), interpolated data/noise states"}

    if name == "diffusion":
        X_train, _ = module._get_data()
        x0 = X_train[:32].to(device)
        betas = module._get_betas()
        _, alphas_bar = module._get_alphas(betas)
        t_values = [0, len(betas) // 2, len(betas) - 1]

        def runner(m):
            for t_val in t_values:
                t_cpu = torch.full((x0.shape[0],), t_val, dtype=torch.long)
                t_dev = t_cpu.to(device)
                alpha_t = alphas_bar[t_cpu].unsqueeze(-1).to(device)
                noise = torch.randn_like(x0)
                x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
                m(x_t, t_dev)

        return None, runner, {"calibration": "runner: diffusion(x_t,t), early/mid/late noised states"}

    return calib_data, None, {"calibration": "tensor sample_input" if calib_data is not None else "none"}


def _write_pilot_profile(name: str, model: torch.nn.Module, suffix: str, analog_domain: str) -> Path:
    """Write a real pilot profile from the loaded checkpoint and analogized module coverage."""
    analog_model = analogize(model, sigma_mismatch=0.0, n_adc_bits=32)
    configure_analog_profile(analog_model, analog_domain)
    coverage = count_analog_vs_digital(analog_model)

    model_params = sum(p.numel() for p in model.parameters())
    analog_params = coverage["analog_params"]
    digital_params = max(model_params - analog_params, 0)
    counted_total = analog_params + digital_params
    analog_frac = analog_params / counted_total if counted_total else 0.0
    digital_frac = digital_params / counted_total if counted_total else 0.0
    layer_count = max(coverage["analog_layers"] + coverage["digital_layers"], 1)
    da_boundaries = coverage["analog_layers"] if analog_domain == "conservative" else min(coverage["analog_layers"], 1)

    dynamics = DynamicsProfile()
    if name == "neural_ode":
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="time_varying_ODE", num_function_evaluations=40, state_dimension=2)
    elif name == "flow":
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="time_varying_ODE", num_function_evaluations=100, state_dimension=2)
    elif name == "diffusion":
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="SDE", num_diffusion_steps=10, is_stochastic=True, state_dimension=64)
    elif name == "deq":
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="implicit_equilibrium", state_dimension=getattr(model, "z_dim", 64))
    elif name == "ssm":
        state_dim = sum(getattr(layer, "d_state", 0) for layer in getattr(model, "layers", []))
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="LTI_ODE", state_dimension=state_dim or None)
    elif name == "ebm":
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="energy_minimization", is_stochastic=True, state_dimension=getattr(model, "n_vis", 64))

    profile = AnalogAmenabilityProfile(
        architecture=_ARCH_FAMILY[name],
        model_name=name,
        model_params=model_params,
        analog_flop_fraction=analog_frac,
        digital_flop_fraction=digital_frac,
        hybrid_flop_fraction=0.0,
        da_boundary_count=da_boundaries,
        layer_count=layer_count,
        min_weight_precision_bits=8,
        min_activation_precision_bits=8,
        dynamics=dynamics,
    )
    profile.compute_scores()

    data = profile.to_dict()
    data["pilot_profile_source"] = {
        "checkpoint_model_params": model_params,
        "analogized_coverage": coverage,
        "analog_domain": analog_domain,
        "profile_basis": "loaded checkpoint + analogize() module coverage + pilot model constants",
    }
    out_path = _PROFILE_DIR / f"{name}_profile{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path




def _build_pilot_graph(name: str, analog_model: torch.nn.Module, analog_domain: str, profile: HardwareProfile) -> AnalogAmenabilityProfile:
    """Build a minimal AnalogGraph from the analogized pilot model and analyze it.

    The pilot models are small and sequential; we introspect the analogized
    modules to create AnalogNodes with FLOPs and seq_len set to the architecture
    iteration count (ODE steps, fixed-point iterations, diffusion steps, etc.).
    This replaces the naive param-count proxy in mismatch_sweep / adc_sweep.
    """
    ITER_COUNTS = {
        "neural_ode": 40,
        "transformer": 64,
        "diffusion": 10,
        "flow": 4,
        "ebm": 500,
        "deq": 30,
        "ssm": 64,
    }
    base_seq_len = ITER_COUNTS.get(name, 1)

    def _seq_len(node_name: str) -> int:
        # DEQ readout is evaluated once; the feedback loop is evaluated iter_count times
        if name == "deq" and "readout" in node_name:
            return 1
        return base_seq_len

    graph = AnalogGraph(
        name=f"{name}_pilot",
        family=_ARCH_FAMILY[name],
        model_params=sum(p.numel() for p in analog_model.parameters()),
    )

    prev_node_id = None
    # Collect leaf modules that should become graph nodes
    # We iterate named_modules() and identify leaves by checking if the module
    # is a known type we want to represent. We do NOT skip by children() because
    # analogize() replaces leaf modules in-place (the parent still lists the
    # replacement as a child). Instead we check: is this module one of the types
    # we care about?
    
    # Known analog types after analogize()
    _ANALOG_TYPES = (AnalogLinear, AnalogTanh, AnalogSigmoid, AnalogReLU, AnalogGELU, AnalogSiLU)
    
    for mod_name, module in analog_model.named_modules():
        seq_len = _seq_len(mod_name)
        node = None
        
        # Skip the top-level model itself
        if module is analog_model:
            continue

        # ── 1. Analog Linear / Crossbar (MVM) ─────────────────────────────
        # Check isinstance first (clean replacement), then duck-typing (monkey-patched)
        is_analog_linear = isinstance(module, AnalogLinear)
        if not is_analog_linear and hasattr(module, "W_nominal") and hasattr(module, "in_features"):
            is_analog_linear = True
            
        if is_analog_linear:
            # Monkey-patched nn.Linear may have in_features/out_features; AnalogLinear has them as attributes
            in_f = getattr(module, "in_features", getattr(module, "weight", torch.empty(0)).shape[1] if hasattr(module, "weight") else 1)
            out_f = getattr(module, "out_features", getattr(module, "weight", torch.empty(0)).shape[0] if hasattr(module, "weight") else 1)
            # Param count: W_nominal for AnalogLinear, weight for monkey-patched
            if hasattr(module, "W_nominal"):
                param_count = module.W_nominal.numel() + (module.bias.numel() if getattr(module, "bias", None) is not None else 0)
            else:
                param_count = sum(p.numel() for p in module.parameters())
            node = AnalogNode(
                name=mod_name,
                op_type=OpType.MVM,
                domain=Domain.ANALOG,
                input_shape=(in_f,),
                output_shape=(out_f,),
                weight_shape=(in_f, out_f),
                seq_len=seq_len,
                flops=seq_len * 2 * in_f * out_f,
                param_count=param_count,
            )
        # ── 2. Analog-native activations ──────────────────────────────────
        elif isinstance(module, (AnalogTanh, AnalogSigmoid, AnalogReLU)):
            # _BaseAnalogActivation has _n_features set lazily on first forward.
            # If not yet set, use in_features from parent Linear or default to 1.
            dim = getattr(module, "_n_features", None)
            if dim is None:
                # Try to infer from parent's output shape, or default safely
                dim = 1
            node = AnalogNode(
                name=mod_name,
                op_type=OpType.ANALOG_SIGMOID if isinstance(module, AnalogSigmoid) else 
                       (OpType.ANALOG_RELU if isinstance(module, AnalogReLU) else OpType.ANALOG_SIGMOID),
                domain=Domain.ANALOG,
                input_shape=(dim,),
                output_shape=(dim,),
                seq_len=seq_len,
                flops=seq_len * dim,
            )
        # ── 3. Digital-required activations (ADC→DAC crossing) ──────────────
        elif isinstance(module, (AnalogGELU, AnalogSiLU)):
            dim = getattr(module, "_n_features", None)
            if dim is None:
                dim = 1
            node = AnalogNode(
                name=mod_name,
                op_type=OpType.GELU if isinstance(module, AnalogGELU) else OpType.SILU,
                domain=Domain.DIGITAL,
                input_shape=(dim,),
                output_shape=(dim,),
                seq_len=seq_len,
                flops=seq_len * dim,
            )
        # ── 4. Normalization layers ─────────────────────────────────────────
        elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            dim = getattr(module, "normalized_shape", [getattr(module, "num_features", 1)])[0]
            node = AnalogNode(
                name=mod_name,
                op_type=OpType.LAYER_NORM,
                domain=Domain.DIGITAL,
                input_shape=(dim,),
                output_shape=(dim,),
                seq_len=seq_len,
                flops=seq_len * 5 * dim,
                param_count=sum(p.numel() for p in module.parameters()),
            )
        # ── 5. Softmax ──────────────────────────────────────────────────────
        elif isinstance(module, torch.nn.Softmax):
            # Softmax operates over the last dimension; infer from dim parameter
            softmax_dim = getattr(module, "dim", -1)
            # Default shape — we'll use (1,) as placeholder; the actual shape
            # depends on the input tensor which we don't have here.
            node = AnalogNode(
                name=mod_name,
                op_type=OpType.SOFTMAX,
                domain=Domain.DIGITAL,
                input_shape=(1,),
                output_shape=(1,),
                seq_len=seq_len,
                flops=seq_len * 10,
            )
        # ── 6. Embedding ────────────────────────────────────────────────────
        elif isinstance(module, torch.nn.Embedding):
            num_embeddings = getattr(module, "num_embeddings", 1)
            embedding_dim = getattr(module, "embedding_dim", 1)
            node = AnalogNode(
                name=mod_name,
                op_type=OpType.EMBEDDING,
                domain=Domain.DIGITAL,
                input_shape=(num_embeddings,),
                output_shape=(embedding_dim,),
                seq_len=seq_len,
                flops=0,
                param_count=sum(p.numel() for p in module.parameters()),
            )
        # ── 7. Dropout (training-only, zero compute at inference) ───────────
        elif isinstance(module, torch.nn.Dropout):
            dim = getattr(module, "_n_features", 1)
            node = AnalogNode(
                name=mod_name,
                op_type=OpType.DROPOUT,
                domain=Domain.DIGITAL,
                input_shape=(dim,),
                output_shape=(dim,),
                seq_len=seq_len,
                flops=0,
            )
        # ── 8. Skip container modules ───────────────────────────────────────
        # nn.Sequential, nn.ModuleList, and other containers have children but
        # are not compute primitives. They should NOT create graph nodes — only
        # their leaf children do. Detect containers by having named_children()
        # but not being one of the explicitly handled types above.
        elif (
            isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict))
            or (bool(list(module.named_children()))
                and not any(isinstance(module, t) for t in (
                    torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                    torch.nn.GroupNorm, torch.nn.Softmax, torch.nn.Embedding,
                    torch.nn.Dropout,
                )))
        ):
            # Container: skip node creation, but children are already iterated
            # by named_modules() so no explicit recursion needed here.
            continue
        # ── 9. Catch-all for any remaining module ───────────────────────────
        # If we reach here, the module wasn't matched by any specific check.
        # Create a generic node so the graph remains complete and connected.
        else:
            # Infer dimension from weight shape if available
            dim = 1
            w = getattr(module, "weight", None)
            if isinstance(w, torch.Tensor) and w.ndim >= 1:
                dim = w.shape[0] if w.ndim == 1 else w.shape[-1]
            node = AnalogNode(
                name=mod_name,
                op_type=OpType.MVM,  # generic compute
                domain=Domain.DIGITAL,
                input_shape=(dim,),
                output_shape=(dim,),
                seq_len=seq_len,
                flops=seq_len * dim,
                param_count=sum(p.numel() for p in module.parameters()),
            )

        if node is not None:
            # Validate node before adding to graph
            if node.seq_len is None:
                node.seq_len = 1  # Default for non-sequential ops
            if node.flops is None:
                node.flops = 0
            if node.param_count is None:
                node.param_count = 0
                
            node_id = graph.add_node(node)
            if prev_node_id is not None:
                graph.add_edge(prev_node_id, node_id)
            prev_node_id = node_id

    # Set dynamics (used by compute_scores but not energy/latency)
    dynamics = DynamicsProfile()
    if name == "neural_ode":
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="time_varying_ODE", num_function_evaluations=40, state_dimension=2)
    elif name == "flow":
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="time_varying_ODE", num_function_evaluations=100, state_dimension=2)
    elif name == "diffusion":
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="SDE", num_diffusion_steps=10, is_stochastic=True, state_dimension=64)
    elif name == "deq":
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="implicit_equilibrium", state_dimension=getattr(analog_model, "z_dim", 64))
    elif name == "ssm":
        state_dim = sum(getattr(layer, "d_state", 0) for layer in getattr(analog_model, "layers", []))
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="LTI_ODE", state_dimension=state_dim or None)
    elif name == "ebm":
        dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="energy_minimization", is_stochastic=True, state_dimension=getattr(analog_model, "n_vis", 64))
    graph.set_dynamics(dynamics)

    return graph.analyze(hardware_profile=profile)


def _compute_hardware_metrics(name: str, model: nn.Module, analog_domain: str, profile: HardwareProfile) -> dict | None:
    """Compute hardware metrics from AnalogGraph analysis of the nominal analog model.

    Returns a dict with energy/latency fields, or None if the graph produces
    physically invalid values (e.g., zero analog latency, negative savings).
    """
    try:
        analog_model = analogize(model, sigma_mismatch=0.0, n_adc_bits=32)
        configure_analog_profile(analog_model, analog_domain)
        profile_obj = _build_pilot_graph(name, analog_model, analog_domain, profile)

        # Validate: reject physically nonsensical values
        # Note: speedup=0.0 is valid when analog latency equals digital latency.
        #       speedup<0 is impossible. energy_saving<0 means analog costs MORE
        #       than digital — physically possible but indicates modeling artifacts.
        if (
            profile_obj.analog_energy_pJ <= 0
            or profile_obj.analog_latency_ns <= 0
            or profile_obj.analog_energy_saving_vs_digital < 0
            or profile_obj.analog_speedup_vs_digital < 0
        ):
            print(
                f"  [Warning] Hardware graph analysis produced invalid values: "
                f"A_energy={profile_obj.analog_energy_pJ:.2e}pJ, A_lat={profile_obj.analog_latency_ns:.2e}ns, "
                f"D_energy={profile_obj.digital_energy_pJ:.2e}pJ, D_lat={profile_obj.digital_latency_ns:.2e}ns, "
                f"saving={profile_obj.analog_energy_saving_vs_digital:.3f}, speedup={profile_obj.analog_speedup_vs_digital:.3f}"
            )
            return None

        return {
            "analog_energy_pJ": profile_obj.analog_energy_pJ,
            "digital_energy_pJ": profile_obj.digital_energy_pJ,
            "analog_latency_ns": profile_obj.analog_latency_ns,
            "digital_latency_ns": profile_obj.digital_latency_ns,
            "energy_saving_vs_digital": profile_obj.analog_energy_saving_vs_digital,
            "speedup_vs_digital": profile_obj.analog_speedup_vs_digital,
        }
    except Exception as e:
        print(f"  Warning: hardware graph analysis failed: {e}")
        return None


def _inject_hardware_metrics(result, metrics: dict | None) -> None:
    """Inject pre-computed hardware metrics into a SweepResult."""
    if metrics is None:
        return
    result.analog_energy_pJ = metrics["analog_energy_pJ"]
    result.digital_energy_pJ = metrics["digital_energy_pJ"]
    result.analog_latency_ns = metrics["analog_latency_ns"]
    result.digital_latency_ns = metrics["digital_latency_ns"]
    result.energy_saving_vs_digital = metrics["energy_saving_vs_digital"]
    result.speedup_vs_digital = metrics["speedup_vs_digital"]


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
    seed: int = 42,
    physical_substrate: str | None = None,
    hwa: bool = False,
    compute_harnessing: bool = False,
) -> None:
    sigma_values = sigma_values if sigma_values is not None else _SIGMA_VALUES
    adc_bits = adc_bits if adc_bits is not None else _BIT_VALUES

    ckpt_name = f"{name}_hwa.pt" if hwa else f"{name}.pt"
    ckpt_path = str(_CKPT_DIR / ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"[{name}] No checkpoint found at {ckpt_path}. Run train_all.py or train_hwa.py first.")
        return

    # Suffix for result filenames:
    #   domain: "" for conservative (default), "_full_analog" for full_analog
    #   substrate: "" for the default substrate of this architecture, else "_{substrate}"
    domain_suffix = "" if analog_domain == "conservative" else f"_{analog_domain}"
    default_sub = _DEFAULT_SUBSTRATE.get(name, "classic")
    substrate_suffix = "" if analog_substrate == default_sub else f"_{analog_substrate}"
    phys_suffix = f"_{physical_substrate}" if physical_substrate is not None else ""
    suffix = domain_suffix + substrate_suffix + phys_suffix

    result_path = _RESULTS_DIR / f"{name}_mismatch{suffix}.json"
    if not adc_only and result_path.exists() and not force:
        print(f"[{name}] Results exist ({analog_domain}/{analog_substrate}), skipping. (--force to re-run)")
        return

    print(f"\n{'='*50}")
    print(f"Sweeping {module.get_family_name()} ({name}), n_trials={n_trials}, domain={analog_domain}, substrate={analog_substrate}")
    if adc_only:
        print(f"  ADC-only mode: adc_sigma={adc_sigma}, bits={adc_bits}")
    print(f"{'='*50}")
    run_seed = _arch_seed(seed, name)
    _set_seed(run_seed)
    print(f"  Seed: {run_seed}")

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

    # Time-dependent models need non-standard calibration call signatures.
    # Calibration runners cover non-standard signatures such as model(t, x) and model(x_t, t).
    calib_data_to_use, calibration_runner, calibration_meta = _build_calibration_runner(
        name, module, calib_data, _DEVICE
    )
    analog_kwargs = {}

    # Hardware profile for energy/latency estimation
    profile = HardwareProfile()
    if physical_substrate is not None:
        analog_kwargs["substrate"] = physical_substrate
        print(f"  Physical substrate: {physical_substrate}")
    if name == "deq":
        deq_probe = analogize(model, sigma_mismatch=0.01, n_adc_bits=32)
        if not hasattr(getattr(deq_probe, "W_z", None), "delta"):
            raise RuntimeError("DEQ recurrent W_z was not analogized; refusing to run pilot sweep")
    profile_path = _write_pilot_profile(name, model, suffix, analog_domain)
    print(f"  Pilot profile: {profile_path}")
    # Collect architecture-specific dynamics diagnostics
    dynamics_diag = {}
    if hasattr(module, "dynamics_metrics"):
        try:
            dynamics_diag = module.dynamics_metrics(model)
            print(f"  Dynamics metrics: {dynamics_diag}")
        except Exception as e:
            print(f"  Warning: dynamics_metrics failed: {e}")

    common_metadata = {
        "arch": name,
        "analog_domain": analog_domain,
        "analog_substrate": analog_substrate,
        "seed": run_seed,
        "pilot_profile_path": str(profile_path),
        "dynamics_diagnostics": dynamics_diag,
        **calibration_meta,
    }

    # ── Compute hardware metrics once from nominal analog graph ─────────
    hw_metrics = _compute_hardware_metrics(name, model, analog_domain, profile)
    if hw_metrics is not None:
        print(f"  Pre-computed hardware metrics: energy_saving={hw_metrics['energy_saving_vs_digital']*100:.1f}%, speedup={hw_metrics['speedup_vs_digital']:.1f}x")

    if not adc_only:
        # 1. Mismatch sweep
        print(f"\n[{name}] Mismatch sweep ({analog_domain})...")
        t0 = time.time()
        result = mismatch_sweep(model, eval_fn, sigma_values=sigma_values, n_trials=n_trials,
                                calibration_data=calib_data_to_use, calibration_runner=calibration_runner,
                                analog_domain=analog_domain, hardware_profile=profile,
                                seed=run_seed, metadata=common_metadata, **analog_kwargs)
        _inject_hardware_metrics(result, hw_metrics)
        result.save(str(_RESULTS_DIR / f"{name}_mismatch{suffix}.json"))
        print(f"  Done in {time.time()-t0:.0f}s. Threshold@10%: {result.degradation_threshold():.3f}")
        # Phase 8: write empirical sigma_10pct back to pilot profile
        try:
            with open(profile_path) as f:
                profile_data = json.load(f)
            profile_data["sigma_10pct"] = float(result.degradation_threshold())
            with open(profile_path, "w") as f:
                json.dump(profile_data, f, indent=2)
        except Exception as e:
            print(f"  Warning: could not write sigma_10pct to profile: {e}")
        if result.energy_saving_vs_digital is not None:
            print(f"  Energy saving vs digital: {result.energy_saving_vs_digital*100:.1f}%")
        if result.speedup_vs_digital is not None:
            print(f"  Speedup vs digital: {result.speedup_vs_digital:.1f}x")

        # 2. Ablation sweep
        print(f"\n[{name}] Ablation sweep ({analog_domain})...")
        t0 = time.time()
        ablation = ablation_sweep(model, eval_fn, sigma_values=sigma_values,
                                  n_trials=max(10, n_trials//5), calibration_data=calib_data_to_use,
                                  calibration_runner=calibration_runner, analog_domain=analog_domain,
                                  hardware_profile=profile, seed=run_seed, metadata=common_metadata,
                                  **analog_kwargs)
        for noise_type, res in ablation.items():
            _inject_hardware_metrics(res, hw_metrics)
            res.save(str(_RESULTS_DIR / f"{name}_ablation_{noise_type}{suffix}.json"))
        print(f"  Done in {time.time()-t0:.0f}s")

    # 3. ADC sweep
    print(f"\n[{name}] ADC precision sweep ({analog_domain}, sigma_mismatch={adc_sigma})...")
    t0 = time.time()
    adc_result = adc_sweep(model, eval_fn, bit_values=adc_bits, sigma_mismatch=adc_sigma,
                           n_trials=max(20, n_trials//2), calibration_data=calib_data_to_use,
                           calibration_runner=calibration_runner, analog_domain=analog_domain,
                           hardware_profile=profile, seed=run_seed, metadata=common_metadata,
                           **analog_kwargs)
    _inject_hardware_metrics(adc_result, hw_metrics)
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

        # Load digital baseline
        digital_model = module.load_model(ckpt_path).to(_DEVICE)

        # MSE sweep: compare analog vs digital outputs at each sigma
        per_trial_mse = np.zeros((len(sigma_values), n_trials), dtype=np.float64)
        mse_trial_seeds = []

        for i, sigma in enumerate(sigma_values):
            row_seeds = []
            for j in range(n_trials):
                trial_seed = run_seed + 900_000 + i * 10_007 + j
                row_seeds.append(trial_seed)
                _set_seed(trial_seed)
                analog_model = analogize(digital_model, sigma_mismatch=sigma, n_adc_bits=8,
                                         **analog_kwargs)
                configure_analog_profile(analog_model, analog_domain)
                if calib_data_to_use is not None or calibration_runner is not None:
                    calibrate_analog_model(analog_model, calib_data_to_use, calibration_runner=calibration_runner)
                resample_all_mismatch(analog_model, sigma=sigma)

                # Call evaluate_output_mse
                if name in _SUBSTRATE_AWARE:
                    mse_val = module.evaluate_output_mse(analog_model, digital_model, analog_substrate=analog_substrate)
                else:
                    mse_val = module.evaluate_output_mse(analog_model, digital_model)
                per_trial_mse[i, j] = mse_val
            mse_trial_seeds.append(row_seeds)

            if i % 3 == 0:
                print(f"  sigma={sigma:.3f}: MSE={per_trial_mse[i].mean():.6f} ± {per_trial_mse[i].std():.6f}")

        # Save MSE results
        from neuro_analog.simulator import SweepResult
        mse_result = SweepResult(
            sigma_values=sigma_values,
            metric_name="output_mse",
            per_trial=per_trial_mse,
            digital_baseline=0.0,  # MSE at sigma=0 is baseline
            ideal_analog_baseline=0.0,
            trial_seeds={"per_trial": mse_trial_seeds},
            metadata=common_metadata,
        )
        mse_result.save(str(_RESULTS_DIR / f"{name}_output_mse{suffix}.json"))
        print(f"  Done in {time.time()-t0:.0f}s")

    # 5. Compute and store acceleration metrics if requested
    if compute_harnessing:
        print(f"\n[{name}] Computing analog acceleration profile...")
        from neuro_analog.simulator.analog_acceleration import compute_acceleration, AnalogAccelerationProfile
        # Reconstruct dynamics profile (mirrors _write_pilot_profile logic)
        dynamics = DynamicsProfile()
        if name == "neural_ode":
            dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="time_varying_ODE", num_function_evaluations=40, state_dimension=2)
        elif name == "flow":
            dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="time_varying_ODE", num_function_evaluations=100, state_dimension=2)
        elif name == "diffusion":
            dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="SDE", num_diffusion_steps=10, is_stochastic=True, state_dimension=64)
        elif name == "deq":
            dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="implicit_equilibrium", state_dimension=getattr(model, "z_dim", 64))
        elif name == "ssm":
            state_dim = sum(getattr(layer, "d_state", 0) for layer in getattr(model, "layers", []))
            dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="LTI_ODE", state_dimension=state_dim or None)
        elif name == "ebm":
            dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="energy_minimization", is_stochastic=True, state_dimension=getattr(model, "n_vis", 64))
        acc = compute_acceleration(_ARCH_FAMILY[name], dynamics=dynamics)
        acc_dict = {
            "architecture": name,
            "family": acc.architecture.value,
            "digital_iterations": acc.digital_iterations,
            "analog_settling_time_constants": acc.analog_settling_time_constants,
            "speedup_settling_vs_digital": acc.speedup_settling_vs_digital,
            "energy_saving_ratio": acc.energy_saving_ratio,
            "digital_energy_per_iteration_pJ": acc.digital_energy_per_iteration_pJ,
            "analog_energy_per_step_pJ": acc.analog_energy_per_step_pJ,
            "speedup_notes": acc.speedup_notes,
            "metadata": common_metadata,
        }
        acc_path = _RESULTS_DIR / f"{name}_acceleration{suffix}.json"
        with open(acc_path, "w") as f:
            json.dump(acc_dict, f, indent=2)
        print(f"  Acceleration: {acc.speedup_settling_vs_digital:.1f}× speedup, {acc.energy_saving_ratio:.0f}× energy saving")
        print(f"  Saved: {acc_path}")

    # 6. DEQ-specific: convergence failure rate + mean iterations + spectral radius
    if not adc_only and name == "deq" and analog_domain == "conservative" and hasattr(module, "evaluate_convergence_stats"):
        print(f"\n[DEQ] Convergence stats sweep (failure rate + mean iters + spectral radius)...")
        failure_rates = []
        mean_iters_list = []
        spectral_radii = []
        from models.deq import _get_data
        (_, _), (X_test, _) = _get_data()
        for sigma in sigma_values:
            rates, iters, rhos = [], [], []
            n_conv_trials = max(10, n_trials // 5)
            for trial_idx in range(n_conv_trials):
                _set_seed(run_seed + 1_100_000 + int(sigma * 1000) * 101 + trial_idx)
                analog_model = analogize(model, sigma_mismatch=sigma)
                configure_analog_profile(analog_model, analog_domain)
                rate, mean_iter = module.evaluate_convergence_stats(analog_model)
                rates.append(rate)
                iters.append(mean_iter)
                # Spectral radius: evaluate on inner _DEQClassifier if wrapped
                inner = analog_model
                if hasattr(analog_model, "convergence_stats"):
                    inner = analog_model
                elif hasattr(analog_model, "module"):
                    inner = analog_model.module
                if hasattr(inner, "spectral_radius_at_equilibrium"):
                    rho = inner.spectral_radius_at_equilibrium(X_test.to(_DEVICE)[:64])
                    rhos.append(rho)
            failure_rates.append(float(sum(rates) / len(rates)))
            mean_iters_list.append(float(sum(iters) / len(iters)))
            if rhos:
                spectral_radii.append(float(sum(rhos) / len(rhos)))
            else:
                spectral_radii.append(None)
            rho_str = f", rho={spectral_radii[-1]:.3f}" if spectral_radii[-1] is not None else ""
            print(f"  sigma={sigma:.3f}: failure_rate={failure_rates[-1]:.3f}, mean_iters={mean_iters_list[-1]:.1f}{rho_str}")

        deq_extra = {
            "sigma_values": sigma_values,
            "convergence_failure_rate": failure_rates,
            "mean_iterations": mean_iters_list,
            "spectral_radii": spectral_radii,
            "metadata": common_metadata,
        }
        with open(_RESULTS_DIR / f"deq_convergence{suffix}.json", "w") as f:
            json.dump(deq_extra, f, indent=2)
        print(f"  DEQ convergence data saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic pilot sweeps")
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
    parser.add_argument(
        "--physical-substrate", type=str, default=None,
        choices=["pcm", "reram", "capacitive"],
        help="Physical substrate model for analog mismatch (PCM / ReRAM / Capacitive). Overrides default sigma_mismatch model.",
    )
    parser.add_argument(
        "--compute-harnessing", action="store_true",
        help="Compute and store analog acceleration metrics (speedup/energy) per architecture.",
    )
    parser.add_argument(
        "--hwa", action="store_true",
        help="Load hardware-aware trained checkpoints ({name}_hwa.pt) instead of standard checkpoints.",
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
                          adc_bits=adc_bits, adc_only=args.adc_only,
                          seed=args.seed, physical_substrate=args.physical_substrate,
                          hwa=args.hwa,
                          compute_harnessing=args.compute_harnessing)

    print(f"\nAll sweeps done in {time.time()-total_t0:.0f}s")
    print(f"Results in: {_RESULTS_DIR}")


if __name__ == "__main__":
    main()
