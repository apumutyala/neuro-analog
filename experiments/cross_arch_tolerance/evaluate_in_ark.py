#!/usr/bin/env python3
"""
Evaluate all analog-native model families natively in JAX/Diffrax via Ark.

For each architecture:
  1. Load the pretrained PyTorch checkpoint.
  2. Export weights to a standalone Ark BaseAnalogCkt subclass (.py file).
  3. Instantiate the generated circuit.
  4. Run n_trials mismatch trials using the correct BaseAnalogCkt.__call__ API.

Runnable families (6):
  neural_ode  — NeuralODEAnalogCkt  (dx/dt = f_theta(x,t), MLP dynamics)
  flow        — FlowAnalogCkt       (same ODE path as neural_ode, v_theta MLP)
  ssm         — SSMAnalogCkt        (S4D real/imag split autonomous ODE)
  deq         — DEQAnalogCkt        (gradient-flow: dz/dt = f(z,x) - z)
  ebm         — HopfieldAnalogCkt   (Hopfield Langevin: dx/dt = -x + tanh(Wx+b))
  diffusion   — DiffusionAnalogCkt  (VP-SDE probability flow ODE)

Analysis-only (transformer FFN has no runnable ODE — attention not expressible
in Ark CDG grammar; see neuro_analog/ark_bridge/transformer_ffn_cdg.py).

Usage:
    python experiments/cross_arch_tolerance/evaluate_in_ark.py
    python experiments/cross_arch_tolerance/evaluate_in_ark.py --model deq
    python experiments/cross_arch_tolerance/evaluate_in_ark.py --n-trials 10
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

try:
    import jax
    import jax.numpy as jnp
    from ark.optimization.base_module import TimeInfo
except ImportError as _e:
    raise ImportError(
        "evaluate_in_ark.py requires JAX and Ark.\n"
        "  JAX:  pip install jax[cuda12]  (or jax[cpu] for CPU-only)\n"
        "  Ark:  pip install -e /path/to/ark\n"
    ) from _e
import torch

_ROOT = Path(__file__).parent.parent.parent
_EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EXP_DIR))

_CKPT_DIR = _EXP_DIR / "checkpoints"
_EXPORT_DIR = _EXP_DIR / "ark_exports"
_SWITCH_EMPTY = jnp.array([])

_ALL_MODELS = ["neural_ode", "flow", "ssm", "deq", "ebm", "diffusion"]


def load_generated_module(module_name: str, file_path: Path):
    """Dynamically load a generated JAX .py file as a module."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_trials(ckt, time_info: TimeInfo, state_dim: int, n_trials: int) -> None:
    """Run n_trials mismatch trials using BaseAnalogCkt.__call__.

    args_seed varies per trial — each seed produces a fresh independent
    mismatch sample from make_args(). noise_seed=0 keeps thermal noise
    deterministic (is_stochastic=False for all exported circuits).
    """
    import equinox as eqx
    import functools
    y0 = jnp.zeros((state_dim,))
    call_jit = eqx.filter_jit(functools.partial(ckt, time_info))
    # Warmup / JIT compile
    _ = call_jit(y0, _SWITCH_EMPTY, args_seed=0, noise_seed=0)
    t0 = time.time()
    for trial in range(n_trials):
        out = call_jit(y0, _SWITCH_EMPTY, args_seed=trial, noise_seed=0)
        print(f"      Trial {trial+1:02d} | args_seed={trial} | Output Norm: {jnp.linalg.norm(out):.6f}")
    print(f"      -> {n_trials} trials in {time.time() - t0:.3f}s")


def run_ark_evaluation(model_type: str, n_trials: int = 5) -> None:
    print(f"\n{'='*55}")
    print(f"  Ark Evaluation: {model_type.upper()}")
    print(f"{'='*55}")

    ckpt_path = _CKPT_DIR / f"{model_type}.pt"
    if not ckpt_path.exists():
        print(f"  [skip] Checkpoint not found: {ckpt_path}. Run train_all.py first.")
        return

    _EXPORT_DIR.mkdir(exist_ok=True)
    export_path = _EXPORT_DIR / f"{model_type}_ark_ckt.py"

    print("[1/3] Exporting PyTorch weights → Ark BaseAnalogCkt...")

    if model_type == "neural_ode":
        from neuro_analog.extractors.neural_ode import NeuralODEExtractor, export_neural_ode_to_ark
        from models import neural_ode as mod
        model = mod.load_model(str(ckpt_path))
        extractor = NeuralODEExtractor.from_module(model, state_dim=2)
        extractor.run()
        export_neural_ode_to_ark(extractor=extractor, output_path=export_path, mismatch_sigma=0.05)
        class_name = "NeuralODEAnalogCkt"
        state_dim = 2
        time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))

    elif model_type == "flow":
        # FlowMLPExtractor wraps NeuralODEExtractor; export delegates to
        # export_neural_ode_to_ark(..., class_name="FlowAnalogCkt")
        from neuro_analog.extractors.flow import FlowMLPExtractor
        ext = FlowMLPExtractor(checkpoint_path=ckpt_path)
        ext.load_model()
        ext.export_to_ark(export_path, mismatch_sigma=0.05)
        class_name = "FlowAnalogCkt"
        state_dim = 2
        time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))

    elif model_type == "ssm":
        from neuro_analog.extractors.ssm import S4DMLPExtractor
        from models import ssm as mod
        model = mod.load_model(str(ckpt_path))
        extractor = S4DMLPExtractor(checkpoint_path=ckpt_path)
        extractor.model = model
        extractor.export_to_ark(export_path, mismatch_sigma=0.05)
        class_name = "SSMAnalogCkt"
        state_dim = 2 * extractor.d_state  # real/imag split: 2 * d_state real values
        time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))

    elif model_type == "deq":
        # export_deq_to_ark accesses model.W_z.weight, model.W_x.weight/.bias directly
        from neuro_analog.ark_bridge.deq_cdg import export_deq_to_ark
        from models import deq as mod
        model = mod.load_model(str(ckpt_path))
        export_deq_to_ark(model, export_path, mismatch_sigma=0.05)
        class_name = "DEQAnalogCkt"
        state_dim = 128  # z_dim=64 + x_dim=64 (augmented state)
        time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.1, saveat=jnp.array([5.0]))

    elif model_type == "ebm":
        # RBM → block-symmetric Hopfield weights → Langevin ODE export
        from neuro_analog.ark_bridge.ebm_cdg import export_hopfield_to_ark, make_rbm_hopfield_weights
        from models import ebm as mod
        model = mod.load_model(str(ckpt_path))
        W_rbm = model.W_fwd.weight.detach().cpu().float().numpy()  # (n_hid, n_vis)
        b_h   = model.W_fwd.bias.detach().cpu().float().numpy()    # (n_hid,) hidden biases
        b_v   = model.W_bwd.bias.detach().cpu().float().numpy()    # (n_vis,) visible biases
        W_block, b_aug = make_rbm_hopfield_weights(W_rbm, b_v, b_h)
        export_hopfield_to_ark(W_block, b_aug, export_path, mismatch_sigma=0.05)
        class_name = "HopfieldAnalogCkt"
        state_dim = W_block.shape[0]  # n_vis + n_hid = 96
        time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.01, saveat=jnp.array([5.0]))

    elif model_type == "diffusion":
        # DiffusionMLPExtractor handles score_net + betas extraction internally
        from neuro_analog.extractors.diffusion import DiffusionMLPExtractor
        ext = DiffusionMLPExtractor(checkpoint_path=ckpt_path)
        ext.load_model()
        ext.export_to_ark(export_path, mismatch_sigma=0.05)
        class_name = "DiffusionAnalogCkt"
        state_dim = 64  # 8x8 MNIST image dimension
        time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.001, saveat=jnp.array([1.0]))

    else:
        raise ValueError(f"Unsupported model: {model_type!r}")

    print(f"      -> Generated {export_path}")

    print("[2/3] Loading generated BaseAnalogCkt subclass...")
    ark_module = load_generated_module(f"{model_type}_ark_ckt", export_path)
    ckt = getattr(ark_module, class_name)()
    print(f"      -> {class_name}  state_dim={state_dim}  a_trainable={ckt.a_trainable.shape}")

    print(f"[3/3] Running {n_trials} mismatch trials in JAX (args_seed varies per trial)...")
    run_trials(ckt, time_info, state_dim, n_trials)
    print(f"\n  SUCCESS: {model_type} exported as {class_name} and ran via JAX/Diffrax.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str,
        choices=_ALL_MODELS + ["all"],
        default="all",
        help="Which model family to evaluate. 'all' runs all 6 runnable families.",
    )
    parser.add_argument("--n-trials", type=int, default=5)
    args = parser.parse_args()

    models_to_run = _ALL_MODELS if args.model == "all" else [args.model]
    for m in models_to_run:
        run_ark_evaluation(m, args.n_trials)
