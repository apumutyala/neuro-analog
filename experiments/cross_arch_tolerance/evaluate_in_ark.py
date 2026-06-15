#!/usr/bin/env python3
"""
Evaluate all analog-native model families natively in JAX/Diffrax via Ark.

For each architecture:
  1. Load the pretrained PyTorch checkpoint.
  2. Compile it directly to an Ark BaseAnalogCkt with neuro_analog.revised_ark_bridge.
  3. Run n_trials conductance-mismatch trials (args_seed varies the realization).

This script is agnostic to any pre-generated export files: every circuit is built
in-memory from the trained PyTorch model by revised_ark_bridge, so the only inputs
are the checkpoints in ./checkpoints/.

Runnable families (6):
  neural_ode  — MLPFieldCkt   (dx/dt = f_theta(x, t), MLP vector field)
  flow        — MLPFieldCkt   (same field circuit, v_theta MLP weights)
  ssm         — Linear SSM    (real diagonal A; dh/dt = A h, RC decay modes)
  deq         — additive CDG  (dz/dt = -z + tanh(W_z z + b_eff), relaxation)
  ebm         — additive CDG  (RBM -> block-symmetric Hopfield, sigmoid mean-field)
  diffusion   — CLDCkt        (critically-damped Langevin probability-flow SDE)

Attention is analysis-only: softmax attention is not expressible in the Ark CDG
grammar (see neuro_analog/revised_ark_bridge/notebooks/04_attention_bottleneck.ipynb).

Usage:
    python experiments/cross_arch_tolerance/evaluate_in_ark.py
    python experiments/cross_arch_tolerance/evaluate_in_ark.py --model deq
    python experiments/cross_arch_tolerance/evaluate_in_ark.py --n-trials 10
"""

import argparse
import functools
import sys
import time
from pathlib import Path

try:
    import jax
    import jax.numpy as jnp
    import diffrax
    import equinox as eqx
    from ark.optimization.base_module import TimeInfo
except ImportError as _e:
    raise ImportError(
        "evaluate_in_ark.py requires JAX and Ark.\n"
        "  JAX:  pip install jax[cuda12]  (or jax[cpu] for CPU-only)\n"
        "  Ark:  pip install -e /path/to/Ark\n"
    ) from _e
import numpy as np
import torch

_ROOT = Path(__file__).parent.parent.parent
_EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_EXP_DIR))

_CKPT_DIR = _EXP_DIR / "checkpoints"
_SWITCH_EMPTY = jnp.array([])
_MISMATCH_SIGMA = 0.05

_ALL_MODELS = ["neural_ode", "flow", "ssm", "deq", "ebm", "diffusion"]


def _instantiate(ckt_class, mgr):
    """Instantiate a compiled CDG class with its initial trainables."""
    return ckt_class(
        init_trainable=mgr.get_initial_vals(),
        is_stochastic=False,
        solver=diffrax.Tsit5(),
    )


def _build_neural_ode(ckpt_path):
    from models import neural_ode as mod
    from neuro_analog.revised_ark_bridge import build_neural_ode

    model = mod.load_model(str(ckpt_path))
    ckt = build_neural_ode(model, mismatch_sigma=_MISMATCH_SIGMA)
    state_dim = ckt._parse_args(ckt.a_trainable)["W3"].shape[0]
    z0 = jnp.zeros((state_dim,))
    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))
    return ckt, z0, time_info


def _build_flow(ckpt_path):
    from models import flow as mod
    from neuro_analog.revised_ark_bridge import build_flow

    model = mod.load_model(str(ckpt_path))
    ckt = build_flow(model, mismatch_sigma=_MISMATCH_SIGMA)
    state_dim = ckt._parse_args(ckt.a_trainable)["W3"].shape[0]
    z0 = jnp.zeros((state_dim,))
    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))
    return ckt, z0, time_info


def _build_ssm(ckpt_path):
    from models import ssm as mod
    from neuro_analog.revised_ark_bridge import build_ssm

    model = mod.load_model(str(ckpt_path))
    ckt_or_class, mgr, used_cdg = build_ssm(
        model, mismatch_sigma=_MISMATCH_SIGMA, force_plain=False
    )
    ckt = _instantiate(ckt_or_class, mgr) if used_cdg else ckt_or_class
    d_state = int(model.layers[0].log_A_real.detach().cpu().numpy().reshape(-1).shape[0])
    z0 = jnp.zeros((d_state,))
    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))
    return ckt, z0, time_info


def _build_deq(ckpt_path):
    from models import deq as mod
    from neuro_analog.revised_ark_bridge import build_deq

    model = mod.load_model(str(ckpt_path))
    ckt_class, mgr, _b_eff, _rho = build_deq(
        model, mismatch_sigma=_MISMATCH_SIGMA, vectorize=True
    )
    ckt = _instantiate(ckt_class, mgr)
    n_state = model.W_z.weight.shape[0]
    z0 = jnp.zeros((n_state,))
    time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.1, saveat=jnp.array([5.0]))
    return ckt, z0, time_info


def _build_ebm(ckpt_path):
    from models import ebm as mod
    from neuro_analog.revised_ark_bridge import build_ebm

    model = mod.load_model(str(ckpt_path))
    ckt_class, mgr = build_ebm(model, mismatch_sigma=_MISMATCH_SIGMA, vectorize=True)
    ckt = _instantiate(ckt_class, mgr)
    n_hid, n_vis = model.W_fwd.weight.shape  # [n_hid, n_vis]
    z0 = jnp.concatenate(
        [jax.random.uniform(jax.random.PRNGKey(0), (n_vis,)), jnp.zeros(n_hid)]
    )
    time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.1, saveat=jnp.array([5.0]))
    return ckt, z0, time_info


def _build_diffusion(ckpt_path):
    from models import diffusion as mod
    from neuro_analog.revised_ark_bridge import build_diffusion

    model = mod.load_model(str(ckpt_path))
    ckt = build_diffusion(model, mismatch_sigma=_MISMATCH_SIGMA)
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    x_dim = linears[-1].out_features  # data dimension = score-net output
    z0 = jnp.zeros((2 * x_dim,))  # CLD state is [position x; momentum v]
    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.array([1.0]))
    return ckt, z0, time_info


_BUILDERS = {
    "neural_ode": _build_neural_ode,
    "flow": _build_flow,
    "ssm": _build_ssm,
    "deq": _build_deq,
    "ebm": _build_ebm,
    "diffusion": _build_diffusion,
}


def run_trials(ckt, z0, time_info, n_trials: int) -> None:
    """Run n_trials mismatch trials using BaseAnalogCkt.__call__.

    The circuit is built once with a fixed mismatch_sigma; args_seed varies per
    trial so each call draws a fresh independent conductance-mismatch realization.
    noise_seed is held fixed so any spread reflects mismatch, not thermal noise.
    """
    call_jit = eqx.filter_jit(functools.partial(ckt, time_info))
    _ = call_jit(z0, _SWITCH_EMPTY, args_seed=0, noise_seed=0)  # warmup / JIT compile
    t0 = time.time()
    for trial in range(n_trials):
        out = call_jit(z0, _SWITCH_EMPTY, args_seed=trial, noise_seed=0)
        print(f"      Trial {trial + 1:02d} | args_seed={trial} | Output Norm: {jnp.linalg.norm(out):.6f}")
    print(f"      -> {n_trials} trials in {time.time() - t0:.3f}s")


def run_ark_evaluation(model_type: str, n_trials: int = 5) -> None:
    print(f"\n{'=' * 55}")
    print(f"  Ark Evaluation: {model_type.upper()}")
    print(f"{'=' * 55}")

    ckpt_path = _CKPT_DIR / f"{model_type}.pt"
    if not ckpt_path.exists():
        print(f"  [skip] Checkpoint not found: {ckpt_path}. Run train_all.py first.")
        return

    print(f"[1/2] Compiling {model_type} -> Ark BaseAnalogCkt (sigma={_MISMATCH_SIGMA}) via revised_ark_bridge...")
    ckt, z0, time_info = _BUILDERS[model_type](ckpt_path)
    print(f"      -> {type(ckt).__name__}  state_dim={z0.shape[0]}  a_trainable={ckt.a_trainable.shape}")

    print(f"[2/2] Running {n_trials} mismatch trials in JAX (args_seed varies per trial)...")
    run_trials(ckt, z0, time_info, n_trials)
    print(f"\n  SUCCESS: {model_type} compiled and ran via JAX/Diffrax.")


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
