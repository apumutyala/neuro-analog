#!/usr/bin/env python3
"""Hardware-aware training: inject analog mismatch noise during training.

Makes models robust to analog device variation by exposing all nn.Linear
layers to multiplicative weight noise during every forward pass:

    W_noisy = W * (1 + ε),   ε ~ N(0, σ²)

The noise is resampled every forward pass.  The optimizer therefore
minimises the expected loss under mismatch, producing weights that are
inherently robust to analog crossbar fabrication variation.

The monkey-patched forward is disabled after training so evaluation
uses clean weights.

Usage:
    python experiments/cross_arch_tolerance/train_hwa.py
    python experiments/cross_arch_tolerance/train_hwa.py --only neural_ode --sigma 0.05
"""

import os
import sys
import time
import random
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

_CKPT_DIR = Path(__file__).parent / "checkpoints"
_CKPT_DIR.mkdir(exist_ok=True)

from models import neural_ode, transformer, diffusion, flow, ebm, deq, ssm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_MODELS = [
    ("neural_ode",   neural_ode),
    ("transformer",  transformer),
    ("diffusion",    diffusion),
    ("flow",         flow),
    ("ebm",          ebm),
    ("deq",          deq),
    ("ssm",          ssm),
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_hwa_noise(model: nn.Module, sigma: float) -> None:
    """Monkey-patch every nn.Linear forward to inject multiplicative weight noise."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            original_forward = module.forward

            def make_forward(m, orig, s):
                def forward(x):
                    if getattr(m, "_hwa_enabled", False):
                        noise = torch.randn_like(m.weight) * s
                        noisy_weight = m.weight * (1 + noise)
                        return F.linear(x, noisy_weight, m.bias)
                    return orig(x)
                return forward

            module._hwa_enabled = True
            module.forward = make_forward(module, original_forward, sigma)


def _disable_hwa(model: nn.Module) -> None:
    """Disable HWA noise injection on all nn.Linear layers."""
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, "_hwa_enabled"):
            module._hwa_enabled = False


def train_one(name: str, module, sigma: float = 0.05, force: bool = False, seed: int = 42) -> None:
    _set_seed(seed)
    ckpt_path = str(_CKPT_DIR / f"{name}_hwa.pt")
    if os.path.exists(ckpt_path) and not force:
        print(f"[{name}] HWA checkpoint exists, skipping. (--force to retrain)")
        model = module.load_model(ckpt_path)
        metric = module.evaluate(model)
        print(f"[{name}] Loaded. {module.get_family_name()} quality: {metric:.4f}")
        return

    print(f"\n{'='*50}")
    print(f"HWA Training {module.get_family_name()} ({name}), σ={sigma}")
    print(f"{'='*50}")
    t0 = time.time()

    model = module.create_model()
    _add_hwa_noise(model, sigma)
    model = module.train_model(model, ckpt_path)

    # Disable noise so subsequent evaluation is clean
    _disable_hwa(model)
    metric = module.evaluate(model)
    elapsed = time.time() - t0
    print(f"[{name}] HWA done in {elapsed:.0f}s. Quality metric: {metric:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    parser.add_argument("--only", type=str, default=None, help="Train only this model")
    parser.add_argument("--sigma", type=float, default=0.05, help="HWA noise level (multiplicative sigma)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    total_t0 = time.time()
    for idx, (name, module) in enumerate(_MODELS):
        if args.only and name != args.only:
            continue
        train_one(name, module, sigma=args.sigma, force=args.force, seed=args.seed + idx * 10_000)

    total_elapsed = time.time() - total_t0
    print(f"\nAll HWA models trained in {total_elapsed:.0f}s total.")
    print(f"Checkpoints saved to: {_CKPT_DIR}")


if __name__ == "__main__":
    main()
