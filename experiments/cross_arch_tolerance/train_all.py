#!/usr/bin/env python3
"""Train all 7 architecture models and save checkpoints.

Usage:
    python experiments/cross_arch_tolerance/train_all.py

Checkpoints saved to: experiments/cross_arch_tolerance/checkpoints/
Each model trains to convergence on CPU in < 5 minutes.
Total time: ~15-25 minutes depending on machine.
"""

import os
import sys
import time
from pathlib import Path

# Make sure project root is on path
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

_CKPT_DIR = Path(__file__).parent / "checkpoints"
_CKPT_DIR.mkdir(exist_ok=True)

# Import all model modules
from models import neural_ode, transformer, diffusion, flow, ebm, deq, ssm

_MODELS = [
    ("neural_ode",   neural_ode),
    ("transformer",  transformer),
    ("diffusion",    diffusion),
    ("flow",         flow),
    ("ebm",          ebm),
    ("deq",          deq),
    ("ssm",          ssm),
]


def train_one(name: str, module, force: bool = False) -> None:
    ckpt_path = str(_CKPT_DIR / f"{name}.pt")
    if os.path.exists(ckpt_path) and not force:
        print(f"[{name}] Checkpoint exists, skipping. (--force to retrain)")
        # Verify it loads
        model = module.load_model(ckpt_path)
        metric = module.evaluate(model)
        print(f"[{name}] Loaded. {module.get_family_name()} quality: {metric:.4f}")
        return

    print(f"\n{'='*50}")
    print(f"Training {module.get_family_name()} ({name})")
    print(f"{'='*50}")
    t0 = time.time()

    model = module.create_model()
    model = module.train_model(model, ckpt_path)
    metric = module.evaluate(model)

    elapsed = time.time() - t0
    print(f"[{name}] Done in {elapsed:.0f}s. Quality metric: {metric:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    parser.add_argument("--only", type=str, default=None, help="Train only this model (e.g. neural_ode)")
    args = parser.parse_args()

    total_t0 = time.time()
    for name, module in _MODELS:
        if args.only and name != args.only:
            continue
        train_one(name, module, force=args.force)

    total_elapsed = time.time() - total_t0
    print(f"\nAll models trained in {total_elapsed:.0f}s total.")
    print(f"Checkpoints saved to: {_CKPT_DIR}")


if __name__ == "__main__":
    main()
