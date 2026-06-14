#!/usr/bin/env python3
"""Minimal diagnostic: DEQ build+solve only, no gradients."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import time
import jax.numpy as jnp
import diffrax
from ark.optimization.base_module import TimeInfo

CKPT_ROOT = Path(r"D:\individual-research\neuro-analog\experiments\cross_arch_tolerance\checkpoints")

print("[1/4] Loading DEQ model...")
t0 = time.time()
from experiments.cross_arch_tolerance.models.deq import load_model as load_deq
model = load_deq(str(CKPT_ROOT / "deq.pt"))
print(f"  Loaded in {time.time()-t0:.1f}s")

print("[2/4] Building DEQ CDG...")
t0 = time.time()
from neuro_analog.revised_ark_bridge.cdg_native.deq import build_deq
CktClass, mgr, b_eff, rho = build_deq(model, mismatch_sigma=0.0, vectorize=True)
print(f"  Built in {time.time()-t0:.1f}s  rho={rho:.4f}")

print("[3/4] Instantiating circuit...")
t0 = time.time()
ckt = CktClass(init_trainable=mgr.get_initial_vals(),
               is_stochastic=False, solver=diffrax.Tsit5())
print(f"  Instantiated in {time.time()-t0:.1f}s")

print("[4/4] Solving (t=0..0.5, dt0=0.1, n_save=5)...")
t0 = time.time()
ti = TimeInfo(t0=0.0, t1=0.5, dt0=0.1,
              saveat=jnp.linspace(0.0, 0.5, 5))
z0 = jnp.zeros(64)
out = ckt(ti, z0, switch=jnp.array([]), args_seed=0, noise_seed=0)
print(f"  Solved in {time.time()-t0:.1f}s  final_norm={float(jnp.linalg.norm(out[-1])):.3f}")

print("[5/5] Solving longer (t=0..5, dt0=0.1)...")
t0 = time.time()
ti = TimeInfo(t0=0.0, t1=5.0, dt0=0.1,
              saveat=jnp.linspace(0.0, 5.0, 50))
out = ckt(ti, z0, switch=jnp.array([]), args_seed=0, noise_seed=0)
print(f"  Solved in {time.time()-t0:.1f}s  final_norm={float(jnp.linalg.norm(out[-1])):.3f}")

print("\nDEQ minimal test PASSED.")
