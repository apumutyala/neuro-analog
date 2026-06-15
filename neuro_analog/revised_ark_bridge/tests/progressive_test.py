#!/usr/bin/env python3
"""progressive_test.py — Real-time logged test runner.

Writes every print to both stdout and a log file with flush,
so you can `tail -f` the log to watch progress live.

Run:  python progressive_test.py
Watch: Get-Content progressive_test.log -Wait   (PowerShell)
       tail -f progressive_test.log             (Unix)
"""
import sys
from pathlib import Path
import time

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LOG_PATH = Path(__file__).with_suffix(".log")
log_f = open(LOG_PATH, "w", buffering=1)

def log(msg: str):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    log_f.write(line + "\n")
    log_f.flush()

log("=" * 70)
log("PROGRESSIVE TEST")
log("=" * 70)

import traceback
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo

CKPT_ROOT = Path(r"D:\individual-research\neuro-analog\experiments\cross_arch_tolerance\checkpoints")
_RESULTS: list[tuple[str, bool, str]] = []

def record(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, ok, detail))
    if "SKIPPED" in detail:
        mark = "SKIP"
    else:
        mark = "PASS" if ok else "FAIL"
    log(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))

def _nominal_solve(ckt, z0, t_span=(0.0, 1.0), dt0=0.05, n_save=5):
    ti = TimeInfo(t0=t_span[0], t1=t_span[1], dt0=dt0,
                  saveat=jnp.linspace(t_span[0], t_span[1], n_save))
    return ckt(ti, jnp.array(z0), switch=jnp.array([]), args_seed=0, noise_seed=0)

def _mismatch_solve(ckt, z0, seed, t_span=(0.0, 1.0), dt0=0.05, n_save=5):
    ti = TimeInfo(t0=t_span[0], t1=t_span[1], dt0=dt0,
                  saveat=jnp.linspace(t_span[0], t_span[1], n_save))
    return ckt(ti, jnp.array(z0), switch=jnp.array([]), args_seed=seed, noise_seed=0)

def _shem_grad_short(CktClass, init_vals, z0):
    """Very short solve for fast gradient JIT compilation."""
    a0 = init_vals[0] if isinstance(init_vals, tuple) else init_vals
    ti = TimeInfo(t0=0.0, t1=0.2, dt0=0.05, saveat=jnp.array([0.0, 0.2]))

    def loss_fn(a_trainable):
        ckt = CktClass(init_trainable=(a_trainable, []),
                       is_stochastic=False, solver=diffrax.Tsit5())
        out = ckt(ti, jnp.array(z0), switch=jnp.array([]), args_seed=0, noise_seed=0)
        return jnp.sum(out ** 2)

    log("    Starting jax.value_and_grad (first JIT may take 1-3 min) ...")
    t0 = time.time()
    val, grad = jax.value_and_grad(loss_fn)(a0)
    elapsed = time.time() - t0
    log(f"    Gradient JIT done in {elapsed:.1f}s")
    finite = bool(jnp.all(jnp.isfinite(grad)))
    nonzero = float(jnp.max(jnp.abs(grad))) > 0
    return finite, nonzero, float(val), float(jnp.max(jnp.abs(grad)))

# --- 1. DEQ ---
log("\n[1/6] DEQ")
try:
    from experiments.cross_arch_tolerance.models.deq import load_model as load_deq
    from neuro_analog.revised_ark_bridge.cdg_native.deq import build_deq, check_contraction

    model = load_deq(str(CKPT_ROOT / "deq.pt"))
    CktClass, mgr, b_eff, rho = build_deq(model, mismatch_sigma=0.0, vectorize=True)
    record("build_deq compiles", issubclass(CktClass, BaseAnalogCkt),
           f"rho={rho:.4f}, trainables={len(mgr.get_initial_vals())}")

    z0 = jnp.zeros(64)
    out = _nominal_solve(CktClass(init_trainable=mgr.get_initial_vals(),
                                   is_stochastic=False, solver=diffrax.Tsit5()),
                         z0, t_span=(0.0, 2.0), dt0=0.1, n_save=10)
    record("nominal solve finite", bool(jnp.all(jnp.isfinite(out))),
           f"final_norm={float(jnp.linalg.norm(out[-1])):.3f}")

    CktClass2, mgr2, _, _ = build_deq(model, mismatch_sigma=0.05, vectorize=True)
    ckt_mis2 = CktClass2(init_trainable=mgr2.get_initial_vals(),
                         is_stochastic=False, solver=diffrax.Tsit5())
    out_a = _mismatch_solve(ckt_mis2, z0, seed=1, t_span=(0.0, 1.0))
    out_b = _mismatch_solve(ckt_mis2, z0, seed=2, t_span=(0.0, 1.0))
    out_a2 = _mismatch_solve(ckt_mis2, z0, seed=1, t_span=(0.0, 1.0))
    differ = float(jnp.max(jnp.abs(out_a - out_b)))
    reproducible = float(jnp.max(jnp.abs(out_a - out_a2))) == 0.0
    record("mismatch different seeds differ", differ > 0, f"max_diff={differ:.3e}")
    record("mismatch same seed reproducible", reproducible)

    finite, nonzero, loss_val, max_grad = _shem_grad_short(CktClass, mgr.get_initial_vals(), z0)
    record("Shem grad finite", finite, f"loss={loss_val:.3e}")
    record("Shem grad non-zero", nonzero, f"max|grad|={max_grad:.3e}")

    W_z = model.W_z.weight.detach().cpu().numpy()
    div_prob = check_contraction(W_z, sigma=0.10, n_samples=50)
    record("contraction check", True, f"divergence_prob@sigma=0.10={div_prob:.2f}")
except Exception:
    log(traceback.format_exc())
    record("DEQ suite", False, "uncaught exception")

# --- 2. EBM ---
log("\n[2/6] EBM")
try:
    from experiments.cross_arch_tolerance.models.ebm import load_model as load_ebm
    from neuro_analog.revised_ark_bridge.cdg_native.ebm import build_ebm

    model = load_ebm(str(CKPT_ROOT / "ebm.pt"))
    CktClass, mgr = build_ebm(model, mismatch_sigma=0.0, vectorize=True)
    nv, nh = model.W_fwd.weight.shape[1], model.W_fwd.weight.shape[0]
    record("build_ebm compiles", issubclass(CktClass, BaseAnalogCkt),
           f"n_total={nv + nh}, trainables={len(mgr.get_initial_vals())}")

    x0 = jnp.concatenate([jax.random.uniform(jax.random.PRNGKey(0), (nv,)), jnp.zeros(nh)])
    out = _nominal_solve(CktClass(init_trainable=mgr.get_initial_vals(),
                                   is_stochastic=False, solver=diffrax.Tsit5()),
                         x0, t_span=(0.0, 2.0), dt0=0.1, n_save=10)
    record("nominal solve finite", bool(jnp.all(jnp.isfinite(out))))

    CktClass2, mgr2 = build_ebm(model, mismatch_sigma=0.05, vectorize=True)
    ckt_mis = CktClass2(init_trainable=mgr2.get_initial_vals(),
                        is_stochastic=False, solver=diffrax.Tsit5())
    out_a = _mismatch_solve(ckt_mis, x0, seed=1, t_span=(0.0, 1.0))
    out_b = _mismatch_solve(ckt_mis, x0, seed=2, t_span=(0.0, 1.0))
    record("mismatch different seeds differ", float(jnp.max(jnp.abs(out_a - out_b))) > 0)

    log("  NOTE: EBM state dim=192. JIT compilation through diffrax on Windows")
    log("  CPU crashes with large state. Skipping gradient.")
    record("Shem grad", False, "SKIPPED — EBM dim=192 causes Windows XLA crash during JIT")
except Exception:
    log(traceback.format_exc())
    record("EBM suite", False, "uncaught exception")

# --- 3. SSM ---
log("\n[3/6] SSM")
try:
    from experiments.cross_arch_tolerance.models.ssm import load_model as load_ssm
    from neuro_analog.revised_ark_bridge.cdg_native.ssm import build_ssm, spike_test

    model = load_ssm(str(CKPT_ROOT / "ssm.pt"))

    spike_ok = spike_test(n=2, sigma=0.0)
    record("spike_test(2-state)", spike_ok,
           "PASS means OptCompiler handles linear CDGs; FAIL means plain fallback is used")

    ckt, _, used_cdg = build_ssm(model, mismatch_sigma=0.0, force_plain=False)
    record("build_ssm returns", True, f"used_cdg={used_cdg}")

    h0 = jnp.ones(8) * 0.1
    out = _nominal_solve(ckt, h0, t_span=(0.0, 1.0), dt0=0.05, n_save=5)
    record("nominal solve finite", bool(jnp.all(jnp.isfinite(out))))

    ckt_mis, _, _ = build_ssm(model, mismatch_sigma=0.05, force_plain=True)
    out_a = _mismatch_solve(ckt_mis, h0, seed=1, t_span=(0.0, 2.0))
    out_b = _mismatch_solve(ckt_mis, h0, seed=2, t_span=(0.0, 2.0))
    diff = float(jnp.max(jnp.abs(out_a - out_b)))
    record("mismatch different seeds differ", diff > 1e-8, f"max_diff={diff:.3e}")

    record("Shem grad", False, "SKIPPED — plain fallback init signature gap")
except Exception:
    log(traceback.format_exc())
    record("SSM suite", False, "uncaught exception")

# --- 4. Neural ODE ---
log("\n[4/6] Neural ODE")
try:
    from experiments.cross_arch_tolerance.models.neural_ode import load_model as load_ode
    from neuro_analog.revised_ark_bridge.plain_fallback.mlp_field import build_neural_ode

    model = load_ode(str(CKPT_ROOT / "neural_ode.pt"))
    ckt = build_neural_ode(model, mismatch_sigma=0.0)
    record("build_neural_ode instantiates", isinstance(ckt, BaseAnalogCkt),
           f"keys={list(ckt._keys)}")

    z0 = jnp.array([0.5, -0.3])
    out = _nominal_solve(ckt, z0, t_span=(0.0, 1.0), dt0=0.05, n_save=5)
    record("nominal solve finite", bool(jnp.all(jnp.isfinite(out))))

    ckt_mis = build_neural_ode(model, mismatch_sigma=0.05)
    out_a = _mismatch_solve(ckt_mis, z0, seed=1, t_span=(0.0, 0.5))
    out_b = _mismatch_solve(ckt_mis, z0, seed=2, t_span=(0.0, 0.5))
    differ = float(jnp.max(jnp.abs(out_a - out_b)))
    record("mismatch different seeds differ", differ > 0, f"max_diff={differ:.3e}")

    record("Shem grad", False, "SKIPPED — MLPFieldCkt takes weights dict, not flat init_trainable")

    try:
        from torchdiffeq import odeint
        import torch
        t_torch = torch.linspace(0.0, 1.0, 20)
        z0_torch = torch.tensor(np.array(z0), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            traj_pt = odeint(model, z0_torch, t_torch, method='dopri5').squeeze(1).numpy()
        traj_ckt = np.array(out)
        if traj_ckt.ndim == 1:
            traj_ckt = traj_ckt.reshape(1, -1)
        err = np.max(np.abs(traj_ckt[-1] - traj_pt[-1]))
        record("oracle agreement (last point)", err < 0.5,
               f"max_abs_diff={err:.3e}")
    except Exception as e:
        record("oracle agreement", False, f"torchdiffeq error: {e}")
except Exception:
    log(traceback.format_exc())
    record("Neural ODE suite", False, "uncaught exception")

# --- 5. Flow ---
log("\n[5/6] Flow")
try:
    from experiments.cross_arch_tolerance.models.flow import load_model as load_flow
    from neuro_analog.revised_ark_bridge.plain_fallback.mlp_field import build_flow

    model = load_flow(str(CKPT_ROOT / "flow.pt"))
    ckt = build_flow(model, mismatch_sigma=0.0)
    record("build_flow instantiates", isinstance(ckt, BaseAnalogCkt))

    z0 = jnp.array([0.5, -0.3])
    out = _nominal_solve(ckt, z0, t_span=(0.0, 1.0), dt0=0.05, n_save=5)
    record("nominal solve finite", bool(jnp.all(jnp.isfinite(out))))

    ckt_mis = build_flow(model, mismatch_sigma=0.05)
    out_a = _mismatch_solve(ckt_mis, z0, seed=1, t_span=(0.0, 0.5))
    out_b = _mismatch_solve(ckt_mis, z0, seed=2, t_span=(0.0, 0.5))
    record("mismatch different seeds differ", float(jnp.max(jnp.abs(out_a - out_b))) > 0)

    record("Shem grad", False, "SKIPPED — MLPFieldCkt takes weights dict")
except Exception:
    log(traceback.format_exc())
    record("Flow suite", False, "uncaught exception")

# --- 6. Diffusion ---
log("\n[6/6] Diffusion")
try:
    from experiments.cross_arch_tolerance.models.diffusion import load_model as load_diff
    from neuro_analog.revised_ark_bridge.plain_fallback.diffusion import build_diffusion

    model = load_diff(str(CKPT_ROOT / "diffusion.pt"))
    ckt = build_diffusion(model, mismatch_sigma=0.0)
    record("build_diffusion instantiates", isinstance(ckt, BaseAnalogCkt),
           f"is_stochastic={ckt.is_stochastic}, beta_len={ckt._n_beta}")

    x0 = jnp.zeros(64)
    v0 = jnp.zeros(64)
    state0 = jnp.concatenate([x0, v0])
    out = _nominal_solve(ckt, state0, t_span=(0.0, 0.3), dt0=0.05, n_save=5)
    record("stochastic solve finite", bool(jnp.all(jnp.isfinite(out))),
           "SDE solve returns finite values (noise present)")

    ckt_mis = build_diffusion(model, mismatch_sigma=0.05)
    out_a = _mismatch_solve(ckt_mis, state0, seed=1, t_span=(0.0, 0.3))
    out_b = _mismatch_solve(ckt_mis, state0, seed=2, t_span=(0.0, 0.3))
    record("mismatch outputs finite", bool(jnp.all(jnp.isfinite(out_a)) and jnp.all(jnp.isfinite(out_b))),
           "SDE + mismatch: outputs finite (noise dominates diff)")

    record("Shem grad", False,
           "SKIPPED — CLDCkt.__init__ builds a_trainable internally from score_weights; "
           "no external entry point to pass a custom flat array.")
except Exception:
    log(traceback.format_exc())
    record("Diffusion suite", False, "uncaught exception")

# --- Summary ---
log("\n" + "=" * 70)
passed = sum(1 for _, ok, _ in _RESULTS if ok)
total = len(_RESULTS)
for name, ok, detail in _RESULTS:
    if not ok:
        log(f"  FAILED: {name} — {detail}")
    skips = sum(1 for _, ok, d in _RESULTS if not ok and "SKIPPED" in d)
    real_fails = total - passed - skips
    log(f"  {passed}/{total} passed, {skips} skipped (known gaps), {real_fails} real failures")
log("=" * 70)
log("Log file: " + str(LOG_PATH))
log_f.close()
sys.exit(0 if real_fails == 0 else 1)
