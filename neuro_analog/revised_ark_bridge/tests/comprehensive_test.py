#!/usr/bin/env python3
"""
comprehensive_test.py — Build, solve, mismatch, and gradient verification
for all 6 neuro-analog families.

Requires: ark, jax, diffrax, equinox, torch, torchdiffeq, scipy
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


# --- Helpers ---


def _nominal_solve(ckt, z0, t_span=(0.0, 2.0), dt0=0.1, n_save=20):
    """Deterministic solve with mismatch off (args_seed=0)."""
    ti = TimeInfo(
        t0=t_span[0], t1=t_span[1], dt0=dt0,
        saveat=jnp.linspace(t_span[0], t_span[1], n_save)
    )
    return ckt(ti, jnp.array(z0), switch=jnp.array([]), args_seed=0, noise_seed=0)


def _mismatch_solve(ckt, z0, seed, t_span=(0.0, 2.0), dt0=0.1, n_save=20):
    """Solve with a specific mismatch realization seed."""
    ti = TimeInfo(
        t0=t_span[0], t1=t_span[1], dt0=dt0,
        saveat=jnp.linspace(t_span[0], t_span[1], n_save)
    )
    return ckt(ti, jnp.array(z0), switch=jnp.array([]), args_seed=seed, noise_seed=0)


def _check_shem_grad(CktClass, init_vals, z0, t_span=(0.0, 1.0), dt0=0.1):
    """Reverse-mode autodiff through diffrax solve."""
    a0 = init_vals[0] if isinstance(init_vals, tuple) else init_vals
    ti = TimeInfo(t0=t_span[0], t1=t_span[1], dt0=dt0,
                  saveat=jnp.linspace(t_span[0], t_span[1], 10))

    def loss_fn(a_trainable):
        ckt = CktClass(init_trainable=(a_trainable, []),
                       is_stochastic=False, solver=diffrax.Tsit5())
        out = ckt(ti, jnp.array(z0), switch=jnp.array([]), args_seed=0, noise_seed=0)
        return jnp.sum(out ** 2)

    val, grad = jax.value_and_grad(loss_fn)(a0)
    finite = bool(jnp.all(jnp.isfinite(grad)))
    nonzero = float(jnp.max(jnp.abs(grad))) > 0
    return finite, nonzero, float(val), float(jnp.max(jnp.abs(grad)))


# --- 1. DEQ (CDG-native) ---


def test_deq():
    print("\n[DEQ] CDG-native Deep Equilibrium Model")
    try:
        from experiments.cross_arch_tolerance.models.deq import load_model as load_deq
        from neuro_analog.revised_ark_bridge.cdg_native.deq import build_deq, check_contraction

        model = load_deq(str(CKPT_ROOT / "deq.pt"))
        CktClass, mgr, b_eff, rho = build_deq(model, mismatch_sigma=0.0, vectorize=True)

        record("build_deq compiles", issubclass(CktClass, BaseAnalogCkt),
               f"rho={rho:.4f}, trainables={len(mgr.get_initial_vals())}")

        # Nominal solve
        z0 = jnp.zeros(64)
        out = _nominal_solve(CktClass(init_trainable=mgr.get_initial_vals(),
                                       is_stochastic=False, solver=diffrax.Tsit5()),
                             z0, t_span=(0.0, 5.0), dt0=0.1, n_save=50)
        record("nominal solve finite", bool(jnp.all(jnp.isfinite(out))),
               f"final_norm={float(jnp.linalg.norm(out[-1])):.3f}")

        # Mismatch sweep
        ckt_mis = CktClass(init_trainable=mgr.get_initial_vals(),
                           is_stochastic=False, solver=diffrax.Tsit5())
        # NOTE: build_deq bakes mismatch_sigma into the CDG's AttrDefMismatch.
        # To test mismatch we need mismatch_sigma > 0 at compile time.
        CktClass2, mgr2, _, _ = build_deq(model, mismatch_sigma=0.05, vectorize=True)
        ckt_mis2 = CktClass2(init_trainable=mgr2.get_initial_vals(),
                             is_stochastic=False, solver=diffrax.Tsit5())
        out_a = _mismatch_solve(ckt_mis2, z0, seed=1, t_span=(0.0, 2.0))
        out_b = _mismatch_solve(ckt_mis2, z0, seed=2, t_span=(0.0, 2.0))
        out_a2 = _mismatch_solve(ckt_mis2, z0, seed=1, t_span=(0.0, 2.0))
        differ = float(jnp.max(jnp.abs(out_a - out_b)))
        reproducible = float(jnp.max(jnp.abs(out_a - out_a2))) == 0.0
        record("mismatch different seeds differ", differ > 0, f"max_diff={differ:.3e}")
        record("mismatch same seed reproducible", reproducible)

        # Shem gradient
        finite, nonzero, loss_val, max_grad = _check_shem_grad(
            CktClass, mgr.get_initial_vals(), z0, t_span=(0.0, 1.0))
        record("Shem grad finite", finite, f"loss={loss_val:.3e}")
        record("Shem grad non-zero", nonzero, f"max|grad|={max_grad:.3e}")

        # Contraction check
        W_z = model.W_z.weight.detach().cpu().numpy()
        div_prob = check_contraction(W_z, sigma=0.10, n_samples=50)
        record("contraction check runs", True, f"divergence_prob@sigma=0.10={div_prob:.2f}")

    except Exception:
        traceback.print_exc()
        record("DEQ suite", False, "uncaught exception")


# --- 2. EBM (CDG-native) ---


def test_ebm():
    print("\n[EBM] CDG-native Energy-Based Model")
    try:
        from experiments.cross_arch_tolerance.models.ebm import load_model as load_ebm
        from neuro_analog.revised_ark_bridge.cdg_native.ebm import build_ebm

        model = load_ebm(str(CKPT_ROOT / "ebm.pt"))
        CktClass, mgr = build_ebm(model, mismatch_sigma=0.0, vectorize=True)

        n_vis = model.W_fwd.weight.shape[1]
        n_hid = model.W_fwd.weight.shape[0]
        record("build_ebm compiles", issubclass(CktClass, BaseAnalogCkt),
               f"n_total={n_vis + n_hid}, trainables={len(mgr.get_initial_vals())}")

        # Nominal solve
        x0 = jnp.concatenate([jax.random.uniform(jax.random.PRNGKey(0), (n_vis,)), jnp.zeros(n_hid)])
        out = _nominal_solve(CktClass(init_trainable=mgr.get_initial_vals(),
                                       is_stochastic=False, solver=diffrax.Tsit5()),
                             x0, t_span=(0.0, 3.0), dt0=0.1, n_save=30)
        record("nominal solve finite", bool(jnp.all(jnp.isfinite(out))))

        # Mismatch
        CktClass2, mgr2 = build_ebm(model, mismatch_sigma=0.05, vectorize=True)
        ckt_mis = CktClass2(init_trainable=mgr2.get_initial_vals(),
                            is_stochastic=False, solver=diffrax.Tsit5())
        out_a = _mismatch_solve(ckt_mis, x0, seed=1, t_span=(0.0, 2.0))
        out_b = _mismatch_solve(ckt_mis, x0, seed=2, t_span=(0.0, 2.0))
        record("mismatch different seeds differ", float(jnp.max(jnp.abs(out_a - out_b))) > 0)

        # Shem gradient
        finite, nonzero, loss_val, max_grad = _check_shem_grad(
            CktClass, mgr.get_initial_vals(), x0, t_span=(0.0, 1.0))
        record("Shem grad finite", finite, f"loss={loss_val:.3e}")
        record("Shem grad non-zero", nonzero, f"max|grad|={max_grad:.3e}")

    except Exception:
        traceback.print_exc()
        record("EBM suite", False, "uncaught exception")


# --- 3. SSM (CDG or plain fallback) ---


def test_ssm():
    print("\n[SSM] State-Space Model")
    try:
        from experiments.cross_arch_tolerance.models.ssm import load_model as load_ssm
        from neuro_analog.revised_ark_bridge.cdg_native.ssm import build_ssm, spike_test

        model = load_ssm(str(CKPT_ROOT / "ssm.pt"))

        # Spike test
        spike_ok = spike_test(n=2, sigma=0.0)
        record("spike_test(2-state)", spike_ok,
               "PASS means OptCompiler handles linear CDGs; FAIL means plain fallback is used")

        result = build_ssm(model, mismatch_sigma=0.0, force_plain=False)
        ckt_or_class, mgr_or_none, used_cdg = result

        record("build_ssm returns", True, f"used_cdg={used_cdg}")

        if not used_cdg:
            ckt = ckt_or_class  # already instantiated LinearSSMCkt
            h0 = jnp.zeros(8)
            out = _nominal_solve(ckt, h0, t_span=(0.0, 1.0), dt0=0.01, n_save=20)
            record("plain fallback solve finite", bool(jnp.all(jnp.isfinite(out))))

            # Mismatch on plain fallback
            CktClass_plain, _, _ = build_ssm(model, mismatch_sigma=0.05, force_plain=True)
            # CktClass_plain IS the instance already when used_cdg=False
            # Wait, build_ssm with force_plain=True returns (ckt, None, False) where ckt is instance
            ckt_mis = ckt_or_class  # but this was compiled with sigma=0, need new one
            # Actually ckt_or_class was built with sigma=0.0. Let's rebuild with sigma=0.05
            ckt_mis_instance, _, _ = build_ssm(model, mismatch_sigma=0.05, force_plain=True)
            out_a = _mismatch_solve(ckt_mis_instance, h0, seed=1, t_span=(0.0, 1.0))
            out_b = _mismatch_solve(ckt_mis_instance, h0, seed=2, t_span=(0.0, 1.0))
            record("mismatch different seeds differ", float(jnp.max(jnp.abs(out_a - out_b))) > 0)

            # Shem gradient on plain fallback
            # LinearSSMCkt init_trainable is a flat array
            a0 = ckt_mis_instance.a_trainable
            ti = TimeInfo(t0=0.0, t1=1.0, dt0=0.01, saveat=jnp.linspace(0.0, 1.0, 10))
            def loss_fn(a_trainable):
                ckt = ckt_mis_instance.__class__(a_trainable, ckt_mis_instance._has_B,
                                                  ckt_mis_instance._u, ckt_mis_instance._sigma,
                                                  solver=diffrax.Tsit5())
                # Can't easily reconstruct — plain classes have complex init signatures
                # Skip Shem grad for SSM plain fallback; document gap
                return jnp.sum(out_a ** 2)
            record("Shem grad", False, "SKIPPED — LinearSSMCkt init signature makes reconstruct-from-flat hard; gap")
        else:
            record("Shem grad", False, "SKIPPED — CDG path succeeded but not tested for gradients yet")

    except Exception:
        traceback.print_exc()
        record("SSM suite", False, "uncaught exception")


# --- 4. Neural ODE (plain BaseAnalogCkt) ---


def test_neural_ode():
    print("\n[Neural ODE] Plain BaseAnalogCkt subclass")
    try:
        from experiments.cross_arch_tolerance.models.neural_ode import load_model as load_ode
        from neuro_analog.revised_ark_bridge.plain_fallback.mlp_field import build_neural_ode

        model = load_ode(str(CKPT_ROOT / "neural_ode.pt"))
        ckt = build_neural_ode(model, mismatch_sigma=0.0)

        record("build_neural_ode instantiates", isinstance(ckt, BaseAnalogCkt),
               f"keys={list(ckt._keys)}")

        # Nominal solve
        z0 = jnp.array([0.5, -0.3])
        out = _nominal_solve(ckt, z0, t_span=(0.0, 1.0), dt0=0.01, n_save=20)
        record("nominal solve finite", bool(jnp.all(jnp.isfinite(out))))

        # Mismatch
        ckt_mis = build_neural_ode(model, mismatch_sigma=0.05)
        out_a = _mismatch_solve(ckt_mis, z0, seed=1, t_span=(0.0, 1.0))
        out_b = _mismatch_solve(ckt_mis, z0, seed=2, t_span=(0.0, 1.0))
        differ = float(jnp.max(jnp.abs(out_a - out_b)))
        record("mismatch different seeds differ", differ > 0, f"max_diff={differ:.3e}")

        # Shem gradient — MLPFieldCkt constructor takes weights dict, not init_trainable tuple.
        # Skipping for plain class; documented as known gap.
        record("Shem grad", False, "SKIPPED — MLPFieldCkt constructor takes weights dict, not flat init_trainable")

        # Oracle verification vs torchdiffeq
        try:
            from torchdiffeq import odeint
            import torch
            t_torch = torch.linspace(0.0, 1.0, 20)
            z0_torch = torch.tensor(np.array(z0), dtype=torch.float32)
            with torch.no_grad():
                traj_pt = odeint(model, z0_torch, t_torch, method='dopri5').numpy()
            traj_ckt = np.array(out)
            if traj_ckt.ndim == 1:
                traj_ckt = traj_ckt.reshape(1, -1)
            # Compare last point
            err = np.max(np.abs(traj_ckt[-1] - traj_pt[-1]))
            record("oracle agreement (last point)", err < 0.5,
                   f"max_abs_diff={err:.3e}  (NOTE: time-augmented MLP may differ due to t-concat)")
        except Exception as e:
            record("oracle agreement", False, f"torchdiffeq error: {e}")

    except Exception:
        traceback.print_exc()
        record("Neural ODE suite", False, "uncaught exception")


# --- 5. Flow (plain BaseAnalogCkt) ---


def test_flow():
    print("\n[Flow] Plain BaseAnalogCkt subclass")
    try:
        from experiments.cross_arch_tolerance.models.flow import load_model as load_flow
        from neuro_analog.revised_ark_bridge.plain_fallback.mlp_field import build_flow

        model = load_flow(str(CKPT_ROOT / "flow.pt"))
        ckt = build_flow(model, mismatch_sigma=0.0)

        record("build_flow instantiates", isinstance(ckt, BaseAnalogCkt))

        z0 = jnp.array([0.5, -0.3])
        out = _nominal_solve(ckt, z0, t_span=(0.0, 1.0), dt0=0.01, n_save=20)
        record("nominal solve finite", bool(jnp.all(jnp.isfinite(out))))

        # Mismatch
        ckt_mis = build_flow(model, mismatch_sigma=0.05)
        out_a = _mismatch_solve(ckt_mis, z0, seed=1, t_span=(0.0, 1.0))
        out_b = _mismatch_solve(ckt_mis, z0, seed=2, t_span=(0.0, 1.0))
        record("mismatch different seeds differ", float(jnp.max(jnp.abs(out_a - out_b))) > 0)

        # Shem gradient — same gap as Neural ODE
        record("Shem grad", False, "SKIPPED — MLPFieldCkt constructor takes weights dict")

    except Exception:
        traceback.print_exc()
        record("Flow suite", False, "uncaught exception")


# --- 6. Diffusion (CLD SDE) ---


def test_diffusion():
    print("\n[Diffusion] CLD SDE plain BaseAnalogCkt")
    try:
        from experiments.cross_arch_tolerance.models.diffusion import load_model as load_diff
        from neuro_analog.revised_ark_bridge.plain_fallback.diffusion import build_diffusion

        model = load_diff(str(CKPT_ROOT / "diffusion.pt"))
        ckt = build_diffusion(model, mismatch_sigma=0.0)

        record("build_diffusion instantiates", isinstance(ckt, BaseAnalogCkt),
               f"is_stochastic={ckt.is_stochastic}, beta_len={ckt._n_beta}")

        # Nominal (deterministic drift only) solve
        x0 = jnp.zeros(64)
        v0 = jnp.zeros(64)
        state0 = jnp.concatenate([x0, v0])

        # For deterministic test, temporarily flip is_stochastic off
        # (BaseAnalogCkt is frozen; we can't mutate. Instead we rely on noise_seed
        # but noise is still injected. Let's just check solve runs.)
        out = _nominal_solve(ckt, state0, t_span=(0.0, 0.5), dt0=0.01, n_save=10)
        record("stochastic solve finite", bool(jnp.all(jnp.isfinite(out))),
               "SDE solve returns finite values (noise present)")

        # Mismatch
        ckt_mis = build_diffusion(model, mismatch_sigma=0.05)
        out_a = _mismatch_solve(ckt_mis, state0, seed=1, t_span=(0.0, 0.5))
        out_b = _mismatch_solve(ckt_mis, state0, seed=2, t_span=(0.0, 0.5))
        # Stochastic outputs differ both from noise AND mismatch; hard to isolate
        # Just verify they are finite
        record("mismatch outputs finite", bool(jnp.all(jnp.isfinite(out_a)) and jnp.all(jnp.isfinite(out_b))),
               "SDE + mismatch: outputs finite (noise dominates diff)")

        # Shem gradient through SDE solve
        # NOTE: CLDCkt builds a_trainable internally from score_weights dict.
        # There is no external flat-array constructor for mismatch-aware retraining.
        record("Shem grad", False,
               "SKIPPED — CLDCkt.__init__ builds a_trainable internally from score_weights; "
               "no external entry point to pass a custom flat array. DESIGN GAP.")

    except Exception:
        traceback.print_exc()
        record("Diffusion suite", False, "uncaught exception")


# --- Summary ---


def main():
    print("=" * 70)
    print("  Comprehensive Test Suite")
    print("=" * 70)

    test_deq()
    test_ebm()
    test_ssm()
    test_neural_ode()
    test_flow()
    test_diffusion()

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"  FAILED: {name} — {detail}")
    print(f"  {passed}/{total} checks passed")
    print("=" * 70)

    # Gap analysis
    print("\n[KNOWN GAPS]")
    print("""
1. SSM Shem gradient (plain fallback):
   LinearSSMCkt.__init__ takes (A, B, u, sigma, solver) and internally flattens
   A and B into a_trainable. There is no constructor accepting a pre-flattened
   a_trainable array. To make it gradient-testable, add either a classmethod
   `from_flat(a_trainable, shapes, ...)` or refactor __init__ to accept
   init_trainable directly.

2. Diffusion (CLD) Shem gradient:
   CLDCkt.__init__ takes score_weights dict and internally flattens to
   a_trainable. No external flat-array constructor exists.
   The CDG-native path avoids this because OptCompiler generates make_args that
   reads self.a_trainable directly.

3. Oracle verification granularity:
   Neural ODE oracle (torchdiffeq) uses dopri5 adaptive stepping; the Ark solve
   uses Tsit5 with fixed saveat. Time-augmented MLP means t is concatenated as
   an input feature. Trajectories are close but not identical due to different
   solvers and time augmentation differences.

4. SSM CDG path:
   spike_test(n=2) currently fails on OptCompiler for linear specs. The fallback
   plain class works. The CDG path for linear SSM remains unvalidated.

5. Diffusion mismatch isolation:
   is_stochastic=True means noise dominates trajectory differences between
   mismatch seeds. Isolating mismatch-only effects requires rebuilding the class
   with is_stochastic=False (not possible on frozen Equinox modules without
   re-instantiation).
""")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
