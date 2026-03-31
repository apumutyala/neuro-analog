"""Transient Stability Experiment: Long-Term Inference Drift Under Fixed Structural Mismatch.

Motivation
----------
Analog hardware has two distinct noise sources:
  1. Structural mismatch (delta ~ N(1, sigma^2)) -- static, baked into fabrication.
     Same defect pattern every inference. Cannot be averaged away.
  2. Thermal noise (Johnson-Nyquist, kT/C) -- dynamic, re-sampled each inference.
     Can be reduced by averaging or wider circuits.

This experiment asks: given a chip with FIXED structural mismatch (sigma=0.05),
how does output quality drift over 5,000 consecutive inferences as thermal
noise accumulates stochastically on top of the fixed defect pattern?

Two architectures are compared:
  - Neural ODE (continuous-depth): RC integrator accumulates thermal noise
    along the ODE trajectory. Error compounds over integration time.
  - DEQ (implicit fixed-point): fixed-point iteration converges to z*(delta),
    a shifted attractor. Mismatch changes the basin structure; thermal noise
    perturbs each iteration step. We track convergence residual and iteration count.

Key quantity: per-inference output deviation from the digital baseline:
  Neural ODE:  Δ_i = (1/N) Σ ||x_gen_analog_i - x_gen_digital||^2
  DEQ:         Δ_i = (1/N) Σ ||z*_analog_i - z*_digital||^2
  DEQ iters:   k_i = number of fixed-point iterations to convergence

The structural mismatch delta is sampled ONCE and held fixed for all 5,000 inferences.
Thermal noise is re-sampled independently each inference (physical reality).

Hardware analogy
----------------
This models a chip running 24/7 on an edge deployment with a manufacturing defect.
The DEQ's fixed-point attractor z*(delta) is systematically shifted -- every inference
produces biased outputs. The Neural ODE's RC integrator adds stochastic drift
ON TOP of the structural bias -- outputs scatter around the biased trajectory.

This is why Hopfield-substrate DEQs (analog ODE relaxation) are superior:
the continuous dynamics naturally average out thermal noise during the relaxation,
whereas discrete fixed-point iteration amplifies per-step noise through the
Jacobian (I - ∂f/∂z)^{-1}.
"""

import math
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuro_analog.simulator import analogize, set_all_noise, analog_odeint

CKPT_DIR = Path(__file__).parent / "checkpoints"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIGMA_FIXED = 0.05       # structural mismatch -- fixed for the whole experiment
N_INFER = 5000           # number of consecutive inferences
BATCH = 64               # samples per inference
SEED_DIGITAL = 0         # fixed seed so digital baseline is deterministic
WINDOW = 50              # smoothing window for rolling mean line


# ── Helpers ───────────────────────────────────────────────────────────────

def _rolling_mean(x, w):
    """Simple 1-D rolling mean."""
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ── Neural ODE transient ──────────────────────────────────────────────────

def run_neural_ode(n_infer=N_INFER):
    """Track output deviation over n_infer inferences on fixed-mismatch analog chip."""
    from experiments.cross_arch_tolerance.models.neural_ode import load_model

    ckpt = CKPT_DIR / "neural_ode.pt"
    if not ckpt.exists():
        print("  [transient] neural_ode.pt not found -- skipping.")
        return None

    print(f"  [NeuralODE] loading checkpoint...")
    # Detect checkpoint hidden size from saved weights (may differ from current create_model())
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    _hidden = state["net.0.bias"].shape[0]
    from experiments.cross_arch_tolerance.models.neural_ode import _TimeAugMLP
    digital = _TimeAugMLP(dim=2, hidden=_hidden)
    digital.load_state_dict(state)
    digital = digital.to(_DEVICE)
    digital.eval()
    print(f"  [NeuralODE] checkpoint hidden={_hidden} (retrain pending if hidden=20)")

    # Analogize once -- mismatch delta is now frozen in the weights
    analog = analogize(digital, sigma_mismatch=SIGMA_FIXED)
    analog.eval()
    # Thermal ON (re-sampled each inference), mismatch already baked in
    set_all_noise(analog, thermal=True, mismatch=True, quantization=False)

    t_span = torch.tensor([0.0, 1.0], device=_DEVICE)
    dt = 0.025  # 40 steps -- matches training

    deviations = []
    rng = np.random.default_rng(1)  # separate rng for z0 across inferences

    print(f"  [NeuralODE] running {n_infer} inferences at sigma={SIGMA_FIXED}...")
    for i in range(n_infer):
        # Fresh z0 each inference (new sample from prior -- simulates real deployment)
        z0_np = rng.standard_normal((BATCH, 2)).astype(np.float32)
        z0 = torch.tensor(z0_np, device=_DEVICE)

        with torch.no_grad():
            # Digital baseline (noiseless, same z0)
            x_dig = analog_odeint(digital, z0, t_span, dt=dt, noise_sigma=0.0)
            # Analog with fixed mismatch + fresh thermal noise
            x_ana = analog_odeint(analog, z0, t_span, dt=dt)

        dev = ((x_dig - x_ana) ** 2).mean().item()
        deviations.append(dev)

        if (i + 1) % 500 == 0:
            print(f"    inference {i+1}/{n_infer}: mean_dev={np.mean(deviations[-500:]):.5f}")

    return np.array(deviations)


# ── DEQ transient ─────────────────────────────────────────────────────────

def run_deq(n_infer=N_INFER):
    """Track fixed-point state deviation and iteration count over n_infer inferences."""
    from experiments.cross_arch_tolerance.models.deq import load_model, _get_data

    ckpt = CKPT_DIR / "deq.pt"
    if not ckpt.exists():
        print("  [transient] deq.pt not found -- skipping.")
        return None, None

    print(f"  [DEQ] loading checkpoint...")
    digital = load_model(str(ckpt))
    digital.eval()

    analog = analogize(digital, sigma_mismatch=SIGMA_FIXED)
    analog.eval()
    set_all_noise(analog, thermal=True, mismatch=True, quantization=False)

    (X_train, _), (X_test, _) = _get_data()
    X_test = X_test.to(_DEVICE)
    n_test = min(BATCH, len(X_test))
    X_batch = X_test[:n_test]

    deviations = []
    iter_counts = []

    print(f"  [DEQ] running {n_infer} inferences at sigma={SIGMA_FIXED}...")
    for i in range(n_infer):
        with torch.no_grad():
            # Digital fixed-point state
            _, z_dig = digital(X_batch)

            # Analog: track how many iterations to convergence
            z = torch.zeros(n_test, digital.z_dim if hasattr(digital, 'z_dim') else 64, device=_DEVICE)
            scale = math.sqrt(z.shape[-1])
            k = 0
            tol = 1e-4
            max_iter = 100  # extend beyond normal 30 to detect non-convergence
            for k in range(max_iter):
                z_next = analog.f_theta(z, X_batch)
                diff = (z_next - z).norm(dim=-1).max().item() / scale
                z = z_next
                if diff < tol:
                    break
            z_star_analog = z

        dev = ((z_dig - z_star_analog) ** 2).mean().item()
        deviations.append(dev)
        iter_counts.append(k + 1)

        if (i + 1) % 500 == 0:
            mean_iters = np.mean(iter_counts[-500:])
            mean_dev = np.mean(deviations[-500:])
            print(f"    inference {i+1}/{n_infer}: mean_dev={mean_dev:.5f}, mean_iters={mean_iters:.1f}")

    return np.array(deviations), np.array(iter_counts)


# ── Plot ──────────────────────────────────────────────────────────────────

def plot(ode_devs, deq_devs, deq_iters):
    n_panels = sum([
        ode_devs is not None,
        deq_devs is not None,
        deq_iters is not None,
    ])
    # Always 3 panels if DEQ ran (devs + iters), else 1
    if deq_devs is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        ax_ode, ax_deq_dev, ax_deq_iter = axes
    else:
        fig, ax_ode = plt.subplots(1, 1, figsize=(6, 4.5))
        ax_deq_dev = ax_deq_iter = None

    fig.suptitle(
        f"Transient Stability Under Fixed Structural Mismatch  (sigma={SIGMA_FIXED}, N={N_INFER:,} inferences)\n"
        "Structural defect delta sampled once (frozen); thermal noise re-sampled each inference.",
        fontsize=11, y=1.03
    )

    infer_idx = np.arange(1, N_INFER + 1)

    # ── Panel 1: Neural ODE output deviation ──────────────────────────────
    if ode_devs is not None:
        ax = ax_ode
        ax.scatter(infer_idx, ode_devs, s=0.5, alpha=0.15, color="#27ae60", rasterized=True)
        if len(ode_devs) >= WINDOW:
            rm = _rolling_mean(ode_devs, WINDOW)
            rm_idx = np.arange(WINDOW, N_INFER + 1)
            ax.plot(rm_idx, rm, color="#1a5e38", linewidth=1.5, label=f"{WINDOW}-infer rolling mean")
        ax.axhline(np.mean(ode_devs), color="#e74c3c", linewidth=1.0, linestyle="--",
                   label=f"global mean = {np.mean(ode_devs):.4f}")
        ax.set_xlabel("Inference #")
        ax.set_ylabel("Mean Squared Deviation  ||x_analog − x_digital||^2")
        ax.set_title("Neural ODE\nRC integrator: thermal noise accumulates along trajectory")
        ax.legend(fontsize=8)

    # ── Panel 2: DEQ fixed-point state deviation ──────────────────────────
    if deq_devs is not None:
        ax = ax_deq_dev
        ax.scatter(infer_idx, deq_devs, s=0.5, alpha=0.15, color="#e67e22", rasterized=True)
        if len(deq_devs) >= WINDOW:
            rm = _rolling_mean(deq_devs, WINDOW)
            rm_idx = np.arange(WINDOW, N_INFER + 1)
            ax.plot(rm_idx, rm, color="#8e4e0e", linewidth=1.5, label=f"{WINDOW}-infer rolling mean")
        ax.axhline(np.mean(deq_devs), color="#e74c3c", linewidth=1.0, linestyle="--",
                   label=f"global mean = {np.mean(deq_devs):.4f}")
        ax.set_xlabel("Inference #")
        ax.set_ylabel("||z*_analog − z*_digital||^2  (fixed-point state)")
        ax.set_title("DEQ (Discrete Fixed-Point)\nMismatch shifts attractor z*(delta); thermal noise scatters around it")
        ax.legend(fontsize=8)

    # ── Panel 3: DEQ iteration count to convergence ───────────────────────
    if deq_iters is not None:
        ax = ax_deq_iter
        ax.scatter(infer_idx, deq_iters, s=0.5, alpha=0.15, color="#8e44ad", rasterized=True)
        if len(deq_iters) >= WINDOW:
            rm = _rolling_mean(deq_iters, WINDOW)
            rm_idx = np.arange(WINDOW, N_INFER + 1)
            ax.plot(rm_idx, rm, color="#5b2c6f", linewidth=1.5, label=f"{WINDOW}-infer rolling mean")
        digital_iters = 30  # nominal max_iter used during eval
        ax.axhline(digital_iters, color="#2980b9", linewidth=1.0, linestyle="--",
                   label=f"digital nominal ({digital_iters} iters)")
        ax.set_xlabel("Inference #")
        ax.set_ylabel("Fixed-point iterations to convergence")
        ax.set_title("DEQ Convergence Cost\nAnalog mismatch may require more iterations (or diverge)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, min(deq_iters.max() * 1.2, 105))

    # Shared note at bottom
    fig.text(
        0.5, -0.04,
        f"Fixed structural mismatch: delta_ij ~ N(1, {SIGMA_FIXED}^2) per weight -- sampled once, held constant for all {N_INFER:,} inferences.\n"
        "Thermal noise: sigma_RC = sqrt(kT/C) ≈ 6.4x10-5 -- re-sampled independently each inference (models real hardware).\n"
        "Neural ODE substrate: RC integrator (Johnson-Nyquist noise per Euler step).  "
        "DEQ substrate: discrete fixed-point iteration (thermal noise per iteration step).",
        ha="center", fontsize=8, color="#555", style="italic"
    )

    fig.tight_layout()
    out_png = FIG_DIR / "fig_transient_stability.png"
    out_pdf = FIG_DIR / "fig_transient_stability.pdf"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_pdf), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-infer", type=int, default=N_INFER,
                        help=f"Number of consecutive inferences (default: {N_INFER})")
    parser.add_argument("--sigma", type=float, default=SIGMA_FIXED,
                        help=f"Fixed structural mismatch sigma (default: {SIGMA_FIXED})")
    parser.add_argument("--only", choices=["neural_ode", "deq"], default=None,
                        help="Run only one architecture")
    args = parser.parse_args()

    SIGMA_FIXED = args.sigma
    N_INFER = args.n_infer

    print(f"\nTransient Stability Experiment")
    print(f"  sigma_structural = {SIGMA_FIXED} (fixed for all inferences)")
    print(f"  N_inferences = {N_INFER:,}")
    print(f"  Device: {_DEVICE}\n")

    ode_devs = None
    deq_devs = None
    deq_iters = None

    if args.only != "deq":
        ode_devs = run_neural_ode(N_INFER)

    if args.only != "neural_ode":
        deq_devs, deq_iters = run_deq(N_INFER)

    plot(ode_devs, deq_devs, deq_iters)

    # Summary statistics
    print("\n-- Summary --")
    if ode_devs is not None:
        print(f"Neural ODE  mean_dev={np.mean(ode_devs):.5f}  std={np.std(ode_devs):.5f}")
    if deq_devs is not None:
        print(f"DEQ         mean_dev={np.mean(deq_devs):.5f}  std={np.std(deq_devs):.5f}")
    if deq_iters is not None:
        frac_slow = (deq_iters > 30).mean()
        frac_maxed = (deq_iters >= 100).mean()
        print(f"DEQ         mean_iters={np.mean(deq_iters):.1f}  "
              f"frac_exceeded_30={frac_slow:.1%}  frac_non_converged={frac_maxed:.1%}")
