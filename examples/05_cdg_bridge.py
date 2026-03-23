#!/usr/bin/env python3
"""
CDG bridge: PyTorch Neural ODE weights -> Ark Constrained Dynamical Graph.

This script shows the full pipeline:
  1. Define a small Hopfield-type Neural ODE (dx/dt = -x + J*tanh(x) + b)
  2. Use neuro_analog.ark_bridge to build an Ark CDGSpec + CDG topology
  3. Compile through Ark's OptCompiler -> BaseAnalogCkt subclass
  4. Run a nominal forward pass (ideal weights, no mismatch sampling)
  5. Run a mismatch forward pass (stochastic, per-weight noise injected)

Architecture
------------
    dx/dt = -x + J * tanh(x) + b

This is the Hopfield / Cohen-Grossberg normal form.  It is structurally
identical to Ark's CNN/CANN paradigm (examples/cnn/spec.py) and maps
exactly to Ark's CDGSpec DSL.

Prerequisites
-------------
    git clone https://github.com/WangYuNeng/Ark
    pip install -e ./Ark
    pip install jax diffrax equinox lineax

Usage
-----
    python examples/05_cdg_bridge.py               # 2-D state, sigma = 0.05
    python examples/05_cdg_bridge.py --n 4         # 4-D state
    python examples/05_cdg_bridge.py --sigma 0.10  # higher mismatch
    python examples/05_cdg_bridge.py --sigma 0.0   # ideal (no mismatch)
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def sep(title=""):
    w = 62
    print(f"\n{'='*w}")
    if title:
        print(f"  {title}")
        print(f"{'='*w}")


# -- Ark availability check ----------------------------------------------------

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


# -- Demo weights --------------------------------------------------------------

def make_demo_weights(n: int, seed: int = 0):
    """Return small random J (n x n) and b (n,) suitable for a stable Hopfield ODE."""
    import numpy as np
    rng = np.random.default_rng(seed)
    # Symmetric J with spectral radius < 1 -> network converges to fixed point
    A = rng.standard_normal((n, n)) * 0.5
    J = (A + A.T) / (2 * n)
    b = rng.standard_normal(n) * 0.1
    return J, b


# -- Step 1: build CDG ---------------------------------------------------------

def build_cdg(n: int, J, b, sigma: float):
    from neuro_analog.ark_bridge import neural_ode_to_cdg

    sep("Step 1 -- Build CDG topology")

    cdg, spec, mgr, state_nodes, inp_nodes = neural_ode_to_cdg(
        J=J,
        b=b,
        K=None,          # autonomous system (no external input)
        mismatch_sigma=sigma,
    )

    n_nodes  = len(cdg.nodes)
    n_edges  = len(cdg.edges)
    n_analog = len(mgr.analog)

    print(f"  CDG nodes:          {n_nodes}   (StateVar + OutUnit = 2 x {n})")
    print(f"  CDG edges:          {n_edges}")
    print(f"    MapEdge  (ReadOut + SelfDecay):  {2*n}")
    nz = sum(1 for i in range(n) for j in range(n) if abs(float(J[i, j])) >= 1e-12)
    print(f"    FlowEdge (JWeight, non-zero):    {nz}  of {n*n}")
    print(f"  Trainable analogs:  {n_analog}   ({n} bias + {nz} weight)")
    print(f"  Mismatch sigma:     {sigma}")

    return cdg, spec, mgr, state_nodes


# -- Step 2: compile CDG -> BaseAnalogCkt --------------------------------------

def compile_cdg(n: int, J, b, sigma: float):
    import diffrax
    from neuro_analog.ark_bridge import compile_neural_ode_cdg
    from ark.optimization.base_module import BaseAnalogCkt

    sep("Step 2 -- Compile CDG -> BaseAnalogCkt")

    CktClass, mgr = compile_neural_ode_cdg(
        J=J,
        b=b,
        K=None,
        mismatch_sigma=sigma,
        prog_name="HopfieldODE",
    )

    assert issubclass(CktClass, BaseAnalogCkt), "Compiled class is not a BaseAnalogCkt"

    # Instantiate with nominal (ideal) weights; Tsit5 is the standard Ark solver
    init_vals = mgr.get_initial_vals()
    ckt_nominal = CktClass(
        init_trainable=init_vals,
        is_stochastic=False,
        solver=diffrax.Tsit5(),
    )

    print(f"  Class name:         {CktClass.__name__}")
    print(f"  is BaseAnalogCkt:   True")
    print(f"  a_trainable shape:  {ckt_nominal.a_trainable.shape}")
    print(f"  is_stochastic:      {ckt_nominal.is_stochastic}")
    print(f"  solver:             {type(ckt_nominal.solver).__name__}")

    if sigma > 0.0:
        ckt_mismatch = CktClass(
            init_trainable=init_vals,
            is_stochastic=True,
            solver=diffrax.Tsit5(),
        )
        print(f"  Mismatch instance:  created (is_stochastic=True, sigma={sigma})")
    else:
        ckt_mismatch = None
        print(f"  Mismatch instance:  skipped (sigma=0)")

    return ckt_nominal, ckt_mismatch


# -- Step 3: forward pass ------------------------------------------------------

def run_forward(n: int, ckt_nominal, ckt_mismatch, sigma: float):
    import jax.numpy as jnp
    from ark.optimization.base_module import TimeInfo

    sep("Step 3 -- Forward pass")

    y0        = jnp.zeros(n)
    switch    = jnp.array([])          # no switching events
    time_info = TimeInfo(t0=0.0, t1=1.0, dt0=0.1, saveat=jnp.array([1.0]))

    # Nominal (ideal) pass
    result_nom = ckt_nominal(
        time_info,
        y0,
        switch=switch,
        args_seed=0,
        noise_seed=0,
    )
    print(f"  Nominal output shape:   {result_nom.shape}")
    print(f"  Nominal output:         {result_nom}")

    # Mismatch pass: different args_seed samples different per-weight mismatch
    if ckt_mismatch is not None:
        result_mis = ckt_mismatch(
            time_info,
            y0,
            switch=switch,
            args_seed=42,   # args_seed drives mismatch sampling
            noise_seed=0,
        )
        delta = jnp.abs(result_mis - result_nom)
        print(f"  Mismatch output:        {result_mis}")
        print(f"  |delta| (nom vs mis.):  {delta}  (mean={float(delta.mean()):.4f})")

    print(f"\n  Forward pass OK  (state_dim={n})")


# -- Main ----------------------------------------------------------------------

def main(n: int = 2, sigma: float = 0.05) -> None:
    check_ark()

    sep(f"05_cdg_bridge.py -- Hopfield ODE CDG  (n={n}, sigma={sigma})")
    print("  Architecture:  dx/dt = -x + J*tanh(x) + b")
    print("  This is Ark's CNN/CANN paradigm; maps 1-to-1 to CDGSpec.")

    J, b = make_demo_weights(n)

    build_cdg(n, J, b, sigma)
    ckt_nominal, ckt_mismatch = compile_cdg(n, J, b, sigma)
    run_forward(n, ckt_nominal, ckt_mismatch, sigma)

    sep("Using a trained PyTorch model")
    print("""
  If you have a trained NeuralODE from the cross-arch experiments:

    import torch
    from experiments.cross_arch_tolerance.models.neural_ode import NeuralODE
    from neuro_analog.ark_bridge import compile_neural_ode_cdg

    model = NeuralODE.load("outputs/neural_ode_run/model.pt")
    J = model.cell.linear.weight.detach().numpy()
    b = model.cell.linear.bias.detach().numpy()

    CktClass, mgr = compile_neural_ode_cdg(J, b, mismatch_sigma=0.05)
    import diffrax
    ckt = CktClass(init_trainable=mgr.get_initial_vals(), is_stochastic=False, solver=diffrax.Tsit5())

  The CDGSpec maps J[i,j] -> FlowEdge.g (with AttrDefMismatch),
  b[i]   -> StateVar.b  (analog trainable),
  and the self-decay (-x) + activation (tanh) wiring is handled by
  the SelfDecay and ReadOut/JWeight production rules automatically.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CDG bridge demo: Hopfield Neural ODE -> Ark BaseAnalogCkt"
    )
    parser.add_argument("--n", type=int, default=2,
                        help="State dimension (default: 2)")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Mismatch sigma (default: 0.05, 0.0 = ideal)")
    args = parser.parse_args()
    main(args.n, args.sigma)
