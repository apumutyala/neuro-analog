"""
Ark CDG bridge: lower neuro-analog IR artifacts into Ark's Constrained
Dynamical Graph representation for hardware-aware retraining.

Supported architecture
----------------------
Hopfield-type Neural ODE:  dx/dt = -x + J * tanh(x) + b + K * u

This is mathematically equivalent to Ark's CNN/CANN paradigm (Cellular /
Continuous Analog Neural Network).  We define a typed CDGSpec for it,
build a CDG from PyTorch weight matrices, and hand off to Ark's OptCompiler
to generate a differentiable BaseAnalogCkt subclass — with per-weight
AttrDefMismatch modelling — that Ark's adjoint optimizer can retrain.

Usage
-----
    from neuro_analog.ark_bridge import compile_neural_ode_cdg

    CktClass, mgr = compile_neural_ode_cdg(J, b, K, mismatch_sigma=0.05)
    import diffrax
    ckt = CktClass(init_trainable=mgr.get_initial_vals(), is_stochastic=False, solver=diffrax.Tsit5())
"""

from .neural_ode_cdg import (
    make_neural_ode_spec,
    neural_ode_to_cdg,
    compile_neural_ode_cdg,
)

__all__ = [
    "make_neural_ode_spec",
    "neural_ode_to_cdg",
    "compile_neural_ode_cdg",
]
