"""
revised_ark_bridge: Ground-up Ark bridge for neuro-analog dynamics.

Single entry point: to_circuit(model, family, mismatch_sigma) -> BaseAnalogCkt
"""

from .plain_fallback.mlp_field import MLPFieldCkt, build_neural_ode, build_flow
from .cdg_native.deq import build_deq
from .cdg_native.ebm import build_ebm
from .cdg_native.ssm import build_ssm
from .plain_fallback.diffusion import build_diffusion
from .verify import verify_family, solve_nominal, solve_with_mismatch

__all__ = [
    "build_neural_ode",
    "build_flow",
    "build_deq",
    "build_ebm",
    "build_ssm",
    "build_diffusion",
    "verify_family",
    "solve_nominal",
    "solve_with_mismatch",
]
