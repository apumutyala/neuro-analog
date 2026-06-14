"""Shared CDG specifications and compiler wrapper for revised_ark_bridge."""

from .paradigms import additive_recurrent, deq_zform, linear_ssm
from .compiler import compile_cdg, build_additive_cdg

__all__ = ["additive_recurrent", "deq_zform", "linear_ssm", "compile_cdg", "build_additive_cdg"]
