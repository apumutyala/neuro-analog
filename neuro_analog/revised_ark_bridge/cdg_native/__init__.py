"""CDG-native compiled circuits (Path A): DEQ, EBM, SSM."""

from .deq import build_deq, check_contraction
from .ebm import build_ebm
from .ssm import build_ssm, spike_test, LinearSSMCkt

__all__ = ["build_deq", "check_contraction", "build_ebm", "build_ssm", "spike_test", "LinearSSMCkt"]
