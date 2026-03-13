from __future__ import annotations
import math

from neuro_analog.ir import AnalogGraph, OpType, NoiseSpec
from neuro_analog.ir.types import IntegratorSpec

class IntegratorMapper:
    """Maps hardware specifications to analog integrator/decay nodes."""
    
    def __init__(self, spec: IntegratorSpec | None = None):
        self.spec = spec or IntegratorSpec()
        
    def annotate_graph(self, graph: AnalogGraph) -> None:
        """Annotate INTEGRATION and DECAY nodes with thermal noise specs."""
        # Thermal kT/C noise model.
        # k_B = 1.38e-23 J/K, T = 300K, C ~ 1pF (typical on-chip capacitor)
        # sigma = sqrt(k_B * T / C)
        k_B = 1.380649e-23
        T = 300.0
        C = 1e-12
        sigma = math.sqrt(k_B * T / C)
        
        noise = NoiseSpec(
            kind="thermal",
            sigma=sigma,
            bandwidth_hz=self.spec.bandwidth_hz,
            corr_length=self.spec.time_constant_s,
        )
        
        for node in graph.nodes:
            if node.op_type in (OpType.INTEGRATION, OpType.DECAY):
                node.noise = noise
