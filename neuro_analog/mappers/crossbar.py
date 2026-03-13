from __future__ import annotations

from neuro_analog.ir import AnalogGraph, OpType, NoiseSpec
from neuro_analog.ir.types import CrossbarSpec

class CrossbarMapper:
    """Maps hardware specifications to analog crossbar MVM nodes."""
    
    def __init__(self, spec: CrossbarSpec | None = None):
        self.spec = spec or CrossbarSpec()
        
    def annotate_graph(self, graph: AnalogGraph) -> None:
        """Annotate MVM nodes with crossbar noise specifications."""
        # Quantization noise from ADC/DAC at the crossbar boundary
        # Assume LSB-equivalent input noise uniformly distributed (variance = LSB^2 / 12)
        # For simplicity in equivalent input units, we use sigma = 1 / (2^bits)
        sigma = 1.0 / (2 ** self.spec.precision_bits)
        bandwidth_hz = 1e6 # Typical 1MHz crossbar operating frequency
        
        noise = NoiseSpec(
            kind="composite",
            sigma=sigma,
            bandwidth_hz=bandwidth_hz,
        )
        
        for node in graph.nodes:
            if node.op_type == OpType.MVM:
                node.noise = noise
