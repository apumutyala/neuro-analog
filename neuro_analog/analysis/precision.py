import math
from neuro_analog.ir import AnalogGraph
from neuro_analog.ir.node import AnalogNode

def compute_snr_db(node: AnalogNode, signal_rms: float) -> float | None:
    """Compute Signal-to-Noise Ratio (SNR) in dB for a node given its hardware NoiseSpec."""
    if not node.noise or node.noise.sigma <= 0:
        return None
    
    return 20.0 * math.log10(signal_rms / node.noise.sigma)

def flag_snr_violations(
    graph: AnalogGraph,
    signal_rms: float = 1.0,
    target_snr_db: float = 30.0,
) -> list[dict]:
    """Identify analog nodes where hardware noise would violate the SNR target."""
    violations = []
    
    for node in graph.nodes:
        if node.domain.name != "ANALOG" or not node.noise:
            continue
            
        snr = compute_snr_db(node, signal_rms)
        if snr is not None and snr < target_snr_db:
            violations.append({
                "node_id": getattr(node, 'node_id', node.name),
                "name": node.name,
                "snr_db": snr,
                "target_snr_db": target_snr_db,
                "margin_db": target_snr_db - snr,
            })
            
    return violations
