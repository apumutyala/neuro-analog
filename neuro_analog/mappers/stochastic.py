from __future__ import annotations
import math

from neuro_analog.ir import AnalogGraph, OpType, NoiseSpec

class StochasticMapper:
    """Maps stochastic nodes to explicit hardware noise targets."""
    
    def __init__(self, i_bias_ua: float = 1.0, bandwidth_hz: float = 1e6):
        self.i_bias_ua = i_bias_ua
        self.bandwidth_hz = bandwidth_hz
        
    def annotate_graph(self, graph: AnalogGraph) -> None:
        """Annotate NOISE_INJECTION and SAMPLE nodes with their noise parameters."""
        # Shot noise model: I_noise = sqrt(2 * q * I_bias * BW)
        q = 1.602e-19
        i_bias_a = self.i_bias_ua * 1e-6
        i_noise_rms = math.sqrt(2 * q * i_bias_a * self.bandwidth_hz)
        
        # In normalized units, we assume this implies a relative sigma (e.g. 1% noise)
        # Here we map the RMS noise to a configurable parameter, default ~1e-3
        sigma_shot = max(1e-3, float(i_noise_rms * 1e6)) # simplified mapping
        
        shot_noise = NoiseSpec(
            kind="shot",
            sigma=sigma_shot,
            bandwidth_hz=self.bandwidth_hz,
        )
        
        thermal_noise = NoiseSpec(
            kind="thermal",
            sigma=0.1,  # typical for sMTJ thermal fluctuation
            bandwidth_hz=self.bandwidth_hz,
        )
        
        for node in graph.nodes:
            if node.op_type == OpType.NOISE_INJECTION:
                node.noise = shot_noise
            elif node.op_type == OpType.SAMPLE:
                node.noise = thermal_noise
            elif node.op_type == OpType.GIBBS_STEP:
                # DTCA thermodynamic sampler: thermal noise is the computational resource.
                # Appendix K (Jelinčič et al. 2025): τ_rng ≈ 100 ns RNG flip time.
                # Appendix E: E_rng ≈ 350 aJ per sampled bit.
                # sigma=0.0 because SNR is not a meaningful metric here — thermal
                # fluctuations are the desired randomness, not a calibration error.
                node.noise = NoiseSpec(
                    kind="thermal",
                    sigma=0.0,
                    bandwidth_hz=1.0 / 100e-9,  # 10 MHz from τ_rng = 100 ns
                )
                node.metadata.setdefault("tau_rng_ns", 100.0)   # Appendix K
                node.metadata.setdefault("E_rng_aJ", 350.0)     # Appendix E
                node.metadata["thermodynamic_sampler"] = True
