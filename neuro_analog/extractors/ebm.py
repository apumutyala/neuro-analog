"""
Energy-Based Model (EBM) theoretical analyzer.

No pretrained model loading needed. Builds the AnalogGraph from the
theoretical Boltzmann machine / Hopfield structure.

Key insight: EBMs are the MOST analog-amenable architecture (score ~0.95)
because Gibbs sampling is physically identical to how RRAM crossbar arrays
with sMTJ (stochastic Magnetic Tunnel Junction) cells naturally operate.

Boltzmann machine Gibbs sampling:
    h_i = Σ_j W_ij x_j + b_i      (ANALOG: crossbar MVM — Kirchhoff's current law)
    x_i ~ Bernoulli(σ(h_i / T))   (ANALOG: p-bit with tunable bias voltage)

Zero D/A boundaries in steady-state operation — fully analog feedback loop.

Modern Hopfield networks (Ramsauer et al. 2020) = transformer attention:
    ξ_new = X · softmax(β · X^T · ξ)
    X^T·ξ → crossbar MVM (ANALOG)
    softmax → DIGITAL (or approximate analog)
    X·softmax → crossbar MVM (ANALOG)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from neuro_analog.ir import (
    AnalogGraph, ArchitectureFamily, DynamicsProfile, OpType, Domain,
    AnalogNode, PrecisionSpec, make_mvm_node, make_activation_node,
)
from .base import BaseExtractor


@dataclass
class EBMConfig:
    """Configuration for a Boltzmann machine / Hopfield network."""
    num_visible: int = 256
    num_hidden: int = 256
    num_layers: int = 1           # Deep Boltzmann machine layers
    temperature: float = 1.0
    hopfield_beta: float = 1.0    # Modern Hopfield retrieval gain
    use_analog_softmax: bool = False  # Use approximate analog softmax?
    model_type: str = "rbm"       # "rbm", "dbm", "hopfield", "extropic_dtm"


class EBMExtractor(BaseExtractor):
    """Theoretical analog analyzer for Energy-Based Models.

    No pretrained model required — builds the graph from known EBM structure.
    Useful for:
      1. Taxonomy completeness (EBM as reference architecture)
      2. Extropic DTM analysis (denoising = EBM chain)
      3. Modern Hopfield → transformer attention analog mapping

    Usage:
        # RBM analysis
        extractor = EBMExtractor(config=EBMConfig(num_visible=512, num_hidden=512))
        profile = extractor.run()

        # Modern Hopfield (attention analog)
        config = EBMConfig(model_type="hopfield", num_visible=768)
        extractor = EBMExtractor(config=config)
    """

    def __init__(
        self,
        config: Optional[EBMConfig] = None,
        model_name: str = "theoretical_ebm",
        device: str = "cpu",
    ):
        super().__init__(model_name, device)
        self.config = config or EBMConfig()

    @classmethod
    def rbm(cls, n_visible: int = 256, n_hidden: int = 256) -> "EBMExtractor":
        return cls(EBMConfig(num_visible=n_visible, num_hidden=n_hidden, model_type="rbm"))

    @classmethod
    def hopfield(cls, pattern_dim: int = 768, num_patterns: int = 64) -> "EBMExtractor":
        return cls(EBMConfig(num_visible=pattern_dim, num_hidden=num_patterns, model_type="hopfield"))

    @classmethod
    def extropic_dtm(cls, dim: int = 256, denoising_steps: int = 10) -> "EBMExtractor":
        return cls(EBMConfig(num_visible=dim, num_layers=denoising_steps, model_type="extropic_dtm"))

    @property
    def family(self) -> ArchitectureFamily:
        return ArchitectureFamily.EBM

    def load_model(self) -> None:
        """No-op: EBM analyzer uses theoretical structure, not a pretrained checkpoint."""
        self.model = self.config  # Use config as a stand-in

    def extract_dynamics(self) -> DynamicsProfile:
        """EBM dynamics: energy minimization via Boltzmann/Gibbs sampling."""
        return DynamicsProfile(
            has_dynamics=True,
            dynamics_type="energy_minimization",
            is_stochastic=True,  # Gibbs sampling requires TRNG
        )

    def build_graph(self) -> AnalogGraph:
        cfg = self.config

        if cfg.model_type == "hopfield":
            return self._build_hopfield_graph(cfg)
        elif cfg.model_type == "extropic_dtm":
            return self._build_dtm_graph(cfg)
        else:
            return self._build_rbm_graph(cfg)

    def _build_rbm_graph(self, cfg: EBMConfig) -> AnalogGraph:
        """Restricted Boltzmann Machine — one hidden layer.

        Gibbs sampling loop (zero D/A boundaries in steady state):

            x (visible) ──▶ h = σ(Wx + b_h)  [crossbar MVM + p-bit]
                         ◀── x = σ(W^T h + b_v) [transposed crossbar + p-bit]
        """
        total_params = cfg.num_visible * cfg.num_hidden + cfg.num_visible + cfg.num_hidden

        graph = AnalogGraph(
            name=f"RBM_{cfg.num_visible}×{cfg.num_hidden}",
            family=ArchitectureFamily.EBM,
            model_params=total_params,
        )

        # Positive phase: visible → hidden
        graph.add_node(make_mvm_node(
            "rbm.W_fwd", cfg.num_visible, cfg.num_hidden,
        ))
        graph.add_node(AnalogNode(
            name="rbm.p_bit_h",
            op_type=OpType.SAMPLE,
            domain=Domain.ANALOG,
            input_shape=(cfg.num_hidden,),
            output_shape=(cfg.num_hidden,),
            flops=cfg.num_hidden,
            metadata={
                "description": "p-bit sampling: x_i ~ Bernoulli(σ(h_i/T))",
                "temperature": cfg.temperature,
                "hardware": "sMTJ (Extropic TSU) or subthreshold MOSFET",
                "da_boundaries": 0,  # Fully analog — no ADC/DAC needed
            },
        ))

        # Negative phase: hidden → visible (transposed crossbar)
        graph.add_node(make_mvm_node(
            "rbm.W_bwd", cfg.num_hidden, cfg.num_visible,
        ))
        graph.add_node(AnalogNode(
            name="rbm.p_bit_v",
            op_type=OpType.SAMPLE,
            domain=Domain.ANALOG,
            input_shape=(cfg.num_visible,),
            output_shape=(cfg.num_visible,),
            flops=cfg.num_visible,
            metadata={
                "description": "Visible unit sampling via p-bit",
                "hardware": "sMTJ",
            },
        ))

        # Wire the Gibbs loop
        graph.add_edge("rbm.W_fwd", "rbm.p_bit_h")
        graph.add_edge("rbm.p_bit_h", "rbm.W_bwd")
        graph.add_edge("rbm.W_bwd", "rbm.p_bit_v")
        # Feedback: p_bit_v → W_fwd (closes the Gibbs loop)
        # NOTE: In hardware this is the analog feedback path — no DAC needed.
        # We do NOT add this edge in the graph (would create a cycle in DAG)
        # but note it in metadata.
        graph.get_node("rbm.W_fwd").metadata["analog_feedback_from"] = "rbm.p_bit_v"

        return graph

    def _build_hopfield_graph(self, cfg: EBMConfig) -> AnalogGraph:
        """Modern Hopfield Network = transformer attention.

        Update rule (Ramsauer et al. 2020):
            ξ_new = X · softmax(β · X^T · ξ)

        Analog partition:
            X^T · ξ → crossbar MVM (ANALOG)
            β · (...)  → programmable gain (ANALOG)
            softmax    → DIGITAL (or ANALOG_SOFTMAX hybrid)
            X · (...)  → crossbar MVM (ANALOG)

        This is the theoretical limit of how analog-amenable attention can be.
        """
        n = cfg.num_visible
        m = cfg.num_hidden  # Number of stored patterns
        total_params = n * m

        graph = AnalogGraph(
            name=f"ModernHopfield_{n}d_{m}patterns",
            family=ArchitectureFamily.EBM,
            model_params=total_params,
        )

        # Store patterns as crossbar conductances
        graph.add_node(make_mvm_node("hopfield.X_T_query", n, m))  # X^T · ξ
        graph.add_node(AnalogNode(
            name="hopfield.beta_scale",
            op_type=OpType.GAIN,
            domain=Domain.ANALOG,
            input_shape=(m,), output_shape=(m,), flops=m,
            metadata={"beta": cfg.hopfield_beta, "description": "Retrieval gain (programmable amp)"},
        ))

        if cfg.use_analog_softmax:
            # Approximate: subthreshold exp + current summing
            graph.add_node(AnalogNode(
                name="hopfield.analog_softmax",
                op_type=OpType.ANALOG_SOFTMAX,
                domain=Domain.HYBRID,
                input_shape=(m,), output_shape=(m,), flops=3 * m,
                metadata={"description": "Approx softmax via subthreshold MOSFET exp + current sum"},
            ))
        else:
            graph.add_node(AnalogNode(
                name="hopfield.softmax",
                op_type=OpType.SOFTMAX,
                domain=Domain.DIGITAL,
                input_shape=(m,), output_shape=(m,), flops=3 * m,
            ))

        graph.add_node(make_mvm_node("hopfield.X_retrieve", m, n))  # X · attn

        sm_name = "hopfield.analog_softmax" if cfg.use_analog_softmax else "hopfield.softmax"
        graph.add_edge("hopfield.X_T_query", "hopfield.beta_scale")
        graph.add_edge("hopfield.beta_scale", sm_name)
        graph.add_edge(sm_name, "hopfield.X_retrieve")

        return graph

    def _build_dtm_graph(self, cfg: EBMConfig) -> AnalogGraph:
        """Extropic DTM (Denoising Thermodynamic Model) — EBM denoising chain.

        Each denoising step is one EBM Gibbs iteration.
        The chain of steps is a sequential analog pipeline.

        No D/A boundaries within the chain — the p-bit outputs feed directly
        into the next crossbar array's input lines.
        """
        n = cfg.num_visible
        steps = cfg.num_layers

        graph = AnalogGraph(
            name=f"Extropic_DTM_{n}d_{steps}steps",
            family=ArchitectureFamily.EBM,
            model_params=n * n * steps,  # One weight matrix per denoising step
        )

        prev_id = None
        for step in range(steps):
            # Energy evaluation: W·x (crossbar)
            mvm_id = f"dtm.step{step}.energy_mvm"
            graph.add_node(make_mvm_node(mvm_id, n, n))

            # Stochastic sampling: p-bit
            sample_id = f"dtm.step{step}.sample"
            graph.add_node(AnalogNode(
                name=sample_id,
                op_type=OpType.SAMPLE,
                domain=Domain.ANALOG,
                input_shape=(n,), output_shape=(n,), flops=n,
                metadata={
                    "step": step,
                    "description": f"Denoising step {step}: p-bit sampling",
                    "hardware": "sMTJ (Extropic TSU)",
                    "analog_feedback": True,
                },
            ))

            graph.add_edge(mvm_id, sample_id)
            if prev_id:
                graph.add_edge(prev_id, mvm_id)
            prev_id = sample_id

        return graph
