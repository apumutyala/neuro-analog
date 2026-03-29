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

import numpy as np

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
    # DTM-specific (Jelinčič et al. 2025)
    connectivity_degree: int = 12  # G_k neighbor count (Table II: G_12 is DTCA default)
    gibbs_steps_K: int = 250       # Gibbs sweeps per denoising step (Appendix E calibration)


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
        if self.config.model_type == "extropic_dtm":
            return ArchitectureFamily.DTM
        return ArchitectureFamily.EBM

    def load_model(self) -> None:
        """No-op: EBM analyzer uses theoretical structure, not a pretrained checkpoint."""
        self.model = self.config  # Use config as a stand-in

    def extract_dynamics(self) -> DynamicsProfile:
        """EBM dynamics: energy minimization via Boltzmann/Gibbs sampling."""
        cfg = self.config
        if cfg.model_type == "extropic_dtm":
            # DTCA: thermal noise is the computational resource, not a nonideality.
            # dynamics_type="thermodynamic_gibbs" triggers noise_score=1.0 in compute_scores().
            # Grid side L = sqrt(N); Appendix E: τ_rng ≈ 100 ns, E_rng ≈ 350 aJ/bit.
            import math
            L = int(math.isqrt(cfg.num_visible))
            return DynamicsProfile(
                has_dynamics=True,
                dynamics_type="thermodynamic_gibbs",
                is_stochastic=True,
                grid_side_L=L,
                connectivity_degree=cfg.connectivity_degree,
                gibbs_steps_K=cfg.gibbs_steps_K,
                denoising_steps_T=cfg.num_layers,
                tau_rng_ns=100.0,       # Appendix K: τ_0 ≈ 100 ns decorrelation time
                energy_per_sample_J=350e-18,  # Appendix E: ~350 aJ/bit (1 aJ = 1e-18 J)
            )
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
                # Calibration target: switching σ ∝ √T at the operating point.
                # A p-bit requires the hardware noise level to match this so that
                # P(x=1|h) = σ(h/T) holds — wrong σ biases the Boltzmann distribution.
                "target_sigma": cfg.temperature ** 0.5,
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
                "target_sigma": cfg.temperature ** 0.5,
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

    def export_to_ark(
        self,
        output_path,
        mismatch_sigma: float = 0.05,
        seed: int = 42,
    ) -> str:
        """Generate a standalone Ark BaseAnalogCkt for this EBM as a Hopfield ODE.

        Reformulates the EBM as the Langevin mean-field ODE:
            dx/dt = -x + tanh(W_sym @ x + b)

        For theoretical models (no pretrained checkpoint), weights are drawn
        from N(0, 1/n) — standard Hopfield random initialization.
        Pass real trained weights via export_hopfield_to_ark() directly.

        Supports model_type: 'hopfield', 'rbm'.
        For 'extropic_dtm', use the thermodynamic substrate (not ODE form).
        """
        from neuro_analog.ark_bridge.ebm_cdg import (
            export_hopfield_to_ark,
            make_rbm_hopfield_weights,
        )
        rng = np.random.default_rng(seed)
        cfg = self.config

        if cfg.model_type == "hopfield":
            n = cfg.num_visible
            W = rng.standard_normal((n, n)) / np.sqrt(n)
            np.fill_diagonal(W, 0.0)  # no self-connections (standard Hopfield)
            b = np.zeros(n)
            class_name = "HopfieldAnalogCkt"

        elif cfg.model_type == "rbm":
            n_v, n_h = cfg.num_visible, cfg.num_hidden
            W_rbm = rng.standard_normal((n_h, n_v)) / np.sqrt(n_v)
            b_v = np.zeros(n_v)
            b_h = np.zeros(n_h)
            W, b = make_rbm_hopfield_weights(W_rbm, b_v, b_h)
            class_name = "RBMHopfieldAnalogCkt"

        else:
            raise ValueError(
                f"model_type='{cfg.model_type}' cannot be exported as a Hopfield ODE. "
                "Use 'hopfield' or 'rbm'. For 'extropic_dtm', the thermodynamic "
                "substrate (sMTJ Gibbs sampling) has no ODE-form Ark export."
            )

        return export_hopfield_to_ark(
            W, b, output_path,
            mismatch_sigma=mismatch_sigma,
            class_name=class_name,
        )

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
        """Extropic DTCA (Denoising Thermodynamic Computer Architecture) — chromatic Gibbs chain.

        Hardware model (Jelinčič et al. 2025, Section II):
          - N = L×L binary spins on a 2D toroidal grid
          - Sparse G_k connectivity: each spin couples to k neighbors via a resistor network
            (no crossbar MVM — Kirchhoff current law on a sparse array, Appendix D Table II)
          - Local field: h_i = Σ_{j∈N_k(i)} J_ij x_j  [ACCUMULATION via resistors]
          - Bias injection: h_i += b_i  [RESISTOR_DAC, Appendix E]
          - Stochastic update: x_i ~ Bernoulli(σ(2β(h_i + b_i)))  [GIBBS_STEP, Fig. 3]

        Parameter count (symmetric Ising coupling):
          - Edge weights: N × k / 2  (G_k sparse, not dense N×N)
          - Biases: N per layer
          Total: T × (N×k/2 + N)

        No D/A boundaries within the chain (Section II.A: fully analog feedback path).
        """
        import math
        n = cfg.num_visible
        k = cfg.connectivity_degree   # G_k: k ∈ {8, 12, 16, 20, 24} (Appendix D, Table II)
        steps = cfg.num_layers        # Denoising steps T

        # Sparse G_k: N×k/2 coupling weights + N biases per layer
        params_per_layer = n * k // 2 + n
        total_params = steps * params_per_layer

        graph = AnalogGraph(
            name=f"Extropic_DTM_N{n}_G{k}_T{steps}",
            family=ArchitectureFamily.DTM,
            model_params=total_params,
        )

        prev_id = None
        for step in range(steps):
            # Sparse G_k neighborhood field summation via resistor network
            # (Kirchhoff's current law — no crossbar MVM needed for sparse connectivity)
            field_id = f"dtm.step{step}.field_sum"
            graph.add_node(AnalogNode(
                name=field_id,
                op_type=OpType.ACCUMULATION,
                domain=Domain.ANALOG,
                input_shape=(n,), output_shape=(n,),
                flops=n * k,  # Each of N spins sums k neighbors
                metadata={
                    "step": step,
                    "description": f"Sparse G_{k} neighbor field: h_i = Σ_{{j∈N_k(i)}} J_ij x_j",
                    "connectivity": f"G_{k}",
                    "hardware": "Resistor network (Kirchhoff current law)",
                    "da_boundaries": 0,
                },
            ))

            # Bias injection from DAC-driven resistor (Appendix E)
            bias_id = f"dtm.step{step}.bias_dac"
            graph.add_node(AnalogNode(
                name=bias_id,
                op_type=OpType.RESISTOR_DAC,
                domain=Domain.ANALOG,
                input_shape=(n,), output_shape=(n,),
                flops=n,
                metadata={
                    "step": step,
                    "description": "Bias loading: h_i += b_i via DAC-driven resistor",
                    "hardware": "Resistor DAC (Appendix E)",
                    "params": n,
                },
            ))

            # Chromatic Gibbs update: subthreshold CMOS RNG (Section II.B, Fig. 3)
            # Appendix K: τ_rng ≈ 100 ns; Appendix E: E_rng ≈ 350 aJ/bit
            gibbs_id = f"dtm.step{step}.gibbs_update"
            graph.add_node(AnalogNode(
                name=gibbs_id,
                op_type=OpType.GIBBS_STEP,
                domain=Domain.ANALOG,
                input_shape=(n,), output_shape=(n,), flops=n,
                metadata={
                    "step": step,
                    "description": f"Chromatic Gibbs: x_i ~ Bernoulli(σ(2β(h_i+b_i)))",
                    "hardware": "Subthreshold CMOS RNG (DTCA §II.B, Fig. 3)",
                    "tau_rng_ns": 100.0,     # Appendix K measurement
                    "E_rng_aJ": 350.0,       # Appendix E energy model
                    "analog_feedback": True,
                    "da_boundaries": 0,
                },
            ))

            graph.add_edge(field_id, bias_id)
            graph.add_edge(bias_id, gibbs_id)
            if prev_id:
                graph.add_edge(prev_id, field_id)
            prev_id = gibbs_id

        return graph
