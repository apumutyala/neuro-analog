"""
Cross-architecture analog amenability taxonomy.

Compares SSMs, diffusion models, flow models, EBMs, and transformers
across quantitative analog feasibility metrics. Generates the unified
taxonomy that is the project's central novel contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from neuro_analog.ir.types import AnalogAmenabilityProfile, ArchitectureFamily, DynamicsProfile


@dataclass
class TaxonomyEntry:
    """One row in the cross-architecture taxonomy."""
    family: ArchitectureFamily
    model_name: str
    profile: AnalogAmenabilityProfile
    
    # Qualitative annotations
    has_native_dynamics: bool = False
    dynamics_description: str = ""
    analog_circuit_primitive: str = ""  # Primary analog primitive used
    key_digital_bottleneck: str = ""    # What forces digital computation
    analog_compiler_fit: str = ""       # How well it maps to Ark's BaseAnalogCkt interface


class AnalogTaxonomy:
    """Cross-architecture comparison framework.
    
    Usage:
        taxonomy = AnalogTaxonomy()
        taxonomy.add_profile(mamba_profile, ...)
        taxonomy.add_profile(sd_profile, ...)
        taxonomy.add_profile(flux_profile, ...)
        
        table = taxonomy.comparison_table()
        ranking = taxonomy.rank_by_analog_amenability()
    """
    
    def __init__(self):
        self.entries: list[TaxonomyEntry] = []
    
    def add_profile(
        self,
        profile: AnalogAmenabilityProfile,
        has_native_dynamics: bool = False,
        dynamics_description: str = "",
        analog_circuit_primitive: str = "",
        key_digital_bottleneck: str = "",
        analog_compiler_fit: str = "",
    ):
        """Add an architecture profile to the taxonomy."""
        self.entries.append(TaxonomyEntry(
            family=profile.architecture,
            model_name=profile.model_name,
            profile=profile,
            has_native_dynamics=has_native_dynamics,
            dynamics_description=dynamics_description,
            analog_circuit_primitive=analog_circuit_primitive,
            key_digital_bottleneck=key_digital_bottleneck,
            analog_compiler_fit=analog_compiler_fit,
        ))
    
    def add_neural_ode_profile(self):
        """Add Neural ODE reference profile — demo centerpiece, highest analog amenability.

        Neural ODE IS an ODE: dx/dt = f_θ(x,t).
        - f_θ is a small MLP (64–256 dim) → fits within demonstrated crossbar scale
        - Adjoint-method training = Ark's gradient computation method
        - Diffrax solver = Ark's internal ODE solver
        - Export IS a valid Ark BaseAnalogCkt today — no approximation needed

        Score ~0.95: highest in the taxonomy because the architecture IS an ODE,
        the exact format Ark compiles.
        """
        from neuro_analog.extractors.neural_ode import NeuralODEExtractor

        ext = NeuralODEExtractor.demo(state_dim=4, hidden_dim=64, num_layers=2, activation="tanh")
        profile = ext.run()

        self.add_profile(
            profile,
            has_native_dynamics=True,
            dynamics_description="dx/dt = f_θ(x,t) (adjoint-trained MLP vector field)",
            analog_circuit_primitive="Crossbar MVM (W·x) + RC integrator (ODE step)",
            key_digital_bottleneck="Adaptive step-size controller (purely digital bookkeeping)",
            analog_compiler_fit=(
                "Perfect — dx/dt = f_θ(x,t) IS Ark's input format. "
                "export_neural_ode_to_ark() produces a runnable BaseAnalogCkt subclass today."
            ),
        )

    def add_reference_profiles(self):
        """Add reference profiles for architectures without empirical extraction.

        These are literature-based estimates for EBMs and generic transformers,
        providing comparison baselines even without pretrained model extraction.
        """
        # Neural ODE (demo centerpiece — empirically extracted, only if not already present)
        if not any(e.family == ArchitectureFamily.NEURAL_ODE for e in self.entries):
            self.add_neural_ode_profile()

        # SSM reference (S4D / Mamba structure)
        ssm_profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.SSM,
            model_name="Reference SSM (S4D)",
            model_params=0,
            analog_flop_fraction=0.88,
            digital_flop_fraction=0.12,
            da_boundary_count=2,  # Input DAC + final ADC; recurrence scan is fully analog
            min_weight_precision_bits=4,
            dynamics=DynamicsProfile(
                has_dynamics=True,
                dynamics_type="LTI_ODE",
                is_stochastic=False,
                state_dimension=64,
            ),
        )
        ssm_profile.compute_scores()
        self.add_profile(
            ssm_profile,
            has_native_dynamics=True,
            dynamics_description="h_{t+1} = A·h_t + B·x_t  [diagonal A → N independent RC circuits]",
            analog_circuit_primitive="RC bank (diagonal A eigenvalues → time constants τ_i = 1/|a_i|)",
            key_digital_bottleneck="B/C/D crossbar MVM at boundaries; LayerNorm around SSM blocks",
            analog_compiler_fit="Strong — diagonal scan maps to RC integrators; export_s4d_to_ark() works today",
        )

        # Flow reference (MLP velocity field — experiment model exports; FLUX is analysis-only)
        flow_profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.FLOW,
            model_name="Reference Flow (MLP velocity)",
            model_params=0,
            analog_flop_fraction=0.80,
            digital_flop_fraction=0.20,
            da_boundary_count=4,  # Input DAC + per-NFE ADC/DAC crossings (4 typical)
            min_weight_precision_bits=4,
            dynamics=DynamicsProfile(
                has_dynamics=True,
                dynamics_type="time_varying_ODE",
                is_stochastic=False,
                num_function_evaluations=4,
            ),
        )
        flow_profile.compute_scores()
        self.add_profile(
            flow_profile,
            has_native_dynamics=True,
            dynamics_description="dx/dt = v_θ(x, t)  [MLP velocity field; same format as Neural ODE]",
            analog_circuit_primitive="Crossbar MVM + RC integrator (identical to Neural ODE path)",
            key_digital_bottleneck="Adaptive NFE controller; FLUX v_θ is 12B MMDiT (too large for fixed CDG)",
            analog_compiler_fit="MLP velocity field: full BaseAnalogCkt via neural_ode path. FLUX: analysis-only.",
        )

        # Diffusion reference (VP-SDE reverse process)
        diffusion_profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.DIFFUSION,
            model_name="Reference Diffusion (VP-SDE)",
            model_params=0,
            analog_flop_fraction=0.70,
            digital_flop_fraction=0.30,
            da_boundary_count=8,   # Conservative: ~2 boundaries/step × several representative steps
            min_weight_precision_bits=6,
            dynamics=DynamicsProfile(
                has_dynamics=True,
                dynamics_type="SDE",
                is_stochastic=True,
                num_diffusion_steps=100,
            ),
        )
        diffusion_profile.compute_scores()
        self.add_profile(
            diffusion_profile,
            has_native_dynamics=True,
            dynamics_description="VP-SDE reverse: dx = [-½β(t)x - β(t)∇logp_t(x)]dt + √β(t)dw̃",
            analog_circuit_primitive="Programmable gain (β schedule) + crossbar score net + TRNG noise",
            key_digital_bottleneck="GroupNorm/AdaLN in score net; 100-step accumulation of mismatch error",
            analog_compiler_fit="Experiment model: BaseAnalogCkt via export_diffusion_to_ark(). SD U-Net 860M: analysis-only.",
        )

        # EBM reference (from Extropic DTM paper + Boltzmann machine theory)
        ebm_profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.EBM,
            model_name="Reference Boltzmann Machine",
            model_params=0,
            analog_flop_fraction=0.95,
            digital_flop_fraction=0.05,
            da_boundary_count=2,  # Input DAC + output ADC only
            min_weight_precision_bits=4,
            dynamics=DynamicsProfile(
                has_dynamics=True,
                dynamics_type="energy_minimization",
                is_stochastic=True,   # Gibbs sampling requires TRNG
            ),
        )
        ebm_profile.compute_scores()
        
        self.add_profile(
            ebm_profile,
            has_native_dynamics=True,
            dynamics_description="Energy minimization: Boltzmann sampling",
            analog_circuit_primitive="p-bit / sMTJ arrays (Extropic TSU)",
            key_digital_bottleneck="Deep energy network evaluation",
            analog_compiler_fit="Indirect — energy landscape, not ODE",
        )
        
        # DEQ reference (Bai et al. 2019; feedback circuit = implicit solver)
        deq_profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.DEQ,
            model_name="Reference DEQ (implicit MLP)",
            model_params=0,
            analog_flop_fraction=0.80,   # Same MVM partition as transformer
            digital_flop_fraction=0.20,
            da_boundary_count=2,         # Only input DAC + output ADC; feedback loop is analog
            min_weight_precision_bits=4,
            dynamics=DynamicsProfile(
                has_dynamics=True,
                dynamics_type="implicit_equilibrium",
                is_stochastic=False,
                state_dimension=64,
            ),
        )
        deq_profile.compute_scores()
        self.add_profile(
            deq_profile,
            has_native_dynamics=True,
            dynamics_description="Implicit fixed-point: z* = f_theta(z*, x), circuit settles naturally",
            analog_circuit_primitive="Feedback MVM loop (op-amp with crossbar in feedback path)",
            key_digital_bottleneck="Convergence guarantee under mismatch (spectral radius may exceed 1)",
            analog_compiler_fit="Strong — dz/dt = f_theta(z,x) - z is native Arco ODE format",
        )

        # Transformer reference (from IBM HERMES + literature)
        tfm_profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.TRANSFORMER,
            model_name="Reference Transformer (IBM HERMES)",
            model_params=0,
            analog_flop_fraction=0.82,
            digital_flop_fraction=0.18,
            da_boundary_count=6,  # Per layer: ~4-6 boundaries
            min_weight_precision_bits=4,
        )
        tfm_profile.compute_scores()
        
        self.add_profile(
            tfm_profile,
            has_native_dynamics=False,
            dynamics_description="No native dynamics (linear algebra)",
            analog_circuit_primitive="Crossbar MVM arrays",
            key_digital_bottleneck="Softmax, LayerNorm, dynamic attention matmuls",
            analog_compiler_fit="None — not an ODE system",
        )
    
    def comparison_table(self) -> str:
        """Generate formatted comparison table."""
        if not self.entries:
            return "No profiles added to taxonomy."
        
        header = (
            f"{'Architecture':<15} {'Model':<25} {'Analog %':>10} "
            f"{'D/A Bound':>10} {'Precision':>10} {'Dynamics':>10} "
            f"{'Noise':>8} {'Score':>8}"
        )
        separator = "-" * len(header)
        
        rows = [header, separator]
        for entry in sorted(self.entries, key=lambda e: -e.profile.overall_score):
            p = entry.profile
            dyn = "YES" if entry.has_native_dynamics else "NO"
            rows.append(
                f"{p.architecture.value:<15} {p.model_name:<25} "
                f"{p.analog_flop_fraction:>9.1%} "
                f"{p.da_boundary_count:>10} "
                f"{p.min_weight_precision_bits:>8}-bit "
                f"{dyn:>10} "
                f"{p.noise_score:>8.2f} "
                f"{p.overall_score:>7.3f}"
            )
        
        return "\n".join(rows)
    
    def rank_by_analog_amenability(self) -> list[TaxonomyEntry]:
        """Rank architectures by overall analog amenability score."""
        return sorted(self.entries, key=lambda e: -e.profile.overall_score)
    
    def to_dict(self) -> list[dict]:
        """Export taxonomy as list of dicts for JSON serialization."""
        return [
            {
                "family": e.family.value,
                "model": e.model_name,
                "analog_flop_fraction": e.profile.analog_flop_fraction,
                "da_boundary_count": e.profile.da_boundary_count,
                "min_precision_bits": e.profile.min_weight_precision_bits,
                "has_dynamics": e.has_native_dynamics,
                "dynamics_type": e.dynamics_description,
                "circuit_primitive": e.analog_circuit_primitive,
                "digital_bottleneck": e.key_digital_bottleneck,
                "analog_compiler_fit": e.analog_compiler_fit,
                "noise_score": e.profile.noise_score,
                "overall_score": e.profile.overall_score,
            }
            for e in self.entries
        ]
    
    def save(self, path: Path | str):
        """Save taxonomy to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        ranked = self.rank_by_analog_amenability()
        lines = [
            "NEURO-ANALOG CROSS-ARCHITECTURE TAXONOMY",
            "=" * 50,
            "",
            self.comparison_table(),
            "",
            "RANKING (most → least analog-amenable):",
        ]
        for i, entry in enumerate(ranked, 1):
            lines.append(
                f"  {i}. {entry.family.value} ({entry.model_name}): "
                f"score={entry.profile.overall_score:.3f}"
            )
            lines.append(f"     Primary analog primitive: {entry.analog_circuit_primitive}")
            lines.append(f"     Key bottleneck: {entry.key_digital_bottleneck}")
            lines.append(f"     Analog compiler fit: {entry.analog_compiler_fit}")
            lines.append("")
        
        return "\n".join(lines)
