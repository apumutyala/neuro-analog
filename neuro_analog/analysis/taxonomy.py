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
    achour_compiler_fit: str = ""       # How well it maps to Arco/Legno/Shem


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
        achour_compiler_fit: str = "",
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
            achour_compiler_fit=achour_compiler_fit,
        ))
    
    def add_neural_ode_profile(self):
        """Add Neural ODE reference profile — demo centerpiece, highest analog amenability.

        Neural ODE IS an ODE: dx/dt = f_θ(x,t).
        - f_θ is a small MLP (64–256 dim) → fits Shem's parameter scale exactly
        - Adjoint-method training = Shem's optimization method
        - Diffrax solver = Shem's internal solver
        - Export IS a valid Shem input today — no approximation needed

        Score ~0.95: highest in the taxonomy because the architecture IS the
        computation Shem was designed for.
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
            achour_compiler_fit=(
                "Perfect — dx/dt = f_θ(x,t) IS Shem's input format. "
                "Export_neural_ode_to_shem() produces runnable Shem code today."
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

        # EBM reference (from Extropic DTM paper + Boltzmann machine theory)
        ebm_profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.EBM,
            model_name="Reference Boltzmann Machine",
            model_params=0,
            analog_flop_fraction=0.95,
            digital_flop_fraction=0.05,
            da_boundary_count=2,  # Input DAC + output ADC only
            min_weight_precision_bits=4,
        )
        ebm_profile.compute_scores()
        
        self.add_profile(
            ebm_profile,
            has_native_dynamics=True,
            dynamics_description="Energy minimization: Boltzmann sampling",
            analog_circuit_primitive="p-bit / sMTJ arrays (Extropic TSU)",
            key_digital_bottleneck="Deep energy network evaluation",
            achour_compiler_fit="Indirect — energy landscape, not ODE",
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
            achour_compiler_fit="Strong — dz/dt = f_theta(z,x) - z is native Arco ODE format",
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
            achour_compiler_fit="None — not an ODE system",
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
                "achour_fit": e.achour_compiler_fit,
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
            lines.append(f"     Achour compiler fit: {entry.achour_compiler_fit}")
            lines.append("")
        
        return "\n".join(lines)
