#!/usr/bin/env python3
"""
Example: Cross-architecture analog amenability comparison.

Runs extraction on multiple architectures and generates the
unified taxonomy — the project's central novel contribution.

Six-family ranking (highest → lowest analog amenability):
  1. Neural ODE  (~0.95) — IS an ODE; f_θ MLP + adjoint = Shem's native format
  2. SSM/Mamba   (~0.86) — diagonal A = independent RC, selective gating is hard
  3. EBM         (~0.85) — Boltzmann = physical equilibrium, p-bits native
  4. Flow/FLUX   (~0.76) — clean ODE but v_θ is a 12B-param MMDiT transformer
  5. Transformer (~0.70) — 80-95% MVM but no dynamics, softmax is the wall
  6. Diffusion   (~0.65) — many steps × many boundaries = high converter overhead

Usage:
    python examples/04_cross_architecture.py
"""

from neuro_analog.ir.types import (
    AnalogAmenabilityProfile, ArchitectureFamily, DynamicsProfile,
)
# ArchitectureFamily now includes: SSM, DIFFUSION, FLOW, EBM, TRANSFORMER, NEURAL_ODE, DEQ
from neuro_analog.analysis.taxonomy import AnalogTaxonomy
from neuro_analog.extractors.neural_ode import NeuralODEExtractor


def main():
    print("\n" + "="*60)
    print("NEURO-ANALOG: Cross-Architecture Taxonomy (6 Families)")
    print("="*60 + "\n")

    taxonomy = AnalogTaxonomy()

    # ── 1. Neural ODE (DEMO CENTERPIECE — empirically extracted) ──
    # dx/dt = f_θ(x,t): the only architecture that IS Shem's input format.
    # export_neural_ode_to_shem() produces syntactically valid Shem code today.
    print("Extracting Neural ODE profile (demo centerpiece)...")
    neural_ode_ext = NeuralODEExtractor.demo(
        state_dim=4, hidden_dim=64, num_layers=2, activation="tanh",
    )
    neural_ode_profile = neural_ode_ext.run()
    taxonomy.add_profile(
        neural_ode_profile,
        has_native_dynamics=True,
        dynamics_description="dx/dt = f_θ(x,t) (adjoint-trained MLP vector field)",
        analog_circuit_primitive="Crossbar MVM (W·x) + RC integrator (one step of ODE)",
        key_digital_bottleneck="Adaptive step-size controller (purely digital bookkeeping)",
        achour_compiler_fit=(
            "Perfect — dx/dt = f_θ(x,t) IS Shem's input format. "
            "export_neural_ode_to_shem() produces runnable Shem code today."
        ),
    )

    # ── 2. SSM (Mamba) ──
    # In production: use MambaExtractor("state-spaces/mamba-370m").run()
    mamba_profile = AnalogAmenabilityProfile(
        architecture=ArchitectureFamily.SSM,
        model_name="Mamba-370M",
        model_params=370_000_000,
        analog_flop_fraction=0.85,
        digital_flop_fraction=0.15,
        da_boundary_count=4,
        min_weight_precision_bits=8,
        dynamics=DynamicsProfile(
            has_dynamics=True, dynamics_type="LTI_ODE",
            state_dimension=16, time_constant_spread=150.0,
        ),
    )
    mamba_profile.compute_scores()
    taxonomy.add_profile(
        mamba_profile,
        has_native_dynamics=True,
        dynamics_description="dx/dt = Ax + Bu (diagonal A, continuous-time)",
        analog_circuit_primitive="RC integrator circuits (1 per state dim)",
        key_digital_bottleneck="Softplus (Δ discretization), SiLU gating",
        achour_compiler_fit="Direct — literal ODE, expressible in Arco/Legno",
    )
    
    # ── Diffusion (Stable Diffusion 1.5) ──
    sd_profile = AnalogAmenabilityProfile(
        architecture=ArchitectureFamily.DIFFUSION,
        model_name="Stable Diffusion 1.5",
        model_params=860_000_000,
        analog_flop_fraction=0.72,
        digital_flop_fraction=0.28,
        da_boundary_count=80,  # ~4 per step × 20 steps
        min_weight_precision_bits=6,
        dynamics=DynamicsProfile(
            has_dynamics=True, dynamics_type="SDE",
            beta_dynamic_range=200.0, num_diffusion_steps=20,
            is_stochastic=True,
        ),
    )
    sd_profile.compute_scores()
    taxonomy.add_profile(
        sd_profile,
        has_native_dynamics=True,
        dynamics_description="VP-SDE: dx = [-½β(t)x - β(t)s_θ]dt + √β(t)dw",
        analog_circuit_primitive="RLC circuit (CLD) + crossbar MVMs",
        key_digital_bottleneck="GroupNorm, SiLU, softmax in U-Net",
        achour_compiler_fit="Partial — ODE/SDE update is analog, score network is mixed",
    )
    
    # ── Flow (FLUX-schnell) ──
    flux_profile = AnalogAmenabilityProfile(
        architecture=ArchitectureFamily.FLOW,
        model_name="FLUX.1-schnell",
        model_params=12_000_000_000,
        analog_flop_fraction=0.78,
        digital_flop_fraction=0.22,
        da_boundary_count=16,  # ~4 per step × 4 steps
        min_weight_precision_bits=8,
        dynamics=DynamicsProfile(
            has_dynamics=True, dynamics_type="time_varying_ODE",
            num_function_evaluations=4,
        ),
    )
    flux_profile.compute_scores()
    taxonomy.add_profile(
        flux_profile,
        has_native_dynamics=True,
        dynamics_description="dx/dt = v_θ(x,t) (rectified flow, near-straight ODE)",
        analog_circuit_primitive="Crossbar MVMs + capacitor accumulation",
        key_digital_bottleneck="Softmax, GELU, LayerNorm, AdaLN in MMDiT",
        achour_compiler_fit="Strong — ODE form matches Arco input exactly",
    )
    
    # ── Add reference profiles (EBM, Transformer) ──
    # Note: Neural ODE is already added above empirically.
    # add_reference_profiles() adds EBM and Transformer baselines only
    # (it calls add_neural_ode_profile() internally — skip to avoid duplicate).
    ebm_profile = AnalogAmenabilityProfile(
        architecture=ArchitectureFamily.EBM,
        model_name="Reference Boltzmann Machine",
        model_params=0,
        analog_flop_fraction=0.95,
        digital_flop_fraction=0.05,
        da_boundary_count=2,
        min_weight_precision_bits=4,
    )
    ebm_profile.compute_scores()
    taxonomy.add_profile(
        ebm_profile,
        has_native_dynamics=True,
        dynamics_description="Energy minimization: Boltzmann sampling (Gibbs)",
        analog_circuit_primitive="p-bit / sMTJ arrays (Extropic TSU)",
        key_digital_bottleneck="Deep energy network evaluation",
        achour_compiler_fit="Indirect — energy landscape, not ODE",
    )

    tfm_profile = AnalogAmenabilityProfile(
        architecture=ArchitectureFamily.TRANSFORMER,
        model_name="Reference Transformer (IBM HERMES)",
        model_params=0,
        analog_flop_fraction=0.82,
        digital_flop_fraction=0.18,
        da_boundary_count=6,
        min_weight_precision_bits=4,
    )
    tfm_profile.compute_scores()
    taxonomy.add_profile(
        tfm_profile,
        has_native_dynamics=False,
        dynamics_description="No native dynamics (linear algebra)",
        analog_circuit_primitive="Crossbar MVM arrays",
        key_digital_bottleneck="Softmax, LayerNorm, dynamic attention matmuls",
        achour_compiler_fit="None — not an ODE system",
    )

    # ── 7. DEQ (Deep Equilibrium Model) ──
    deq_profile = AnalogAmenabilityProfile(
        architecture=ArchitectureFamily.DEQ,
        model_name="Reference DEQ (implicit MLP)",
        model_params=0,
        analog_flop_fraction=0.80,
        digital_flop_fraction=0.20,
        da_boundary_count=2,   # Input DAC + output ADC; feedback loop is analog
        min_weight_precision_bits=4,
        dynamics=DynamicsProfile(
            has_dynamics=True,
            dynamics_type="implicit_equilibrium",
            is_stochastic=False,
            state_dimension=64,
        ),
    )
    deq_profile.compute_scores()
    taxonomy.add_profile(
        deq_profile,
        has_native_dynamics=True,
        dynamics_description="Implicit fixed-point: z* = f_theta(z*, x), circuit settles naturally",
        analog_circuit_primitive="Feedback MVM loop (op-amp with crossbar in feedback path)",
        key_digital_bottleneck="Convergence guarantee under mismatch (spectral radius rho may exceed 1)",
        achour_compiler_fit="Strong — dz/dt = f_theta(z,x) - z is native Arco ODE format",
    )

    # ── Generate outputs ──
    print(taxonomy.summary())
    
    # Save to JSON
    from pathlib import Path
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    taxonomy.save(out_dir / "taxonomy.json")
    print(f"\nSaved taxonomy to outputs/taxonomy.json")


if __name__ == "__main__":
    main()
