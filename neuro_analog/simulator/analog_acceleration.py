"""
Analog acceleration / harnessing models for cross-architecture comparison.

When a digital iterative algorithm is mapped to analog hardware, the analog
physics replaces the discrete solver with a continuous-time dynamical system.
The speedup and energy savings are determined by how many digital iterations
can be replaced by a single analog settling/integration step.

Reference architectures and their analog-native counterparts:

1. DEQ  → Feedback analog circuit (Bai et al. 2019; Liao & Poggio 2020)
   Digital: fixed-point iteration (30-100 steps)
   Analog:  RC/OTA feedback loop settles to equilibrium in ~5τ
   Speedup: O(iterations)

2. Neural ODE → Analog integrator (Chen et al. 2018; Hasani et al. 2022)
   Digital: adaptive RK45 (20-100 function evaluations)
   Analog:  single RC/OTA integration step per τ
   Speedup: O(function_evaluations)

3. Diffusion → Langevin dynamics / SDE sampler (Ho et al. 2020; Song et al. 2021)
   Digital: 1000 DDPM/DDIM steps
   Analog:  physical Langevin with thermal noise (1 step per noise level)
   Speedup: O(num_steps)

4. Flow Matching → Continuous normalizing flow (Lipman et al. 2023)
   Digital: ODE solver (50-100 evals)
   Analog:  analog flow integrator (same speedup as Neural ODE)

5. EBM (Gibbs) → Thermodynamic compute / DTCA (Jelinčič et al. 2025)
   Digital: 1000 Gibbs sweeps
   Analog:  subthreshold CMOS RNG, each sweep is O(1) thermal fluctuation
   Speedup: O(gibbs_steps)

6. SSM → Continuous-time LTI filter (Gu et al. 2022)
   Digital: discretized recurrence (S4/S5)
   Analog:  continuous-time S4/S5 state update via RC/OTA
   Speedup: O(state_dim) for parallel mode, 1× for recurrent mode

7. Transformer → Kernel attention + analog MVM
   Digital: O(n²) attention
   Analog:  AIMC kernel approximation (Büchel et al. 2024, IBM HERMES)
            MVMs are native; softmax is digital bottleneck
   Speedup: limited to linear layers; attention still digital

All speedup and energy metrics are relative to a calibrated digital baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..ir.types import DynamicsProfile, ArchitectureFamily


@dataclass
class AnalogAccelerationProfile:
    """Quantitative acceleration metrics for one architecture on analog hardware.

    Attributes:
        architecture: Architecture family (e.g., DEQ, Neural ODE, etc.)
        digital_iterations: Number of discrete solver steps in digital baseline.
        analog_settling_time_constants: Number of RC time constants for analog settling.
        speedup_settling_vs_digital: digital_iterations / analog_settling_time_constants
        digital_energy_per_iteration_pJ: Energy per digital iteration (MACs + memory).
        analog_energy_per_step_pJ: Energy per analog step (integrator + crossbar).
        energy_saving_ratio: digital_total / analog_total (per inference).
        speedup_notes: Human-readable explanation.
    """

    architecture: ArchitectureFamily
    digital_iterations: int = 1
    analog_settling_time_constants: float = 5.0

    digital_energy_per_iteration_pJ: float = 0.0
    analog_energy_per_step_pJ: float = 0.0

    speedup_settling_vs_digital: float = 1.0
    energy_saving_ratio: float = 1.0

    speedup_notes: str = ""

    # Architectural-specific sub-metrics
    deq_spectral_radius: float | None = None
    ode_function_evaluations: int | None = None
    diffusion_num_steps: int | None = None
    ebm_gibbs_steps: int | None = None
    ssm_state_dim: int | None = None


# Backward-compatible alias
AnalogHarnessingProfile = AnalogAccelerationProfile


def compute_acceleration(
    family: ArchitectureFamily,
    dynamics: DynamicsProfile,
    digital_mac_energy_pJ: float = 10.0,
    analog_mac_energy_pJ: float = 0.10,
    digital_mac_throughput: float = 1e11,
    analog_mac_throughput: float = 1e12,
) -> AnalogAccelerationProfile:
    """Compute analog acceleration profile for a given architecture family.

    Args:
        family: Architecture family.
        dynamics: DynamicsProfile with solver/step counts.
        digital_mac_energy_pJ: Baseline digital MAC energy [pJ].
        analog_mac_energy_pJ: Analog MAC energy [pJ] (e.g., HERMES PCM).
        digital_mac_throughput: Digital MAC/s.
        analog_mac_throughput: Analog MAC/s.

    Returns:
        AnalogHarnessingProfile with speedup and energy metrics.
    """

    # Default profile (single-pass, no dynamics)
    profile = AnalogAccelerationProfile(
        architecture=family,
        digital_iterations=1,
        analog_settling_time_constants=1.0,
        speedup_settling_vs_digital=1.0,
        energy_saving_ratio=1.0,
        speedup_notes="Single-pass: no analog speedup for purely feedforward logic.",
    )

    if family == ArchitectureFamily.DEQ:
        # DEQ: digital fixed-point iteration vs analog feedback settling
        n_iter = 30  # default DEQ iteration count
        if dynamics.num_function_evaluations:
            n_iter = dynamics.num_function_evaluations
        tau = 5.0  # 5τ settling to within 0.67% of equilibrium
        speedup = n_iter / tau

        # Energy: each digital iter does a full forward pass (MACs + memory)
        # Analog: one forward pass but with continuous signal (no ADC/DAC per iter)
        digital_e = n_iter * digital_mac_energy_pJ
        analog_e = 1 * analog_mac_energy_pJ  # single analog settling pass

        profile = AnalogAccelerationProfile(
            architecture=family,
            digital_iterations=n_iter,
            analog_settling_time_constants=tau,
            speedup_settling_vs_digital=speedup,
            digital_energy_per_iteration_pJ=digital_mac_energy_pJ,
            analog_energy_per_step_pJ=analog_mac_energy_pJ,
            energy_saving_ratio=digital_e / analog_e if analog_e > 0 else 1.0,
            speedup_notes=(
                f"DEQ: {n_iter} digital fixed-point iterations → "
                f"analog feedback settling in {tau}τ. "
                f"Speedup ≈ {speedup:.1f}×. "
                "Note: spectral radius > 1 causes analog instability."
            ),
            deq_spectral_radius=getattr(dynamics, "stiffness_ratio", None),
        )

    elif family == ArchitectureFamily.NEURAL_ODE:
        # Neural ODE: RK solver steps vs analog integrator
        n_eval = 40  # default
        if dynamics.num_function_evaluations:
            n_eval = dynamics.num_function_evaluations
        tau = 3.0  # 3τ for ODE integration horizon
        speedup = n_eval / tau

        digital_e = n_eval * digital_mac_energy_pJ
        analog_e = analog_mac_energy_pJ  # continuous integration

        profile = AnalogAccelerationProfile(
            architecture=family,
            digital_iterations=n_eval,
            analog_settling_time_constants=tau,
            speedup_settling_vs_digital=speedup,
            digital_energy_per_iteration_pJ=digital_mac_energy_pJ,
            analog_energy_per_step_pJ=analog_mac_energy_pJ,
            energy_saving_ratio=digital_e / analog_e if analog_e > 0 else 1.0,
            speedup_notes=(
                f"Neural ODE: {n_eval} digital RK evaluations → "
                f"analog integrator in {tau}τ. "
                f"Speedup ≈ {speedup:.1f}×. "
                "Stiff systems (large Lipschitz) require fast integrators."
            ),
            ode_function_evaluations=n_eval,
        )

    elif family == ArchitectureFamily.DIFFUSION:
        # Diffusion: DDPM/DDIM steps vs Langevin analog dynamics
        n_steps = 100  # default (can be 50 with DDIM)
        if dynamics.num_diffusion_steps:
            n_steps = dynamics.num_diffusion_steps
        tau = 1.0  # Each noise level is one analog Langevin step
        speedup = n_steps / tau

        digital_e = n_steps * digital_mac_energy_pJ
        analog_e = analog_mac_energy_pJ

        profile = AnalogAccelerationProfile(
            architecture=family,
            digital_iterations=n_steps,
            analog_settling_time_constants=tau,
            speedup_settling_vs_digital=speedup,
            digital_energy_per_iteration_pJ=digital_mac_energy_pJ,
            analog_energy_per_step_pJ=analog_mac_energy_pJ,
            energy_saving_ratio=digital_e / analog_e if analog_e > 0 else 1.0,
            speedup_notes=(
                f"Diffusion: {n_steps} digital sampling steps → "
                f"analog Langevin dynamics in {int(tau)} physical step. "
                f"Speedup ≈ {speedup:.0f}×. "
                "Thermal noise IS the sampler in analog (no RNG overhead)."
            ),
            diffusion_num_steps=n_steps,
        )

    elif family == ArchitectureFamily.FLOW:
        # Flow Matching: same as Neural ODE (velocity field integration)
        n_eval = 50
        if dynamics.num_function_evaluations:
            n_eval = dynamics.num_function_evaluations
        tau = 3.0
        speedup = n_eval / tau

        digital_e = n_eval * digital_mac_energy_pJ
        analog_e = analog_mac_energy_pJ

        profile = AnalogAccelerationProfile(
            architecture=family,
            digital_iterations=n_eval,
            analog_settling_time_constants=tau,
            speedup_settling_vs_digital=speedup,
            digital_energy_per_iteration_pJ=digital_mac_energy_pJ,
            analog_energy_per_step_pJ=analog_mac_energy_pJ,
            energy_saving_ratio=digital_e / analog_e if analog_e > 0 else 1.0,
            speedup_notes=(
                f"Flow: {n_eval} ODE solver steps → "
                f"analog flow integrator in {tau}τ. "
                f"Speedup ≈ {speedup:.1f}×. "
                "Straight trajectories (high flow_straightness) reduce analog latency."
            ),
        )

    elif family == ArchitectureFamily.EBM:
        # EBM / Gibbs: digital MCMC vs thermodynamic compute
        n_steps = 100  # default Gibbs sweeps
        if dynamics.gibbs_steps_K:
            n_steps = dynamics.gibbs_steps_K
        elif dynamics.denoising_steps_T:
            n_steps = dynamics.denoising_steps_T
        tau = 1.0  # Each sweep is one thermal fluctuation cycle
        speedup = n_steps / tau

        digital_e = n_steps * digital_mac_energy_pJ
        # Thermodynamic compute: ~350 aJ per sampled bit (DTCA Appendix E)
        analog_e = 0.00035  # 350 aJ = 3.5e-4 fJ = 3.5e-7 pJ per bit
        # But typical EBM has many bits; scale to per-iteration equivalent
        analog_e = analog_mac_energy_pJ  # use same unit for comparison

        profile = AnalogAccelerationProfile(
            architecture=family,
            digital_iterations=n_steps,
            analog_settling_time_constants=tau,
            speedup_settling_vs_digital=speedup,
            digital_energy_per_iteration_pJ=digital_mac_energy_pJ,
            analog_energy_per_step_pJ=analog_e,
            energy_saving_ratio=digital_e / analog_e if analog_e > 0 else 1.0,
            speedup_notes=(
                f"EBM/DTCA: {n_steps} digital Gibbs sweeps → "
                f"thermodynamic compute in {int(tau)} physical step. "
                f"Speedup ≈ {speedup:.0f}×. "
                "Thermal noise IS the computational resource (DTCA §I)."
            ),
            ebm_gibbs_steps=n_steps,
        )

    elif family == ArchitectureFamily.SSM:
        # SSM: discretized recurrence vs continuous-time analog filter
        # For recurrent mode: no iteration speedup (1 step per token)
        # For parallel scan: analog doesn't help (still need sequential scan)
        # But continuous-time analog SSM avoids discretization errors
        state_dim = dynamics.state_dimension or 64
        speedup = 1.0  # No iteration reduction in recurrent mode

        # Energy: analog SSM uses integrator per state variable
        digital_e = digital_mac_energy_pJ
        analog_e = state_dim * 0.5  # integrator energy per state

        profile = AnalogAccelerationProfile(
            architecture=family,
            digital_iterations=1,
            analog_settling_time_constants=1.0,
            speedup_settling_vs_digital=speedup,
            digital_energy_per_iteration_pJ=digital_mac_energy_pJ,
            analog_energy_per_step_pJ=analog_e,
            energy_saving_ratio=digital_e / analog_e if analog_e > 0 else 1.0,
            speedup_notes=(
                f"SSM: Recurrent mode has no iteration reduction, "
                f"but continuous-time analog avoids discretization artifacts. "
                f"State dim = {state_dim}."
            ),
            ssm_state_dim=state_dim,
        )

    elif family == ArchitectureFamily.TRANSFORMER:
        # Transformer: linear layers are native MVM (speedup = throughput ratio)
        # Attention is still digital bottleneck (O(n²) softmax)
        throughput_ratio = analog_mac_throughput / digital_mac_throughput
        digital_e = digital_mac_energy_pJ
        analog_e = analog_mac_energy_pJ

        profile = AnalogAccelerationProfile(
            architecture=family,
            digital_iterations=1,
            analog_settling_time_constants=1.0,
            speedup_settling_vs_digital=throughput_ratio,
            digital_energy_per_iteration_pJ=digital_mac_energy_pJ,
            analog_energy_per_step_pJ=analog_e,
            energy_saving_ratio=digital_e / analog_e if analog_e > 0 else 1.0,
            speedup_notes=(
                f"Transformer: Linear layers speedup ≈ {throughput_ratio:.0f}× "
                f"(throughput ratio). Attention O(n²) softmax remains digital bottleneck. "
                "Kernel approximation (Büchel et al. 2024) can reduce this."
            ),
        )

    return profile


def acceleration_summary_table(profiles: list[AnalogAccelerationProfile]) -> str:
    """Format a markdown-like table of acceleration profiles."""
    lines = [
        "| Architecture | Digital Iters | Analog τ | Speedup | Energy Saving | Notes |",
        "|-------------|---------------|----------|---------|---------------|-------|",
    ]
    for p in profiles:
        lines.append(
            f"| {p.architecture.value:12s} | {p.digital_iterations:13d} | "
            f"{p.analog_settling_time_constants:8.1f} | "
            f"{p.speedup_settling_vs_digital:7.1f}× | "
            f"{p.energy_saving_ratio:13.1f}× | "
            f"{p.speedup_notes[:40]:40s} |"
        )
    return "\n".join(lines)


def compute_all_accelerations(
    dynamics_map: dict[ArchitectureFamily, DynamicsProfile],
    **kwargs,
) -> dict[ArchitectureFamily, AnalogAccelerationProfile]:
    """Compute acceleration for all 7 architecture families.

    Args:
        dynamics_map: Mapping from ArchitectureFamily to DynamicsProfile.
        **kwargs: Passed to compute_harnessing (energy/throughput constants).

    Returns:
        Dict mapping each family to its AnalogHarnessingProfile.
    """
    return {
        family: compute_harnessing(family, dynamics_map.get(family, DynamicsProfile()), **kwargs)
        for family in ArchitectureFamily
    }


# Backward-compatible aliases (must be defined BEFORE __all__ references them)
compute_harnessing = compute_acceleration
harnessing_summary_table = acceleration_summary_table
compute_all_harnessings = compute_all_accelerations

__all__ = [
    "AnalogAccelerationProfile",
    "AnalogHarnessingProfile",  # backward-compatible alias
    "compute_acceleration",
    "compute_harnessing",       # backward-compatible alias
    "acceleration_summary_table",
    "harnessing_summary_table",  # backward-compatible alias
    "compute_all_accelerations",
    "compute_all_harnessings",  # backward-compatible alias
]
