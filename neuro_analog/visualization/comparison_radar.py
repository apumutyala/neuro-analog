"""
Six-axis radar chart comparing all architecture families on analog amenability.

Axes:
  1. Analog FLOP %          — more = better (weights on physics)
  2. D/A boundary score     — fewer boundaries = better (inverted count)
  3. Precision tolerance     — lower bits needed = better
  4. Dynamics naturalness    — ODE/SDE fit to Ark (Arco/Legno)
  5. Mismatch resilience     — from propagate_mismatch() analysis
  6. Ark compatibility       — can we generate a valid BaseAnalogCkt subclass today?
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuro_analog.ir.types import AnalogAmenabilityProfile, ArchitectureFamily


# Architecture display names and colors
_ARCH_COLORS = {
    ArchitectureFamily.NEURAL_ODE: "#9b59b6",   # Purple
    ArchitectureFamily.SSM: "#2ecc71",           # Green
    ArchitectureFamily.EBM: "#1abc9c",           # Teal
    ArchitectureFamily.FLOW: "#3498db",          # Blue
    ArchitectureFamily.TRANSFORMER: "#e67e22",   # Orange
    ArchitectureFamily.DIFFUSION: "#e74c3c",     # Red
    ArchitectureFamily.DEQ: "#f39c12",           # Yellow-orange
    ArchitectureFamily.DTM: "#fd79a8",           # Pink (Extropic thermodynamic)
}

_ARCH_NAMES = {
    ArchitectureFamily.NEURAL_ODE: "Neural ODE",
    ArchitectureFamily.SSM: "SSM (Mamba)",
    ArchitectureFamily.EBM: "EBM",
    ArchitectureFamily.FLOW: "Flow (FLUX)",
    ArchitectureFamily.TRANSFORMER: "Transformer",
    ArchitectureFamily.DIFFUSION: "Diffusion (SD)",
    ArchitectureFamily.DEQ: "DEQ",
    ArchitectureFamily.DTM: "DTM (Extropic)",
}

AXES = [
    "Analog\nFLOP %",
    "D/A Boundary\nScore",
    "Precision\nTolerance",
    "Dynamics\nNaturalness",
    "Mismatch\nResilience",
    "Ark\nCompatibility",
]


def _compute_radar_scores(
    profile: AnalogAmenabilityProfile,
    mismatch_resilience: float = 0.5,
    ark_compat: float = 0.5,
) -> list[float]:
    """Convert an AnalogAmenabilityProfile to a 6-axis radar score vector (all in [0,1])."""

    # Axis 1: Analog FLOP fraction (direct)
    analog_frac = min(1.0, profile.analog_flop_fraction)

    # Axis 2: D/A boundary score — invert: 0 boundaries = 1.0, 100+ boundaries = 0.0
    boundary_score = max(0.0, 1.0 - profile.da_boundary_count / 100.0)

    # Axis 3: Precision tolerance — fewer bits needed = better
    # 4 bits = 1.0 (excellent), 16 bits = 0.0 (needs full precision)
    precision_score = max(0.0, 1.0 - (profile.min_weight_precision_bits - 4) / 12.0)

    # Axis 4: Dynamics naturalness (directly from compute_scores)
    dynamics_score = profile.dynamics_score

    # Axis 5: Mismatch resilience (externally provided — from propagate_mismatch())
    mismatch_score = max(0.0, min(1.0, mismatch_resilience))

    # Axis 6: Ark compatibility (externally provided — 1.0 = valid BaseAnalogCkt today, 0.0 = not applicable)
    ark_score = max(0.0, min(1.0, ark_compat))

    return [analog_frac, boundary_score, precision_score, dynamics_score, mismatch_score, ark_score]


def plot_radar(
    profiles: list[tuple[AnalogAmenabilityProfile, float, float]],
    output_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 10),
    title: str = "Analog Amenability: Cross-Architecture Comparison",
) -> plt.Figure:
    """Generate a six-axis radar chart.

    Args:
        profiles: List of (profile, mismatch_resilience, ark_compat) tuples.
                  mismatch_resilience: 0-1 score from mismatch analysis.
                  ark_compat: 0-1 score (1.0 = valid BaseAnalogCkt subclass today).
        output_path: If provided, save PNG to this path.
        figsize: Figure dimensions.
        title: Chart title.

    Returns:
        matplotlib Figure.
    """
    n_axes = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Draw gridlines
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, color="white", fontsize=10, fontweight="bold")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#888", fontsize=7)
    ax.set_ylim(0, 1)
    ax.grid(color="#444", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.spines["polar"].set_color("#444")

    # Plot each architecture
    for profile, mismatch_r, ark_c in profiles:
        scores = _compute_radar_scores(profile, mismatch_r, ark_c)
        scores += scores[:1]  # close polygon
        color = _ARCH_COLORS.get(profile.architecture, "#aaa")
        name = _ARCH_NAMES.get(profile.architecture, profile.architecture.value)
        overall = profile.overall_score

        ax.plot(angles, scores, color=color, linewidth=2.0, linestyle="-", label=f"{name} ({overall:.2f})")
        ax.fill(angles, scores, color=color, alpha=0.12)

        # Mark each vertex
        for angle, score in zip(angles[:-1], scores[:-1]):
            ax.plot(angle, score, "o", color=color, markersize=5)

    # Legend
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.1),
        facecolor="#16213e",
        labelcolor="white",
        fontsize=9,
        framealpha=0.9,
        title="Architecture (overall score)",
        title_fontsize=9,
    )
    legend.get_title().set_color("white")

    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_radar_from_taxonomy(
    taxonomy,
    mismatch_scores: Optional[dict[str, float]] = None,
    ark_scores: Optional[dict[str, float]] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Convenience wrapper: build radar from an AnalogTaxonomy object."""
    # Ark compatibility — does the EXPERIMENT MODEL for this architecture produce a valid
    # BaseAnalogCkt subclass today?  All experiment models are small MLPs so the CDG
    # capacity constraint (which rules out 12B FLUX or 860M SD U-Net at production scale)
    # does not apply here.
    #
    #   Neural ODE — export_neural_ode_to_ark()  → NeuralODEAnalogCkt  ✓
    #   SSM (S4D)  — export_s4d_to_ark()         → SSMAnalogCkt        ✓ (real/imag split)
    #   DEQ        — export_deq_to_ark()          → DEQAnalogCkt        ✓ (gradient-flow ODE)
    #   EBM        — export_hopfield_to_ark()     → HopfieldAnalogCkt   ✓ (CDG bridge)
    #   Diffusion  — export_diffusion_to_ark()    → DiffusionAnalogCkt  ✓ (VP-SDE prob-flow)
    #   Flow (MLP) — FlowMLPExtractor → export_neural_ode_to_ark()      ✓ (velocity IS MLP)
    #   Transformer— export_ffn_to_ark()          → analysis doc only   ✗ (no ODE dynamics;
    #                softmax requires global comparison, no analog equivalent)
    default_ark = {
        ArchitectureFamily.NEURAL_ODE:  1.0,
        ArchitectureFamily.SSM:         1.0,
        ArchitectureFamily.DEQ:         1.0,
        ArchitectureFamily.EBM:         1.0,
        ArchitectureFamily.DIFFUSION:   1.0,
        ArchitectureFamily.FLOW:        1.0,   # FlowMLP experiment model exports via neural_ode path
        ArchitectureFamily.TRANSFORMER: 0.2,   # No ODE form; FFN partition is analysis-only
        ArchitectureFamily.DTM:         0.0,   # Thermodynamic substrate (sMTJ Gibbs) has no ODE-form Ark export
    }
    # Mismatch resilience — normalised from measured σ thresholds at 5% degradation.
    # These are estimates pending a full rerun of the cross-arch sweep with bug fixes.
    # Neural ODE: σ≈0.10–0.12 (small MLP, well-conditioned, easy calibration)
    # DEQ:        σ≈0.10 (spectral norm enforces ρ<1 even under mismatch)
    # EBM:        high (Hopfield energy landscape wide; noise just blurs attractor basins)
    # SSM:        moderate (A_re decay structure helps, but B mismatch accumulates over sequence)
    # Flow/Transformer: moderate (many NFE accumulate; large weight count)
    # Diffusion:  lowest (100 NFE steps; each adds mismatch error; score net sensitive)
    # DTM:        highest — thermal noise IS the desired randomness, not a nonideality (Jelinčič 2025)
    # Mismatch resilience — measured σ threshold at 10% quality loss, normalized to [0,1]
    # by dividing by 0.15 (max tested sigma). ≥0.15 ceiling architectures score 1.0.
    # Source: cross_arch_tolerance sweep, 50 trials, conservative profile.
    default_mismatch = {
        ArchitectureFamily.NEURAL_ODE:  1.00,  # ≥σ=0.15 (no measurable threshold)
        ArchitectureFamily.EBM:         1.00,  # ≥σ=0.15
        ArchitectureFamily.SSM:         1.00,  # ≥σ=0.15
        ArchitectureFamily.DIFFUSION:   1.00,  # ≥σ=0.15
        ArchitectureFamily.TRANSFORMER: 1.00,  # ≥σ=0.15 (FFN-only analog partition)
        ArchitectureFamily.DEQ:         0.67,  # σ=0.10 threshold
        ArchitectureFamily.FLOW:        0.47,  # σ=0.07 threshold (mismatch-only ablation)
        ArchitectureFamily.DTM:         1.0,   # Thermal noise is the computation — mismatch is by design
    }

    profiles_input = []
    for entry in taxonomy.entries:
        mr = (mismatch_scores or {}).get(entry.model_name, default_mismatch.get(entry.family, 0.5))
        sc = (ark_scores or {}).get(entry.model_name, default_ark.get(entry.family, 0.5))
        profiles_input.append((entry.profile, mr, sc))

    return plot_radar(profiles_input, output_path=output_path)
