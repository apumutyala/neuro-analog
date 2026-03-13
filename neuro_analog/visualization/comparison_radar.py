"""
Six-axis radar chart comparing all architecture families on analog amenability.

Axes:
  1. Analog FLOP %          — more = better (weights on physics)
  2. D/A boundary score     — fewer boundaries = better (inverted count)
  3. Precision tolerance     — lower bits needed = better
  4. Dynamics naturalness    — ODE/SDE fit to Arco/Legno/Shem
  5. Mismatch resilience     — from propagate_mismatch() analysis
  6. Shem compatibility      — can we generate a valid Shem input today?
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
}

_ARCH_NAMES = {
    ArchitectureFamily.NEURAL_ODE: "Neural ODE",
    ArchitectureFamily.SSM: "SSM (Mamba)",
    ArchitectureFamily.EBM: "EBM",
    ArchitectureFamily.FLOW: "Flow (FLUX)",
    ArchitectureFamily.TRANSFORMER: "Transformer",
    ArchitectureFamily.DIFFUSION: "Diffusion (SD)",
}

AXES = [
    "Analog\nFLOP %",
    "D/A Boundary\nScore",
    "Precision\nTolerance",
    "Dynamics\nNaturalness",
    "Mismatch\nResilience",
    "Shem\nCompatibility",
]


def _compute_radar_scores(
    profile: AnalogAmenabilityProfile,
    mismatch_resilience: float = 0.5,
    shem_compat: float = 0.5,
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

    # Axis 6: Shem compatibility (externally provided — 1.0 = direct input, 0.0 = not applicable)
    shem_score = max(0.0, min(1.0, shem_compat))

    return [analog_frac, boundary_score, precision_score, dynamics_score, mismatch_score, shem_score]


def plot_radar(
    profiles: list[tuple[AnalogAmenabilityProfile, float, float]],
    output_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (10, 10),
    title: str = "Analog Amenability: Cross-Architecture Comparison",
) -> plt.Figure:
    """Generate a six-axis radar chart.

    Args:
        profiles: List of (profile, mismatch_resilience, shem_compat) tuples.
                  mismatch_resilience: 0-1 score from mismatch analysis.
                  shem_compat: 0-1 score (1.0 = valid Shem input today).
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
    for profile, mismatch_r, shem_c in profiles:
        scores = _compute_radar_scores(profile, mismatch_r, shem_c)
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
    shem_scores: Optional[dict[str, float]] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Convenience wrapper: build radar from an AnalogTaxonomy object."""
    # Default Shem compatibility by architecture
    default_shem = {
        ArchitectureFamily.NEURAL_ODE: 1.0,   # Direct Shem input
        ArchitectureFamily.SSM: 0.75,          # ODE form; scale challenge
        ArchitectureFamily.EBM: 0.4,           # Energy, not ODE
        ArchitectureFamily.FLOW: 0.7,          # ODE form; v_θ scale challenge
        ArchitectureFamily.TRANSFORMER: 0.2,   # No ODE form
        ArchitectureFamily.DIFFUSION: 0.5,     # SDE; CLD maps well
    }
    default_mismatch = {
        ArchitectureFamily.NEURAL_ODE: 0.9,   # Small params → easy calibration
        ArchitectureFamily.SSM: 0.65,
        ArchitectureFamily.EBM: 0.8,           # p-bits inherently stochastic
        ArchitectureFamily.FLOW: 0.6,
        ArchitectureFamily.TRANSFORMER: 0.55,
        ArchitectureFamily.DIFFUSION: 0.45,    # Many noisy steps
    }

    profiles_input = []
    for entry in taxonomy.entries:
        mr = (mismatch_scores or {}).get(entry.model_name, default_mismatch.get(entry.family, 0.5))
        sc = (shem_scores or {}).get(entry.model_name, default_shem.get(entry.family, 0.5))
        profiles_input.append((entry.profile, mr, sc))

    return plot_radar(profiles_input, output_path=output_path)
