"""
Layer-by-layer partition heatmap visualization.

Green  = ANALOG   (runs on physics — crossbar, RC, p-bit)
Red    = DIGITAL  (requires digital logic — softmax, LayerNorm, GELU)
Yellow = HYBRID   (analog-possible with quality tradeoff)

Vertical dashed lines mark D/A boundaries (ADC/DAC locations).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from neuro_analog.ir import AnalogGraph
from neuro_analog.ir.types import Domain


# Domain color scheme
_DOMAIN_COLORS = {
    Domain.ANALOG: "#2ecc71",    # Green
    Domain.DIGITAL: "#e74c3c",   # Red
    Domain.HYBRID: "#f1c40f",    # Yellow
}
_DOMAIN_LABELS = {
    Domain.ANALOG: "Analog (physics)", Domain.DIGITAL: "Digital", Domain.HYBRID: "Hybrid",
}


def plot_partition_map(
    graph: AnalogGraph,
    output_path: Optional[str | Path] = None,
    max_nodes: int = 80,
    figsize: tuple[int, int] = (16, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """Generate a layer-by-layer analog/digital partition heatmap.

    Each column is one operation node. Color = domain.
    Red dashed vertical lines = D/A boundaries (ADC/DAC required).

    Args:
        graph: AnalogGraph from any extractor.
        output_path: If provided, save PNG to this path.
        max_nodes: Truncate to first N nodes for readability.
        figsize: Figure size (width, height) in inches.
        title: Override plot title.

    Returns:
        matplotlib Figure object.
    """
    nodes = graph.nodes[:max_nodes]
    if not nodes:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
        return fig

    boundaries = graph.find_da_boundaries()
    boundary_node_ids = {b.source_node_id for b in boundaries}

    # Build color array
    colors = [_DOMAIN_COLORS.get(n.domain, "#95a5a6") for n in nodes]
    flops = [max(n.flops, 1) for n in nodes]  # avoid log(0)
    log_flops = np.log1p(flops)
    bar_heights = log_flops / log_flops.max()

    fig, (ax_bar, ax_label) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [4, 1]}, sharex=True
    )
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax_bar, ax_label):
        ax.set_facecolor("#16213e")

    x = np.arange(len(nodes))
    bars = ax_bar.bar(x, bar_heights, color=colors, edgecolor="none", width=0.9, alpha=0.9)

    # D/A boundary lines
    boundary_positions = [i for i, n in enumerate(nodes) if n.node_id in boundary_node_ids]
    for pos in boundary_positions:
        ax_bar.axvline(x=pos + 0.5, color="#e67e22", linewidth=1.5, linestyle="--", alpha=0.8)

    # FLOP annotation
    ax_bar.set_ylabel("log(FLOPs) — normalized", color="white", fontsize=9)
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#444")

    title_str = title or f"Analog/Digital Partition — {graph.name}"
    ax_bar.set_title(title_str, color="white", fontsize=11, fontweight="bold", pad=8)

    # Legend
    patches = [mpatches.Patch(color=c, label=_DOMAIN_LABELS[d]) for d, c in _DOMAIN_COLORS.items()]
    patches.append(mpatches.Patch(color="#e67e22", label=f"D/A boundary ({len(boundaries)} total)"))
    ax_bar.legend(handles=patches, loc="upper right", facecolor="#16213e",
                  labelcolor="white", fontsize=8, framealpha=0.9)

    # Domain-colored label strip
    import matplotlib.colors as mcolors
    domain_colors_strip = [mcolors.to_rgb(_DOMAIN_COLORS.get(n.domain, "#95a5a6")) for n in nodes]
    ax_label.imshow(
        [domain_colors_strip], aspect="auto", extent=[-0.5, len(nodes) - 0.5, 0, 1]
    )
    ax_label.set_yticks([])
    ax_label.set_xlabel("Operation index (left→right = model forward pass)", color="white", fontsize=9)
    ax_label.tick_params(colors="white")
    ax_label.spines[:].set_color("#444")

    # Stats annotation
    fracs = graph.flop_fractions()
    analog_pct = fracs.get(Domain.ANALOG, 0.0) * 100
    digital_pct = fracs.get(Domain.DIGITAL, 0.0) * 100
    stats_text = (
        f"Analog: {analog_pct:.1f}%  |  Digital: {digital_pct:.1f}%  |  "
        f"D/A boundaries: {len(boundaries)}  |  Nodes shown: {len(nodes)}/{graph.node_count}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", color="#aaa", fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def plot_partition_comparison(
    graphs: dict[str, AnalogGraph],
    output_path: Optional[str | Path] = None,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Side-by-side partition bars for multiple architectures.

    One row per architecture. Each bar shows analog/digital/hybrid fractions.
    """
    from neuro_analog.ir.types import Domain

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    names = list(graphs.keys())
    y = np.arange(len(names))
    bar_height = 0.6

    for i, (name, graph) in enumerate(graphs.items()):
        fracs = graph.flop_fractions()
        analog = fracs.get(Domain.ANALOG, 0.0)
        hybrid = fracs.get(Domain.HYBRID, 0.0)
        digital = fracs.get(Domain.DIGITAL, 0.0)

        ax.barh(i, analog, height=bar_height, color=_DOMAIN_COLORS[Domain.ANALOG], label="Analog" if i == 0 else "")
        ax.barh(i, hybrid, height=bar_height, left=analog, color=_DOMAIN_COLORS[Domain.HYBRID], label="Hybrid" if i == 0 else "")
        ax.barh(i, digital, height=bar_height, left=analog + hybrid, color=_DOMAIN_COLORS[Domain.DIGITAL], label="Digital" if i == 0 else "")
        ax.text(1.01, i, f"{analog*100:.1f}% analog", va="center", color="white", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(names, color="white", fontsize=10)
    ax.set_xlabel("FLOP fraction", color="white")
    ax.set_xlim(0, 1.25)
    ax.set_title("Cross-Architecture Partition Comparison", color="white", fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#16213e", labelcolor="white", fontsize=9)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig
