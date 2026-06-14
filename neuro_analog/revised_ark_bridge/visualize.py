"""
visualize.py — CDG visualization helpers for the notebooks.

Produces directed, labeled circuit diagrams showing:
- Nodes color-coded by type (StateVar=blue, OutUnit=green, InpNode=yellow)
- Edges labeled with name + weight value
- Self-loops shown as curved arcs
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_cdg(cdg, title="CDG Circuit Diagram", figsize=(10, 7)):
    """
    Draw a directed CDG with labeled edges and color-coded nodes.

    Args:
        cdg: Ark CDG instance (with .nodes and .edges)
        title: figure title
        figsize: matplotlib figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Collect nodes
    nodes = list(cdg.nodes)
    node_names = [n.name for n in nodes]
    name_to_node = {n.name: n for n in nodes}

    # Position nodes in a circle
    n_nodes = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 3.0
    positions = {
        name: (radius * np.cos(a), radius * np.sin(a))
        for name, a in zip(node_names, angles)
    }

    # Color by node type name
    type_colors = {
        "StateVar": "#4A90D9",
        "OutUnit":  "#5DBB63",
        "InpNode":  "#F4D03F",
        "Osc":      "#E74C3C",
    }

    # Draw edges (directed)
    for edge in cdg.edges:
        src_name = edge.src.name
        dst_name = edge.dst.name
        if src_name not in positions or dst_name not in positions:
            continue

        x1, y1 = positions[src_name]
        x2, y2 = positions[dst_name]

        # Self-loop
        if src_name == dst_name:
            # Draw a curved arc above the node
            arc = mpatches.FancyArrowPatch(
                (x1 + 0.3, y1 + 0.3),
                (x1 - 0.3, y1 + 0.3),
                connectionstyle="arc3,rad=0.5",
                arrowstyle="->",
                color="gray",
                lw=1.5,
                mutation_scale=12,
            )
            ax.add_patch(arc)
            # Label
            ax.text(x1, y1 + 0.9, f"{edge.name}",
                    fontsize=7, ha="center", va="bottom", color="dimgray")
            continue

        # Regular edge — offset arrow slightly so bidirectional edges don't overlap
        dx, dy = x2 - x1, y2 - y1
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            continue

        # Shorten by node radius so arrow doesn't overlap node circle
        node_r = 0.35
        fx = node_r / dist
        x1_a = x1 + fx * dx
        y1_a = y1 + fx * dy
        x2_a = x2 - fx * dx
        y2_a = y2 - fx * dy

        # Curvature for bidirectional edges
        edge_key = tuple(sorted([src_name, dst_name]))
        # Simple curvature: offset perpendicular
        perp_x, perp_y = -dy / dist * 0.15, dx / dist * 0.15
        # Check if reverse edge exists
        has_reverse = any(e.src.name == dst_name and e.dst.name == src_name for e in cdg.edges)
        if has_reverse and src_name > dst_name:
            perp_x, perp_y = -perp_x, -perp_y

        ax.annotate(
            "",
            xy=(x2_a + perp_x, y2_a + perp_y),
            xytext=(x1_a + perp_x, y1_a + perp_y),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5,
                           connectionstyle=f"arc3,rad={0.15 if has_reverse else 0}"),
        )

        # Edge label — weight if available
        label = edge.name
        if "g" in edge.attrs and hasattr(edge.attrs["g"], "mean"):
            w = float(edge.attrs["g"].mean)
            label += f"\n{w:.2f}"
        elif "a" in edge.attrs and hasattr(edge.attrs["a"], "mean"):
            w = float(edge.attrs["a"].mean)
            label += f"\n{w:.2f}"

        mid_x = (x1 + x2) / 2 + perp_x * 2
        mid_y = (y1 + y2) / 2 + perp_y * 2
        ax.text(mid_x, mid_y, label, fontsize=6, ha="center", va="center",
                color="dimgray", bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                            edgecolor="none", alpha=0.8))

    # Draw nodes
    for node in nodes:
        x, y = positions[node.name]
        color = type_colors.get(node.cdg_type.name, "#95A5A6")
        circle = plt.Circle((x, y), 0.35, color=color, ec="black", lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, node.name, fontsize=8, ha="center", va="center",
                fontweight="bold", color="white", zorder=6)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#4A90D9", label="StateVar (ODE state)"),
        mpatches.Patch(color="#5DBB63", label="OutUnit (activation)"),
        mpatches.Patch(color="#F4D03F", label="InpNode (fixed input)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    ax.set_xlim(-radius - 1.5, radius + 1.5)
    ax.set_ylim(-radius - 1.5, radius + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def visualize_mlp_field(weights: dict, title="MLP Field Circuit", figsize=(10, 4)):
    """
    Schematic of MLPFieldCkt as a crossbar array diagram.

    Shows 3-layer MLP with time-augmented input as a stacked crossbar.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Layer dims
    z_dim = weights["W3"].shape[0]
    h_dim = weights["W1"].shape[0]
    in_dim = weights["W1"].shape[1]  # z_dim + 1 (time)

    layer_dims = [in_dim, h_dim, h_dim, z_dim]
    layer_names = ["Input\n[z, t]", "Hidden 1\ntanh", "Hidden 2\ntanh", "Output\ndz/dt"]
    colors = ["#F4D03F", "#5DBB63", "#5DBB63", "#4A90D9"]

    x_positions = [1, 3, 5, 7]
    y_scale = 0.15

    for x, dim, name, color in zip(x_positions, layer_dims, layer_names, colors):
        # Draw column of neurons
        y_positions = np.linspace(-(dim - 1) * y_scale / 2, (dim - 1) * y_scale / 2, dim)
        for y in y_positions:
            circle = plt.Circle((x, y), 0.08, color=color, ec="black", lw=1, zorder=5)
            ax.add_patch(circle)
        # Label
        ax.text(x, max(y_positions) + 0.25, name, ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    # Draw connections (crossbar lines)
    for i in range(len(x_positions) - 1):
        x1, x2 = x_positions[i], x_positions[i + 1]
        n1 = layer_dims[i]
        n2 = layer_dims[i + 1]
        y1s = np.linspace(-(n1 - 1) * y_scale / 2, (n1 - 1) * y_scale / 2, n1)
        y2s = np.linspace(-(n2 - 1) * y_scale / 2, (n2 - 1) * y_scale / 2, n2)

        for y1 in y1s:
            for y2 in y2s:
                ax.plot([x1 + 0.08, x2 - 0.08], [y1, y2], color="lightgray", lw=0.3, zorder=1)

        # Weight matrix label
        w_key = f"W{i+1}"
        w_shape = weights[w_key].shape
        ax.text((x1 + x2) / 2, min(y1s[0], y2s[0]) - 0.15,
                f"{w_key}: {w_shape}", ha="center", va="top",
                fontsize=7, color="dimgray")

    ax.set_xlim(0, 8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def visualize_cld(title="CLD Diffusion Circuit", figsize=(10, 4)):
    """
    Schematic of CLDCkt as an RLC-circuit-equivalent diagram.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Two masses (x and v) with coupling
    ax.text(1, 0.5, "x\n(position)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#4A90D9", edgecolor="black"))

    ax.text(4, 0.5, "v\n(velocity)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E74C3C", edgecolor="black"))

    # dx/dt = (beta/M) * v  (spring-like coupling)
    ax.annotate("", xy=(3.5, 0.5), xytext=(1.5, 0.5),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(2.5, 0.7, "dx/dt = (β/M)·v", ha="center", va="bottom", fontsize=8, color="green")

    # dv/dt = -beta*x - (Gamma*beta/M)*v - score  (damped oscillator)
    ax.annotate("", xy=(1.5, 0.3), xytext=(3.5, 0.3),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(2.5, 0.1, "dv/dt = -β·x - Γβ/M·v - score(x)", ha="center", va="top", fontsize=8, color="red")

    # Noise on v (Johnson-Nyquist)
    ax.text(4, 0, "+ η(t)·√(2Γβ)", ha="center", va="top",
            fontsize=8, color="purple", style="italic")
    ax.annotate("", xy=(4, 0.35), xytext=(4, 0.05),
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.5, ls="--"))

    # Score net
    ax.text(2.5, -0.5, "Score MLP:\neps_θ(x, t)", ha="center", va="center",
            fontsize=8, color="dimgray",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray"))
    ax.annotate("", xy=(3.5, 0.3), xytext=(2.8, -0.3),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1, connectionstyle="arc3,rad=0.2"))

    ax.set_xlim(0, 5.5)
    ax.set_ylim(-0.8, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig
