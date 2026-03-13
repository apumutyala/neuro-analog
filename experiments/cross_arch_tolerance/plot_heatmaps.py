#!/usr/bin/env python3
"""
Generate layer-by-layer analog/digital partition heatmaps for all architectures.
This script uses the theoretical/reference graph builders from the extractors
to generate the heatmaps without requiring full pre-trained HuggingFace checkpoints.
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from neuro_analog.extractors.neural_ode import NeuralODEExtractor
from neuro_analog.extractors.transformer import TransformerExtractor
from neuro_analog.extractors.diffusion import DiTExtractor, StableDiffusionExtractor
from neuro_analog.extractors.flow import FLUXExtractor
from neuro_analog.extractors.ebm import EBMExtractor
from neuro_analog.extractors.deq import DEQExtractor
from neuro_analog.extractors.ssm import MambaExtractor
from neuro_analog.visualization.partition_map import plot_partition_map

_FIGURES_DIR = Path(__file__).parent / "figures"
_FIGURES_DIR.mkdir(exist_ok=True)

def main():
    print("Generating analog partition heatmaps...")
    
    graphs_to_plot = []
    
    # 1. Neural ODE
    print("Building Neural ODE reference graph...")
    node_ext = NeuralODEExtractor.demo()
    graphs_to_plot.append(("neural_ode", node_ext.build_graph()))
    
    # 2. Transformer
    print("Building Transformer reference graph...")
    graphs_to_plot.append(("transformer", TransformerExtractor.reference()._graph))
    
    # 3. EBM
    print("Building EBM (Boltzmann) reference graph...")
    graphs_to_plot.append(("ebm_rbm", EBMExtractor.rbm().build_graph()))
    
    # 4. DEQ
    print("Building DEQ reference graph...")
    deq_ext = DEQExtractor.reference()
    graphs_to_plot.append(("deq", deq_ext.build_graph()))
    
    # 5. Diffusion (DiT)
    print("Building Diffusion (DiT) reference graph...")
    # DiTExtractor can build_graph without loading a model
    dit = DiTExtractor("dit")
    graphs_to_plot.append(("diffusion_dit", dit.build_graph()))
    
    # 6. Flow (FLUX)
    print("Building Flow (FLUX) reference graph...")
    # FLUXExtractor can build_graph without loading a model
    flux = FLUXExtractor("flux")
    flux.model = None
    graphs_to_plot.append(("flow_flux", flux.build_graph()))
    
    # 7. SSM (Mamba)
    print("Building SSM (Mamba) graph... (requires HF download, skipping to avoid hang)")

    for name, graph in graphs_to_plot:
        if graph is None:
            print(f"[{name}] Failed to extract graph.")
            continue
            
        out_path = _FIGURES_DIR / f"heatmap_partition_{name}.png"
        try:
            plot_partition_map(graph, output_path=str(out_path), title=f"Analog/Digital Partition — {name.upper()}")
            print(f"[{name}] Saved heatmap to {out_path}")
        except Exception as e:
            print(f"[{name}] Error generating heatmap: {e}")

if __name__ == "__main__":
    main()
