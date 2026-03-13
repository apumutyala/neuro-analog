"""Base extractor interface for architecture-specific parameter extraction."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import torch
from neuro_analog.ir import AnalogGraph, AnalogAmenabilityProfile, ArchitectureFamily, DynamicsProfile, PrecisionSpec

class BaseExtractor(ABC):
    """Abstract base class for architecture-specific extractors."""
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model: Any = None
        self._graph: AnalogGraph | None = None

    @property
    @abstractmethod
    def family(self) -> ArchitectureFamily: ...

    @abstractmethod
    def load_model(self) -> None:
        """Load the pretrained model from HuggingFace."""
        ...

    @abstractmethod
    def extract_dynamics(self) -> DynamicsProfile:
        """Extract continuous-time dynamics parameters."""
        ...

    @abstractmethod
    def build_graph(self) -> AnalogGraph:
        """Decompose model into AnalogGraph of primitive operations."""
        ...

    def extract_weight_statistics(self) -> dict[str, PrecisionSpec]:
        """Extract weight distribution statistics from all named parameters."""
        assert self.model is not None, "Call load_model() first"
        stats = {}
        for name, param in self.model.named_parameters():
            data = param.detach().float()
            stats[name] = PrecisionSpec(
                weight_min=float(data.min()), weight_max=float(data.max()),
                weight_std=float(data.std()),
            )
        return stats

    def calibrate_activations(self, calibration_data: torch.Tensor | None = None) -> dict[str, PrecisionSpec]:
        """Run forward passes to measure activation dynamic ranges via hooks."""
        assert self.model is not None, "Call load_model() first"
        activation_stats: dict[str, list] = {}
        hooks = []
        def make_hook(name):
            def hook_fn(mod, inp, out):
                if isinstance(out, torch.Tensor):
                    activation_stats.setdefault(name, []).append(
                        (float(out.min()), float(out.max()), float(out.std()))
                    )
            return hook_fn
        for name, module in self.model.named_modules():
            if not list(module.children()):
                hooks.append(module.register_forward_hook(make_hook(name)))
        self.model.eval()
        with torch.no_grad():
            if calibration_data is not None:
                self.model(calibration_data)
        for h in hooks:
            h.remove()
        specs = {}
        for name, slist in activation_stats.items():
            specs[name] = PrecisionSpec(
                activation_min=min(s[0] for s in slist),
                activation_max=max(s[1] for s in slist),
                activation_std=sum(s[2] for s in slist) / len(slist),
            )
        return specs

    def run(self) -> AnalogAmenabilityProfile:
        """Full pipeline: load → extract → build graph → analyze."""
        print(f"[neuro-analog] Loading {self.model_name}...")
        self.load_model()
        print("[neuro-analog] Extracting dynamics...")
        dynamics = self.extract_dynamics()
        print("[neuro-analog] Building IR graph...")
        graph = self.build_graph()
        graph.set_dynamics(dynamics)
        self._graph = graph
        print("[neuro-analog] Analyzing...")
        profile = graph.analyze()
        print(f"[neuro-analog] Done. Score: {profile.overall_score:.3f}")
        return profile

    @property
    def graph(self) -> AnalogGraph | None:
        return self._graph
