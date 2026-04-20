"""Base extractor interface for architecture-specific parameter extraction."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import math
import torch
from neuro_analog.ir import AnalogGraph, AnalogAmenabilityProfile, ArchitectureFamily, DynamicsProfile, PrecisionSpec

class BaseExtractor(ABC):
    """Abstract base class for architecture-specific extractors."""
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", seq_len: int | None = None):
        self.model_name = model_name
        self.device = device
        self.model: Any = None
        self._graph: AnalogGraph | None = None
        self._activation_specs: dict[str, PrecisionSpec] | None = None
        self.seq_len = seq_len  # Sequence length for FLOP calculations (None = use architecture default)

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

    def calibrate_activations(
        self,
        calibration_data: torch.Tensor | None = None,
        percentile: float = 99.9,
    ) -> dict[str, PrecisionSpec]:
        """Run a forward pass and measure per-layer activation dynamic ranges via hooks.

        Uses percentile-based clipping (industry standard for PTQ and analog V_ref
        selection) rather than absolute min/max.  Absolute extremes are dominated by
        rare outliers and cause the ADC to waste most of its range on headroom — the
        same failure mode TensorRT's entropy calibrator was designed to fix.

        Args:
            calibration_data: Representative input batch.  The batch should cover the
                typical operating distribution; a randomly-selected 256–512-sample
                slice of the training set is sufficient for most models.  A single
                forward pass is run; hooks record the full distribution of activation
                values at every leaf module.
            percentile: Upper percentile (and its symmetric lower counterpart) used
                to clip the activation range.  Default 99.9 follows TensorRT/PyTorch
                FX convention: covers 99.9% of inference inputs while discarding the
                0.1% outlier tail that would otherwise inflate V_ref.

        Returns:
            dict mapping leaf module names to PrecisionSpec with activation_min,
            activation_max (at the requested percentile), activation_std, and
            activation_bits computed from the crest factor.
        """
        assert self.model is not None, "Call load_model() first"
        activation_vals: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def make_hook(name):
            def hook_fn(mod, inp, out):
                if isinstance(out, torch.Tensor):
                    activation_vals.setdefault(name, []).append(
                        out.detach().float().reshape(-1)
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

        lo_q = (100.0 - percentile) / 100.0
        hi_q = percentile / 100.0
        specs = {}
        for name, tensors in activation_vals.items():
            all_vals = torch.cat(tensors)
            act_min = float(torch.quantile(all_vals, lo_q))
            act_max = float(torch.quantile(all_vals, hi_q))
            act_std = float(all_vals.std())
            # Crest-factor-based ADC bit requirement:
            #   bits = ceil(log2(activation_max / activation_std)) + 4
            # Rationale: ceil(log2(crest)) bits cover the dynamic range headroom;
            # +4 provides baseline resolution for the signal body.  Crest=10 → 8 bits
            # (the empirical standard for analog crossbar MVMs).
            crest = act_max / max(act_std, 1e-10) if act_max > 0 else 1.0
            act_bits = max(4, min(16, math.ceil(math.log2(max(crest, 1.0))) + 4))
            specs[name] = PrecisionSpec(
                activation_min=act_min,
                activation_max=act_max,
                activation_std=act_std,
                activation_bits=act_bits,
            )
        return specs

    def _apply_activation_specs(
        self, profile: AnalogAmenabilityProfile
    ) -> AnalogAmenabilityProfile:
        """Incorporate activation calibration data into an existing profile.

        Computes ``min_activation_precision_bits`` from the crest-factor-based
        bit estimates in ``_activation_specs``, writes it onto the profile, and
        re-runs ``compute_scores()`` so ``precision_score`` reflects the joint
        weight + activation precision demand.
        """
        if not self._activation_specs:
            return profile
        bits_list = [
            spec.activation_bits
            for spec in self._activation_specs.values()
            if spec.activation_bits > 0
        ]
        if bits_list:
            # Consistent with graph.analyze() convention: use min (most-favourable
            # layer) as the headline figure so the score rewards architectures where
            # at least some layers tolerate low precision.
            profile.min_activation_precision_bits = min(bits_list)
            profile.compute_scores()
        return profile

    def run(self, calibration_data: torch.Tensor | None = None, log_profile_path: str | None = None) -> AnalogAmenabilityProfile:
        """Full pipeline: load → extract → build graph → analyze.

        Args:
            calibration_data: Optional representative input batch (e.g. X_train[:32]).
                If provided, a single forward pass is run with forward hooks to measure
                per-layer activation min/max/std under real data.  These values populate
                ``self.activation_specs`` and directly determine the per-layer V_ref
                needed for ADC calibration (V_ref = 1.1 × activation_max).  Without
                calibration data the activation fields in each PrecisionSpec remain zero
                and ``AnalogLinear.calibrate()`` will fall back to a fixed default V_ref,
                which may cause clipping (ADC saturation) or waste precision on headroom.
            log_profile_path: Optional path to save profile as JSON. If None, saves to
                outputs/profiles/{model_name}_{timestamp}.json by default.
        """
        print(f"[neuro-analog] Loading {self.model_name}...")
        self.load_model()
        print("[neuro-analog] Extracting dynamics...")
        dynamics = self.extract_dynamics()
        print("[neuro-analog] Building IR graph...")
        graph = self.build_graph()
        graph.set_dynamics(dynamics)
        self._graph = graph
        if calibration_data is not None:
            print("[neuro-analog] Calibrating activations...")
            self._activation_specs = self.calibrate_activations(calibration_data)
        print("[neuro-analog] Analyzing...")
        profile = graph.analyze()
        profile = self._apply_activation_specs(profile)
        print(f"[neuro-analog] Done. Score: {profile.overall_score:.3f}")

        # Save profile to JSON (always-on with default path)
        if log_profile_path is None:
            from pathlib import Path
            import time
            output_dir = Path("outputs/profiles")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            safe_name = self.model_name.replace("/", "_").replace("\\", "_")
            log_profile_path = str(output_dir / f"{safe_name}_{timestamp}.json")

        import json
        with open(log_profile_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        print(f"[neuro-analog] Profile saved to: {log_profile_path}")

        return profile

    @property
    def graph(self) -> AnalogGraph | None:
        return self._graph

    @property
    def activation_specs(self) -> dict[str, PrecisionSpec] | None:
        """Per-layer activation statistics from the last ``run(calibration_data=...)`` call.

        Keys are ``model.named_modules()`` leaf names.  Each ``PrecisionSpec`` carries:
        - ``activation_min`` / ``activation_max``: observed output range across the
          calibration batch — sets the V_ref window for ADC/DAC conversion.
        - ``activation_std``: average output standard deviation — a proxy for SNR
          under thermal noise (ε ~ N(0, σ_th²)); layers with low std relative to
          their range are more sensitive to quantization error.

        Returns None if ``run()`` was called without calibration data.
        """
        return self._activation_specs
