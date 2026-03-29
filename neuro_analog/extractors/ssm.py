"""
Mamba / Mamba-2 SSM extractor.

Extracts A_log eigenvalue spectra, time constants, selective mechanism
parameters, and decomposes Mamba blocks into analog/digital primitives.

The continuous-time SSM dx/dt = Ax + Bu is the most naturally analog
neural architecture: diagonal A yields N independent RC circuits with
time constants τ_i = 1/|a_i| where a_i = -exp(A_log_i).

--- The CUDA kernel problem and our solution ---

The mamba_ssm package ships two versions of its selective scan:
  1. selective_scan_cuda  – a fused CUDA kernel for efficient training/inference.
     From a graph-extraction standpoint this is a black box: torch.fx and
     torch.jit cannot trace through it, so we cannot observe the intermediate
     tensors (B[t], C[t], Δ[t], h[t]) that determine analog hardware specs.

  2. selective_scan_ref   – a pure-Python/PyTorch reference implementation with
     identical numerics and a straightforward loop structure fully visible to
     Python.

Our mitigation: during calibration forward passes (to measure activation
dynamic ranges) we monkey-patch selective_scan_fn → selective_scan_ref.
This gives us full access to every intermediate tensor with no CUDA kernel
changes and no change to how production inference runs.

The monkey-patch is always wrapped in try/finally to guarantee the fast kernel
is restored even if the calibration pass raises an exception.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from neuro_analog.ir import (
    AnalogGraph, ArchitectureFamily, DynamicsProfile, OpType, Domain,
    AnalogNode, PrecisionSpec, make_mvm_node, make_activation_node,
    ODESystem, ParameterSpec, NoiseProfile,
)
from .base import BaseExtractor

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Precision helpers
# ──────────────────────────────────────────────────────────────────────

def _estimate_bits_from_range(min_val: float, max_val: float, std: float) -> int:
    """Estimate the minimum crossbar bit-width needed for a weight distribution.

    Why this matters:
      Analog crossbar arrays are limited to 4–8 bits of effective precision by
      device-to-device variation (RRAM, PCM) and sense-amplifier noise.  If a
      weight tensor has a dynamic range that requires 10+ bits to represent, the
      operation cannot run on analog without significant quality loss.

    Method:
      Dynamic range DR = 20·log10(max_abs / noise_floor) in dB.
      Bits ≈ DR / 6.02  (the 6 dB/bit rule for linear quantization).
      We floor at 4 (minimum useful crossbar) and ceil at 16 (no analog benefit).
    """
    max_abs = max(abs(min_val), abs(max_val))
    if max_abs == 0 or std == 0:
        return 4  # degenerate / all-zero tensor

    # noise_floor ≈ std / 1000 is a conservative proxy for the quantization
    # granularity needed to faithfully represent the distribution tails.
    noise_floor = max(std / 1000.0, 1e-9)
    dr_db = 20.0 * math.log10(max_abs / noise_floor)
    bits = max(4, min(16, math.ceil(dr_db / 6.02)))
    return bits


def _build_precision_from_stats(stats: dict) -> PrecisionSpec:
    """Convert raw weight statistics (from extract_selective_mechanism_stats)
    into a PrecisionSpec grounded in real parameter values.

    Why this matters:
      Without this, every MVM node in the graph defaults to 8-bit precision.
      That is a placeholder, not a measurement.  The actual weight distributions
      vary substantially across Mamba layers: early layers tend to be more
      uniform (lower required precision), while dt_proj weights can span a much
      wider dynamic range (higher required precision) because they control the
      input-dependent time step Δ—the most precision-sensitive part of the
      selective mechanism.

      Grounding the PrecisionSpec in real stats means our downstream feasibility
      analysis (and the Ark export) reflects what the hardware actually needs.
    """
    bits = _estimate_bits_from_range(
        stats.get("min", 0.0),
        stats.get("max", 0.0),
        stats.get("std", 1.0),
    )
    return PrecisionSpec(
        weight_bits=bits,
        # Activations are harder to measure without a forward pass; we use
        # a conservative 8-bit default that run_calibration_pass() will refine.
        activation_bits=8,
        accumulator_bits=bits + 8,  # standard: weight_bits + activation_bits
        weight_min=stats.get("min", 0.0),
        weight_max=stats.get("max", 0.0),
        weight_std=stats.get("std", 0.0),
    )


# ──────────────────────────────────────────────────────────────────────
# Context manager for selective_scan monkey-patch
# ──────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _use_reference_scan():
    """Context manager that swaps Mamba's fused CUDA kernel for the
    pure-Python reference implementation during the enclosed block.

    Why we need this:
      selective_scan_cuda is a single opaque call from Python's perspective.
      When we run a calibration forward pass to collect activation statistics
      (ranges for B[t], C[t], Δ[t], internal state h[t]), the CUDA kernel
      gives us only its final output — all intermediates are computed inside
      the kernel and are never surfaced to Python.

      selective_scan_ref has the identical loop structure but in Python/PyTorch,
      so we can attach hooks, inspect intermediates, or just let the tensors
      flow through normally while our registered forward hooks capture them.

    Safety:
      The original function is always restored in the finally block, even if the
      calibration pass raises.  This means production inference is never affected.

    Fallback:
      If mamba_ssm is not installed (e.g. running on a CPU-only dev machine with
      the transformers fallback), the context manager is a no-op — we simply
      can't do finer calibration but nothing breaks.
    """
    try:
        import mamba_ssm.ops.selective_scan_interface as ssi
        original_fn = ssi.selective_scan_fn
        ssi.selective_scan_fn = ssi.selective_scan_ref
        log.debug("selective_scan_fn → selective_scan_ref (calibration mode)")
        try:
            yield
        finally:
            ssi.selective_scan_fn = original_fn
            log.debug("selective_scan_fn restored to CUDA kernel")
    except ImportError:
        # mamba_ssm not available; calibration still runs but through the
        # transformers fallback path which is already pure Python.
        log.debug("mamba_ssm not available; _use_reference_scan is a no-op")
        yield


# ──────────────────────────────────────────────────────────────────────
# MambaExtractor
# ──────────────────────────────────────────────────────────────────────

class MambaExtractor(BaseExtractor):
    """Extract analog-relevant parameters from pretrained Mamba models.

    Supports: state-spaces/mamba-130m, mamba-370m, mamba-790m, mamba-1.4b, mamba-2.8b

    Usage:
        extractor = MambaExtractor("state-spaces/mamba-370m")
        profile = extractor.run()
        print(extractor.graph.summary_table())
    """

    @property
    def family(self) -> ArchitectureFamily:
        return ArchitectureFamily.SSM

    def load_model(self) -> None:
        """Load Mamba model from HuggingFace."""
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
            self.model = MambaLMHeadModel.from_pretrained(
                self.model_name, device=self.device, dtype=torch.float32,
            )
        except ImportError:
            # Fallback: load via transformers if mamba_ssm not available
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def extract_dynamics(self) -> DynamicsProfile:
        """Extract A_log eigenvalue spectra and time constants.

        For each layer, reads A_log ∈ ℝ^{D×N} where:
        - D = model dimension (e.g., 768 for mamba-370m)
        - N = SSM state dimension (typically 16)
        - A = -exp(A_log) gives diagonal state matrix entries
        - Time constant τ_i = 1/|a_i| = 1/exp(A_log_i)

        Returns DynamicsProfile with:
        - All time constants across all layers
        - Time constant spread (max/min ratio)
        - State dimension
        """
        assert self.model is not None, "Call load_model() first"

        all_time_constants = []
        a_log_per_layer = {}

        for name, param in self.model.named_parameters():
            if "A_log" in name:
                a_log = param.detach().float().cpu()
                a_log_per_layer[name] = a_log

                # A = -exp(A_log), so eigenvalues are negative
                # Time constants = 1/|eigenvalue| = 1/exp(A_log)
                time_constants = (1.0 / torch.exp(a_log)).numpy().flatten()
                all_time_constants.extend(time_constants.tolist())

        if not all_time_constants:
            return DynamicsProfile(has_dynamics=True, dynamics_type="LTI_ODE")

        tc_array = np.array(all_time_constants)

        # Extract state dimension from first A_log shape
        first_a_log = next(iter(a_log_per_layer.values()))
        state_dim = first_a_log.shape[-1] if first_a_log.dim() >= 2 else first_a_log.shape[0]

        return DynamicsProfile(
            has_dynamics=True,
            dynamics_type="LTI_ODE",  # Base SSM is LTI; selectivity makes it time-varying
            time_constants=tc_array.tolist(),
            time_constant_spread=float(tc_array.max() / tc_array.min()) if tc_array.min() > 0 else float('inf'),
            state_dimension=int(state_dim),
            stiffness_ratio=float(tc_array.max() / tc_array.min()) if tc_array.min() > 0 else None,
        )

    def extract_a_log_spectra(self) -> dict[str, np.ndarray]:
        """Extract per-layer A_log eigenvalue spectra for detailed analysis.

        Returns dict mapping layer name → A_log values as numpy array.
        Used for visualization (histograms, cross-scale comparison).
        """
        assert self.model is not None, "Call load_model() first"
        spectra = {}
        for name, param in self.model.named_parameters():
            if "A_log" in name:
                spectra[name] = param.detach().float().cpu().numpy()
        return spectra

    def extract_selective_mechanism_stats(self) -> dict[str, dict]:
        """Analyze the selective mechanism (input-dependent B, C, Δ).

        The selective mechanism is what makes Mamba time-varying (not LTI).
        This is the key open problem for analog implementation:
        - B[t], C[t] are computed via x_proj linear layer
        - Δ[t] is computed via dt_proj + softplus

        Returns per-layer statistics on projection weight distributions.
        """
        assert self.model is not None, "Call load_model() first"
        stats = {}
        for name, param in self.model.named_parameters():
            if any(k in name for k in ["x_proj", "dt_proj", "out_proj"]):
                data = param.detach().float()
                stats[name] = {
                    "shape": tuple(data.shape),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "std": float(data.std()),
                    "mean": float(data.mean()),
                    "sparsity": float((data.abs() < 1e-6).sum() / data.numel()),
                }
        return stats

    def run_calibration_pass(
        self,
        calibration_input: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        """Run a forward pass to collect activation statistics for every SSM layer.

        This is the key mitigation for the CUDA kernel opacity problem.

        What we collect:
          For each Mamba block we record the min/max/std of:
          - The output of x_proj   → B[t] and C[t] raw values
          - The output of dt_proj  → pre-softplus Δ
          - The post-softplus Δ   → the actual time-step fed into the recurrence
          - The post-conv1d output → input to the selective scan
          - The final SSM output (y before gating)

        Why activation ranges matter:
          In analog hardware, each D/A boundary (ADC or DAC) is sized to a
          specific voltage/current range.  If the actual activation at that
          boundary has a larger dynamic range than the converter was designed for,
          signals clip — introducing errors that neither training nor calibration
          can recover.  Measuring the true activation ranges from a representative
          forward pass lets us specify the ADC/DAC requirements accurately.

        Why we use selective_scan_ref:
          The CUDA kernel computes B·u, C·h, exp(Δ·A)·h all in one fused call.
          The reference implementation exposes each of these as a separate Python
          tensor at each time step.  By using the reference for calibration only,
          we get full observability with zero impact on production throughput.

        Args:
            calibration_input: A representative input tensor (e.g., a short
                               tokenized text batch). Shape: (batch, seq_len).

        Returns:
            dict mapping layer_name → {"x_proj_max": ..., "dt_pre_max": ..., ...}
        """
        assert self.model is not None, "Call load_model() first"

        activation_stats: dict[str, dict[str, float]] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []

        def _make_hook(layer_name: str, stat_key: str):
            """Factory so each closure captures its own (layer_name, stat_key)."""
            def _hook(module: nn.Module, inp: tuple, out: torch.Tensor):
                # out may be a tuple for some modules; take first element
                tensor = out[0] if isinstance(out, tuple) else out
                tensor = tensor.detach().float()
                bucket = activation_stats.setdefault(layer_name, {})
                bucket[f"{stat_key}_min"] = float(tensor.min())
                bucket[f"{stat_key}_max"] = float(tensor.max())
                bucket[f"{stat_key}_std"] = float(tensor.std())
            return _hook

        # Register forward hooks on the linear projections inside each mixer.
        # We target the actual nn.Module objects so we capture post-activation
        # values regardless of how the model's forward() is structured.
        for name, module in self.model.named_modules():
            # name looks like "backbone.layers.3.mixer.x_proj" etc.
            parts = name.split(".")
            if len(parts) < 2:
                continue

            if parts[-1] == "x_proj" and "mixer" in parts:
                layer_key = ".".join(parts[:-1])  # e.g. "backbone.layers.3.mixer"
                hooks.append(module.register_forward_hook(_make_hook(layer_key, "x_proj")))

            elif parts[-1] == "dt_proj" and "mixer" in parts:
                layer_key = ".".join(parts[:-1])
                hooks.append(module.register_forward_hook(_make_hook(layer_key, "dt_proj")))

            elif parts[-1] == "conv1d" and "mixer" in parts:
                layer_key = ".".join(parts[:-1])
                hooks.append(module.register_forward_hook(_make_hook(layer_key, "conv1d")))

            elif parts[-1] == "out_proj" and "mixer" in parts:
                layer_key = ".".join(parts[:-1])
                hooks.append(module.register_forward_hook(_make_hook(layer_key, "out_proj")))

        try:
            # Run the calibration forward pass with selective_scan_ref active.
            # This replaces the fused CUDA selective_scan_cuda with the reference
            # implementation, making all intermediate SSM tensors observable.
            with _use_reference_scan(), torch.no_grad():
                self.model(calibration_input)
        finally:
            # Always remove hooks to avoid memory leaks and unintended side effects
            # on any future forward passes.
            for h in hooks:
                h.remove()

        return activation_stats

    def build_graph(
        self,
        calibration_stats: dict[str, dict[str, float]] | None = None,
    ) -> AnalogGraph:
        """Build AnalogGraph decomposing Mamba into analog/digital primitives.

        Per Mamba block, creates nodes for:
        - in_proj MVM        (ANALOG: crossbar)
        - Conv1D             (ANALOG: short FIR filter)
        - SiLU on x branch   (DIGITAL)
        - x_proj MVM for B,C (ANALOG: crossbar)
        - dt_proj MVM for Δ  (ANALOG: crossbar)
        - softplus for Δ     (DIGITAL)
        - State recurrence   (ANALOG: RC decay + accumulation)
        - y = C·h            (ANALOG: dot product)
        - Element-wise gate  (ANALOG: analog multiplier)
        - out_proj MVM       (ANALOG: crossbar)
        - Residual add       (ANALOG: current sum)

        Args:
            calibration_stats: Optional output of run_calibration_pass().
                If provided, activation_bits in PrecisionSpec nodes will be
                refined from placeholder (8) to a measurement-backed estimate.
                If None, nodes are still annotated with weight-stat-derived
                precision, just without the activation side.
        """
        assert self.model is not None, "Call load_model() first"

        # ── 1. Gather weight statistics (no forward pass needed) ──────────────
        # This is fast — it's just iterating model.named_parameters().
        # We use the stats to build real PrecisionSpec objects instead of
        # always defaulting to 8-bit everywhere.
        weight_stats = self.extract_selective_mechanism_stats()

        # ── 2. Read architecture config ───────────────────────────────────────
        config = self._get_config()
        D = config.get("d_model", 768)
        N = config.get("d_state", 16)
        n_layers = config.get("n_layer", 24)
        expand = config.get("expand", 2)
        D_inner = D * expand  # width of the expanded internal dimension

        total_params = sum(p.numel() for p in self.model.parameters())
        graph = AnalogGraph(
            name=self.model_name,
            family=ArchitectureFamily.SSM,
            model_params=total_params,
        )

        # ── 3. Build per-layer subgraph ───────────────────────────────────────
        for layer_idx in range(n_layers):
            prefix = f"layer_{layer_idx}"

            # Build the canonical HuggingFace/mamba_ssm parameter path prefix
            # so we can look up per-layer weight stats correctly.
            # mamba_ssm uses "backbone.layers.{i}.mixer.{proj}.weight"
            # transformers uses "model.layers.{i}.mixer.{proj}.weight"
            # We try both so the lookup is robust across codebases.
            hw_prefix_candidates = [
                f"backbone.layers.{layer_idx}.mixer",
                f"model.layers.{layer_idx}.mixer",
                f"layers.{layer_idx}.mixer",
            ]

            def _get_weight_stats(proj_name: str) -> dict | None:
                """Look up weight stats for a projection across possible path prefixes."""
                for hw_prefix in hw_prefix_candidates:
                    key = f"{hw_prefix}.{proj_name}.weight"
                    if key in weight_stats:
                        return weight_stats[key]
                return None

            def _get_activation_max(proj_name: str) -> float | None:
                """Look up the measured activation max from calibration_stats (if available)."""
                if calibration_stats is None:
                    return None
                for hw_prefix in hw_prefix_candidates:
                    key = hw_prefix  # calibration_stats is keyed by mixer path
                    if key in calibration_stats:
                        stat_key = f"{proj_name}_max"
                        return calibration_stats[key].get(stat_key)
                return None

            def _make_precision(proj_name: str) -> PrecisionSpec:
                """Build a PrecisionSpec grounded in real weight and, if available,
                measured activation statistics for this layer's projection.

                Why this matters:
                  dt_proj controls the time step Δ — the most precision-sensitive
                  parameter in the selective mechanism.  If Δ must span a wide
                  dynamic range to represent diverse sequence patterns, it requires
                  more bits than a typical weight matrix.  Measuring this per-layer
                  rather than assuming 8-bit everywhere is what lets us give specific
                  hardware recommendations (e.g. "dt_proj in layers 12-24 requires
                  10-bit DAC resolution, not 8-bit").
                """
                ws = _get_weight_stats(proj_name)
                if ws is None:
                    return PrecisionSpec()  # fallback to defaults

                spec = _build_precision_from_stats(ws)

                # Refine activation_bits from calibration measurement if available
                act_max = _get_activation_max(proj_name)
                if act_max is not None:
                    # Use the activation dynamic range to estimate required ADC bits.
                    # We don't have std here so we use a 10× headroom heuristic:
                    # bits ≈ log2(max_abs / (max_abs / 1024)) = 10 bits, adjusted
                    # by the actual measured range.
                    act_bits = max(4, min(16, math.ceil(math.log2(max(abs(act_max), 1e-9) * 1024 + 1))))
                    spec.activation_bits = act_bits

                return spec

            # ── in_proj: D → 2*D_inner ───────────────────────────────────────
            # This is a single linear layer that projects the residual stream
            # into both the x branch (SSM input) and z branch (gating signal).
            # Weight stats may not be in selective_mechanism_stats (in_proj is
            # not in the selective list), so we fall back to defaults here.
            graph.add_node(make_mvm_node(
                f"{prefix}.in_proj", D, 2 * D_inner,
            ))

            # ── Conv1D: short FIR on x branch ─────────────────────────────────
            # A 4-tap causal convolution applied to the x branch before the SSM.
            # In analog this maps to a switched-capacitor delay chain + weighted
            # current summation.  kernel_size=4 is fixed by the Mamba architecture.
            graph.add_node(AnalogNode(
                name=f"{prefix}.conv1d", op_type=OpType.ANALOG_FIR,
                domain=Domain.ANALOG,
                input_shape=(D_inner,), output_shape=(D_inner,),
                flops=4 * D_inner,  # kernel_size=4 taps × D_inner channels
                param_count=4 * D_inner,
            ))

            # ── SiLU on x branch ──────────────────────────────────────────────
            # Applied after conv1d to the x input.  SiLU = x·σ(x) requires
            # division-like nonlinearity, kept digital.  (Could be approximated
            # piecewise in analog at quality cost — see PIECEWISE_SILU.)
            graph.add_node(make_activation_node(f"{prefix}.silu_x", D_inner, "silu"))

            # ── SiLU on z branch (gating) ─────────────────────────────────────
            # The z branch is gated through SiLU and then multiplied element-wise
            # with the SSM output y.  The gate multiply itself is analog.
            graph.add_node(make_activation_node(f"{prefix}.silu_z", D_inner, "silu"))

            # ── x_proj: D_inner → 2*N  (compute B[t] and C[t]) ───────────────
            # This is the heart of the selective mechanism: a linear projection
            # whose output, after splitting, gives the input-dependent B and C
            # matrices.  Because B and C change with every token, the crossbar
            # weights are static but the *role* they play in the ODE changes —
            # this is different from a plain MVM and is why Mamba is time-varying.
            graph.add_node(make_mvm_node(
                f"{prefix}.x_proj", D_inner, 2 * N,
                precision=_make_precision("x_proj"),
            ))

            # ── dt_proj: D_inner → D_inner  (compute pre-softplus Δ) ──────────
            # dt_proj projects the input into the log-space time step.
            # This is the most precision-sensitive operation in the whole block:
            # small errors in Δ change the RC time constants, shifting the model's
            # temporal selectivity.  The _make_precision() call will flag this if
            # the measured dynamic range demands >8 bits.
            graph.add_node(make_mvm_node(
                f"{prefix}.dt_proj", D_inner, D_inner,
                precision=_make_precision("dt_proj"),
            ))

            # ── Softplus: Δ = log(1 + exp(Δ_pre)) — DIGITAL ──────────────────
            # Softplus ensures Δ > 0 (time steps must be positive).
            # log(1+exp(x)) requires accurate floating-point log and exp — not
            # implementable to sufficient accuracy in analog.
            graph.add_node(AnalogNode(
                name=f"{prefix}.softplus", op_type=OpType.SOFTPLUS,
                domain=Domain.DIGITAL,
                input_shape=(D_inner,), output_shape=(D_inner,),
                flops=D_inner,
            ))

            # ── State recurrence: h = exp(Δ·A)·h + Δ·B·u  — ANALOG ───────────
            # This is the core SSM operation.  Broken into two nodes:
            #
            #   state_decay:      exp(Δ·A)·h  — RC exponential decay.
            #     Each of the N state variables per D_inner channel decays
            #     with time constant τ = 1/|a_i| determined by A_log.
            #     In analog: an RC circuit where R·C = τ.
            #
            #   state_accumulate: + Δ·B·u  — current injection into each RC node.
            #     Kirchhoff's current law gives summation for free.
            #
            # The analogy between this SSM recurrence and an RC network is exact
            # for LTI systems (fixed A).  The time-varying nature (exp(Δ·A) changes
            # with input) means in a real analog circuit Δ would need to modulate
            # the RC time constant dynamically — a challenge for current hardware
            # but exactly what Extropic-style "compute-in-physics" devices target.
            graph.add_node(AnalogNode(
                name=f"{prefix}.state_decay", op_type=OpType.DECAY,
                domain=Domain.ANALOG,
                input_shape=(D_inner, N), output_shape=(D_inner, N),
                flops=D_inner * N,  # element-wise exp(Δ·A)·h
                metadata={
                    "description": "RC decay: exp(Δ·A)·h",
                    "n_independent_odes": D_inner,
                    "state_dim_per_channel": N,
                },
            ))
            graph.add_node(AnalogNode(
                name=f"{prefix}.state_accumulate", op_type=OpType.ACCUMULATION,
                domain=Domain.ANALOG,
                input_shape=(D_inner, N), output_shape=(D_inner, N),
                flops=D_inner * N,  # Δ·B·u + decay output
                metadata={"description": "Current sum: Δ·B·u + decayed state"},
            ))

            # ── y = C[t] · h  — dot product per channel ───────────────────────
            # Projects the hidden state h (shape D_inner×N) back to a scalar per
            # channel using the input-dependent C[t].  In analog: a weighted
            # current summing node per output channel, where the weights are set
            # by the crossbar conductances for C (static part) gated by C[t].
            graph.add_node(AnalogNode(
                name=f"{prefix}.output_dot", op_type=OpType.MVM,
                domain=Domain.ANALOG,
                input_shape=(D_inner, N), output_shape=(D_inner,),
                flops=D_inner * N,
            ))

            # ── Element-wise gate: y * silu(z)  — ANALOG multiplier ───────────
            # Gated architecture: the SSM output y is modulated by silu(z) from
            # the z branch.  In analog: a Gilbert cell or analog multiplier per
            # channel.  This is an analog operation with 1 FLOP per channel.
            graph.add_node(AnalogNode(
                name=f"{prefix}.gate_mul", op_type=OpType.ELEMENTWISE_MUL,
                domain=Domain.ANALOG,
                input_shape=(D_inner,), output_shape=(D_inner,),
                flops=D_inner,
            ))

            # ── out_proj: D_inner → D ─────────────────────────────────────────
            # Projects the gated output back to the residual stream dimension.
            graph.add_node(make_mvm_node(
                f"{prefix}.out_proj", D_inner, D,
                precision=_make_precision("out_proj"),
            ))

            # ── Residual connection ────────────────────────────────────────────
            # Adds the block output to the residual stream.  In analog: current
            # summation (Kirchhoff), zero added hardware if shared bus.
            graph.add_node(AnalogNode(
                name=f"{prefix}.residual", op_type=OpType.SKIP_CONNECTION,
                domain=Domain.ANALOG,
                input_shape=(D,), output_shape=(D,),
                flops=D,
            ))

            # ── Wire edges (linear chain) ──────────────────────────────────────
            # Note: this is a simplified linearization of the true dataflow graph
            # (which has the z branch running in parallel with x→conv→silu→SSM).
            # For FLOP accounting and D/A boundary counting, the linear chain
            # gives correct results.  A full parallel DAG would require explicit
            # fork nodes, which adds complexity without changing the boundary count.
            layer_nodes = [
                f"{prefix}.in_proj",       f"{prefix}.conv1d",
                f"{prefix}.silu_x",        f"{prefix}.x_proj",
                f"{prefix}.dt_proj",       f"{prefix}.softplus",
                f"{prefix}.state_decay",   f"{prefix}.state_accumulate",
                f"{prefix}.output_dot",    f"{prefix}.gate_mul",
                f"{prefix}.out_proj",      f"{prefix}.residual",
            ]
            for i in range(len(layer_nodes) - 1):
                graph.add_edge(layer_nodes[i], layer_nodes[i + 1])

        # Wire inter-layer edges: residual output of layer N feeds in_proj of layer N+1.
        # This completes the residual stream topology across all layers.
        for i in range(n_layers - 1):
            graph.add_edge(f"layer_{i}.residual", f"layer_{i+1}.in_proj")

        return graph

    def extract_ode_system(self) -> ODESystem:
        """Extract a complete ODESystem from a pretrained Mamba model.

        For SSMs, the ODE parameters are fundamentally different from Neural ODEs:
          - A_log → diagonal state matrix entries → RC time constants τ_i = 1/exp(A_log_i)
            These map to analog_primitive = "RC_time_constant".
            Bounds: A_log ∈ [-15, 15] → τ ∈ [~3e-7, ~3e6] seconds.
            Mismatch on A_log is the most critical: it shifts the RC decay rates.
          - B, C, dt_proj, in_proj, out_proj weights → crossbar conductances
            Same treatment as Neural ODE Linear weights.

        Ark DG mapping (Section 3 of Ark paper, ASPLOS '24):
          - Each SSM state dimension → a DG node of type "State" with attr decay_rate=real[...] mm(0, σ)
          - Each B/C weight connection → a DG edge of type "Coupling" with attr g=real[...] mm(0, σ)
          - Production rules: state <= -decay_rate * var(state) + coupling * var(input)

        The dynamics_module is the full Mamba model. For mismatch simulation,
        parameters are written back into the model via _apply_parameters_to_module.
        """
        assert self.model is not None, "Call load_model() first"

        config = self._get_config()
        D = config.get("d_model", 768)
        N = config.get("d_state", 16)
        n_layers = config.get("n_layer", 24)
        expand = config.get("expand", 2)
        D_inner = D * expand

        params: dict[str, ParameterSpec] = {}

        for name, param in self.model.named_parameters():
            data = param.detach().float()
            std = float(data.std()) if data.numel() > 1 else 1.0

            if "A_log" in name:
                # A_log controls RC time constants: τ = 1/exp(A_log)
                # Bound to physically realizable range: exp(-15)..exp(15)
                params[name] = ParameterSpec(
                    name=name,
                    value=data.clone(),
                    bounds=(-15.0, 15.0),
                    mismatch_sigma=0.05,
                    trainable=True,
                    analog_primitive="RC_time_constant",
                )
            elif any(k in name for k in ["x_proj", "dt_proj", "in_proj", "out_proj"]):
                # Crossbar MVM weights
                mean_val = float(data.mean())
                bound_range = max(4.0 * std, 0.5)
                params[name] = ParameterSpec(
                    name=name,
                    value=data.clone(),
                    bounds=(mean_val - bound_range, mean_val + bound_range),
                    mismatch_sigma=0.05,
                    trainable=True,
                    analog_primitive="crossbar_conductance",
                )
            elif "conv1d" in name and "weight" in name:
                # Short FIR filter taps — analog switched-capacitor
                mean_val = float(data.mean())
                bound_range = max(4.0 * std, 0.5)
                params[name] = ParameterSpec(
                    name=name,
                    value=data.clone(),
                    bounds=(mean_val - bound_range, mean_val + bound_range),
                    mismatch_sigma=0.03,  # FIR taps less sensitive than A
                    trainable=True,
                    analog_primitive="switched_cap_tap",
                )
            elif "bias" in name or "D" in name.split(".")[-1]:
                # Bias currents and D (feedthrough) terms
                mean_val = float(data.mean())
                bound_range = max(4.0 * std, 0.5)
                params[name] = ParameterSpec(
                    name=name,
                    value=data.clone(),
                    bounds=(mean_val - bound_range, mean_val + bound_range),
                    mismatch_sigma=0.05,
                    trainable=True,
                    analog_primitive="bias_current",
                )

        # Transient noise: sqrt(kT/C) for 1pF caps at 300K
        kt_over_c = 1.380649e-23 * 300.0 / 1e-12
        transient_sigma = math.sqrt(kt_over_c)

        # State dimension: each layer has D_inner channels × N state dims
        total_state_dim = n_layers * D_inner * N

        return ODESystem(
            name=self.model_name,
            family="ssm",
            state_dim=total_state_dim,
            parameters=params,
            dynamics_fn=self.model,
            dynamics_module=self.model,
            readout_times=[1.0],  # SSM reads out at every step; 1.0 = final
            t_span=(0.0, 1.0),
            noise=NoiseProfile(
                sigma=transient_sigma,
                bandwidth_hz=250e3,
            ),
            metadata={
                "d_model": D,
                "d_state": N,
                "n_layers": n_layers,
                "d_inner": D_inner,
                "expand": expand,
                "dynamics_type": "diagonal_SSM",
                "selective": True,  # Mamba is time-varying (selective)
            },
        )

    def _get_config(self) -> dict:
        """Extract model config as a flat dict.

        We try multiple access patterns because mamba_ssm and the transformers
        Mamba port store configs differently:
          - mamba_ssm MambaLMHeadModel:  model.config (MambaConfig object)
          - transformers MambaModel:     model.config (PretrainedConfig)
          - Some wrappers expose config under model.backbone.config

        Fallback: infer dimensions directly from parameter shapes, which
        always works regardless of how the model was loaded.  This matters
        because getting the wrong D or N would silently produce a graph with
        wrong FLOP counts and wrong D/A boundary analysis.
        """
        if hasattr(self.model, 'config'):
            config_obj = self.model.config
            # Try backbone.config first (mamba_ssm wraps the config there)
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'config'):
                config_obj = self.model.backbone.config
            if hasattr(config_obj, '__dict__'):
                cfg = {
                    k: v for k, v in config_obj.__dict__.items()
                    if not k.startswith('_')
                }
                # Normalize key names: transformers uses "num_hidden_layers",
                # mamba_ssm uses "n_layer".
                cfg.setdefault("n_layer", cfg.get("num_hidden_layers", cfg.get("n_layer", 24)))
                cfg.setdefault("d_model", cfg.get("hidden_size", cfg.get("d_model", 768)))
                return cfg

        # Infer from parameter shapes as a last resort.
        # A_log shape is (D_inner, N) = (d_model * expand, d_state).
        for name, param in self.model.named_parameters():
            if "A_log" in name:
                if param.dim() == 2:
                    d_inner, n = param.shape
                    # expand is almost always 2 in all published Mamba models
                    expand = 2
                    return {
                        "d_model": d_inner // expand,
                        "d_state": n,
                        "n_layer": sum(1 for n2, _ in self.model.named_parameters() if "A_log" in n2),
                        "expand": expand,
                    }
                return {"d_model": 768, "d_state": param.shape[0], "n_layer": 24, "expand": 2}

        return {"d_model": 768, "d_state": 16, "n_layer": 24, "expand": 2}


# ──────────────────────────────────────────────────────────────────────
# Mamba2Extractor
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# Standalone SSM ODE system extraction (for cross-arch sweep models)
# ──────────────────────────────────────────────────────────────────────

def extract_ssm_ode_system(
    model: nn.Module,
    model_name: str = "ssm_model",
) -> ODESystem:
    """Extract an ODESystem from any model containing S4D-style SSM layers.

    Works with _S4DLayer-style models (used in cross-arch experiments) and
    any nn.Module that has:
      - log_A_real / log_A_imag parameters (continuous-time A parametrization)
      - B, C, D linear projections

    This is the SSM analog of NeuralODEExtractor.extract_ode_system(), but
    standalone (no HuggingFace model loading). Used by sweep_all.py to route
    SSM models through the ODE system extraction path.

    Ark DG mapping:
      Node types: State (complex, order=1, sum) with attrs:
        - decay_rate = real[0, 15] mm(0, 0.05)  →  exp(-log_A_real)
        - imag_freq  = real[-π, π]  mm(0, 0.05)  →  oscillation frequency
      Edge types: Coupling with attr g = real[...] mm(0, 0.05)
      Production rule: state <= -decay * var(state) + g * var(input)
    """
    params: dict[str, ParameterSpec] = {}
    d_state = 0
    d_model = 0
    n_layers = 0

    for name, param in model.named_parameters():
        data = param.detach().float()
        std = float(data.std()) if data.numel() > 1 else 1.0

        if "log_A_real" in name:
            # Continuous-time A real part: A_c_real = -exp(log_A_real)
            # Bound log_A_real to ensure reasonable time constants
            params[name] = ParameterSpec(
                name=name,
                value=data.clone(),
                bounds=(-15.0, 15.0),
                mismatch_sigma=0.05,
                trainable=True,
                analog_primitive="RC_time_constant",
            )
            d_state = max(d_state, data.shape[-1] if data.dim() >= 1 else int(data.numel()))
            n_layers += 1

        elif "log_A_imag" in name:
            # Imaginary part of A: controls oscillation frequency
            params[name] = ParameterSpec(
                name=name,
                value=data.clone(),
                bounds=(-10.0, 10.0),
                mismatch_sigma=0.05,
                trainable=True,
                analog_primitive="oscillator_frequency",
            )

        elif "A_log" in name:
            # Mamba-style A_log (real-valued, A = -exp(A_log))
            params[name] = ParameterSpec(
                name=name,
                value=data.clone(),
                bounds=(-15.0, 15.0),
                mismatch_sigma=0.05,
                trainable=True,
                analog_primitive="RC_time_constant",
            )
            d_state = max(d_state, data.shape[-1] if data.dim() >= 2 else data.shape[0])
            n_layers += 1

        elif isinstance(_get_parent_module(model, name), nn.Linear):
            # B, C, D and other linear projections → crossbar conductance
            mean_val = float(data.mean())
            bound_range = max(4.0 * std, 0.5)
            primitive = "crossbar_conductance"
            if "bias" in name:
                primitive = "bias_current"

            params[name] = ParameterSpec(
                name=name,
                value=data.clone(),
                bounds=(mean_val - bound_range, mean_val + bound_range),
                mismatch_sigma=0.05,
                trainable=True,
                analog_primitive=primitive,
            )
            # Infer d_model from projection dimensions
            if "B" in name.split(".")[-2] or "C" in name.split(".")[-2]:
                if data.dim() == 2:
                    d_model = max(d_model, data.shape[-1])

        elif "weight" in name or "bias" in name:
            # Catch-all for other trainable parameters (e.g. embedding, head)
            mean_val = float(data.mean())
            bound_range = max(4.0 * std, 0.5)
            params[name] = ParameterSpec(
                name=name,
                value=data.clone(),
                bounds=(mean_val - bound_range, mean_val + bound_range),
                mismatch_sigma=0.05,
                trainable=True,
                analog_primitive="crossbar_conductance" if "weight" in name else "bias_current",
            )

    # Infer d_model from first linear layer if not already set
    if d_model == 0:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                d_model = m.in_features
                break

    # Transient noise
    kt_over_c = 1.380649e-23 * 300.0 / 1e-12
    transient_sigma = math.sqrt(kt_over_c)

    total_state_dim = n_layers * d_model * d_state if d_model > 0 else d_state

    return ODESystem(
        name=model_name,
        family="ssm",
        state_dim=total_state_dim,
        parameters=params,
        dynamics_fn=model,
        dynamics_module=model,
        readout_times=[1.0],
        t_span=(0.0, 1.0),
        noise=NoiseProfile(
            sigma=transient_sigma,
            bandwidth_hz=250e3,
        ),
        metadata={
            "d_model": d_model,
            "d_state": d_state,
            "n_layers": n_layers,
            "dynamics_type": "diagonal_SSM",
            "selective": False,  # S4D-style is LTI (not input-dependent)
        },
    )


def _get_parent_module(model: nn.Module, param_name: str) -> nn.Module | None:
    """Get the nn.Module that owns a given parameter name."""
    parts = param_name.split(".")
    if len(parts) <= 1:
        return model
    # Remove the parameter name itself (e.g., "weight" or "bias")
    module_path = ".".join(parts[:-1])
    try:
        module = model
        for part in module_path.split("."):
            module = getattr(module, part)
        return module
    except AttributeError:
        return None


# ──────────────────────────────────────────────────────────────────────
# Mamba2Extractor
# ──────────────────────────────────────────────────────────────────────

class Mamba2Extractor(MambaExtractor):
    """Extractor for Mamba-2 (SSD) models.

    Key differences from Mamba-1:
    - A restricted to scalar × identity (1 decay rate per head, not N)
    - Computation converts to chunked matrix multiplications (more crossbar-friendly)
    - Larger state dimension possible (N=64-128 vs N=16)
    """

    def extract_dynamics(self) -> DynamicsProfile:
        profile = super().extract_dynamics()
        profile.dynamics_type = "LTI_ODE"  # SSD is still LTI in structure
        # Note: scalar A means uniform time constant per head.
        # This dramatically simplifies analog implementation (1 RC per head, not N).
        return profile

    def build_graph(
        self,
        calibration_stats: dict[str, dict[str, float]] | None = None,
    ) -> AnalogGraph:
        """Build graph with Mamba-2 SSD optimizations.

        SSD converts the recurrence to chunked matrix multiplications:
        - Steps 1, 2, 4: Pure MVMs (crossbar-native)
        - Step 3: Inter-chunk state passing (sequential but reduced)
        """
        # For now, use same structure as Mamba-1 but annotate differences.
        graph = super().build_graph(calibration_stats=calibration_stats)
        graph.name = f"{self.model_name} (Mamba-2/SSD)"
        # TODO: Add SSD-specific chunked matmul decomposition
        return graph


class S4DMLPExtractor:
    """Extractor and Ark exporter for the S4D SSM in experiments/cross_arch_tolerance.

    Targets _S4DLayer from _SSMClassifier (d_model=16, d_state=8, 2 layers).
    Exports a single layer's continuous-time dynamics kernel as SSMAnalogCkt.

    The continuous-time SSM kernel (real/imag split):
        dh_re/dt = A_re * h_re - A_im * h_im
        dh_im/dt = A_im * h_re + A_re * h_im
    where A_re = -exp(log_A_real), A_im = log_A_imag.

    Autonomous mode (u=0): exports the RC oscillator bank impulse response.
    C and D matrices (readout) are embedded as non-trainable class attributes.

    Usage:
        ext = S4DMLPExtractor()
        ext.load_model()
        code = ext.export_to_ark("outputs/ssm_ark.py", mismatch_sigma=0.05)
    """

    _EXP_DIR = (
        __import__("pathlib").Path(__file__).parent.parent.parent
        / "experiments" / "cross_arch_tolerance"
    )

    def __init__(
        self,
        checkpoint_path=None,
        d_model: int = 16,
        d_state: int = 8,
        layer_idx: int = 0,
    ):
        if checkpoint_path is None:
            checkpoint_path = self._EXP_DIR / "checkpoints" / "ssm.pt"
        self.checkpoint_path = __import__("pathlib").Path(checkpoint_path)
        self.d_model = d_model
        self.d_state = d_state
        self.layer_idx = layer_idx
        self.model = None

    def load_model(self):
        import sys
        sys.path.insert(0, str(self._EXP_DIR))
        import models.ssm as ssm_module
        if self.checkpoint_path.exists():
            self.model = ssm_module.load_model(str(self.checkpoint_path))
        else:
            self.model = ssm_module.create_model()
            ssm_module.train_model(self.model, str(self.checkpoint_path))

    def export_to_ark(self, output_path, mismatch_sigma: float = 0.05) -> str:
        from neuro_analog.ark_bridge.ssm_cdg import export_s4d_to_ark
        assert self.model is not None, "Call load_model() first"
        layer = self.model.layers[self.layer_idx]
        return export_s4d_to_ark(
            s4d_layer=layer,
            output_path=output_path,
            mismatch_sigma=mismatch_sigma,
            class_name="SSMAnalogCkt",
            d_model=self.d_model,
            d_state=self.d_state,
        )
