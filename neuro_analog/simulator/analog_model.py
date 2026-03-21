"""
Architecture-agnostic analog model wrapper.

analogize() recursively replaces PyTorch modules with their analog equivalents:
  nn.Linear  → AnalogLinear   (crossbar MVM with mismatch + thermal + ADC)
  nn.Tanh    → AnalogTanh     (diff pair with gain/offset mismatch)
  nn.Sigmoid → AnalogSigmoid
  nn.ReLU    → AnalogReLU
  nn.GELU    → AnalogGELU     (digital + ADC→DAC crossing penalty)
  nn.SiLU    → AnalogSiLU

Everything else (LayerNorm, Softmax, Dropout, BatchNorm, Embedding, etc.)
stays digital — there is no efficient analog implementation.

analogize() returns a new model and does not modify the original.
copy.deepcopy() ensures the two are completely independent — changes to mismatch
in the analog copy do not affect the digital original.

The v_ref calibration problem. When replacing nn.Linear with AnalogLinear,
we don't know what activation range the layer will see until we run a forward pass.
We initialize v_ref=1.0 (matches HCDCv2 hardware spec: ±1V voltage range). For more
accurate simulation, call calibrate_analog_model(model, sample_input) after analogize().

nn.Linear can appear in attention mechanisms as Q/K/V/O projections.
These should be analogized (they ARE crossbar operations). Our recursive traversal
will catch all of them since we replace at the nn.Linear level regardless of context.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from .analog_linear import AnalogLinear
from .analog_activation import (
    AnalogTanh, AnalogSigmoid, AnalogReLU, AnalogGELU, AnalogSiLU,
    AnalogELU, AnalogLeakyReLU, AnalogHardswish, AnalogMish,
)
from .analog_conv import analog_conv_from_module
from .analog_attention import AnalogMultiheadAttention

# Mapping from PyTorch module class to analog replacement factory
# Activations only — Linear, Conv, and MHA are handled with special cases below.
_ACT_REPLACEMENTS = {
    nn.Tanh:       lambda m, **kw: AnalogTanh(sigma_mismatch=kw["sigma_mismatch"]),
    nn.Sigmoid:    lambda m, **kw: AnalogSigmoid(sigma_mismatch=kw["sigma_mismatch"]),
    nn.ReLU:       lambda m, **kw: AnalogReLU(rail=kw["v_ref"], sigma_mismatch=kw["sigma_mismatch"]),
    nn.ELU:        lambda m, **kw: AnalogELU(alpha=m.alpha, sigma_mismatch=kw["sigma_mismatch"]),
    nn.LeakyReLU:  lambda m, **kw: AnalogLeakyReLU(negative_slope=m.negative_slope, sigma_mismatch=kw["sigma_mismatch"]),
    nn.GELU:       lambda m, **kw: AnalogGELU(
                       sigma_mismatch=kw["sigma_mismatch"], n_bits=kw["n_adc_bits"],
                       v_ref=kw["v_ref"], temperature_K=kw["temperature_K"], cap_F=kw["cap_F"]),
    nn.SiLU:       lambda m, **kw: AnalogSiLU(
                       sigma_mismatch=kw["sigma_mismatch"], n_bits=kw["n_adc_bits"],
                       v_ref=kw["v_ref"], temperature_K=kw["temperature_K"], cap_F=kw["cap_F"]),
    nn.Hardswish:  lambda m, **kw: AnalogHardswish(
                       sigma_mismatch=kw["sigma_mismatch"], n_bits=kw["n_adc_bits"],
                       v_ref=kw["v_ref"], temperature_K=kw["temperature_K"], cap_F=kw["cap_F"]),
    nn.Mish:       lambda m, **kw: AnalogMish(
                       sigma_mismatch=kw["sigma_mismatch"], n_bits=kw["n_adc_bits"],
                       v_ref=kw["v_ref"], temperature_K=kw["temperature_K"], cap_F=kw["cap_F"]),
}

# Conv module types that map to AnalogConvNd via analog_conv_from_module()
_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def analogize(
    model: nn.Module,
    sigma_mismatch: float = 0.05,
    n_adc_bits: int = 8,
    temperature_K: float = 300.0,
    cap_F: float = 1e-12,
    v_ref: float = 1.0,
    v_ref_input: float | None = None,
) -> nn.Module:
    """Convert any PyTorch model to analog-simulated execution.

    Recursively replaces digital modules with analog equivalents.
    Pretrained weights are copied into analog buffers.
    Returns a new model — original is not modified.

    Args:
        model: Any nn.Module (trained or untrained).
        sigma_mismatch: Conductance mismatch std dev (0 = perfect hardware).
        n_adc_bits: ADC/DAC bit width (32 = effectively noiseless quantization).
        temperature_K: Operating temperature for thermal noise.
        cap_F: Integration capacitance for thermal noise floor.
        v_ref: ADC reference voltage (V). Use calibrate_analog_model() to set
               this from actual activation ranges.

    Returns:
        New nn.Module with analog-replaced layers.
    """
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    analog_model = copy.deepcopy(model)
    _replace_recursive(
        analog_model,
        sigma_mismatch=sigma_mismatch,
        n_adc_bits=n_adc_bits,
        temperature_K=temperature_K,
        cap_F=cap_F,
        v_ref=v_ref,
    )
    analog_model = analog_model.to(device)

    # Override input DAC range if the model's first-layer input distribution differs
    # from the output distribution (e.g. z0 ~ N(0,I) for generative models vs v_ref=1.0).
    if v_ref_input is not None:
        from .analog_conv import AnalogConv1d, AnalogConv2d, AnalogConv3d
        crossbar_types = (AnalogLinear, AnalogConv1d, AnalogConv2d, AnalogConv3d)
        for m in analog_model.modules():
            if isinstance(m, crossbar_types):
                m.v_ref_input = v_ref_input

    return analog_model


def _replace_recursive(module: nn.Module, **kwargs) -> None:
    """In-place recursive module replacement.

    Handles, in priority order:
      1. nn.MultiheadAttention  — special case: fused weights need splitting
      2. nn.Conv{1,2,3}d        — special case: conv weight tensor + receptive-field N_in
      3. nn.Linear              — standard crossbar MVM replacement
      4. Activation functions   — from _ACT_REPLACEMENTS registry
      5. Everything else        — recurse into children
    """
    for name, child in list(module.named_children()):

        # ── 1. MultiheadAttention — must be handled before recursing, because
        #        its out_proj is an nn.Linear that would otherwise get caught
        #        by case 3 without the Q/K/V weights being handled. ──────────
        if isinstance(child, nn.MultiheadAttention):
            replacement = AnalogMultiheadAttention.from_module(child, **{
                k: kwargs[k] for k in
                ("sigma_mismatch", "n_adc_bits", "temperature_K", "cap_F", "v_ref")
            })
            setattr(module, name, replacement)

        # ── 2. Convolutional layers ──────────────────────────────────────────
        elif isinstance(child, _CONV_TYPES):
            replacement = analog_conv_from_module(child, **{
                k: kwargs[k] for k in
                ("sigma_mismatch", "n_adc_bits", "temperature_K", "cap_F", "v_ref")
            })
            setattr(module, name, replacement)

        # ── 3. Linear ────────────────────────────────────────────────────────
        elif isinstance(child, nn.Linear):
            replacement = AnalogLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                weight=child.weight.data,
                bias=child.bias.data if child.bias is not None else None,
                sigma_mismatch=kwargs["sigma_mismatch"],
                n_adc_bits=kwargs["n_adc_bits"],
                temperature_K=kwargs["temperature_K"],
                cap_F=kwargs["cap_F"],
                v_ref=kwargs["v_ref"],
            )
            setattr(module, name, replacement)

        # ── 4. Activation functions ──────────────────────────────────────────
        elif type(child) in _ACT_REPLACEMENTS:
            replacement = _ACT_REPLACEMENTS[type(child)](child, **kwargs)
            setattr(module, name, replacement)

        # ── 5. Recurse ───────────────────────────────────────────────────────
        else:
            _replace_recursive(child, **kwargs)


# ── Monte Carlo utilities ────────────────────────────────────────────────

def resample_all_mismatch(model: nn.Module, sigma: float | None = None) -> None:
    """Re-roll static mismatch δ for all analog layers.

    Call between Monte Carlo trials to get independent mismatch realizations.
    If sigma is provided, updates σ_mismatch for all layers simultaneously.

    After resampling all layers, calls sync_mismatch_pairs() on any module
    that defines it — used by architectures with physically tied crossbars
    (e.g. RBM W_fwd / W_bwd, which are the same RRAM array read bidirectionally).
    """
    for module in model.modules():
        if hasattr(module, "resample_mismatch"):
            module.resample_mismatch(sigma)
    # Allow models with tied crossbars to enforce consistent δ after resampling
    for module in model.modules():
        if hasattr(module, "sync_mismatch_pairs"):
            module.sync_mismatch_pairs()


def set_all_noise(
    model: nn.Module,
    thermal: bool = True,
    quantization: bool = True,
    mismatch: bool = True,
) -> None:
    """Global noise source toggle. Useful for ablation experiments.

    Set specific combinations to isolate which nonideality dominates:
      Mismatch only:     mismatch=True,  thermal=False, quantization=False
      Thermal only:      mismatch=False, thermal=True,  quantization=False
      Quantization only: mismatch=False, thermal=False, quantization=True
    """
    for module in model.modules():
        if hasattr(module, "set_noise_config"):
            module.set_noise_config(
                thermal=thermal,
                quantization=quantization,
                mismatch=mismatch,
            )


def calibrate_analog_model(model: nn.Module, sample_input: torch.Tensor) -> None:
    """Set v_ref and v_ref_input for all crossbar layers from a representative forward pass.

    Runs a noiseless forward pass (mismatch+thermal+quantization off) through the model,
    capturing activation ranges at each AnalogLinear/AnalogConv layer via hooks.
      v_ref       = 1.1 * max(|output|)  — output ADC range
      v_ref_input = 1.1 * max(|input|)   — input DAC range

    Both are set per layer, since input and output distributions can differ
    significantly (e.g. first layer sees raw data, intermediate layers see
    bounded activations).
    """
    # Temporarily disable noise for calibration pass
    set_all_noise(model, thermal=False, quantization=False, mismatch=False)

    hooks = []

    from .analog_conv import AnalogConv1d, AnalogConv2d, AnalogConv3d
    calibrate_types = (AnalogLinear, AnalogConv1d, AnalogConv2d, AnalogConv3d)

    for name, module in model.named_modules():
        if isinstance(module, calibrate_types):
            def make_hook(n, m):
                def hook(mod, inp, out):
                    peak_in = float(inp[0].detach().abs().max().item())
                    m.v_ref_input = peak_in * 1.1 if peak_in > 0 else 1.0
                    peak_out = float(out.detach().abs().max().item())
                    m.v_ref = peak_out * 1.1 if peak_out > 0 else 1.0
                return hook
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()

    # Re-enable all noise
    set_all_noise(model, thermal=True, quantization=True, mismatch=True)


def configure_analog_profile(model: nn.Module, profile: str) -> None:
    """Set the ADC quantization profile for an analogized model.

    Two profiles representing physical upper and lower bounds on quantization
    sensitivity:

    'conservative' (default / upper bound)
        ADC is applied at the output of every AnalogLinear layer. Models a
        digital-analog hybrid chip where each crossbar unit is paired with a
        dedicated ADC and DAC. Every layer boundary is a domain crossing.
        Quantization errors compound across all layers and across all
        iterations for iterative architectures (Diffusion DDPM steps, DEQ
        fixed-point, Neural ODE integration steps). This is the worst-case
        assumption and sets the ceiling on observed quantization sensitivity.

    'full_analog' (lower bound)
        ADC is applied only at the final AnalogLinear layer of the model —
        the readout. All intermediate layers keep their signal in the
        continuous analog domain: mismatch and thermal noise still apply
        (they are physical and unavoidable), but no discretization occurs
        until the computation is complete. Models a true analog compute
        substrate of the kind targeted by Shem and ARK, where the signal
        lives in conductance/voltage space throughout the forward pass and
        is only read out digitally at the end.

        For iterative architectures the "last AnalogLinear" fires once per
        iteration (100× for Diffusion, 40× for Neural ODE, 30× for DEQ,
        100× for EBM Gibbs), so this is still more conservative than a
        fully-analog-memory chip where even iteration boundaries stay
        analog. It is nonetheless a substantial improvement over conservative
        for any multi-layer architecture.

    Args:
        model: An analogized nn.Module (output of analogize()).
        profile: 'conservative' or 'full_analog'.
    """
    from .analog_conv import AnalogConv1d, AnalogConv2d, AnalogConv3d
    readout_types = (AnalogLinear, AnalogConv1d, AnalogConv2d, AnalogConv3d)

    readout_layers = [(n, m) for n, m in model.named_modules()
                      if isinstance(m, readout_types)]

    if profile == "conservative":
        for _, m in readout_layers:
            m._is_readout = True

    elif profile == "full_analog":
        for _, m in readout_layers:
            m._is_readout = False
        # Mark only the last layer as the readout (final domain crossing)
        if readout_layers:
            readout_layers[-1][1]._is_readout = True

    else:
        raise ValueError(f"Unknown profile {profile!r}. Use 'conservative' or 'full_analog'.")


def count_analog_vs_digital(model: nn.Module) -> dict:
    """Count analog-replaced vs digital-remaining layers.

    Returns:
        dict with keys:
          analog_layers: number of analog-replaced modules
          digital_layers: number of remaining digital modules with parameters
          analog_params: total parameters in analog layers
          digital_params: total parameters in digital layers
          analog_layer_names: list of names of analog layers
          digital_layer_names: list of names of non-trivial digital layers
          coverage_pct: analog_params / (analog_params + digital_params) * 100
    """
    from .analog_conv import AnalogConv1d, AnalogConv2d, AnalogConv3d
    # AnalogMultiheadAttention is intentionally excluded: named_modules() traverses
    # into its q/k/v/out_proj children (all AnalogLinear), which are counted directly.
    # Including the MHA container would inflate analog_layers by 1 per attention block.
    analog_types = (
        AnalogLinear,
        AnalogConv1d, AnalogConv2d, AnalogConv3d,
        AnalogTanh, AnalogSigmoid, AnalogReLU,
        AnalogELU, AnalogLeakyReLU,
        AnalogGELU, AnalogSiLU, AnalogHardswish, AnalogMish,
    )
    analog_conv_types = (AnalogConv1d, AnalogConv2d, AnalogConv3d)
    digital_non_trivial = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
                           nn.BatchNorm3d, nn.GroupNorm, nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm,
                           nn.Softmax, nn.Embedding)

    analog_layers = 0
    digital_layers = 0
    analog_params = 0
    digital_params = 0
    analog_names = []
    digital_names = []

    for name, module in model.named_modules():
        if isinstance(module, analog_types):
            analog_layers += 1
            analog_names.append(name)
            if isinstance(module, AnalogLinear):
                analog_params += module.W_nominal.numel()
                if module.bias is not None:
                    analog_params += module.bias.numel()
            elif isinstance(module, analog_conv_types):
                analog_params += module.W_nominal.numel()
                if module.bias is not None:
                    analog_params += module.bias.numel()
        elif isinstance(module, digital_non_trivial):
            digital_layers += 1
            digital_names.append(name)
            for p in module.parameters():
                digital_params += p.numel()

    total = analog_params + digital_params
    return {
        "analog_layers": analog_layers,
        "digital_layers": digital_layers,
        "analog_params": analog_params,
        "digital_params": digital_params,
        "analog_layer_names": analog_names,
        "digital_layer_names": digital_names,
        "coverage_pct": 100.0 * analog_params / total if total > 0 else 0.0,
    }
