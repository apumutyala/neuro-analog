"""neuro-analog: Cross-architecture neural-to-analog compilation framework.

Convert any trained PyTorch model to physics-grounded analog hardware simulation,
measure degradation under realistic nonidealities, and export to Ark-compatible
BaseAnalogCkt specifications.

Quick start::

    from neuro_analog import analogize, mismatch_sweep

    analog = analogize(model, sigma_mismatch=0.05, n_adc_bits=8)
    result = mismatch_sweep(model, eval_fn, sigma_values=[0.0, 0.05, 0.10])

Analog nonidealities modeled:
  - Conductance mismatch:  W_eff = W * delta, delta ~ N(1, sigma^2) per cell
  - Thermal read noise:    epsilon ~ N(0, kT/C * sqrt(N_in)) per output
  - ADC quantization:      hard uniform quantization at n_adc_bits resolution

Hardware parameters default to HCDCv2 / typical RRAM:
  T = 300 K, C = 1 pF, V_ref = 1.0 V
"""

from .simulator import (
    # Core API — what most users need
    analogize,
    mismatch_sweep,
    adc_sweep,
    ablation_sweep,
    calibrate_analog_model,
    count_analog_vs_digital,
    configure_analog_profile,
    resample_all_mismatch,
    set_all_noise,
    SweepResult,
    # Analog layer classes (for isinstance checks / advanced use)
    AnalogLinear,
    AnalogTanh,
    AnalogSigmoid,
    AnalogReLU,
    AnalogGELU,
    AnalogSiLU,
)

__version__ = "0.1.0"

__all__ = [
    "analogize",
    "mismatch_sweep",
    "adc_sweep",
    "ablation_sweep",
    "calibrate_analog_model",
    "count_analog_vs_digital",
    "configure_analog_profile",
    "resample_all_mismatch",
    "set_all_noise",
    "SweepResult",
    "AnalogLinear",
    "AnalogTanh",
    "AnalogSigmoid",
    "AnalogReLU",
    "AnalogGELU",
    "AnalogSiLU",
]
