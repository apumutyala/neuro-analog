"""neuro_analog.simulator — Physics-grounded analog forward-pass simulator.

Architecture-agnostic PyTorch module replacement layer for simulating
inference through degraded analog hardware (RRAM crossbars, RC integrators,
ADC/DAC converters).

Quick start:
    from neuro_analog.simulator import analogize, mismatch_sweep

    digital_model = MyModel()
    analog_model = analogize(digital_model, sigma_mismatch=0.05, n_adc_bits=8)
    result = mismatch_sweep(digital_model, evaluate_fn, n_trials=50)
"""

from .analog_linear import AnalogLinear
from .analog_activation import AnalogTanh, AnalogSigmoid, AnalogReLU, AnalogGELU, AnalogSiLU
from .analog_ode_solver import analog_odeint, analog_odeint_with_logdet
from .analog_ssm_solver import apply_ssm_mismatch, analog_ssm_recurrence
from .analog_model import (
    analogize,
    resample_all_mismatch,
    set_all_noise,
    calibrate_analog_model,
    count_analog_vs_digital,
)
from .sweep import SweepResult, mismatch_sweep, adc_sweep, ablation_sweep

__all__ = [
    "AnalogLinear",
    "AnalogTanh", "AnalogSigmoid", "AnalogReLU", "AnalogGELU", "AnalogSiLU",
    "analog_odeint", "analog_odeint_with_logdet",
    "apply_ssm_mismatch", "analog_ssm_recurrence",
    "analogize", "resample_all_mismatch", "set_all_noise",
    "calibrate_analog_model", "count_analog_vs_digital",
    "SweepResult", "mismatch_sweep", "adc_sweep", "ablation_sweep",
]
