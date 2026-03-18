# Simulator

`neuro_analog.simulator` — physics-grounded analog forward-pass simulation via PyTorch module replacement.

---

## How it works

`analogize(model, sigma_mismatch, n_adc_bits)` walks a PyTorch model and swaps digital modules for analog equivalents:

| Digital | Analog replacement | Physics |
|---|---|---|
| `nn.Linear` | `AnalogLinear` | Conductance mismatch + thermal noise + ADC quantization |
| `nn.Tanh` | `AnalogTanh` | MOSFET diff pair gain/offset mismatch, swing saturation |
| `nn.Sigmoid` | `AnalogSigmoid` | Same as Tanh, single-ended output |
| `nn.ReLU` | `AnalogReLU` | Diode-connected transistor offset |
| `nn.GELU/SiLU` | `AnalogGELU/SiLU` | Digital implementation + ADC→DAC domain crossing penalty |

Everything else (LayerNorm, Softmax, Embedding) stays digital.

---

## AnalogLinear

Three noise sources applied in order during each forward pass:

**1. Conductance mismatch (static, sampled once at construction)**

```
W_device = W_nominal ⊙ δ,   δ ~ N(1, σ²·I)
```

Same δ persists across all inferences — it represents baked-in fabrication variance of RRAM/PCM conductance cells. Call `resample_mismatch()` between Monte Carlo trials.

**2. Thermal read noise (dynamic, fresh each forward pass)**

```
y = W_device @ x + ε,   ε ~ N(0, σ_thermal² · I)
σ_thermal = sqrt(kT/C) · sqrt(N_in)
```

At C = 1 pF, T = 300 K: σ_thermal ≈ 6.4×10⁻⁵ · √N_in volts. The √N_in factor follows independent Johnson-Nyquist contributions from N_in column currents summing at the sense node. At demo-scale model sizes this is negligible (~5×10⁻⁴ V for N_in=64).

**3. ADC quantization (deterministic)**

```
scale = (2^n_bits - 1) / (2 · V_ref)
y_q = clamp(y, -V_ref, V_ref)
y_q = round(y_q · scale) / scale
```

V_ref is set per-layer from calibration (max absolute activation). Hard quantization models inference behavior; Shem uses Gumbel-Softmax for differentiable training through discrete parameters.

---

## Analog activations

| Class | Analog primitive | Notes |
|---|---|---|
| `AnalogTanh` | MOSFET differential pair | Gain mismatch + output swing saturation |
| `AnalogSigmoid` | Single-ended diff pair | Same physics as Tanh |
| `AnalogReLU` | Diode-connected transistor | Threshold offset shifts the zero crossing |
| `AnalogGELU` / `AnalogSiLU` | Digital + domain crossing penalty | No analog GELU circuit; models the DAC→compute→ADC roundtrip cost |

---

## ODE solver

`analog_odeint` and `analog_odeint_with_logdet` wrap torchdiffeq's Euler/Dopri5 solver with analog-perturbed dynamics. Used by Neural ODE and Flow models.

The log-det computation (`analog_odeint_with_logdet`) runs on nominal weights during backprop — there's no way to hardware-backpropagate through a physical crossbar. Mismatch is applied to the forward dynamics only.

---

## SSM solver

`apply_ssm_mismatch` and `analog_ssm_recurrence` handle state-space model recurrences. The A/B/C matrices of an S4D-style SSM each get independent δ mismatch. The diagonal recurrence h[t] = A_bar·h[t-1] + B_bar·u[t] means state dimensions degrade independently without cross-contamination — one reason SSMs are robust.

---

## Simulation profiles

Two profiles bound the plausible range of quantization behavior:

**Conservative** — ADC at every AnalogLinear output. Models current digital-analog hybrid chips (per-crossbar ADC, digital routing between arrays). Quantization compounds across depth × iterations.

**Full-analog** — ADC only at the final readout layer. Models a true analog substrate (Shem/Ark target) where intermediate activations pass as continuous voltages. Mismatch and thermal noise still apply everywhere.

For iterative architectures (Neural ODE: ~40 solver steps, DEQ: ~30 fixed-point iterations, Diffusion: 100 DDPM steps), the profiles can produce very different results. Configure with `configure_analog_profile(model, profile='conservative' | 'full_analog')`.

---

## Sweep API

```python
from neuro_analog.simulator import mismatch_sweep, adc_sweep, ablation_sweep, SweepResult

# σ sweep (main sweep)
result = mismatch_sweep(
    model, eval_fn,
    sigma_values=[0.0, 0.05, 0.10, 0.15],
    n_trials=50,
    n_adc_bits=8,
)
print(result.degradation_threshold(max_relative_loss=0.10))  # σ at 90% quality

# ADC bit-width sweep at fixed σ
result = adc_sweep(model, eval_fn, bits=[2, 4, 6, 8, 12, 16], sigma_mismatch=0.05)

# Noise attribution (isolates each source)
ablation = ablation_sweep(model, eval_fn, sigma_values=[0.0, 0.05, 0.10], n_trials=10)
# ablation keys: 'mismatch', 'thermal', 'quantization'
```

`SweepResult` holds:
- `.sigma_values` / `.mean` / `.std` — raw measurements
- `.normalized_mean` — relative to σ=0 baseline (1.0 = no degradation)
- `.degradation_threshold(max_relative_loss)` — interpolated σ at quality threshold

---

## Other utilities

```python
from neuro_analog.simulator import (
    count_analog_vs_digital,     # layer count + param coverage breakdown
    resample_all_mismatch,       # re-draw all δ tensors (between MC trials)
    set_all_noise,               # set sigma/bits on every AnalogLinear
    calibrate_analog_model,      # set per-layer V_ref from a calibration batch
)
```

---

## Design decisions

**Why multiplicative mismatch?**
Conductance cells have a fractional variance — a 10% mismatch means each cell deviates by ±10% of its nominal value, regardless of whether that value is large or small. Additive models misrepresent this for cells near zero.

**Why static δ?**
Fabrication mismatch is permanent. The same δ is re-used across all inferences to model a deployed chip. Monte Carlo trials resample δ to estimate the distribution over chips off a fabrication line.

**Why hard quantization?**
We simulate inference, not training. Shem uses Gumbel-Softmax for differentiable optimization through discrete ADC levels; our simulator uses deterministic rounding since we're measuring, not optimizing.

**Why not simulate Softmax/LayerNorm in analog?**
No practical analog circuit implements them efficiently. Softmax requires a normalization that touches all elements simultaneously; LayerNorm requires mean/variance computation. Both are computed digitally in current academic analog AI designs.
