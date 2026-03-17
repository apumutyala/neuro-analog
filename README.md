# neuro-analog

A cross-architecture analog hardware tolerance simulator and IR extraction framework for neural networks. Measures how seven families of neural networks degrade under realistic fabrication nonidealities, extracts their dynamics into a typed intermediate representation, and generates Shem-compatible JAX/Diffrax code for hardware-aware optimization.

**Research context:** This work sits upstream of [Shem](https://arxiv.org/abs/2411.03557) (Achour & Wang, 2024) and [Ark](https://arxiv.org/abs/2309.08774). It answers the pre-compilation question — *which architectures are worth compiling to analog, and at what noise level does each one break?* — while producing the IR and exports needed to feed the Shem optimizer.

---

## Table of Contents

1. [What This Does](#what-this-does)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)
4. [Core Library: `neuro_analog/`](#core-library-neuro_analog)
   - [Simulator](#simulator)
   - [Intermediate Representation](#intermediate-representation-ir)
   - [Extractors](#extractors)
   - [Analysis & Taxonomy](#analysis--taxonomy)
   - [Mappers](#mappers)
   - [Visualization](#visualization)
   - [Pipeline](#pipeline)
5. [Experiments: `cross_arch_tolerance/`](#experiments-cross_arch_tolerance)
   - [The Seven Models](#the-seven-models)
   - [Running the Sweep](#running-the-sweep)
   - [Understanding the Results](#understanding-the-results)
6. [Shem Export Pipeline](#shem-export-pipeline)
7. [Tests](#tests)
8. [Key Design Decisions](#key-design-decisions)
9. [Findings Summary](#findings-summary)
10. [What's Complete vs. In Progress](#whats-complete-vs-in-progress)
11. [Connection to Shem / Ark / Unconventional AI](#connection-to-shem--ark--unconventional-ai)
12. [Next Steps for Compilation to Shem](#next-steps-for-compilation-to-shem)

---

## What This Does

Analog hardware — crossbar arrays (RRAM, PCM), RC integrators, differential-pair activations — runs neural network operations at radically lower energy than digital, but introduces three unavoidable nonidealities:

| Nonideality | Source | Effect |
|---|---|---|
| **Conductance mismatch** δ~N(1,σ²) | Fabrication variance in W/L ratios, oxide thickness | Static weight corruption, baked in at manufacture |
| **Thermal read noise** ε~N(0,kT/C·N_in) | Johnson-Nyquist noise at sense capacitor | Dynamic noise per inference, scales with array width |
| **ADC quantization** | Finite precision of analog-to-digital conversion | Deterministic rounding, degrades with fewer bits |

This codebase:

1. **Simulates** these nonidealities inside any PyTorch model by replacing `nn.Linear`, `nn.Tanh`, `nn.Sigmoid`, etc. with physics-grounded analog equivalents
2. **Measures** how 7 neural network families degrade as σ increases from 0% → 15%
3. **Extracts** each architecture into a typed IR classifying every operation as ANALOG_NATIVE, DIGITAL_REQUIRED, or HYBRID
4. **Exports** Neural ODE and SSM models to Shem-compatible JAX/Diffrax code for hardware-aware optimization

---

## Installation

```bash
git clone https://github.com/apumutyala/neuro-analog
cd neuro-analog
pip install -e ".[dev]"
```

With optional backends for Mamba extraction and Shem evaluation:
```bash
pip install -e ".[mamba,jax]"
```

**Core dependencies:** `torch>=2.1`, `numpy>=1.24`, `matplotlib>=3.8`, `seaborn>=0.13`, `rich>=13.0`, `torchdiffeq`, `torchvision`, `scikit-learn`, `scipy`

**Optional:** `mamba-ssm>=1.0` (Mamba extraction), `jax`, `jaxlib`, `diffrax` (Shem evaluation), `transformers>=4.36` (HuggingFace models), `plotly>=5.18` (interactive plots)

---

## Repository Structure

```
neuro-analog/
│
├── neuro_analog/                   # Core installable library
│   ├── simulator/                  # Physics-based analog forward-pass simulation
│   │   ├── analog_linear.py        # AnalogLinear: mismatch + thermal + ADC
│   │   ├── analog_activation.py    # AnalogTanh/Sigmoid/ReLU/GELU/SiLU/ELU/LeakyReLU/Hardswish/Mish
│   │   ├── analog_conv.py          # AnalogConv1d/2d/3d: convolutional crossbar simulation
│   │   ├── analog_attention.py     # AnalogMultiheadAttention: analog Q/K/V/O projections
│   │   ├── analog_model.py         # analogize(), resample_all_mismatch(), set_all_noise()
│   │   ├── analog_ode_solver.py    # analog_odeint(), analog_odeint_with_logdet()
│   │   ├── analog_ssm_solver.py    # analog_ssm_recurrence(), apply_ssm_mismatch()
│   │   └── sweep.py                # mismatch_sweep(), adc_sweep(), ablation_sweep(), SweepResult
│   │
│   ├── ir/                         # Intermediate representation
│   │   ├── types.py                # OpType (22 ops), Domain, ArchitectureFamily, all spec types
│   │   ├── node.py                 # AnalogNode, factory functions
│   │   ├── graph.py                # AnalogGraph, DABoundary
│   │   ├── ode_system.py           # ODESystem, ParameterSpec — bridge to Shem
│   │   └── shem_export.py          # JAX/Diffrax code generation
│   │
│   ├── extractors/                 # Architecture-specific IR extraction
│   │   ├── base.py                 # BaseExtractor abstract interface
│   │   ├── neural_ode.py           # NeuralODEExtractor (most complete)
│   │   ├── ssm.py                  # MambaExtractor (with CUDA kernel patch)
│   │   ├── transformer.py          # TransformerExtractor (HuggingFace)
│   │   ├── diffusion.py            # DiffusionExtractor
│   │   ├── flow.py                 # FlowExtractor
│   │   ├── ebm.py                  # EBMExtractor
│   │   └── deq.py                  # DEQExtractor
│   │
│   ├── analysis/
│   │   ├── taxonomy.py             # AnalogTaxonomy, TaxonomyEntry, cross-arch comparison
│   │   └── precision.py            # Dynamic range analysis, effective bit-width
│   │
│   ├── mappers/                    # Circuit-level component specs
│   │   ├── crossbar.py             # MVM → crossbar array (RRAM/PCM/SRAM) specs
│   │   ├── integrator.py           # ODE integration → RC time constant values
│   │   └── stochastic.py           # Stochastic sampling → p-bit / sMTJ specs
│   │
│   ├── nonidealities/              # Reusable physics models
│   │   ├── mismatch.py             # δ~N(1,σ²), MismatchReport
│   │   ├── noise.py                # Johnson-Nyquist, shot noise
│   │   ├── quantization.py         # ADC/DAC uniform quantization
│   │   └── signal_scaling.py       # Gain/offset calibration utilities
│   │
│   ├── visualization/
│   │   ├── comparison_radar.py     # 6-axis radar chart (all 7 architectures)
│   │   ├── noise_budget.py         # Noise source contribution bar charts
│   │   └── partition_map.py        # Layer-wise analog/digital partition heatmaps
│   │
│   └── pipeline.py                 # Unified run_pipeline() entry point
│
├── experiments/
│   └── cross_arch_tolerance/       # Main experiment — 7 architectures
│       ├── models/                 # One file per architecture (standard interface)
│       │   ├── neural_ode.py       # CNF on make_circles
│       │   ├── ssm.py              # S4D diagonal SSM on sequence classification
│       │   ├── transformer.py      # 2-layer attention on pattern detection
│       │   ├── diffusion.py        # DDPM on 8×8 MNIST
│       │   ├── flow.py             # Rectified flow on make_moons
│       │   ├── ebm.py              # RBM on 8×8 MNIST
│       │   └── deq.py              # Implicit MLP on 8×8 MNIST
│       ├── train_all.py            # Train and save all 7 models
│       ├── sweep_all.py            # Run all sweeps (--force --n-trials N)
│       ├── plot_results.py         # Generate 5 publication figures
│       ├── evaluate_in_shem.py     # JAX/Diffrax native evaluation of exports
│       ├── diagnose_analog.py      # Debug and calibration utilities
│       ├── checkpoints/            # Saved model weights (gitignored)
│       ├── results/                # JSON sweep outputs (~49 files per full run)
│       ├── figures/                # Generated PNG/PDF plots
│       └── TECHNICAL_NOTE.md       # Full methodology and errata log
│
├── outputs/
│   └── ssm_shem_test.py            # Example Shem export (SSM, real weights)
│
├── examples/
│   ├── 01_analyze_mamba.py         # Load Mamba → extract → score
│   ├── 04_cross_architecture.py    # Build full taxonomy across all 7
│   └── 05_shem_pipeline_demo.py    # End-to-end: train → sweep → export
│
├── tests/
│   ├── test_simulator.py           # Simulator invariant tests (10+ tests)
│   ├── test_extractors.py
│   ├── test_ir.py
│   └── test_nonidealities.py
│
├── validate_shem_export.py         # Neural ODE Shem export validation
├── validate_ssm_shem_export.py     # SSM Shem export validation
├── verify.py                       # Sanity check suite
├── pyproject.toml
└── docs/
    └── PROJECT_DESIGN.md           # Full system specification
```

---

## Core Library: `neuro_analog/`

### Simulator

The simulator is the most-used component. It instruments any PyTorch model with physics-grounded analog nonidealities, with zero changes to the model definition.

#### `analog_linear.py` — AnalogLinear

Replaces `nn.Linear`. Applies three noise sources in order on every forward pass:

```
1. Conductance mismatch (static — baked in at fabrication time):
   W_device = W_nominal ⊙ δ,   δ ~ N(1, σ²·I)
   Same δ persists across all inferences for the lifetime of one device.

2. Thermal read noise (dynamic — different every inference):
   y = W_device @ x + ε,   ε ~ N(0, σ_th² · I)
   σ_th = sqrt(kT/C) · sqrt(N_in)
   The sqrt(N_in) factor: N_in independent column currents sum at the sense
   capacitor (Johnson-Nyquist at C=1pF, T=300K).

3. ADC quantization (deterministic):
   y_q = round(clamp(y, -V_ref, V_ref) · scale) / scale
   scale = (2^n_bits - 1) / (2·V_ref)
```

Key methods:
```python
layer = AnalogLinear(in_features, out_features, sigma_mismatch=0.05, n_adc_bits=8)
layer.resample_mismatch(sigma=0.10)   # re-roll δ for new MC trial
layer.set_noise_config(thermal=False, quantization=True, mismatch=True)  # ablation
layer.calibrate(x)                   # set V_ref = 1.1×max(|activation|) from data
```

**Why static mismatch matters:** Real fabricated hardware has a fixed δ per device. Every inference uses the same corrupted weights. This means the error is systematic, not averaged out over multiple calls — and is the dominant noise source in 6 of 7 architectures.

#### `analog_activation.py` — Analog Activation Replacements

| Digital | Analog Replacement | Circuit | Nonidealities modeled |
|---|---|---|---|
| `nn.Tanh` | `AnalogTanh` | MOSFET differential pair | α~N(1,σ²) gain, β~N(0,(0.5σ)²) offset, clip to ±0.95 |
| `nn.Sigmoid` | `AnalogSigmoid` | Single-ended diff pair | Same as tanh, clip to [0.025, 0.975] |
| `nn.ReLU` | `AnalogReLU` | Diode-connected transistor | Offset threshold variation δ_th~N(0,(0.05σ)²) |
| `nn.ELU` | `AnalogELU` | Diode exponential region | Gain α mismatch + output swing clamp |
| `nn.LeakyReLU` | `AnalogLeakyReLU` | Asymmetric diode pair | Slope mismatch on negative region |
| `nn.GELU` | `AnalogGELU` | Digital + ADC→DAC crossing | quantize(GELU(x), n_bits) + ε_thermal |
| `nn.SiLU` | `AnalogSiLU` | Digital + ADC→DAC crossing | quantize(SiLU(x), n_bits) + ε_thermal |
| `nn.Hardswish` | `AnalogHardswish` | Digital + ADC→DAC crossing | quantize(Hardswish(x), n_bits) + ε_thermal |
| `nn.Mish` | `AnalogMish` | Digital + ADC→DAC crossing | quantize(Mish(x), n_bits) + ε_thermal |

GELU/SiLU/Hardswish/Mish have no efficient analog implementation — they're modeled as a full domain crossing (ADC in, digital compute, DAC out), with quantization and DAC output noise.

#### `analog_conv.py` — AnalogConv1d / AnalogConv2d / AnalogConv3d

Drop-in replacements for `nn.Conv{1,2,3}d`. Applies the same three noise sources as `AnalogLinear` to the convolution weight tensor.

The physical model treats convolution as a tiled MVM: the input is unfolded (im2col) and the kernel weights form one row of the crossbar per output channel. Thermal noise uses `N_in = in_channels × kH × kW / groups` (the receptive field size — the number of input values summed at one output node).

```python
from neuro_analog.simulator.analog_conv import AnalogConv2d, analog_conv_from_module

# From an existing module
analog_c = analog_conv_from_module(conv_layer, sigma_mismatch=0.05, n_adc_bits=8)

# Direct construction
layer = AnalogConv2d(in_channels=3, out_channels=64, kernel_size=3, sigma_mismatch=0.05)
```

`analogize()` automatically replaces all `nn.Conv1d/2d/3d` with the appropriate `AnalogConvNd` variant.

#### `analog_attention.py` — AnalogMultiheadAttention

Drop-in replacement for `nn.MultiheadAttention`. Applies analog noise to the Q/K/V/O linear projections; Q·Kᵀ and Attn·V stay digital (dynamic matmuls — inputs change every forward pass, no static crossbar mapping possible).

PyTorch's built-in MHA fuses Q/K/V into a single `in_proj_weight` tensor of shape `(3E, E)`. The recursive walk in `analogize()` cannot reach it as a plain `nn.Linear`. `AnalogMultiheadAttention.from_module()` handles this by splitting `in_proj_weight` into three slices and wrapping each with `AnalogLinear`.

```python
from neuro_analog.simulator.analog_attention import AnalogMultiheadAttention

analog_mha = AnalogMultiheadAttention.from_module(mha_layer, sigma_mismatch=0.05, n_adc_bits=8)
analog_mha.resample_mismatch(sigma=0.10)
analog_mha.set_noise_config(thermal=False, quantization=True, mismatch=True)
```

`analogize()` handles `nn.MultiheadAttention` before recursing into children to ensure Q/K/V weights are split correctly.

**Important:** `analogize()` only replaces *registered* `nn.Module` children. Functional calls like `torch.tanh()` or `torch.sigmoid()` in `forward()` are invisible to it and will not be replaced. All model files use `nn.Tanh()` / `nn.Sigmoid()` as registered modules specifically to avoid this.

#### `analog_model.py` — Model-Level Utilities

```python
from neuro_analog.simulator import analogize, resample_all_mismatch, set_all_noise

# Wrap any model — original is untouched (returns deepcopy)
analog_model = analogize(model, sigma_mismatch=0.05, n_adc_bits=8)

# Monte Carlo trial: re-roll all δ tensors
resample_all_mismatch(analog_model, sigma=0.10)

# Ablation: isolate one noise source at a time
set_all_noise(analog_model, thermal=False, quantization=False, mismatch=True)

# Per-layer V_ref calibration from real data
calibrate_analog_model(analog_model, sample_input=X_train[:32])
```

`analogize()` recursively replaces modules in priority order:
1. `nn.MultiheadAttention` → `AnalogMultiheadAttention` (handled first; fused in_proj_weight needs splitting)
2. `nn.Conv{1,2,3}d` → `AnalogConv{1,2,3}d`
3. `nn.Linear` → `AnalogLinear`
4. Activations (Tanh, Sigmoid, ReLU, ELU, LeakyReLU, GELU, SiLU, Hardswish, Mish) → analog equivalents
5. Everything else (LayerNorm, Softmax, Embedding, BatchNorm, ...) stays digital

`count_analog_vs_digital(model)` returns a coverage report: number of analog/digital layers, parameter counts, and `coverage_pct` = analog_params / total_params × 100.

#### `analog_ode_solver.py` — Analog ODE Integration

```python
from neuro_analog.simulator import analog_odeint, analog_odeint_with_logdet

# Euler integration through an analogized ODE
z_T = analog_odeint(f_theta, z0, t_span=(0.0, 1.0), dt=0.1, noise_sigma=0.05)

# With log-determinant tracking (for CNF likelihood computation)
# Thermal noise is automatically disabled during log-det to prevent Jacobian corruption
z_T, delta_logp = analog_odeint_with_logdet(f_theta, z0, t_span, dt, noise_sigma)
```

**Why thermal noise is disabled in log-det:** The log-determinant of the Jacobian requires a deterministic function. Stochastic thermal noise breaks the trace estimator (Hutchinson estimator assumes E[tr(J)] = tr(E[J])). Mismatch (static) is kept on because it's a fixed perturbation of f_θ, not a random draw per evaluation.

#### `sweep.py` — Monte Carlo Sweeps

The core measurement loop. All sweep functions follow the same pattern: for each σ, create an analogized model, resample δ n_trials times, call the provided eval_fn, record the scalar result.

```python
from neuro_analog.simulator.sweep import mismatch_sweep, adc_sweep, ablation_sweep, SweepResult

# Main sweep: mismatch tolerance curve
result = mismatch_sweep(
    model=model,
    eval_fn=evaluate,                        # fn(model) → float, higher=better
    sigma_values=[0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15],
    n_trials=50,
    n_adc_bits=8,
    calibration_data=X_train[:32],           # None for time-dependent models
)

# SweepResult properties
result.mean           # shape (n_sigma,) — mean metric per sigma
result.std            # shape (n_sigma,) — std across trials
result.normalized_mean  # 1.0 at σ=0, decreases as quality degrades
result.degradation_threshold(max_relative_loss=0.10)  # σ at 90% quality

# ADC precision tradeoff
adc_result = adc_sweep(model, evaluate, bit_values=[2,4,6,8,10,12,16], sigma_mismatch=0.05)

# Isolate noise sources
ablation = ablation_sweep(model, evaluate, sigma_values=[...])
# Returns dict: {"mismatch": SweepResult, "thermal": SweepResult, "quantization": SweepResult}
```

`SweepResult.normalized_mean` formula: `1.0 + (mean - digital_baseline) / abs(digital_baseline)`. This gives 1.0 at σ=0 and decreases correctly regardless of the sign of the underlying metric (important: all our metrics are negative numbers like negative cross-entropy, so naive `mean/abs(baseline)` gives negative normalized values and breaks the threshold check).

---

### Intermediate Representation (IR)

The IR classifies every operation in a neural network into one of 22 primitive types across three domains, then computes a composite analog amenability score.

#### `ir/types.py` — Operation Types and Domains

```python
from neuro_analog.ir.types import OpType, Domain

# 22 operation primitives
class OpType(Enum):
    # ANALOG_NATIVE — can run directly on crossbar / diff pair / RC circuit
    MVM               # Matrix-vector multiply → crossbar array
    INTEGRATION       # ODE integration → RC circuit (τ = RC)
    DECAY             # Exponential decay → leaky capacitor
    ACCUMULATION      # Summation → charge accumulation
    ELEMENTWISE_MUL   # Element-wise multiply → Gilbert cell
    ANALOG_SIGMOID    # σ(x) → differential pair
    ANALOG_EXP        # exp(x) → translinear circuit
    ANALOG_RELU       # ReLU → diode
    NOISE_INJECTION   # Stochastic sampling → sMTJ / p-bit
    ANALOG_FIR        # Finite impulse response → CCD delay line
    SAMPLE            # Sample from distribution
    SKIP_CONNECTION   # Residual addition → wire (free)
    GAIN              # Scalar multiply → variable gain amplifier

    # DIGITAL_REQUIRED — no efficient analog implementation
    SOFTMAX           # Requires exponents + normalization
    LAYER_NORM        # Requires statistics over activations
    GROUP_NORM
    RMS_NORM
    SOFTPLUS
    SILU              # Sigmoid×x, non-separable
    GELU              # Gaussian CDF, requires DAC/ADC crossing
    ADALN             # Adaptive LayerNorm (diffusion conditioning)
    DYNAMIC_MATMUL    # QK^T — weights change per input, not static

    # HYBRID — partial analog implementations exist
    KERNEL_ATTENTION  # FAVOR+ / Performer — φ(Q)·(φ(K)^T·V), ~90% analog
    APPROX_NORM       # Layer norm with analog approximation
    PIECEWISE_SILU    # Piecewise-linear SiLU on analog
    ANALOG_SOFTMAX    # Log-domain softmax on analog
```

The `AnalogAmenabilityProfile` synthesizes the graph into 5 scores (0.0–1.0):
- `dynamics_score`: does the architecture have native continuous-time dynamics?
- `precision_score`: how many bits does it need?
- `boundary_score`: how many ADC/DAC domain crossings?
- `noise_score`: how noise-tolerant is the algorithm?
- `overall_score`: weighted combination

#### `ir/graph.py` — AnalogGraph

```python
from neuro_analog.ir.graph import AnalogGraph

graph = AnalogGraph(name="neural_ode", family=ArchitectureFamily.NEURAL_ODE)
node_id = graph.add_node(make_mvm_node("W0", in_features=64, out_features=64))
graph.add_edge(source_id, target_id)

profile = graph.analyze()          # → AnalogAmenabilityProfile
print(graph.summary_table())        # ASCII bar chart of ANALOG/DIGITAL/HYBRID FLOPs
boundaries = graph.find_da_boundaries()  # list of DABoundary (ADC/DAC crossing points)
```

#### `ir/ode_system.py` — ODESystem

The bridge between extraction and Shem export. Wraps a model's parameters as `ParameterSpec` objects with mismatch annotations, so Shem can treat them as trainable with noise-aware gradients.

```python
from neuro_analog.ir.ode_system import ODESystem, ParameterSpec

ode_sys = ODESystem(
    name="neural_ode_cnf",
    family=ArchitectureFamily.NEURAL_ODE,
    state_dim=2,
    parameters={"W0": ParameterSpec("W0", value=W0_tensor, mismatch_sigma=0.05, trainable=True)},
    dynamics_fn=f_theta,
    t_span=(0.0, 1.0),
)

# Monte Carlo trial: re-draw δ for all parameters
perturbed = ode_sys.sample_mismatch(sigma=0.10)     # returns new ODESystem
ode_sys.resample_mismatch_inplace(sigma=0.10)        # in-place (faster in sweep loops)
```

---

### Extractors

Each extractor reads a pretrained model, runs it on sample data, and builds the IR graph. They all implement `BaseExtractor`:

```python
ext.load_model()
ext.extract_dynamics()      # → DynamicsProfile (ODE type, stiffness, time constants, etc.)
ext.build_graph()           # → AnalogGraph (22-op typed IR)
ext.extract_weight_statistics()  # → dict[str, PrecisionSpec] (weight min/max/std per layer)
profile = ext.run()                                        # Full pipeline, no activation calibration
profile = ext.run(calibration_data=X[:256])               # Full pipeline + activation calibration
profile = ext.run(calibration_data=X[:256], ...)          # NeuralODE: z sampled from X, t swept
specs   = ext.activation_specs                            # dict[str, PrecisionSpec] or None
specs   = ext.calibrate_activations(X[:256], percentile=99.0)  # manual call, custom percentile
```

#### Activation calibration — `calibrate_activations` / `run(calibration_data=...)`

Passing a representative input batch to `run()` triggers a single forward pass with PyTorch forward hooks attached to every leaf module. Each hook collects the full distribution of activation values across the batch. Results are stored in `ext.activation_specs` and exposed as `PrecisionSpec.activation_min/max/std/bits` per layer.

**Industry practice — percentile-based clipping:**

Production quantization frameworks (TensorRT, PyTorch FX, Quanto) never use absolute min/max for calibration. Absolute extremes are dominated by outliers that appear on < 0.1% of inputs but inflate V_ref so much that 99.9% of inferences waste ADC headroom. The standard is **99.9th-percentile clipping**: V_ref is set to cover 99.9% of activation values, accepting a tiny clipping error on the outlier tail in exchange for 1–2 extra bits of effective resolution everywhere else. This is the default in `calibrate_activations(percentile=99.9)`.

**What the numbers mean and how they affect noise:**

| Field | What it captures | Hardware consequence |
|---|---|---|
| `activation_max` | 99.9th-percentile output magnitude | Sets V_ref for ADC/DAC calibration (V_ref = activation_max). Too low → clipping (hard saturation). Too high → wasted LSB precision. |
| `activation_min` | 0.1th-percentile output | Determines whether ADC range must be bipolar. One-sided distributions (post-ReLU) allow a unipolar ADC with double the effective resolution. |
| `activation_std` | Standard deviation of the output distribution | SNR proxy: SNR ≈ activation_std / σ_thermal. Low-std layers are disproportionately hurt by both thermal noise (ε ~ N(0, kT/C · N_in)) and quantization error (Δ = 2·V_ref / 2^n_bits). |
| `activation_bits` | `ceil(log2(crest)) + 4` where crest = activation_max / activation_std | Estimated minimum ADC bits. Crest ≈ 10 → 8 bits (the empirical crossbar standard). Tanh outputs (crest ≈ 3) only need 6 bits. Large residual streams can demand 10+. |

**Crest factor and its Shem / Ark relevance:**

The crest factor (activation_max / activation_std) quantifies how many bits the ADC wastes on headroom vs. resolution. This is the same quantity Shem's V_ref calibration step (`calibrate_analog_model`) targets per inference — but the extractor measures it once at analysis time, before any analog conversion. A low crest factor means:

- Fewer ADC bits needed → lower domain-crossing cost at every D/A boundary
- Higher precision_score (since `activation_bits` is low)
- Architecture is more "worth the Shem compilation effort" — the hardware can be configured with a smaller, lower-power ADC

**How input choice affects the calibration:**

The activation statistics are calibration-set dependent. A narrow batch (only one data mode) under-estimates V_ref and will clip OOD inputs at inference. The mitigations already applied:

1. **Percentile clipping** (default 99.9) removes outliers without requiring a large batch.
2. For representative coverage: use a diverse batch of 256–512 samples spanning the training distribution. `X_train[:32]` is the minimum; `X_train[:256]` is preferred.

**Neural ODE — multi-timestep calibration:**

`NeuralODEExtractor.calibrate_activations()` is overridden. The dynamics function `f_theta(t, z)` is called at 5 uniformly-spaced time points across `t_span`, collecting activations at each step. This matters because the ODE vector field visits different activation regimes at different times (e.g., near the data manifold at t=1 vs. near the prior at t=0). A static single-t call would miss most of the trajectory's activation range.

```python
# Neural ODE: pass data samples as z0 for calibration
ext = NeuralODEExtractor.from_module(f_theta, state_dim=2, t_span=(0.0, 1.0))
profile = ext.run(calibration_data=X_train[:256])   # z sampled from data; t swept automatically
```

For SSM and Transformer models, the standard `run(calibration_data=X_train[:32])` path works directly.

For `StableDiffusionExtractor` and `FLUXExtractor` (multi-argument UNet/transformer forward), calibration via `run()` is not supported — call `calibrate_activations()` manually with a wrapper that constructs the correct `(noisy_sample, timestep, encoder_hidden_states)` tuple.

**Relationship to the precision score:**

`AnalogAmenabilityProfile.precision_score` is a **joint weight + activation** score (60/40 blend):

```
precision_score = 0.6 × weight_score + 0.4 × activation_score
weight_score     = max(0, 1 - (min_weight_precision_bits - 4) / 12)
activation_score = max(0, 1 - (min_activation_precision_bits - 4) / 12)
```

When no calibration data is provided, `min_activation_precision_bits` stays at the default of 8 (activation_score = 0.67), which neither rewards nor penalises uncalibrated profiles. When calibrated:
- Low-crest architectures (e.g. tanh-only Neural ODEs) get `activation_bits ≈ 6` → activation_score ≈ 0.83 → precision_score improves
- High-crest architectures (e.g. diffusion residual streams) get `activation_bits ≈ 10` → activation_score ≈ 0.5 → precision_score is penalised

#### `extractors/neural_ode.py` — NeuralODEExtractor

The most complete extractor. Supports torchdiffeq, torchcde, or any module with a `.dynamics()` or `.f()` method.

```python
from neuro_analog.extractors.neural_ode import NeuralODEExtractor

# From a pretrained model
ext = NeuralODEExtractor(model, state_dim=2, t_span=(0.0, 1.0))

# Or use the built-in demo (makes_circles CNF, matches the experiment model)
ext = NeuralODEExtractor.demo(state_dim=2, hidden_dim=32, num_layers=2, activation="tanh")
profile = ext.run()

# After run(), extract ODESystem for Shem
ode_sys = ext.extract_ode_system()
```

Stiffness estimation: `estimate_jacobian_stiffness(f_theta, state_dim)` computes λ_max/λ_min of the Jacobian numerically — a proxy for how hard the ODE is to integrate and how sensitive it is to perturbations.

#### `extractors/ssm.py` — MambaExtractor

The hardest extraction problem: the Mamba selective scan is a fused CUDA kernel that hides all intermediate tensors (B[t], C[t], Δ[t], h[t]).

**Solution:** A context manager `_use_reference_scan()` monkey-patches `selective_scan_fn` → `selective_scan_ref` (pure Python reference implementation with identical numerics) during calibration. The original CUDA kernel is restored immediately afterward in a `finally` block, so production inference is unaffected.

```python
from neuro_analog.extractors.ssm import MambaExtractor

ext = MambaExtractor("state-spaces/mamba-370m")
profile = ext.run()

# Direct access to extracted eigenvalues, time constants
tau = ext.dynamics_profile.time_constants  # τ_i = 1/|a_i| per state dimension
```

#### `extractors/transformer.py` — TransformerExtractor

Works with any HuggingFace model via `AutoModel`. Partitions each transformer block:

| Operation | Classification | Reason |
|---|---|---|
| Q/K/V projections | ANALOG | Static weight MVM — crossbar-native |
| Q·K^T (attention scores) | DIGITAL | Dynamic matmul — weights change per input |
| Softmax | DIGITAL | Requires normalization over sequence |
| Attn·V | DIGITAL | Dynamic matmul |
| FFN linear 1 (d→4d) | ANALOG | Static weight MVM |
| FFN activation (GELU/SiLU) | DIGITAL | No analog circuit; requires ADC→DAC crossing |
| FFN activation (ReLU) | ANALOG | Diode-connected transistor |
| FFN linear 2 (4d→d) | ANALOG | Static weight MVM |
| LayerNorm | DIGITAL | Requires mean/variance over activations |

The extractor also notes FAVOR+ (kernel attention) as an upgrade path: replacing softmax attention with random feature projections φ(Q)·(φ(K)^T·V) pushes analog fraction from ~75% → ~90%.

**No Shem export for Transformer.** Transformers have no native continuous-time ODE dynamics — every operation is a discrete matrix multiply or normalization step over a fixed sequence. There is no `dx/dt = f_θ(x, t)` to hand to Shem. The extractor's value is purely in the analog/digital partition analysis (which ~75-85% of FLOPs can physically live on a crossbar) and in the taxonomy scoring. A Shem export path would first require a continuous reparameterization of the architecture (e.g., treating the transformer as a Neural ODE with depth as time), which is a separate research problem.

#### `extractors/diffusion.py` — DiffusionExtractor

**Status: structural only — score network is an external oracle.**

Extracts the continuous-time SDE / VP-SDE framework around a diffusion model's score network `s_θ(x, t)`:

- `β(t)` schedule (linear, cosine, or learned) → noise budget per timestep
- Forward process: `dx = -½β(t)x dt + √β(t) dW`
- Reverse SDE: `dx = [-½β(t)x - β(t)s_θ(x,t)] dt + √β(t) dW`
- `build_graph()` partitions the score network's internal structure: ResBlock MVMs (ANALOG), GroupNorm (DIGITAL), SiLU (DIGITAL), attention Q/K/V projections (ANALOG), softmax (DIGITAL)

What the export generates: the SDE dynamics struct with the β schedule wired in, plus a call-out stub `s_theta(x, t)` where the score network goes. The score network itself (the U-Net or DiT backbone, which is 99.9%+ of compute) is not extracted into real weights — it's too large (512M–2B params) and architecturally equivalent to the Transformer case for its internal compute.

The analog story for diffusion is: the SDE integration loop is an analog circuit, but every denoising step calls a large digital subgraph. The bottleneck is not the ODE solver — it's the number of ADC/DAC crossings per step.

#### `extractors/flow.py` — FlowExtractor

**Status: structural only — velocity network is an external oracle.**

Extracts the rectified flow / flow matching ODE: `dx/dt = v_θ(x, t)`, where `x₀ ~ p_noise`, `x₁ ~ p_data`, and the velocity field is learned to be as straight as possible.

- `straightness`: trajectory curvature (0 = perfectly straight; lower = fewer Euler steps needed)
- `lipschitz`: Jacobian spectral norm of `v_θ` (upper bounds error accumulation per step)
- `nfe`: number of function evaluations (4 for FLUX schnell, 28 for dev)
- `build_graph()`: same transformer-block partition as TransformerExtractor

What the export generates: an Euler integration loop with a `v_theta(x, t)` call-out stub. The velocity network for production flow models (FLUX.1 = 12B params) is not extracted. For the toy `make_moons` experiment model (a 3-layer tanh MLP), the weights *are* small enough to extract in full — but the current extractor doesn't implement that path; it uses the same oracle-call pattern for consistency with the production case.

#### `extractors/ebm.py` — EBMTheoreticalAnalyzer

**Status: theoretical reference only — no pretrained weight extraction.**

EBMs have no standard forward pass and no canonical pretrained checkpoint format, so this extractor uses a theoretical profile rather than extracting from a real model. The experiment model (`experiments/.../ebm.py`) is an RBM; the extractor's `build_graph()` describes the general EBM case.

The reason EBMs are included despite being theoretical: they represent the **analog ceiling**. An EBM samples from `p(x) ∝ exp(-E_θ(x))` via Gibbs / Langevin MCMC. Every step is:
1. Local field computation: `h = W·x + b` → crossbar MVM (ANALOG)
2. Stochastic bit flip: sample `x_i ~ σ(h_i)` → p-bit / sMTJ (ANALOG)

Zero digital operations. Zero D/A boundaries. The entire sampling chain is analog-native. This is the reference point for the taxonomy: any other architecture with D/A boundaries is a step down from the EBM ceiling.

No Shem export planned — there are no ODE dynamics; the system is a discrete MCMC chain, not a continuous-time ODE.

#### `extractors/deq.py` — DEQExtractor

**Status: structural, no real weights — needs a pretrained DEQ checkpoint.**

Deep Equilibrium models find the fixed point `z* = f_θ(z*, x)` implicitly, which can be rewritten as an ODE `dz/dt = f_θ(z, x) - z` that drives the residual to zero. This is directly Shem-compilable: an analog circuit settles to equilibrium in continuous time, which *is* the fixed-point iteration.

What the extractor extracts:
- `spectral_radius`: ρ(∂f/∂z) at the fixed point — must be < 1 for convergence. If σ_mismatch shifts ρ above 1, the circuit oscillates instead of converging.
- `fixed_point_residual`: ‖f_θ(z*,x) - z*‖ — should be near zero; increases under mismatch
- `build_graph()`: MVM nodes for the weight matrices in `f_θ`, feedback edges for the recurrence

What's missing: the extractor currently uses zero-initialized placeholder weights for the MVM nodes. A real DEQ checkpoint (e.g., from the `deq` library) would replace these. The experiment model is trained from scratch in `train_all.py`, so a checkpoint exists at `checkpoints/deq.pt` after training — wiring `load_model()` to read those weights is the remaining step (~1 hour of work).

---

### Analysis & Taxonomy

```python
from neuro_analog.analysis.taxonomy import AnalogTaxonomy

tax = AnalogTaxonomy()
tax.add_reference_profiles()   # Loads all 7 architectures with empirical or literature-based profiles
print(tax.comparison_table())  # Full ASCII ranking table
ranked = tax.rank_by_analog_amenability()  # sorted list of TaxonomyEntry
```

Each `TaxonomyEntry` includes:
- `family`, `model_name`, `profile` (AnalogAmenabilityProfile with all 5 scores)
- Qualitative: `has_native_dynamics`, `dynamics_type`, `analog_circuit_primitive`, `key_digital_bottleneck`, `achour_compiler_fit`

---

### Mappers

Circuit-level mappings from IR operations to physical analog components. Used by the taxonomy and hardware estimates in AnalogGraph.

- **`crossbar.py`**: MVM → crossbar array specs. Input: weight matrix dimensions and precision. Output: `CrossbarSpec` with rows, cols, precision_bits, technology (RRAM/PCM/SRAM/NOR_FLASH/CAPACITIVE), area_mm², power_mW.

- **`integrator.py`**: ODE integration node → RC circuit spec. Maps time_constant_s → C·R values, bandwidth_hz → pole frequency, outputs `IntegratorSpec`.

- **`stochastic.py`**: Stochastic sampling nodes → p-bit / sMTJ spec. Used for EBM Gibbs sampling and diffusion noise injection.

---

### Visualization

```python
from neuro_analog.visualization.comparison_radar import plot_radar_comparison
from neuro_analog.visualization.noise_budget import plot_noise_budget
from neuro_analog.visualization.partition_map import plot_partition_map

# 6-axis radar chart: all 7 architectures on analog suitability axes
plot_radar_comparison(taxonomy, output_path="figures/radar.pdf")

# Noise source contribution at σ=0.05
plot_noise_budget(ablation_results, output_path="figures/noise_budget.pdf")

# Layer-by-layer analog/digital partition heatmap
plot_partition_map(graph, output_path="figures/partition.pdf")
```

The radar chart axes: (1) Analog FLOP %, (2) D/A boundary score, (3) Precision tolerance, (4) Dynamics naturalness, (5) Mismatch resilience, (6) Shem compatibility.

---

### Pipeline

The unified entry point for running the full analysis on a single model:

```python
from neuro_analog.pipeline import run_pipeline
from neuro_analog.extractors.neural_ode import NeuralODEExtractor
from experiments.cross_arch_tolerance.models.neural_ode import evaluate

result = run_pipeline(
    model=model,
    family="neural_ode",
    extractor_class=NeuralODEExtractor,
    eval_fn=evaluate,
    calibration_data=X_train[:32],
    output_dir="outputs/",
    n_trials=50,
)
# result.graph           → AnalogGraph IR
# result.ode_system      → ODESystem (for ODE/SSM families)
# result.sweep_results   → dict[str, SweepResult]
# result.shem_export_path → path to generated JAX file
```

---

## Experiments: `cross_arch_tolerance/`

### The Seven Models

Each model file exposes a standard interface:
```python
create_model() → nn.Module
train_model(model, save_path) → nn.Module
load_model(save_path) → nn.Module
evaluate(model) → float          # higher = better; MUST be a continuous metric
evaluate_output_mse(model, digital_baseline) → float  # negative MSE vs digital
get_family_name() → str
```

#### Why continuous metrics, not accuracy?

With accuracy (argmax), a Transformer showed only 0.1% drop at σ=15% — logit *rankings* are preserved even with 76% magnitude corruption. Cross-entropy exposes the magnitude degradation: the model becomes less confident, even if still correct. This matters for Shem: gradient-based optimization needs a smooth signal.

| Model | Task | Metric | Why this metric |
|---|---|---|---|
| **neural_ode.py** | 2D density estimation on make_circles | Log-likelihood | Continuous; ODE integration directly maps log-likelihood → sensitivity to vector field corruption |
| **ssm.py** | Sequence classification (pattern detection) | Negative cross-entropy | Continuous logit distribution; smoother than accuracy |
| **transformer.py** | Sequence classification (pattern detection) | Negative cross-entropy | Same task as SSM for direct comparison |
| **diffusion.py** | Image generation (8×8 MNIST) | Negative nearest-neighbor distance | Sample quality proxy; continuous |
| **flow.py** | 2D density estimation on make_moons | Negative Wasserstein distance | OT distance between generated and true distribution |
| **ebm.py** | Free generation (8×8 MNIST) | Negative nearest-neighbor dist. | Free-chain quality; lower NN dist. = better generative model |
| **deq.py** | Classification (8×8 MNIST) | Negative cross-entropy + convergence failure rate | CE for quality; failure rate for analog-specific convergence risk |

#### Why small models near capacity?

Overparameterized models have noise margin — redundant weights absorb mismatch without affecting output. We shrink models to operate near capacity (tight weight–output coupling) so the degradation curve is actually informative:

- Transformer: d_model shrunk from 64 → 24
- SSM: d_model shrunk from 32 → 16
- Neural ODE: hidden_dim 64 → 20

At larger scale on harder tasks, the degradation curves would shift right (more tolerant). Our results bound the *worst case* for a given architecture family.

#### Model-specific design notes

**neural_ode.py** — Continuous Normalizing Flow. The ODE dynamics f_θ is a `[z_dim+1 → 64 → 64 → z_dim]` tanh MLP (the +1 is time augmentation). Training runs the ODE *backward* with log-det Jacobian tracking. Evaluation uses `analog_odeint_with_logdet()` with thermal noise disabled (see design decisions below).

**ssm.py** — S4D-style diagonal SSM. The state h is complex-valued (8 complex = 16 real eigenvalues). Only B and C projections (`nn.Linear`) are analogized — A_bar is element-wise complex multiplication, which maps to passive RC decay (the analog circuit IS the computation, no replacement needed). Bilinear (Tustin) discretization ensures |A_bar| < 1 for all σ.

**transformer.py** — Standard self-attention. Everything except Q/K/V projections and FFN linears is digital (LayerNorm, Softmax, dynamic Q·K^T). The model intentionally uses ReLU not GELU in the FFN, making the FFN fully analogizable.

**diffusion.py** — DDPM with T=100 denoising steps. Each step runs through the analogized score network. The 100-step cascade is what makes diffusion uniquely sensitive: mismatch errors compound multiplicatively across steps. Also the only architecture where quantization is a co-dominant noise source.

**flow.py** — Rectified flow (straight-line ODE from noise to data). Uses 4 Euler steps. Metric uses random z₀ sampling for Wasserstein estimation, which introduces stochastic baseline variance — this is why the baseline sample count fix (see TECHNICAL_NOTE §6) was critical.

**ebm.py** — Restricted Boltzmann Machine. Evaluation runs 500 Gibbs burn-in steps from random init, then measures nearest-neighbor distance of 200 free-generated samples to the test set. `W_fwd` and `W_bwd` are separate `nn.Linear` modules (both analogized independently), modeling the realistic case where a crossbar read in forward vs. transpose mode has different noise characteristics. Activations use registered `nn.Sigmoid()` modules (not `torch.sigmoid()`) so `analogize()` can replace them.

**deq.py** — Deep Equilibrium Model. Fixed-point iteration `z_{k+1} = tanh(W_z·z + W_x·x)` run until convergence. Spectral normalization on W_z ensures ρ(∂f/∂z) < 1 (contraction), guaranteeing the fixed point exists. Under mismatch, the effective spectral radius can grow, potentially causing divergence. Uses registered `nn.Tanh()` (not `torch.tanh()`) for the same reason as EBM above.

---

### Running the Sweep

**Step 1: Train all 7 models** (~15–25 min on CPU, ~5 min on GPU)
```bash
cd experiments/cross_arch_tolerance
python train_all.py
# Saves: checkpoints/neural_ode.pt, checkpoints/ssm.pt, etc.
# Detects existing checkpoints and skips. Use --force to retrain.
```

**Step 2: Run all sweeps** (~90 min on CPU, ~20 min on GPU with n_trials=50)
```bash
python sweep_all.py --force --n-trials 50
# --force: rerun even if results/*.json already exist
# --n-trials: MC trials per sigma point (50 recommended, 20 minimum)
# Outputs: results/{name}_{sweep_type}.json for all 7 models × 5 sweep types
```

**Step 3: Generate figures**
```bash
python plot_results.py
# Outputs: figures/fig1_mismatch_tolerance.png/.pdf
#          figures/fig2_noise_ablation.png/.pdf
#          figures/fig3_adc_precision.png/.pdf
#          figures/fig4_deq_convergence.png/.pdf
#          figures/fig5_visual_samples.png/.pdf
```

**Step 4 (optional): Evaluate continuous models natively in JAX/Diffrax**
```bash
python evaluate_in_shem.py --model ssm        # SSM with real B/C weights
python evaluate_in_shem.py --model neural_ode  # Neural ODE CNF
python evaluate_in_shem.py --model all         # Both
```

This runs the complete Shem-compatible evaluation loop: extracts the PyTorch model → exports to JAX/Diffrax via `shem_export.py` → dynamically loads the generated code → runs JIT-compiled mismatch trials natively in JAX. Requires `jax`, `jaxlib`, `diffrax`.

**Quick summary of thresholds (after results are in):**
```bash
python -c "
import json, glob
for f in sorted(glob.glob('results/*_mismatch.json')):
    d = json.load(open(f))
    name = f.split('/')[-1].replace('_mismatch.json', '')
    print(f'{name:15s} threshold={d.get(\"degradation_threshold_10pct\", \"N/A\"):.3f}  baseline={d.get(\"digital_baseline\", 0):.4f}')
"
```

---

### Understanding the Results

**Results directory structure (49 files per full run):**
```
results/
├── {name}_mismatch.json              # sigma: 0→15%, 50 trials — THE main result
├── {name}_ablation_mismatch.json     # mismatch source only
├── {name}_ablation_thermal.json      # thermal noise only
├── {name}_ablation_quantization.json # quantization only
├── {name}_adc.json                   # bit-width sweep at σ=5%
├── {name}_output_mse.json            # output corruption (deviation from digital)
└── {name}_convergence.json           # DEQ only: fixed-point failure rates
```

**Each JSON schema:**
```json
{
  "sigma_values": [0.0, 0.01, 0.02, ...],
  "metric_name": "cross_entropy",
  "per_trial": [[trial1_σ0, trial2_σ0, ...], [trial1_σ1, ...]],
  "digital_baseline": -0.165,
  "mean": [-0.165, -0.166, ...],
  "std": [0.001, 0.003, ...],
  "normalized_mean": [1.000, 0.994, ...],
  "degradation_threshold_10pct": 0.15
}
```

**What the figures show:**

- **Figure 1** (mismatch tolerance): 7 curves plotting normalized quality vs σ. Horizontal line at 0.9 (10% loss threshold). The x-intercept of each curve with this line is the architecture's tolerance threshold. Higher and further right = better for analog deployment.

- **Figure 2** (noise ablation): For each architecture, 3 bars at σ=5% showing the contribution of mismatch-only, thermal-only, and quantization-only. Tells you which noise source to prioritize in Shem optimization.

- **Figure 3** (ADC precision): Quality vs bit-width at σ=5%. Finds the minimum bits needed before quality degrades more than 5%. SSM shows catastrophic 2-bit failure (state explosion in recurrence).

- **Figure 4** (DEQ convergence): Dual-axis: CE loss + convergence failure rate vs σ. Shows whether fixed-point iteration diverges under mismatch, independent of output quality.

- **Figure 5** (visual samples): For generative models (diffusion, flow), shows generated samples at σ=0%, 5%, 10%, 15% to visualize degradation qualitatively.

---

## Shem Export Pipeline

Shem takes an ODE system described in JAX and optimizes its parameters to be robust to mismatch δ~N(1,σ²) via the adjoint method. We generate the input code:

```python
from neuro_analog.ir.shem_export import export_neural_ode_to_shem, export_ssm_to_shem
from neuro_analog.extractors.neural_ode import NeuralODEExtractor

# Export Neural ODE → Shem-compatible JAX class
ext = NeuralODEExtractor.demo()
ext.run()
export_neural_ode_to_shem(ext, "outputs/neural_ode_shem.py", mismatch_sigma=0.05)

# Export SSM with real pretrained weights
ext = MambaExtractor(model_path="checkpoints/ssm.pt")
ext.run()
export_ssm_to_shem(ext.graph, ext, "outputs/ssm_shem.py", mismatch_sigma=0.05)
```

The generated file (see `outputs/ssm_shem_test.py` for an example) is a JAX class implementing the `BaseAnalogCkt` interface from Ark:

```python
class SSMAnalogCkt(BaseAnalogCkt):
    def dynamics(self, t, x, u):
        # dx/dt = A·x + B·u  (extracted A, B matrices with mismatch annotations)
        ...
    def solve(self, y0):
        # Diffrax Tsit5 solver
        ...
    def mismatch(self, sigma):
        # Returns AnalogTrainable parameters for Shem optimizer
        ...
```

**Validate exports:**
```bash
python validate_shem_export.py       # Neural ODE
python validate_ssm_shem_export.py   # SSM
```

**What's exported today:**
- Neural ODE: complete (all f_θ weights, mismatch annotations, diffrax solve block)
- SSM: complete with real B/C weights from pretrained model

**Structural templates — ODE/SDE shell without full weights:**

These three architectures have real ODE dynamics (Shem-compilable in principle) but the export is not yet runnable end-to-end:

| Architecture | What's generated | What's missing | Why it's a stub |
|---|---|---|---|
| Flow | Euler integration loop, `v_theta(x, t)` call-out | Velocity network weights | Production v_θ is 12B params (FLUX); toy model weights not wired to extractor |
| Diffusion | VP-SDE dynamics, β schedule, `s_theta(x, t)` call-out | Score network weights | Score network is 99.9%+ of compute; architecturally the Transformer problem |
| DEQ | Fixed-point ODE `dz/dt = f_θ(z,x) - z`, convergence struct | Real MVM weights | Extractor uses zero-init placeholders; checkpoint exists but not wired |

**No Shem export path:**

| Architecture | Reason |
|---|---|
| Transformer | No native ODE dynamics — all discrete matmuls and normalization; would require continuous reparameterization first |
| EBM | Discrete MCMC chain (Gibbs sampling), not a continuous-time ODE; analog-native but not Shem-compilable |

---

## Tests

```bash
pytest tests/                          # all tests
pytest tests/test_simulator.py -v      # simulator tests
pytest tests/test_ir.py -v             # IR tests
```

**Core invariants tested in `test_simulator.py`:**

1. `sigma=0, all noise off` → output matches `F.linear` exactly (machine epsilon)
2. Mismatch δ is static — same output on repeated forward passes without resample
3. `resample_mismatch()` changes the output
4. Thermal noise is stochastic — different output each forward pass
5. `n_bits=32` quantization ≈ noiseless (< 1e-5 error)
6. ADC clips correctly at ±V_ref
7. `calibrate()` sets V_ref to 1.1×max(|activation|)
8. `ablation_sweep` correctly isolates noise sources (mismatch-only > thermal-only at large σ)
9. `degradation_threshold` returns correct σ at 90% normalized quality
10. `SweepResult` serializes and deserializes correctly from JSON

---

## Key Design Decisions

### 1. Static vs. Dynamic Mismatch

Mismatch δ is resampled once per `analogize()` call or explicit `resample_mismatch()` call, then held fixed for all forward passes. This is physically correct: fabricated conductances are fixed at manufacture. Re-sampling δ per forward pass would model a different (incorrect) regime where every read has independent noise.

### 2. Thermal Noise Disabled During Log-Det Computation

The log-determinant of the Jacobian uses a Hutchinson trace estimator: `tr(J) ≈ εᵀ·J·ε` for random ε. This requires E[tr(J)] = tr(E[J]), which only holds if f_θ is deterministic given the same input. Thermal noise (stochastic per inference) breaks this. Mismatch (fixed perturbation of weights) preserves it. So we call `set_all_noise(thermal=False, mismatch=True)` before log-det computation and restore `thermal=True` immediately after.

### 3. Mamba CUDA Kernel Monkey-Patch

`selective_scan_cuda` is a fused CUDA kernel — intermediate tensors B[t], C[t], Δ[t], h[t] are not accessible via standard PyTorch hooks. Solution: during calibration only, swap the implementation to `selective_scan_ref` (the Python reference that Mamba ships for testing) using a context manager that always restores the original in a `finally` block. This gives full tensor visibility with zero impact on production inference.

### 4. Baseline Sample Count Must Equal n_trials

Early versions used `min(5, n_trials)` baseline samples for the digital reference. For stochastic-metric models (Flow uses random z₀ in Wasserstein, Diffusion uses random noise), 5 samples produced an unstable baseline with high variance. Setting baseline samples = n_trials everywhere makes the normalized_mean formula stable. This was one of the critical bugs causing incorrect results (see TECHNICAL_NOTE §6.2 Bug 3).

### 5. SweepResult.normalized_mean Formula

The correct formula is `1.0 + (mean - baseline) / abs(baseline)`. Naive `mean / abs(baseline)` gives negative normalized values when the underlying metric is negative (all our metrics are things like negative cross-entropy = −0.165). Negative normalized values make `q >= 0.9` never true, causing all threshold values to falsely report 0.0. See TECHNICAL_NOTE §6.2 Bug 1.

### 6. Registered Modules vs. Functional Calls

`analogize()` walks `nn.Module` children recursively and replaces registered submodules. It cannot see Python function calls inside `forward()`. If `deq.py` used `torch.tanh()` instead of `self.act = nn.Tanh()`, the tanh would never become `AnalogTanh` and the DEQ experiment would be undersimulated. All model files use registered modules for every operation that should be analogized.

### 7. Inference-Only Simulation (No Analog Training)

The simulator instruments the forward pass only. There is no analog backward pass, no noise-aware gradient, and no support for training through the simulated hardware. This was a deliberate scope decision: characterizing *inference* degradation is the prerequisite question ("does this architecture survive fabrication at all?") before the more expensive question of hardware-aware training. Shem handles the optimization side; neuro-analog handles the measurement side.

A consequence: mismatch δ is treated as a fixed, frozen perturbation of the weights during a sweep — not as a parameter to differentiate through. If you want to train a model to be robust to δ, you export to Shem and use its adjoint optimizer.

### 8. Layer-Wise Nonidealities Only

The simulator models three nonidealities at the layer boundary (mismatch, thermal noise, ADC quantization) and explicitly excludes:

- **IR drop across crossbar arrays**: Metal interconnect resistance causes the effective input voltage to attenuate along a crossbar row. This is a strong function of array size and position — significant for large (512×512+) arrays, negligible for the small arrays in our experiment models.
- **1/f (flicker) noise**: Frequency-dependent noise dominant at low frequencies / slow inference rates. Requires a power spectral density model not warranted for first-order characterization.
- **PCM/RRAM conductance drift**: Post-fabrication, phase-change materials drift on a power-law timescale (hours to months). Critical for deployed hardware but out of scope for static tolerance characterization.
- **Multi-layer coupling**: The simulator treats each layer independently. In practice, a large mismatch-induced signal at layer L can shift the input distribution to layer L+1 in ways that compound non-linearly. This is captured partially by the sweep (which measures end-to-end output degradation) but not in the per-layer model.

These effects are well-understood and modeled in the HCDCv2 / IBM HERMES literature. Extending the simulator to include them is the natural next step for hardware validation studies.

### 9. Architecture Scope: 7 Families, No Recurrent Networks

The seven architectures (Neural ODE, SSM, Diffusion, Flow, EBM, Transformer, DEQ) were chosen to span the space of continuous-time vs. discrete, generative vs. discriminative, and ODE-native vs. discrete-sequence. Explicitly excluded:

- **GRU / LSTM**: Gated recurrent units share the static-weight MVM structure but add element-wise gating operations. The simulator already supports these (AnalogLinear handles all MVMs, AnalogSigmoid handles the gates), but no extractor or experiment was built. The analog story is similar to SSM but with more DIGITAL_REQUIRED gates per step.
- **Convolutional architectures (CNNs, U-Nets)**: AnalogConv is fully implemented and tested; CNN models were excluded from the cross-architecture experiment to keep the study focused on temporal/generative dynamics. CNNs as a class are straightforwardly analog-amenable (static weight crossbar MVMs throughout).
- **Hybrid architectures (Jamba, Zamba, MambaFormer)**: SSM + Transformer interleaved. Extractors for pure SSM and pure Transformer exist; hybrid models would need a composite extractor. Left for future work.

---

## Findings Summary

Results from full rerun, 50 trials (conservative profile). See TECHNICAL_NOTE §3 and §6 for full tables. Four sweep types per architecture: mismatch tolerance, noise source ablation, ADC precision, output MSE.

### Mismatch tolerance (conservative profile, `*_mismatch.json` / `*_ablation_mismatch.json`)

| Architecture | σ threshold @ 10% loss | σ=5% quality | σ=15% quality | Notes |
|---|---|---|---|---|
| **EBM** | **≥15%** | 0.999 | 0.995 | Flattest mismatch curve; Gibbs sampling self-corrects per step |
| **SSM** | **≥15%** | 0.999 | 0.996 | Diagonal state recurrence; clean monotonic degradation |
| **Transformer** | **≥15%** | 0.997 | 0.981 | LayerNorm absorbs scale perturbations |
| **Neural ODE** | **≥15%** | 1.008 | 1.016 | High-variance metric; both runs (1.016, 0.902) consistent with near-zero trend |
| **Diffusion** | **≥15%†** | ~1.000 | ~1.000 | **Mismatch-immune** (ablation flat 0.997–1.005); quantization is the only failure mode |
| **DEQ** | **12%** | 0.982 | 0.766 | Spectral norm keeps ρ<1 but threshold degrades; 0.766 at σ=15% |
| **Flow** | **10%†** | 0.978 | 0.585 | **Worst mismatch tolerance**; velocity field drift, no error correction |

†Values from `*_ablation_mismatch.json` — generation metric baseline variance makes full sweep unreliable for these two.

**Thermal noise: zero effect on all 7 architectures.** σ_thermal = √(kT/C)·√N_in ≈ 5×10⁻⁴ V at C=1pF is negligible vs. weight mismatch at any σ tested. Flow is the weakest architecture; EBM is the most robust.

### ADC minimum bits at σ=5% (`*_adc.json`)

| Architecture | Min bits (≥95%) | 2-bit quality | Failure mode at 2-bit |
|---|---|---|---|
| **Neural ODE** | **2 bits** | 0.998 | None — log-det correctly isolates quantization |
| **EBM** | **2 bits** | 0.988 | None — Gibbs sampling inherently near-binary |
| **Transformer** | **4 bits** | 0.942 | Attention pattern collapse |
| **SSM** | **4 bits** | 0.762 | State recurrence diverges across 64 timesteps |
| **DEQ** | **6 bits** | **−2.333** | Fixed-point **limit cycle** — z_k oscillates between quantization bins |
| **Diffusion** | **N/A** | 0.927 | Conservative profile floor (per-layer ADC × 100 DDPM steps); use full-analog profile |
| **Flow** | **2 bits†** | 0.990 | None (Wasserstein metric insensitive to quantization) |

DEQ's 6-bit minimum is the hardest hardware constraint of all 7 architectures. The 2-bit failure is not accuracy degradation — it's a **convergence failure**: the iterative solver oscillates rather than converges.

### Full-analog profile reveals ranking reversals

The conservative profile (ADC at every layer) is the wrong model for ODE-native analog substrates. Under full-analog (single ADC at final readout):

| Architecture | Conservative threshold | Full-analog threshold | Change |
|---|---|---|---|
| EBM | ≥15% (flat) | **10%** | **Degrades** — removing per-step ADC binarization exposes mismatch in continuous sigmoid activations |
| DEQ | 12% | **≥15%** | **Improves** — removing per-iteration ADC eliminates limit cycles |
| Diffusion | N/A (ADC floor) | **≥15%, all bit-widths OK** | Resolves entirely — ADC floor disappears |
| SSM, Transformer, Neural ODE, Flow | Similar | Similar | Minimal change |

The EBM/DEQ reversal is the most hardware-architecturally significant finding: which substrate model you use changes the **ranking**, not just the numbers.

### CLD substrate: Diffusion becomes fully noise-agnostic

Under the CLD (RLC/Langevin circuit) substrate — where thermal fluctuations are the diffusion noise, not a separate noise source:

- Mismatch: flat 0.998–1.000 (same as classic)
- **Thermal: flat 1.000** (circuit noise = generation noise = no net effect)
- **Quantization: flat 0.998 at all bit-widths including 2-bit**

CLD diffusion is 2-bit tolerant and immune to all three noise sources simultaneously. This is the appropriate model for hardware where the score network runs on a physical Langevin substrate.

### Output MSE: how much does each architecture's output trajectory diverge?

| Architecture | MSE at σ=0 | MSE at σ=15% | Interpretation |
|---|---|---|---|
| EBM | ~0.000 | 0.0009 | Gibbs equilibrium absorbs perturbations |
| Transformer | 0.00002 | 0.019 | Clean monotonic growth — most interpretable signal |
| SSM | 0.001 | 0.023 | Similar to Transformer |
| Flow | 0.029 | 0.091 | Moderate growth; z0-seed baseline offset |
| Neural ODE | **0.070** | **0.071** | **Flat** — structural offset from log-det path differences, not mismatch |
| DEQ | 0.003 | **0.234** | Steepest growth — fixed-point iteration amplifies output divergence |
| Diffusion | **0.695** | **0.694** | **Flat** — conservative ADC floor dominates; mismatch adds nothing |

Two architectures show flat output MSE: Neural ODE (structural log-det difference at σ=0) and Diffusion (ADC quantization floor at σ=0). DEQ shows 78× MSE growth from σ=0 to σ=15% — the largest amplification factor of any architecture.

### DEQ hopfield substrate outperforms discrete fixed-point

| Substrate | σ threshold | σ=15% quality | Mechanism |
|---|---|---|---|
| Discrete (fixed-point iteration) | 12% | 0.766 | No damping; mismatch pushes ρ toward instability |
| **Hopfield (damped RC relaxation)** | **≥15%** | **0.849** | `-z` damping term acts as physical regularizer |

The Hopfield substrate brings DEQ from a mid-table result to the top tier. Both require 6-bit ADC.

---

## What's Complete vs. In Progress

**Complete and tested:**
- `AnalogLinear`, `AnalogTanh/Sigmoid/ReLU/ELU/LeakyReLU/GELU/SiLU/Hardswish/Mish` — all 3 noise sources, ablation, calibration
- `AnalogConv1d/2d/3d` — convolutional crossbar simulation with receptive-field thermal noise
- `AnalogMultiheadAttention` — analog Q/K/V/O projections with fused weight splitting
- `apply_ssm_mismatch()`, `analog_ssm_recurrence()` — SSM-specific A_bar mismatch + transient noise
- `analogize()`, `resample_all_mismatch()`, `set_all_noise()`, `calibrate_analog_model()`, `count_analog_vs_digital()`
- `mismatch_sweep`, `adc_sweep`, `ablation_sweep`, `SweepResult`
- `analog_odeint_with_logdet()` — ODE integration with log-det tracking
- `ODESystem`, `ParameterSpec`, `NoiseProfile` — IR bridge to Shem
- `run_pipeline()` — unified extraction + sweep + export in one call
- All 7 experiment models with standard interface
- `train_all.py`, `sweep_all.py`, `plot_results.py`, `evaluate_in_shem.py`
- Neural ODE Shem export with real weights
- SSM Shem export with real B/C weights
- `NeuralODEExtractor` including stiffness estimation and `extract_ode_system()`
- `MambaExtractor` including CUDA kernel monkey-patch and `extract_ode_system()`
- `TransformerExtractor` (HuggingFace)
- IR types, nodes, graph, amenability scoring
- Taxonomy and radar chart

**Structural templates (ODE/SDE shell generated, not fully runnable):**
- **Flow Shem export** — Euler loop + `v_theta` call-out stub; toy model weights not wired to extractor
- **Diffusion Shem export** — VP-SDE dynamics + β schedule + `s_theta` call-out stub; score network external
- **DEQ Shem export** — fixed-point ODE form + convergence struct; uses zero-init weight placeholders (checkpoint exists but not connected)
- **EBM extractor** — theoretical profile only; no pretrained weight extraction (EBM is the analog ceiling reference)

**No export path (by design):**
- **Transformer** — no native ODE dynamics; all discrete matmuls; would require continuous reparameterization first
- **EBM** — discrete MCMC chain, not an ODE; analog-native but not Shem-compilable

**Physics not modeled (scope decisions — see Key Design Decision §8):**
- PCM/RRAM conductance drift over time (power-law decay)
- 1/f (flicker) noise
- IR drop across crossbar arrays
- Multi-layer nonideality coupling

**Architectures not covered (scope decisions — see Key Design Decision §9):**
- GRU / LSTM (simulator supports them; no extractor or experiment)
- CNN / U-Net (AnalogConv is implemented; excluded from cross-arch study)
- Hybrid SSM+Transformer architectures (Jamba, Zamba, MambaFormer)

---

## Connection to Shem / Ark / Unconventional AI

**Shem** ([arXiv 2411.03557](https://arxiv.org/abs/2411.03557)) is a hardware-aware optimization framework that takes a parametrized ODE system and uses the adjoint method to find parameters robust to fabrication mismatch δ~N(1,σ²) and transient SDE noise. It models mismatch as multiplicative perturbations (same model we use) and uses Gumbel-Softmax for discrete parameters.

**Ark** ([arXiv 2309.08774](https://arxiv.org/abs/2309.08774)) is the DSL/compiler that specifies analog compute paradigms as ODEs and validates/simulates them. Its `nacs_as_nn` example is the closest to our work — neural networks implemented as analog ODE systems, optimized with Shem.

**Unconventional AI** (founded 2025, $475M seed at $4.5B valuation) is commercializing this research direction: silicon circuits that run neural networks "on the physics directly" rather than simulating physics on digital computers.

**Where neuro-analog fits:**

```
[neuro-analog]                      [Ark + Shem]
analogize(model, σ)  ─────────────→ shem_export.py generates
measures degradation                 JAX/Diffrax ODE spec
                                        ↓
                                   Shem adjoint optimizer
                                   finds δ-robust parameters
                                        ↓
                                   re-run analogize() sweep
                                   → degradation curve shifts right
```

This codebase answers: *which architectures are worth the Shem compilation effort, and what does the degradation gap look like before optimization?* The Shem exports (Neural ODE, SSM) are the first step toward closing that loop end-to-end on a generative task — which has not been demonstrated in the Ark paper.

The `nacs_as_nn` paradigm in Ark specifically targets neural networks as analog ODEs. Our `analogize()` + sweep is the measurement instrument that quantifies what Shem needs to fix, and our Neural ODE Shem export is the scaffold for running Shem on a model whose analog tolerance we have already characterized.

Sources:
- [Shem paper](https://arxiv.org/abs/2411.03557)
- [Ark paper](https://arxiv.org/abs/2309.08774)
- [WangYuNeng/Ark on GitHub](https://github.com/WangYuNeng/Ark)

---

## Next Steps for Compilation to Shem

Each of the seven architectures has a different relationship to the ODE form that Shem requires. The table below summarizes current status and what remains.

| Architecture | ODE-compatible | Current status | What's needed |
|---|---|---|---|
| Neural ODE | Yes | **Done** — runs through `analog_odeint_with_logdet`; RC integrator substrate implemented | Nothing; ready for Shem adjoint today |
| Flow | Yes | **Done** — `v_theta(t, x)` runs through `analog_odeint`; RC integrator substrate implemented | Nothing; ready for Shem adjoint today |
| Diffusion | Effectively yes | **Done** — three substrates: classic DDIM (deterministic reverse ODE), CLD (RLC/Langevin), Extropic DTM (Langevin MCMC per step) | Probability flow ODE form could replace manual DDIM loop for cleaner `analog_odeint` integration; not blocking |
| DEQ | Yes | **Done** — Hopfield substrate IS the continuous-time ODE: `dz/dt = -z + f(z,x)`; fixed-point iteration is its discrete approximation | Hopfield substrate already runs the continuous-time form; Shem export scaffold exists |
| SSM | Yes, partially | **Gap** — `A_c` continuous-time parameters are already learned; discrete bilinear recurrence is a numerical approximation of `dh/dt = A_c·h + B_c·u(t)` | `analog_odeint` needs time-varying forcing term support (`f(t, h, u_t)`); once added, SSM becomes a linear ODE driven by the input sequence — cleanly Shem-exportable |
| Transformer | No | **Structural mismatch** — attention is a set operation over tokens, not a dynamical system; sequence dimension is not continuous time | Would require continuous-depth reformulation (e.g., neural ODE over layers); entirely different architecture |
| EBM (RBM) | Possible via Langevin | **Major redesign** — discrete Gibbs chain is not an ODE; the energy landscape supports Langevin dynamics `dx = -∇E(x)dt + √(2T)dW`, which is Shem-compatible and Extropic-aligned | Replace CD training with score matching; replace Gibbs sampler with Langevin integrator; substantial but well-defined path |

### Immediate next step: SSM forcing term

The highest-value near-term addition is extending `analog_odeint` to accept a time-varying input signal `u(t)`, enabling the SSM to run as a continuous-time ODE:

```
dh/dt = A_c · h(t) + B_c · u(t)
```

The `A_c` and `B_c` parameters are already learned and available in `_S4DLayer._get_discrete_params()`. The bilinear transform to `A_bar` was always a discretization convenience — the continuous-time form is more physically accurate for analog RC integrators and directly maps onto the Shem optimizer's adjoint method.

### EBM Langevin path

Replacing the RBM's Gibbs sampler with Langevin dynamics would make the EBM the most physically grounded model in the suite — the thermal noise `√(2T)dW` is not injected but physically present in subthreshold transistors, exactly as described in the Extropic DTM paper (arXiv:2510.23972). This is a meaningful architectural change but the energy function `E(v,h) = -v^T W h - b^T v - c^T h` is already implemented and differentiable.
- [Unconventional AI](https://unconv.ai)
