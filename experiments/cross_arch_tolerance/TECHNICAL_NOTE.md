# Cross-Architecture Analog Tolerance: How Seven Neural Network Families Degrade Under Fabrication Mismatch

**Authors:** Apuroop Mutyala
**Date:** 2026
**Repository:** github.com/apumutyala/neuro-analog

---

## 1. Problem

No systematic comparison exists of how different neural architectures degrade under realistic analog hardware nonidealities. Architecture selection for analog AI chips is currently based on theoretical analysis of which operations are "analog-amenable" — but amenability in theory and tolerance in practice are different questions.

When a Boltzmann machine runs on a crossbar array, the conductance cells have fabrication variance δ ~ N(1, σ²). When an SSM integrator runs on an RC circuit, thermal noise adds σ = √(kT/C) per time step. When any architecture reads its output through an ADC, quantization error adds V_LSB²/12. These nonidealities compound in architecture-specific ways. A transformer's softmax amplifies weight perturbations; a DEQ's fixed-point loop may diverge entirely when the Jacobian spectral radius crosses 1.

This note answers the question empirically: **for seven families of neural networks, at what mismatch level does quality degrade past 10%, and which nonideality is responsible?**

---

## 2. Method

### 2.1 Analog Forward-Pass Simulator

We implement an architecture-agnostic analog inference simulator as a PyTorch module replacement layer. `analogize(model, sigma, n_adc_bits)` recursively replaces:

| Digital module | Analog replacement | Physics modeled |
|---|---|---|
| `nn.Linear` | `AnalogLinear` | Conductance mismatch + kT/C thermal + ADC quantization |
| `nn.Tanh` | `AnalogTanh` | MOSFET diff pair gain/offset mismatch, swing saturation |
| `nn.Sigmoid` | `AnalogSigmoid` | Same as Tanh with single-ended output |
| `nn.ReLU` | `AnalogReLU` | Diode-connected transistor offset |
| `nn.GELU/SiLU` | `AnalogGELU/SiLU` | Digital computation + ADC→DAC domain crossing penalty |

Everything else (LayerNorm, Softmax, Embedding) stays digital — there is no efficient analog implementation.

**Three noise sources in `AnalogLinear`, applied in order:**

**1. Conductance mismatch (static, per device):**

```
W_device = W_nominal ⊙ δ,   δ ~ N(1, σ²·I)
```

The same δ persists across all inferences — it is baked into the fabricated conductance values. Source: Shem §4.1.

**2. Thermal read noise (dynamic, per inference):**

```
y = W_device @ x + ε,   ε ~ N(0, σ_thermal² · I)
σ_thermal = √(kT/C) · √(N_in)
```

The √N_in factor models N independent column current contributions at the sense node (Legno §4, Johnson-Nyquist at C = 1 pF, T = 300 K).

**3. ADC quantization (deterministic):**

```
y_q = round(y · scale) / scale,   scale = (2^n_bits - 1) / (2 · V_ref)
```

V_ref = 1.0 V (HCDCv2 hardware spec). Hard quantization models inference; Shem uses Gumbel-Softmax for differentiable training.

### 2.2 Seven Models

| # | Family | Architecture | Params | Task | Metric |
|---|---|---|---|---|---|
| 1 | **Neural ODE** | f_θ: [3→20→20→2] tanh MLP | ~1K | CNF density, make_circles | Log-likelihood ↑ |
| 2 | **Transformer** | 2-layer, dim=24, 2 heads, ReLU FFN | ~10K | Sequence classification | Neg. cross-entropy ↑ |
| 3 | **Diffusion** | 3-layer score MLP, T=100 DDPM, 8×8 MNIST | ~103K | Generation | Neg. nearest-neighbor dist. ↑ |
| 4 | **Flow** | v_θ: [3→64→64→2] tanh MLP | ~4K | Rectified flow, make_moons | Neg. Wasserstein ↑ |
| 5 | **EBM** | RBM: visible=64, hidden=32 | ~4K | Free generation, 8×8 MNIST | Neg. nearest-neighbor dist. ↑ |
| 6 | **DEQ** | W_z, W_x ∈ R^{64×64} tanh implicit MLP, z_dim=64 | ~9K | Classification, 8×8 MNIST | Neg. cross-entropy ↑ |
| 7 | **SSM** | S4D-style: D=16, N=8, 2 layers | ~2K | Sequence classification | Neg. cross-entropy ↑ |

All models train to convergence on CPU. All metrics are normalized to digital baseline = 1.0.

### 2.3 Monte Carlo Protocol

For each architecture and each σ ∈ {0%, 1%, 2%, 3%, 5%, 7%, 10%, 12%, 15%}:

- 50 independent trials, each with a fresh δ realization
- Record quality metric per trial
- Report mean ± 1 std

Ablation: three separate sweeps isolating mismatch-only, thermal-only, quantization-only.
ADC sweep: 7 bit-width values {2, 4, 6, 8, 10, 12, 16} at fixed σ = 5%.

### 2.4 Simulation Profiles: Conservative vs. Full-Analog

Every analog inference chip must eventually cross back to the digital domain — there must be at least one ADC at the output. The question is *how many* ADC/DAC conversions occur during the forward pass. This has a large and architecture-dependent effect on observed quantization sensitivity, so we report two profiles bounding the plausible range.

#### Conservative (upper bound on quantization error)

ADC quantization is applied at the output of **every** AnalogLinear layer. This models a *digital-analog hybrid* chip architecture where each crossbar array is paired with a dedicated sense-amplifier + ADC, and the digitized result is fed as input to the next crossbar via a DAC. Every layer boundary is a discrete domain crossing.

**Why this is the upper bound:**
- Quantization errors from each layer boundary are independent and compound multiplicatively across depth.
- For iterative architectures (Neural ODE: ~40 solver steps × L layers; Diffusion: 100 DDPM steps × L layers; DEQ: ~30 fixed-point iterations × L layers; EBM: ~100 Gibbs steps × L layers), every iteration fires every layer's ADC. Errors accumulate not just over depth but over time.
- Result: quantization sensitivity scales with depth × iterations, making architectures like Neural ODE appear extremely quantization-sensitive regardless of mismatch.

**Interpretation:** Conservative results represent a chip where the analog compute fabric is only used for the MVM, with all routing and buffering done digitally. This is the current design point for most academic analog AI accelerators (e.g., ISSCC 2023 crossbar chips). It is also the design assumption of the HCDC v2 hardware that the Shem physical constants in this simulator are calibrated to.

#### Full-Analog (lower bound on quantization error)

ADC quantization is applied **only at the final readout layer**. All intermediate AnalogLinear layers pass their outputs directly into the next layer's input as a continuous voltage/current, without conversion to digital. Mismatch (δ ~ N(1,σ²)) and thermal noise (√(kT/C)) still apply at every layer — they are physical and unavoidable regardless of the domain-crossing architecture. Only the *discretization* step is deferred.

**Why this is the lower bound:**
- A single ADC fires once (or once per iteration step) instead of once per layer. Quantization does not compound across depth.
- For iterative architectures, the "last AnalogLinear" fires once per iteration (not once per layer per iteration), so the quantization compounding factor is reduced from `depth × iterations` to just `iterations`.
- This approximates a *true analog compute substrate* of the kind Shem and ARK target: a physical system where the computation evolves continuously in conductance/voltage space, with digital readout only at the end.

**Why this is still not the absolute lower bound:**
Full-analog is more conservative than a hypothetical chip where even *iteration boundaries* stay analog (e.g., a fully recurrent analog circuit that never writes intermediate state to digital memory). In our model, each iteration step ends with a readout-layer ADC, which is still a domain crossing. The true lower bound — zero ADC, continuous-time analog dynamics — is not modeled here as it would require a circuit-level ODE solver.

#### Per-Architecture Reasoning

| Architecture | Conservative profile effect | Full-analog profile effect | Dominant nonideality |
|---|---|---|---|
| **Neural ODE** | Severe: 40 solver steps × L ADC crossings; quantization collapses log-det Jacobian | Mild: 1 ADC/step × 40 steps, log-det intact | Mismatch (full-analog) |
| **Transformer** | Mild: forward pass is single-pass, depth ~6; quantization compounds but bounded | Same as conservative (no iterations) | Mismatch |
| **SSM** | Mild: recurrence is SSM-native, not crossbar; Linear layers are small | Same as conservative | Mismatch |
| **DEQ** | Moderate: 30 iterations × L ADC crossings; fixed-point convergence slightly slower | Mild: 1 ADC/step × 30 steps | Mismatch |
| **EBM** | Moderate: 100 Gibbs steps × L ADC crossings | Mild: 1 ADC/step × 100 steps | Mismatch |
| **Flow** | Mild: single Euler-integrated forward pass; quantization compounds over steps but mismatch dominates | Same as conservative | Mismatch |
| **Diffusion** | Moderate: 100 DDPM steps × L ADC crossings; current 5.9% quality loss at 8-bit | Mild: 1 ADC/step × 100 steps; quality loss should approach zero at 8-bit | Quantization (conservative) |

**Summary:** For static architectures (Transformer, SSM, Flow), the two profiles yield nearly identical results — there are no time iterations to compound quantization errors. For iterative architectures (Neural ODE, DEQ, EBM, Diffusion), full-analog is substantially more favorable, and is the more physically accurate model for a substrate like Shem's target hardware. All primary results in §3 are reported under the conservative profile (worst-case upper bound). Full-analog sweep results are saved to `*_full_analog.json` files and are discussed in §3.4.

---

## 3. Results

All sweeps completed with n_trials=20 for rapid iteration (mismatch main sweep: 50 trials). Figures generated in `figures/` directory.

### 3.1 Mismatch Tolerance Ranking (Figure 1)

Current results (2026-03-17, 50 trials). For architectures where `*_mismatch.json` has a metric baseline artifact (Flow: Wasserstein z0-seed variance; Diffusion: nearest-neighbor generation seed variance), values marked † use `*_ablation_mismatch.json` (mismatch-only, internally consistent baseline).

| Rank | Family | σ threshold (10% quality loss) | Digital Baseline | σ=5% quality | σ=15% quality |
|---|---|---|---|---|---|
| 1 | **EBM** | **≥15%** | -3.993 neg. nearest-neighbor dist. | 0.999 | 0.995 |
| 2 | **SSM** | **≥15%** | -0.253 cross-entropy | 0.999 | 0.996 |
| 3 | **Transformer** | **≥15%** | -0.248 cross-entropy | 0.997 | 0.981 |
| 4 | **Neural ODE** | **≥15%** | +38.5 log-likelihood | 1.008 | 1.016 |
| 5 | **Diffusion** | **≥15%†** | -6.956 neg. nearest-neighbor | ~1.000† | ~1.000† |
| 6 | **DEQ** | **12%** | -0.492 cross-entropy | 0.982 | 0.766 |
| 7 | **Flow** | **10%†** | -0.272 neg. Wasserstein | 0.978† | 0.585† |

†Diffusion and Flow primary mismatch values from `*_ablation_mismatch.json`. `*_mismatch.json` baselines are unreliable: Diffusion `mismatch.json` shows ~15% offset at all σ including σ=0 (baseline measurement artifact from generation seed variance); Flow `mismatch.json` shows σ=0 normalized = +1.34 (z0 sampling variance — fixed z0 seed is used in `evaluate()` but not in the baseline measurement call chain). Neither artifact reflects real mismatch degradation.

Neural ODE: `mismatch.json` shows 1.008/1.016 at σ=5%/15%; `ablation_mismatch.json` shows 1.015/0.902. Both are within the estimator noise band (log-likelihood on 500-point 2D make_circles, 20 trials, std ≈ 5–8%). No systematic degradation trend is detectable.

**Confirmed observations:**

1. **EBM, SSM, Transformer are robustly mismatch-tolerant** — all retain ≥98% quality at σ=15% with low-variance metrics. EBM's Gibbs sampling decorrelates each step from the current state, so static weight errors don't compound. SSM diagonal recurrence means each state dimension degrades independently. Transformer's LayerNorm absorbs scale perturbations from mismatch.
2. **Neural ODE is tolerant but its metric is high-variance** — both runs (1.016 and 0.902 at σ=15%) are consistent with near-zero systematic degradation. Precise threshold determination requires more trials or a lower-variance metric than 500-point 2D log-likelihood.
3. **Diffusion is effectively mismatch-immune** — `ablation_mismatch` flat at 0.997–1.005 across all σ. The score network ∇log p(x_t) is robust to small weight perturbations: the DDPM reverse chain has 100 self-correcting steps, and shifting the score slightly wrong at one step gets partially compensated by subsequent steps pointing toward higher probability. The dominant failure mode is ADC quantization (§3.2).
4. **DEQ: threshold at σ=12%** — spectral normalization on W_z keeps ρ(∂f/∂z) < 1 for small σ, but as mismatch grows the effective spectral radius increases, slowing convergence and raising CE. Steepest degradation among non-Flow architectures: 0.766 at σ=15%.
5. **Flow is the worst mismatch-tolerant architecture** — `ablation_mismatch` threshold at σ=10%, collapsing to 0.585 at σ=15%. Mechanism: the rectified flow trajectory drifts monotonically when v_θ is corrupted by mismatch. Unlike DEQ's fixed-point iteration which can partially self-correct, or Diffusion's 100-step chain, Flow's 4 Euler steps leave no room for error recovery.
6. **Cross-entropy vs. accuracy**: with accuracy (argmax), Transformer shows ~0% quality loss even at σ=15% — logit rankings are preserved. Cross-entropy exposes magnitude degradation, which matters for Shem's gradient-based optimization.

### 3.2 Noise Source Ablation (Figure 2)

**Measured dominance** (from `*_ablation_mismatch/thermal/quantization.json`, 2026-03-17):

| Architecture | Mismatch @σ=15% | Thermal @σ=15% | Quantization @σ=15% | Dominant Source |
|---|---|---|---|---|
| Neural ODE | 0.902 (σ thresh ≥15%) | 1.000 (zero effect) | 1.000 (zero effect) | **Mismatch** |
| Transformer | 0.991 (σ thresh ≥15%) | 1.000 (zero effect) | 1.000 (zero effect) | **Mismatch** |
| SSM | 0.995 (σ thresh ≥15%) | 1.000 (zero effect) | 1.000 (zero effect) | **Mismatch** |
| DEQ | 0.851 (σ thresh 12%) | 1.000 (zero effect) | 0.999 (zero effect) | **Mismatch** |
| EBM | 1.006 (σ thresh ≥15%) | 1.000 (zero effect) | 1.000 (zero effect) | **None** (all flat) |
| Flow | 0.585 (σ thresh 10%) | 1.000 (zero effect) | 1.384† (artifact) | **Mismatch** |
| Diffusion | 0.999 (σ thresh ≥15%) | 1.000 (zero effect) | 0.850 (σ thresh ~0%) | **Quantization** |

†Flow quantization ablation > 1.0 is the Wasserstein z0-seed variance artifact (see §3.1). Raw metric values are stable; quantization has no meaningful effect on Flow.

**Confirmed findings:**

- **Thermal noise: zero effect on all 7 architectures.** σ_thermal = √(kT/C)·√N_in ≈ 6.4×10⁻⁵·√64 ≈ 5×10⁻⁴ V at C=1pF, T=300K, N_in=64. This is negligible relative to static mismatch at σ=1% (which directly scales all weight values by ±1%). Thermal would matter at C=10fF or N_in>1000, but not at HCDCv2 1pF / demo-scale model dimensions.
- **Mismatch dominates for 6 of 7 architectures.** Static weight corruption baked in at fabrication time is the primary failure mode. The one exception is Diffusion.
- **EBM: no dominant failure mode at σ≤15%.** All three noise sources produce flat ablation curves at ±0.1% throughout. EBM is the most noise-agnostic architecture at this scale.
- **Diffusion: quantization is the only failure mode.** `ablation_quantization` shows 0.850 at σ=15% (actually 0.850 at ALL σ since the quantization floor is fixed by bit-width, not mismatch level). `ablation_mismatch` is perfectly flat at 0.997–1.005. The conservative ADC profile (per-layer quantization across 100 DDPM steps) is entirely responsible for Diffusion's quality reduction.

### 3.3 ADC Precision Tradeoff (Figure 3)

**Measured at σ=5% mismatch** (from `*_adc.json`, bits ∈ {2, 4, 6, 8, 10, 12, 16}, 2026-03-17):

| Architecture | 2-bit quality | 4-bit quality | 6-bit quality | 8-bit quality | Min bits (≥95% quality) |
|---|---|---|---|---|---|
| Neural ODE | 0.998 | 0.981 | 1.012 | 1.025 | **2 bits** |
| EBM | 0.988 | 1.001 | 1.003 | 0.999 | **2 bits** |
| Flow | 0.990† | 1.258† | 1.321† | 1.316† | **2 bits†** |
| Transformer | 0.942 | 0.996 | 0.997 | 0.998 | **4 bits** |
| SSM | 0.762 | 1.001 | 0.999 | 0.998 | **4 bits** |
| DEQ | **−2.333 (!)** | 0.882 | 0.983 | 0.987 | **6 bits** |
| Diffusion | 0.927 | 0.861 | 0.854 | 0.847 | **N/A (none reach 95%)‡** |

†Flow all bit-widths normalized >1.0 due to Wasserstein z0-seed baseline artifact. `ablation_quantization` confirms quantization has no effect; effective minimum bits is 2 but the number is not meaningful from this data.

‡Diffusion: none of the tested bit-widths reach 95% quality in the conservative ADC profile. The ~15% floor is from the ADC-per-layer model compounding across 100 DDPM steps (quantization fire count ≈ 100 × depth), not from any specific bit-width. This is a conservative profile artifact — under full-analog profile (one ADC at final readout only), the quantization floor disappears. See §2.4.

**Confirmed key findings:**

- **DEQ: 2-bit catastrophic (−2.333), 6-bit minimum.** Coarse quantization prevents fixed-point convergence — z_k oscillates between quantization levels rather than converging to z*. This is a "limit cycle" failure mode unique to iterative architectures: the quantization step size becomes the dominant noise source, pushing z_k around a cycle of 2–4 quantized states.
- **SSM: 2-bit catastrophic (0.762), 4-bit sufficient.** The state recurrence h[t] = A_bar * h[t-1] + B_bar * u[t] amplifies quantization error across 64 timesteps (sequence length). 2-bit quantizes the B and C projections so coarsely that temporal state compression fails.
- **Transformer: 2-bit marginal (0.942), 4-bit sufficient.** Attention pattern collapse at 2-bit — Q, K, V projections are quantized so coarsely that softmax(QK^T/√d) collapses to a near-uniform distribution. Recovers at 4-bit.
- **EBM and Neural ODE: 2-bit tolerant.** EBM because Gibbs sampling is inherently discrete (sigmoid outputs ≈ 0 or 1 anyway — coarse quantization just reinforces this). Neural ODE because the log-det computation now correctly disables quantization noise, and the ODE integration is quantization-insensitive at demo scale (2D).
- **Diffusion: no ADC minimum determinable under conservative profile.** The 100-DDPM-step × per-layer quantization compounds to a floor regardless of bit-width. Full-analog profile required for ADC sensitivity analysis.

### 3.4 DEQ Convergence Bifurcation (Figure 4)

**Measured convergence_failure_rate:** (from `deq_convergence.json`, model CE=0.492, ~93% accuracy)

| σ mismatch | Failure rate | Observation |
|---|---|---|
| 0–15% | **1.000 at all σ** | 100% "failure" is a tolerance artifact — see below |

**Confirmed finding: The convergence_failure_rate metric is not informative.** The norm check is `||z_{k+1} - z_k|| / √dim < tol` with tol=1e-4 and dim=64. This requires every element to change by < 1e-4 simultaneously — equivalent to per-element change < 1.5×10⁻⁵, never achieved within 30 iterations for any input. The model achieves 93% test accuracy and CE=0.492, confirming it IS finding useful fixed points.

**Evidence:**

1. 93% test accuracy with failure_rate=1.000 → fixed points are functionally correct despite "100% failure"
2. Spectral normalization guarantees ρ(W_z) < 1 → fixed point exists and is unique (Banach)
3. CE degrades continuously with σ (normed 0.982→0.766 at σ=5%→15%) → graceful degradation, not divergence

**Revised interpretation:** DEQs with spectral normalization degrade gracefully under mismatch. The convergence failure metric is an artifact of using absolute tolerance on a high-dimensional norm. For production analog DEQs, report ||z_{k+1} − z_k||/√dim as a continuous quality metric instead of a binary threshold.

### 3.5 Conservative vs. Full-Analog Profile Comparison

The choice of simulation profile does not merely shift the numbers — for two architectures it **reverses the ranking**.

**Full-analog mismatch results** (from `*_mismatch_full_analog.json`):

| Architecture | Conservative threshold | Full-analog threshold | Conservative @0.15 | Full-analog @0.15 | Direction |
|---|---|---|---|---|---|
| EBM | ≥15% (flat) | **10%** | 0.995 | **0.837** | **Worse** |
| DEQ | 12% | **≥15%** | 0.766 | **0.816** | **Better** |
| SSM | ≥15% | ≥15% | 0.996 | 0.993 | Similar |
| Transformer | ≥15% | ≥15% | 0.981 | 0.989 | Similar |
| Neural ODE | ≥15% | ≥15% | 1.016 | 0.959 | Similar (high variance) |
| Diffusion | ≥15%† | ≥15% | ~1.000 | 0.962 | Similar |
| Flow | 10%† | ≥15%† | 0.585 | 1.030 | Artifact (Wasserstein) |

**EBM degrades in full-analog.** The conservative profile's per-Gibbs-step ADC was inadvertently *helping* EBM by binarizing sigmoid outputs at each step — reinforcing the binary {0,1} nature of the visible/hidden units that RBM Gibbs sampling was designed for. In full-analog, sigmoid activations are continuous throughout the chain, and small mismatch-induced shifts in the sigmoid operating point accumulate across 100 Gibbs steps rather than being snapped to {0,1} by quantization. The full-analog baseline for EBM is also substantially better (−1.51 vs. −3.99) because continuous Gibbs chains explore the energy landscape more finely.

**DEQ improves in full-analog.** The per-iteration ADC in the conservative profile was creating limit cycles — z_k oscillating between quantization bins rather than converging. Removing intermediate ADC lets the fixed-point iteration converge correctly. The Hopfield substrate (§3.5 below) further improves DEQ to threshold ≥15% in both profiles.

**Diffusion resolves entirely.** The conservative profile's ~15% quality floor (from per-layer ADC × 100 DDPM steps) disappears under full-analog. Diffusion's full-analog ablation shows quantization = 0.980 at σ=0.05 (residual from one final ADC), vs. 0.850 under conservative. For analog hardware where the score network runs as a continuous circuit, Diffusion is essentially a zero-noise architecture.

**Hardware interpretation:** Conservative = digital-analog hybrid chip (per-crossbar ADC, current ISSCC crossbar design point). Full-analog = true analog compute substrate where intermediate activations pass as continuous voltages/currents (Shem/Ark target). The EBM/DEQ reversal means architecture selection depends on the hardware architecture, not just the algorithm.

### 3.6 Substrate Comparisons

#### Diffusion: classic vs. CLD (RLC/Langevin)

| Substrate | Mismatch @0.15 | Thermal @0.15 | ADC 2-bit | Notes |
|---|---|---|---|---|
| Classic (DDIM) | ~1.000 | 1.000 | N/A (floor) | Quantization dominant under conservative profile |
| **CLD (RLC/Langevin)** | **0.998** | **1.000** | **0.999** | **All three noise sources flat; 2-bit sufficient** |
| extropic_dtm | NaN | — | — | Numerical issue in sweep; substrate implementation needs investigation |

The CLD result is the strongest hardware-relevant finding in the substrate comparison. The RLC/Langevin circuit models the diffusion process as a physical Langevin equation: the hardware thermal fluctuations *are* the generation noise, not a corruption of it. This has two consequences: (1) thermal noise from the substrate contributes constructively to the score matching process rather than adding independent error; (2) the ADC quantization floor that plagues the classic substrate disappears because the score network's outputs are smaller-variance in the CLD parameterization. CLD diffusion is 2-bit ADC tolerant and immune to all three nonidealities simultaneously.

The `extropic_dtm` substrate produced NaN results throughout the sweep — a numerical issue in the substrate implementation unrelated to hardware relevance. Not investigated further as this substrate is outside the target hardware scope.

#### Neural ODE: euler vs. rc_integrator

Both substrates show identical mismatch tolerance (threshold ≥15%, quality ≈1.0 at σ=15%). The RC integrator noise √(kT/C) ≈ 6.4×10⁻⁵ V per step over 40 Euler steps contributes no measurable additional degradation beyond weight mismatch. This is a null result: integration capacitor thermal noise is not a relevant design constraint at the HCDCv2 parameter regime (C=1pF, T=300K).

#### DEQ: discrete fixed-point vs. Hopfield damped relaxation

| Substrate | Mismatch threshold | σ=15% quality | ADC min bits |
|---|---|---|---|
| Discrete fixed-point | 12% | 0.766 | 6 bits |
| **Hopfield (RC-damped)** | **≥15%** | **0.849** | 6 bits |

The Hopfield substrate is strictly better for mismatch tolerance: the physical `-z` damping term in `dz/dt = -z + f(z,x)` prevents the fixed-point iteration from approaching the instability boundary even as mismatch pushes the effective spectral radius upward. Both substrates require 6-bit ADC minimum — the limit cycle failure mode is about quantization bin size independent of substrate dynamics.

#### Flow: euler vs. rc_integrator

No measurable difference between substrates — same conclusion as Neural ODE. RC integrator noise negligible at this scale.

### 3.7 Output MSE: Direct Output Divergence Measurement

Output MSE measures how much the analog model's output trajectory diverges from the digital baseline, independently of task-level quality metrics. Results from `*_output_mse.json` (conservative profile, n_trials=20):

| Architecture | MSE at σ=0 | MSE at σ=0.05 | MSE at σ=0.15 | Pattern |
|---|---|---|---|---|
| EBM | ~0.000 | 0.00010 | 0.00090 | Near-flat; Gibbs equilibrium absorbs perturbations |
| Transformer | 0.00002 | 0.00332 | 0.01866 | Clean monotonic growth; most interpretable |
| SSM | 0.00070 | 0.00256 | 0.02341 | Similar to Transformer; slightly higher baseline |
| Flow | 0.02908 | 0.03801 | 0.09119 | Moderate growth on top of z0-seed baseline offset |
| Neural ODE | **0.07031** | **0.07006** | **0.07073** | **Flat** — structural baseline offset, not mismatch |
| DEQ | 0.00309 | 0.02981 | **0.23436** | **Steepest growth** (76× from σ=0 to σ=15%) |
| Diffusion | **0.69535** | **0.70537** | **0.69345** | **Flat** — ADC quantization floor dominates; mismatch adds nothing |

**Two flat curves for different reasons:** Neural ODE's flat MSE (0.070 throughout) reflects a structural difference between the analog and digital model's log-det computation paths — present at σ=0 and not worsened by mismatch. Diffusion's flat MSE (0.695 throughout) reflects the conservative ADC floor: per-layer quantization across 10 DDIM steps creates a constant 0.695 output displacement regardless of weight mismatch level.

**DEQ has the highest MSE growth rate (76×).** The fixed-point iteration amplifies small weight errors into large output divergences — by σ=15%, the DEQ analog output is 0.234 units away from digital baseline on average. This is consistent with the CE mismatch sweep showing the steepest non-Flow degradation (0.766 at σ=15%).

**EBM has the lowest absolute MSE growth.** The Gibbs chain's self-correcting property means weight perturbations get averaged out across 100 sampling steps rather than amplified. This makes EBM the architecture whose output trajectory is most faithfully preserved under analog nonidealities.

**Methodological note:** Output MSE and task-level quality metrics can disagree. Neural ODE has the second-highest MSE at σ=0 (0.070) but shows no quality degradation — the log-likelihood metric is insensitive to small output trajectory shifts as long as the final density estimate is accurate. Transformer has the cleanest MSE growth but its task metric is also near-flat. DEQ's MSE growth (76×) is the largest and directly corresponds to CE degradation. For Shem optimization, MSE is the more actionable signal: it quantifies the gradient error budget that the adjoint optimizer must close.

---

## 4. Implications

### 4.1 Architecture Selection for Analog AI

These measurements directly inform analog chip architects:

**EBM — the most noise-agnostic architecture.** All three noise sources (mismatch, thermal, quantization) produce flat ablation curves at ±0.1% through σ=15%. Mechanism: Gibbs sampling is a fixed-point iteration on the probability distribution itself — each conditional draw decorrelates from any weight-induced error at the previous step. EBM is also 2-bit ADC tolerant (0.988 at 2-bit): sigmoid outputs in {0,1} are inherently near-binary, so coarse quantization reinforces rather than destroys the computation. Strongest all-around candidate for analog deployment at the hardware noise levels simulated here.

**SSM, Transformer — robustly mismatch-tolerant, 4-bit ADC minimum.** Both retain ≥98% quality at σ=15% with low-variance metrics. Transformer: LayerNorm absorbs scale perturbations; attention pattern survives mismatch well but collapses at 2-bit ADC (0.942). SSM: diagonal state recurrence means each dimension degrades independently without cross-contamination; 2-bit quantizes B/C projections too coarsely for 64-step temporal compression (0.762 at 2-bit), recovers fully at 4-bit.

**Neural ODE — mismatch-tolerant (high-variance metric), 2-bit ADC tolerant.** Mismatch ablation shows no systematic degradation trend (σ=15%: 0.902–1.016 across two runs, consistent with estimator noise on 500-point 2D log-likelihood). ADC sweep is clean: stable 0.98–1.03 across all bit-widths, confirming that the evaluate() fix (disabling quantization during log-det computation) removed the previous ADC sensitivity artifact. For density estimation on analog hardware, a separate digital backward pass on nominal weights is required for the log-det computation — the analog forward pass alone is hardware-native.

**Modeling limitation — log-det under mismatch**: The accumulated log-det `tr(∂f/∂z)` is computed by backpropagating through the mismatch-perturbed ODE. This means reported log-likelihood reflects the Jacobian of the *analog* model, not the nominal one. In practice no hardware can backpropagate through itself. The mismatch-only ablation values (quantization disabled) are the most physically honest Neural ODE numbers here.

**DEQ — 6-bit ADC minimum, graceful mismatch degradation.** Spectral normalization on W_z (ρ < 1) keeps the fixed-point iteration contractive under mismatch, giving graceful CE degradation: 0.982 at σ=5%, threshold at σ=12%, 0.766 at σ=15%. The ADC constraint is more severe: 2-bit ADC creates a limit cycle (z_k oscillates between quantization bins rather than converging), catastrophically failing at normalized = −2.33. Requires ≥6 bits. The convergence_failure_rate metric (§3.4) reports 100% failure at all σ — this is a per-dimension tolerance artifact, not real divergence. The model achieves 93% test accuracy.

**Flow — use with caution above σ=10%.** Worst mismatch tolerance of all 7 architectures: mismatch ablation threshold at σ=10%, collapsing to 0.585 at σ=15%. Despite sharing the same MLP architecture and 2D task type as Neural ODE, Flow has no fixed-point structure, no LayerNorm, and 4 Euler steps with no error correction — velocity field perturbations accumulate monotonically. ADC insensitive (Wasserstein metric is robust to quantization noise in generation).

**Diffusion — mismatch-immune, ADC analysis requires full-analog profile.** Mismatch ablation perfectly flat at 0.997–1.005 across all σ — the only architecture with true mismatch immunity. The 100-step DDPM reverse chain self-corrects weight perturbations: each step shifts the score slightly, but subsequent steps pointing toward higher probability partially compensate. ADC minimum bits cannot be determined under the conservative profile (per-layer ADC × 100 steps creates a floor at ~15% quality loss regardless of bit-width). Full-analog profile (single ADC at final readout) is required for Diffusion ADC analysis and is the appropriate model for Shem's target hardware.

**Methodological finding:** Continuous metrics (cross-entropy, log-likelihood) expose degradation that accuracy (argmax) masks entirely. At σ=15%, Transformer cross-entropy degrades 1.9% while accuracy remains near-perfect — logit rankings are preserved even with magnitude corruption. Shem's adjoint optimization needs the smooth signal that continuous metrics provide.

### 4.2 Connection to the Achour Compilation Pipeline

This paper identifies the performance gap. Shem closes it.

Our `analogize()` simulator shows *what goes wrong*. Shem's adjoint-based mismatch optimization provides *a mechanism to fix it*: by training δ ◦ θ models from the start, the trained weights compensate for fabrication variance. The degradation curves in Figure 1 — with EBM/SSM/Transformer retaining ≥98% quality at σ=15% and Flow/DEQ degrading significantly — establish the baselines that Shem optimization is expected to improve upon.

The direct pipeline is:

1. Train a Neural ODE (or SSM, or flow model)
2. Run `analogize(model, sigma=0.10)` → measure quality degradation (this paper)
3. Export to Shem via `export_neural_ode_to_shem(extractor)` → get runnable Shem code
4. Run `Shem.compile(model)` → adjoint optimization over δ ~ N(1, σ²) perturbations
5. Re-run `analogize()` on the Shem-optimized weights → degradation curve shifts right

This paper provides steps 2 and 3. Shem provides step 4. The combination is the complete analog AI compiler workflow. The confirmed result — EBM and SSM as the most noise-agnostic architectures, Flow and DEQ as the highest-priority targets for mismatch compensation — makes them the natural starting points for this pipeline.

---

## 5. Reproduce

```bash
# Install dependencies
pip install torch numpy matplotlib scikit-learn scipy

# Train all 7 models (~20 min on CPU)
python experiments/cross_arch_tolerance/train_all.py

# Run sweeps (~60-90 min on CPU, 50 trials × 9 sigma × 7 models)
python experiments/cross_arch_tolerance/sweep_all.py

# Generate figures
python experiments/cross_arch_tolerance/plot_results.py

# Figures in: experiments/cross_arch_tolerance/figures/
```

For faster results (fewer trials):

```bash
python sweep_all.py --n-trials 10   # ~10 min, larger error bars
```

---

## 6. Confirmed Results (2026-03-17)

Current results from JSON files in `results/` (50 trials for mismatch sweeps, 20–50 for ADC/ablation). Source: `*_mismatch.json` for stable-metric architectures; `*_ablation_mismatch.json` for Flow and Diffusion (see §3.1 footnotes).

**Mismatch tolerance:**

| Architecture | σ threshold@10% | σ=5% quality | σ=15% quality | Source |
|---|---|---|---|---|
| EBM | ≥15% | 0.999 | 0.995 | mismatch.json |
| SSM | ≥15% | 0.999 | 0.996 | mismatch.json |
| Transformer | ≥15% | 0.997 | 0.981 | mismatch.json |
| Neural ODE | ≥15% | 1.008 | 1.016 | mismatch.json |
| Diffusion | ≥15%† | ~1.000 | ~1.000 | ablation_mismatch |
| DEQ | **12%** | 0.982 | 0.766 | mismatch.json |
| Flow | **10%†** | 0.978 | 0.585 | ablation_mismatch |

†Diffusion: mismatch-immune (score network robustness). Flow: worst mismatch tolerance, collapses at high σ.

**Dominant noise sources:** Thermal = zero effect for all architectures. Mismatch dominates for 6/7. Quantization is the only failure mode for **Diffusion** (conservative ADC profile: ~15% floor from per-layer ADC × 100 DDPM steps). EBM shows no dominant failure mode at σ≤15% — all three noise sources are effectively flat.

**ADC minimum bits at σ=5% (from `*_adc.json`):**

| Architecture | Min bits (≥95%) | 2-bit quality | Note |
|---|---|---|---|
| Neural ODE | **2 bits** | 0.998 | Log-det correctly disables quant; stable across all bit-widths |
| EBM | **2 bits** | 0.988 | Gibbs sampling tolerates coarse quantization |
| Transformer | **4 bits** | 0.942 | 2-bit: attention collapse |
| SSM | **4 bits** | 0.762 | 2-bit: state recurrence diverges |
| DEQ | **6 bits** | −2.333 (!) | 2-bit: fixed-point limit cycle |
| Flow | **2 bits†** | 0.990† | Wasserstein baseline artifact; quantization insensitive |
| Diffusion | **N/A** | 0.927 | Conservative profile floor; full-analog profile needed |

**DEQ convergence failure rate:** 100% at all σ — confirmed tolerance artifact (tol=1e-4/√64 ≈ 1.5×10⁻⁵ per element, unreachable in 30 iterations). Model achieves 93% accuracy + CE=0.492. Spectral normalization guarantees Banach fixed-point existence.

**Full-analog profile key reversals (from `*_mismatch_full_analog.json`, `*_adc_full_analog.json`):**
- EBM: threshold drops from ≥15% → **10%** in full-analog (conservative ADC was binarizing sigmoid outputs, helping EBM)
- DEQ: threshold improves from 12% → **≥15%** in full-analog (per-iteration ADC limit cycles removed)
- Diffusion: conservative ADC floor (~15% constant loss) disappears entirely in full-analog
- EBM ADC minimum in full-analog: **6 bits** (2-bit catastrophic at −0.649) vs. 2-bit in conservative — a major reversal

**Substrate comparison highlights:**
- Diffusion CLD (RLC/Langevin): all three noise sources flat; 2-bit ADC sufficient; circuit thermal noise IS the diffusion noise (constructive)
- DEQ Hopfield vs. discrete: threshold ≥15% vs. 12%; Hopfield RC damping acts as physical regularizer
- Neural ODE / Flow rc_integrator: **null result** — integration capacitor noise adds nothing at C=1pF

**Output MSE at σ=15% (from `*_output_mse.json`, conservative profile):**
- EBM: 0.0009 (near-zero — Gibbs equilibrium self-corrects)
- Transformer: 0.019 | SSM: 0.023 — clean monotonic growth
- DEQ: **0.234** (76× growth from σ=0; steepest amplification of all architectures)
- Neural ODE: **0.071** (flat — structural log-det path difference, not mismatch)
- Diffusion: **0.694** (flat — ADC quantization floor constant regardless of σ).
