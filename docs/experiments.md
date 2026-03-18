# Experiments

Cross-architecture analog tolerance sweep — 7 neural network families, 50 trials, conservative ADC profile.

Full methodology and errata: [`experiments/cross_arch_tolerance/TECHNICAL_NOTE.md`](../experiments/cross_arch_tolerance/TECHNICAL_NOTE.md)

---

## The 7 models

| # | Family | Architecture | Params | Task | Metric |
|---|---|---|---|---|---|
| 1 | **Neural ODE** | f_θ: [3→20→20→2] tanh MLP | ~1K | CNF density, make_circles | Log-likelihood ↑ |
| 2 | **Transformer** | 2-layer, dim=24, 2 heads, ReLU FFN | ~10K | Sequence classification | Neg. cross-entropy ↑ |
| 3 | **Diffusion** | 3-layer score MLP, T=100 DDPM, 8×8 MNIST | ~103K | Generation | Neg. nearest-neighbor dist. ↑ |
| 4 | **Flow** | v_θ: [3→64→64→2] tanh MLP | ~4K | Rectified flow, make_moons | Neg. Wasserstein ↑ |
| 5 | **EBM** | RBM: visible=64, hidden=32 | ~4K | Free generation, 8×8 MNIST | Neg. nearest-neighbor dist. ↑ |
| 6 | **DEQ** | W_z, W_x ∈ R^{64×64} tanh implicit MLP | ~9K | Classification, 8×8 MNIST | Neg. cross-entropy ↑ |
| 7 | **SSM** | S4D-style: D=16, N=8, 2 layers | ~2K | Sequence classification | Neg. cross-entropy ↑ |

All models train to convergence on CPU. All quality metrics are normalized to digital baseline = 1.0.

---

## Run the sweeps

```bash
cd experiments/cross_arch_tolerance
python train_all.py          # ~20 min CPU
python sweep_all.py          # ~90 min CPU, 50 trials, all 7 models
python plot_results.py       # regenerate figures
```

Single architecture, faster:
```bash
python sweep_all.py --only neural_ode --n-trials 10
python sweep_all.py --only deq --analog-substrate hopfield
python sweep_all.py --only diffusion --analog-substrate cld
```

Key flags: `--only <arch>`, `--n-trials N`, `--analog-substrate <name>`, `--n-adc-bits N`, `--profile conservative|full_analog`

---

## Results (2026-03-17, 50 trials)

### Mismatch tolerance

| Architecture | σ threshold @ 10% loss | σ=5% quality | σ=15% quality | Dominant noise |
|---|---|---|---|---|
| EBM | **≥15%** | 0.999 | 0.995 | None (all flat) |
| SSM | **≥15%** | 0.999 | 0.996 | Mismatch |
| Transformer | **≥15%** | 0.997 | 0.981 | Mismatch |
| Neural ODE | **≥15%** | 1.008 | 1.016 | Mismatch (high-variance metric) |
| Diffusion | **≥15%†** | ~1.000 | ~1.000 | Quantization (conservative profile) |
| DEQ | **12%** | 0.982 | 0.766 | Mismatch |
| Flow | **10%†** | 0.978 | 0.585 | Mismatch |

†Diffusion and Flow primary values from `*_ablation_mismatch.json` — the main `*_mismatch.json` baseline has a measurement artifact. See TECHNICAL_NOTE §3.1.

### Noise source attribution

Thermal noise: zero effect on all 7 architectures. σ_thermal ≈ 5×10⁻⁴ V at C=1pF, N_in=64 — negligible vs. 1% weight mismatch.

Mismatch dominates for 6/7. Diffusion is the exception — its score network is mismatch-immune (100-step DDPM chain self-corrects). Diffusion's only failure mode is ADC quantization under the conservative (per-layer) profile.

### ADC minimum bit-width (at σ=5%)

| Architecture | Min bits (≥95% quality) | 2-bit quality | Why |
|---|---|---|---|
| Neural ODE | **2 bits** | 0.998 | Log-det computation disables quantization; ODE integration insensitive at demo scale |
| EBM | **2 bits** | 0.988 | Gibbs sigmoid outputs near-binary; coarse quantization reinforces rather than destroys |
| Flow | **2 bits†** | 0.990† | Wasserstein baseline artifact; quantization has no real effect |
| Transformer | **4 bits** | 0.942 | 2-bit: Q/K/V projections quantized too coarsely → attention pattern collapses |
| SSM | **4 bits** | 0.762 | 2-bit: B/C projections fail temporal state compression across 64 timesteps |
| DEQ | **6 bits** | −2.333 (!) | 2-bit: fixed-point limit cycle — z_k oscillates between quantization bins |
| Diffusion | **N/A** | 0.927 | Conservative ADC floor from 100-step × per-layer compounding. Use full-analog profile. |

### Conservative vs. full-analog profile

The two profiles don't just shift numbers — for EBM and DEQ they reverse the ranking:

| Architecture | Conservative threshold | Full-analog threshold | Direction |
|---|---|---|---|
| EBM | ≥15% | **10%** | **Worse** — conservative ADC was binarizing sigmoid outputs, helping EBM |
| DEQ | 12% | **≥15%** | **Better** — per-iteration ADC was causing limit cycles |
| Diffusion | ≥15%† | ≥15% | Better — conservative ADC floor disappears entirely |
| Others | ≥15% or similar | ≥15% or similar | Minimal change (no iterations to compound) |

**Hardware interpretation:** Conservative = digital-analog hybrid chip (current ISSCC design point). Full-analog = true continuous analog substrate (Shem/Ark target).

### DEQ convergence note

The `convergence_failure_rate` metric reports 100% failure at all σ — this is a tolerance artifact (tol=1e-4 / √64 ≈ 1.5×10⁻⁵ per element, unreachable in 30 iterations). The model achieves 93% accuracy and CE=0.492. Spectral normalization on W_z guarantees a unique fixed point exists (Banach). See TECHNICAL_NOTE §3.4.

---

## Status by architecture

| Architecture | Trained | Sweeps done | Notes |
|---|---|---|---|
| Neural ODE | ✓ | ✓ | High-variance log-lik metric; more trials needed for precise threshold |
| Transformer | ✓ | ✓ | Clean results |
| Diffusion | ✓ | ✓ | Full-analog profile needed for ADC analysis |
| Flow | ✓ | ✓ | Wasserstein z0-seed artifact in main sweep; use ablation values |
| EBM | ✓ | ✓ | Clean results; conservative/full-analog reversal is the key finding |
| DEQ | ✓ | ✓ | 6-bit minimum and 12% threshold confirmed |
| SSM | ✓ | ✓ | Clean results |
| FLUX.1 extractor | — | — | FLUXExtractor in `extractors/flow.py` — flow_straightness/Lipschitz TODO |
| Mamba extractor | — | — | Requires `[ssm]` extra; extractor in `extractors/ssm.py` |

---

## Output files

All results saved to `experiments/cross_arch_tolerance/results/`:

| File pattern | Contents |
|---|---|
| `*_mismatch.json` | Main σ sweep (50 trials) |
| `*_ablation_mismatch/thermal/quantization.json` | Isolated noise source sweeps |
| `*_adc.json` | ADC bit-width sweep at σ=5% |
| `*_mismatch_full_analog.json` | Full-analog profile σ sweep |
| `*_adc_full_analog.json` | Full-analog ADC sweep |
| `deq_convergence.json` | DEQ fixed-point convergence stats |
| `*_output_mse.json` | Direct output MSE vs. σ |
