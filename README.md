# neuro-analog

Analog hardware — crossbar arrays, RC integrators, differential-pair activations — can run neural network inference at orders-of-magnitude lower energy than digital, but it introduces unavoidable physical nonidealities: fabrication mismatch bakes static weight errors into every device, thermal noise corrupts every readout, and ADC quantization discretizes every layer boundary. Which neural architectures actually survive these conditions, and at what noise level does each one break?

This framework answers that empirically. It instruments any PyTorch model with physics-grounded analog nonidealities, measures how task-level quality degrades as noise increases, and extracts the model into a typed IR classifying each operation as analog-native, digital-required, or hybrid — then exports compatible models to [Ark](https://arxiv.org/abs/2309.08774) (Wang & Achour, ASPLOS '24) as runnable `BaseAnalogCkt` subclasses. The noise physics follows the mismatch/SDE/discrete-optimization model described in Wang & Achour (arXiv:2411.03557).

---

## Install

```bash
git clone https://github.com/apumutyala/neuro-analog
cd neuro-analog
pip install -e ".[dev]"
```

Optional extras: `[ssm]` for Mamba extraction, `[jax]` for Diffrax evaluation, `[full]` for everything.

**Ark export** (`ark_bridge/`, `evaluate_in_ark.py`) requires Ark. Install order matters — torch before JAX:

```bash
apt-get install -y graphviz
git clone https://github.com/WangYuNeng/Ark.git
cd Ark && pip install -r requirement_torch.txt && pip install -r requirement.txt && pip install -e .
```

---

## Quick start

```python
from neuro_analog.simulator import analogize, mismatch_sweep

analog_model = analogize(model, sigma_mismatch=0.05, n_adc_bits=8)
result = mismatch_sweep(model, eval_fn, sigma_values=[0.0, 0.05, 0.10, 0.15], n_trials=50)
print(result.degradation_threshold(max_relative_loss=0.10))  # σ at 90% quality
```

`analogize()` replaces `nn.Linear`, `nn.Conv*`, `nn.MultiheadAttention`, and analog-implementable activations with physics-grounded equivalents. Everything without an efficient analog circuit (LayerNorm, Softmax, dynamic Q·Kᵀ) stays digital.

See [`examples/01_quickstart.py`](examples/01_quickstart.py) for a full walkthrough.

---

## Experiment: Analog Tolerance across Neural Architecture Families

Seven neural network families trained on small-scale tasks, swept over conductance mismatch σ with 50–200 Monte Carlo trials per point (DEQ: 200 trials on a dense σ grid; others: 50 trials). Three noise sources ablated independently. ADC bit-width swept separately. Two simulation profiles: conservative (ADC at every layer boundary) and full-analog (ADC at final readout only).

The goal is to establish empirical baselines for which architectures are worth compiling to analog hardware and where the failure points are before committing to a chip design or running analog-aware adjoint optimization within Ark.

**Disclaimer:** These are simulation results on demo-scale models (1K–103K parameters) trained near capacity on synthetic and small benchmark tasks. They represent a worst-case bound for each architecture family — production-scale overparameterized models will degrade more gracefully. This is not hardware validation. Several physical effects are not simulated (see Nonidealities section). Results should be interpreted as characterizing the analog sensitivity of the architecture's computational structure, not a specific chip.

> **Work in progress — results will be updated.** The models in this experiment are demo-scale and several are still being fine-tuned for accuracy. The tolerance metric measures *relative* degradation (analog vs. digital baseline for the same model), so architectural rankings are meaningful even when the underlying models are not fully optimized. However, three specific results have known data quality issues and should be treated as preliminary until the planned re-run is complete:
>
> - **Diffusion (DDIM):** The 3-layer MLP score network does not generate visually recognizable 8×8 MNIST digits even at σ=0. The structural quantization-accumulation finding (quality floor across all DDIM steps) is architecturally real, but the absolute quality numbers are inflated by the model's limited capacity. Fig 5 reflects this — the σ=0 baseline for Diffusion is already poor. Planned fix: larger score network or U-Net backbone.
>
> - **EBM:** The current checkpoint's sweep baseline and the local `evaluate()` differ by ~6%, attributable to Gibbs chain stochasticity during evaluation. The binarized 8×8 test sample used in Fig 5 also happens to be nearly empty (3 non-zero pixels after downsampling and binarization), making the reconstruction in Fig 5 unrepresentative. Planned fix: fix the Fig 5 test sample seed and tighten evaluation by averaging over multiple Gibbs chains.
>
> - **DEQ:** The sweep baseline stored in the result files (−0.5673) does not match the current checkpoint's `evaluate()` output (−0.4447) — a 21% discrepancy on a fully deterministic metric — indicating the checkpoint on disk was updated after the RunPod sweep was run. The σ≈11% threshold finding is internally consistent within the sweep data but was computed against a different model version than what is currently saved. Planned fix: re-run the full DEQ sweep from the current checkpoint.
>
> Neural ODE, SSM, Transformer, and Flow baselines are stable and consistent (stored sweep baselines match current `evaluate()` output within 1%).

### Results (conservative profile)

| Architecture | σ threshold @ 10% loss | Dominant noise | Min ADC bits | Quality at max σ tested |
|---|---|---|---|---|
| Neural ODE | ≥ 15% | negligible | 2 | ≥ 0.90 at σ=15% |
| S4D (diagonal SSM) | ≥ 15% | negligible | 4 | ≥ 0.90 at σ=15% |
| EBM | ≥ 15% | negligible | 4 | ≥ 0.90 at σ=15% |
| Flow | ~15–20% ² | mismatch | 4 | 0.905 ± 0.096 at σ=15% |
| Transformer | ≥ 15% | negligible | 4 | ≥ 0.90 at σ=15% |
| DEQ | **~11%** | mismatch | 6 ³ | 0.854 ± 0.088 at σ=15% |
| Diffusion | — ¹ | quantization | — ¹ | — ¹ |

¹ Diffusion never reaches the 90% quality threshold at any σ tested. The score network's output is quantized at every DDIM step and errors accumulate across 20 inference steps, creating a ~12% quality floor even at σ=0. This is structural, not a mismatch problem — see ablation below.

² Flow threshold uses the mismatch-only ablation sweep (ADC off) as the baseline; the combined mismatch sweep shows a σ=0 artifact (1.085 normalized quality due to quantization discretizing the Wasserstein metric). At σ=15% the ablation mean is 0.905 ± 0.096 — barely above the 0.90 threshold with wide variance, indicating a high-sensitivity transition zone rather than a clean threshold. Extended sweep to σ=30% confirms full degradation (0.842 at σ=20%).

³ DEQ 6-bit minimum from ADC sweep at σ=0.05 mismatch. A σ=0 pure-quantization sweep (no mismatch) produced non-monotonic, high-variance results and is pending re-validation before any architectural precision-floor claim can be made. Run `sweep_all.py --only deq --adc-only --adc-sigma 0.0 --n-trials 200` to re-validate.

**Key finding (conservative):** Five single-pass architectures (Neural ODE, SSM, EBM, Transformer, Flow) show no clean failure threshold within the σ=0–15% range, though Flow enters a high-variance transition zone near σ=15–20%. DEQ — the only architecture requiring iterative fixed-point convergence — shows a threshold near σ≈11% in the current sweep data (200 trials: mean=0.914 at σ=11%, drops to 0.892 at σ=12%), though this result is preliminary pending re-validation from the current checkpoint (see disclaimer). Diffusion's failure mode is structurally different: quantization accumulates across 20 DDIM inference steps regardless of mismatch level. Thermal noise is negligible across all architectures at demo-scale layer widths.

> **SSM scope note:** The SSM experiment uses a diagonal S4D model (Gu et al. 2022) with fixed A, B, C matrices — not Mamba's selective SSM. Mamba's input-dependent B[t], C[t], Δ[t] projections introduce additional analog failure modes (mismatch in `x_proj`/`dt_proj` compounds through the selective mechanism per token) that are not captured here. These results characterize the recurrence structure shared by all diagonal SSMs; Mamba-specific tolerance is a separate study requiring instrumentation of the selective scan kernel.

### Figures (conservative profile)

<table>
<tr>
<td><img src="experiments/cross_arch_tolerance/figures/fig1_mismatch_tolerance.png" width="340"/><br><sub>Fig 1 — Mismatch tolerance curves</sub></td>
<td><img src="experiments/cross_arch_tolerance/figures/fig2_ablation.png" width="340"/><br><sub>Fig 2 — Noise source attribution</sub></td>
</tr>
<tr>
<td><img src="experiments/cross_arch_tolerance/figures/fig3_adc_precision.png" width="340"/><br><sub>Fig 3 — ADC bit-width sweep</sub></td>
<td><img src="experiments/cross_arch_tolerance/figures/fig4_deq_convergence.png" width="340"/><br><sub>Fig 4 — DEQ convergence failure rate (100% rate is a metric artifact — see note below)</sub></td>
</tr>
<tr>
<td><img src="experiments/cross_arch_tolerance/figures/fig5_visual_results.png" width="340"/><br><sub>Fig 5 — Generated sample quality vs σ (see disclaimer above: Diffusion and EBM baselines are preliminary)</sub></td>
<td><img src="experiments/cross_arch_tolerance/figures/fig6_output_mse.png" width="340"/><br><sub>Fig 6 — Output MSE vs σ</sub></td>
</tr>
</table>

**Fig 4 note — DEQ convergence "100% failure rate":** This is a measurement artifact, not a real failure. The convergence check uses tol=1e-4 on a 64-dim vector (per-element threshold ~1.5×10⁻⁵), which is never satisfied in 30 unroll steps at any σ — including σ=0 with a perfectly trained model. Spectral norm on W_z guarantees the fixed point exists and is unique. The figure correctly shows that the failure rate is *flat across all σ* (structural — not noise-induced), which is the actual finding. Note: the DEQ checkpoint has a known provenance issue (see disclaimer above); the convergence flatness finding holds regardless, but the mismatch threshold (σ≈11%) should be treated as preliminary pending a re-sweep from the current checkpoint.

---

### Results (full-analog profile)

Full-analog defers ADC to the final readout only, removing per-layer quantization boundaries. For architectures compiled to crossbar arrays where a single output digitization step is feasible, this is the more physically realistic operating point.

| Architecture | σ threshold @ 10% loss | Min ADC bits |
|---|---|---|
| Neural ODE | ≥ 15% | 2 |
| S4D (diagonal SSM) | ≥ 15% | 4 |
| EBM | ≥ 15% | 4 |
| Flow | ~15–20% ² | 4 |
| Transformer | ≥ 15% | 2 |
| DEQ | **~11%** | **4** |
| Diffusion | — ¹ | — ¹ |

¹ Diffusion's structural quantization floor persists even under full-analog: deferring ADC to the final readout doesn't help because the score network's output (not the state) is quantized at each DDIM step.

² See conservative profile footnote ².

**Key finding (full-analog):** Removing per-layer ADC does not change the mismatch ranking. DEQ's ADC requirement drops from 6 to 4 bits when quantization is deferred to readout, consistent with per-iteration ADC creating fixed-point limit cycles. All other thresholds are unchanged — confirming mismatch, not quantization, is the dominant analog failure mode for these architectures.

---

## Nonidealities modeled

| Nonideality | Coverage | What's simulated |
|---|---|---|
| **Process variation** (mismatch) | Full | δ~N(1,σ²) per weight, static across inferences — dominant failure mode in 6/7 architectures |
| **Quantization error** | Full | Hard ADC quantization; swept over {2,4,6,8,10,12,16} bits; conservative and full-analog profiles |
| **Thermal noise** | Full | Johnson-Nyquist ε~N(0, kT/C·N_in) per readout, dynamic per inference. Negligible at C=1pF / demo-scale widths |
| **Operating range** | Partial | Output saturation at ±V_ref (1V); activation swing clipping (AnalogTanh ±0.95, AnalogSigmoid [0.025, 0.975]). IR drop along crossbar rows not modeled |
| **Frequency / bandwidth** | Not modeled | No settling time, RC bandwidth limits, 1/f noise, or clock-rate vs. precision tradeoff |

Out of scope: PCM/RRAM conductance drift over time, multi-layer nonideality coupling.

---

## Run the experiments

```bash
cd experiments/cross_arch_tolerance
python train_all.py          # train all 7 models (CPU: ~2–4 hrs with updated configs; GPU: ~3 min)
python sweep_all.py          # run all sweeps (~90 min CPU, 50 trials; requires trained checkpoints)
python plot_results.py       # generate figures from sweep results
```

> **Note:** Model training times increased significantly after architecture/hyperparameter updates (Flow: 5000 epochs, Diffusion: 5000 epochs, EBM: 2000 epochs with CD-5). GPU recommended. The quickstart example (`examples/01_quickstart.py`) trains its own small model and runs in ~2 min on CPU with no checkpoints required.

Single architecture, faster:

```bash
python sweep_all.py --only neural_ode --n-trials 20
python sweep_all.py --only diffusion --analog-substrate cld
```

---

## Ark integration

neuro-analog connects to [Ark](https://github.com/WangYuNeng/Ark) (Wang & Achour, ASPLOS '24) via two export paths. The CDG bridge mirrors the node/edge/production-rule structure from Wang & Achour (Ark, ASPLOS '24) — each exported file is a valid `BaseAnalogCkt` subclass immediately runnable by `OptCompiler`.

**Path 1 — Direct code generation** (`neuro_analog/ark_bridge/`): Reads trained weights from a PyTorch model and emits a Python/JAX file containing a `BaseAnalogCkt` subclass with `make_args`, `ode_fn`, `noise_fn`, and `readout` implemented. The generated file is immediately runnable by Ark's `OptCompiler` and trains with the adjoint method via JAX/Diffrax.

```bash
python examples/03_ark_pipeline.py           # Neural ODE — full sweep + Ark export
python examples/06_ebm_ark.py               # EBM (Hopfield) — CDG bridge + Ark export
python examples/07_flow_ark.py              # Flow — MLP velocity field export
python examples/08_diffusion_ark.py         # Diffusion — VP-SDE probability flow ODE
python examples/09_transformer_ffn_ark.py   # Transformer FFN — crossbar partition
python examples/10_deq_ark.py              # DEQ — gradient flow fixed-point ODE
python examples/11_ssm_ark.py              # SSM — S4D real/imag split dynamics
```

**Path 2 — CDG bridge** (`neuro_analog/ark_bridge/neural_ode_cdg.py`): Converts Neural ODE weights in Hopfield/Cohen-Grossberg normal form (dx/dt = −x + J·tanh(x) + b + K·u) to a proper Ark `CDGSpec` → `CDG` → `OptCompiler` → `BaseAnalogCkt` subclass. This uses Ark's full compiler pipeline with typed node/edge production rules, per-weight `TrainableMgr` registration, and optional mismatch tagging.

```bash
python examples/05_cdg_bridge.py --n 4 --sigma 0.10
```

**Which architectures export to Ark:**

| Architecture | Export tier | Generated class | Notes |
|---|---|---|---|
| Neural ODE | Runnable `BaseAnalogCkt` | `NeuralODEAnalogCkt` | Two paths: direct + CDG bridge |
| SSM (S4D) | Runnable `BaseAnalogCkt` | `SSMAnalogCkt` | Real/imag split; diagonal A → RC bank |
| DEQ | Runnable `BaseAnalogCkt` | `DEQAnalogCkt` | Fixed-point → ODE form (dz/dt = f−z) |
| Diffusion | Runnable `BaseAnalogCkt` | `DiffusionAnalogCkt` | VP-SDE probability flow ODE |
| EBM (Hopfield) | Runnable `BaseAnalogCkt` | `HopfieldAnalogCkt` | CDG bridge path |
| Flow | Analysis document | `FlowODE` (plain class) | Velocity field MLP; no fixed-point ODE structure for CDG mapping |
| Transformer | Analysis document | — | FFN crossbar partition; attention softmax stays digital |

Requires Ark, JAX, Diffrax, Equinox, Lineax (`pip install -e ".[jax]"`).
