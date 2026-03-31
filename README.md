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

Seven neural network families trained on small-scale tasks, then swept over conductance mismatch σ ∈ {0–15%} with 50 Monte Carlo trials per point. Three noise sources ablated independently. ADC bit-width swept separately. Two simulation profiles: conservative (ADC at every layer boundary) and full-analog (ADC at final readout only).

The goal is to establish empirical baselines for which architectures are worth compiling to analog hardware and where the failure points are before committing to a chip design or running analog-aware adjoint optimization within Ark.

**Disclaimer:** These are simulation results on demo-scale models (1K–103K parameters) trained near capacity on synthetic and small benchmark tasks. They represent a worst-case bound for each architecture family — production-scale overparameterized models will degrade more gracefully. This is not hardware validation. Several physical effects are not simulated (see Nonidealities section). Results should be interpreted as characterizing the analog sensitivity of the architecture's computational structure, not a specific chip.

### Results (50 trials, conservative profile)

| Architecture | σ threshold @ 10% loss | Dominant noise | Min ADC bits |
|---|---|---|---|
| Neural ODE | ≥ 15% | negligible | 2 |
| EBM | ≥ 15% | negligible | 4 |
| SSM | ≥ 15% | negligible | 4 |
| Diffusion | ≥ 15% | negligible | 4 |
| Transformer | ≥ 15% | negligible | 4 |
| DEQ | 10% | mismatch | **6** |
| Flow | 7% | mismatch | 2 |

**Key finding (conservative):** Five architectures show no measurable degradation up to σ=15%. Flow and DEQ are the only architectures with identifiable mismatch thresholds within the tested range: Flow breaks down at σ=7% and DEQ at σ=10%, both mismatch-dominated. All other noise sources (thermal, quantization) are negligible at demo-scale layer widths.

> **Reproducibility note:** DEQ (10%) and SSM/Transformer (≥15%) results are from fully reproducible sweeps on stable checkpoints. Neural ODE, EBM, and Diffusion model configurations were updated after the initial sweep run (hidden size and training hyperparameter changes); their ≥15% thresholds are directionally valid as lower bounds but have not been re-swept on the updated checkpoints. Flow (7%) used a revised ODE integration step (dt 0.25→0.01) in the current codebase; the threshold direction is expected to hold but the exact value is provisional. All results will be refreshed on next retraining.

### Figures (conservative profile)

> **Figure note:** Figures reflect sweep data from the initial checkpoint run. Curves for Neural ODE, EBM, and Diffusion correspond to the prior model configurations (see reproducibility note above). DEQ and SSM/Transformer curves are current. Figures will be regenerated after retraining.

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
<td><img src="experiments/cross_arch_tolerance/figures/fig5_visual_results.png" width="340"/><br><sub>Fig 5 — Generated sample quality vs σ</sub></td>
<td><img src="experiments/cross_arch_tolerance/figures/fig6_output_mse.png" width="340"/><br><sub>Fig 6 — Output MSE vs σ</sub></td>
</tr>
</table>

**Fig 4 note — DEQ convergence "100% failure rate":** This is a measurement artifact, not a real failure. The convergence check uses tol=1e-4 on a 64-dim vector (per-element threshold ~1.5×10⁻⁵), which is never satisfied in 30 unroll steps at any σ — including σ=0 with a perfectly trained model. The DEQ achieves ~93% test accuracy; spectral norm on W_z guarantees the fixed point exists and is unique. The figure correctly shows that the failure rate is *flat across all σ* (structural — not noise-induced), which is the actual finding.

---

### Results (50 trials, full-analog profile) — Preliminary

> **Preliminary:** Full-analog results share the same sweep run as the conservative profile above and carry the same reproducibility caveats. Numbers for Neural ODE, EBM, Diffusion, and Flow are directional only and will be updated on re-sweep.

Full-analog defers ADC to the final readout only, removing per-layer quantization boundaries. For architectures compiled to crossbar arrays where a single output digitization step is feasible, this is the more physically realistic operating point.

| Architecture | σ threshold @ 10% loss | Min ADC bits | Status |
|---|---|---|---|
| Neural ODE | ≥ 15% | 2 | Preliminary |
| SSM | ≥ 15% | 4 | Current |
| Diffusion | ≥ 15% | 4 | Preliminary |
| Transformer | ≥ 15% | 2 | Current |
| EBM | ≥ 15% | 4 | Preliminary |
| DEQ | 10% | **4** | Current |
| Flow | 5% | 2 | Provisional |

**Key finding (full-analog):** Removing per-layer ADC does not change the ranking — mismatch remains the dominant failure mode, only in Flow and DEQ. DEQ's ADC requirement drops from 6 to 4 bits when quantization is deferred to readout, consistent with per-iteration ADC creating fixed-point limit cycles. Flow's threshold tightens from 7% to 5%, suggesting per-layer quantization was inadvertently regularizing the velocity field.

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
