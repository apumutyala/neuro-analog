# neuro-analog

Analogizes PyTorch models into a typed analog IR, exports Ark-compatible JAX/Diffrax ODE specs, and simulates analog hardware degradation across 7 neural architecture families.

Sits upstream of [Shem](https://arxiv.org/abs/2411.03557) (Achour & Wang, 2024) and [Ark](https://arxiv.org/abs/2309.08774) — answers the pre-compilation question (*which architectures survive fabrication noise, and at what level do they break?*) before you commit to a CDG design.

---

## Install

```bash
git clone https://github.com/apumutyala/neuro-analog
cd neuro-analog
pip install -e ".[dev]"
```

Optional extras: `[ssm]` for Mamba extraction, `[jax]` for Diffrax evaluation, `[full]` for everything.

---

## Quick start

```python
from neuro_analog.simulator import analogize, mismatch_sweep

analog_model = analogize(model, sigma_mismatch=0.05, n_adc_bits=8)
result = mismatch_sweep(model, eval_fn, sigma_values=[0.0, 0.05, 0.10, 0.15], n_trials=50)
print(result.degradation_threshold(max_relative_loss=0.10))  # σ at 90% quality
```

See [`examples/01_quickstart.py`](examples/01_quickstart.py) for a full walkthrough.

---

## Results

Mismatch tolerance across 7 architectures (50 trials, conservative profile):

| Architecture | σ threshold @ 10% loss | Dominant noise | Min ADC bits |
|---|---|---|---|
| EBM | ≥ 15% | mismatch | 2 |
| SSM | ≥ 15% | mismatch | 4 |
| Transformer | ≥ 15% | mismatch | 4 |
| Neural ODE | ≥ 15% | mismatch | 2 |
| Diffusion | ≥ 15%† | **quantization** | N/A† |
| DEQ | 12% | mismatch | **6** |
| Flow | 10% | mismatch | 2 |

†Diffusion is mismatch-immune but quantization-limited under conservative (per-layer ADC) profile. Resolves completely under full-analog profile.

**Key finding:** Substrate model matters more than architecture. EBM drops from ≥15% → 10% in full-analog profile (binarization was doing work); DEQ improves from 12% → ≥15% (limit cycles disappear). Thermal noise is negligible across all 7.

---

## Figures

<table>
<tr>
<td><img src="experiments/cross_arch_tolerance/figures/fig1_mismatch_tolerance.png" width="340"/><br><sub>Fig 1 — Mismatch tolerance curves</sub></td>
<td><img src="experiments/cross_arch_tolerance/figures/fig2_ablation.png" width="340"/><br><sub>Fig 2 — Noise source attribution</sub></td>
</tr>
<tr>
<td><img src="experiments/cross_arch_tolerance/figures/fig3_adc_precision.png" width="340"/><br><sub>Fig 3 — ADC bit-width sweep</sub></td>
<td><img src="experiments/cross_arch_tolerance/figures/fig4_deq_convergence.png" width="340"/><br><sub>Fig 4 — DEQ convergence failure rate</sub></td>
</tr>
<tr>
<td><img src="experiments/cross_arch_tolerance/figures/fig5_visual_results.png" width="340"/><br><sub>Fig 5 — Generated sample quality vs σ</sub></td>
<td><img src="experiments/cross_arch_tolerance/figures/fig6_output_mse.png" width="340"/><br><sub>Fig 6 — Output MSE vs σ</sub></td>
</tr>
</table>

---

## Run the experiments

```bash
cd experiments/cross_arch_tolerance
python train_all.py          # train all 7 models (~20 min CPU)
python sweep_all.py          # run all sweeps (~90 min CPU, 50 trials)
python plot_results.py       # generate figures
```

Single architecture, faster:
```bash
python sweep_all.py --only neural_ode --n-trials 20
python sweep_all.py --only diffusion --analog-substrate cld
```

---

## Docs

- [`docs/simulator.md`](docs/simulator.md) — AnalogLinear, activations, ODE solver, sweep API, design decisions
- [`docs/experiments.md`](docs/experiments.md) — the 7 models, full results, complete vs. in-progress
- [`docs/shem_export.md`](docs/shem_export.md) — Shem/Ark connection, export pipeline, next steps per architecture

Full methodology and errata: [`experiments/cross_arch_tolerance/TECHNICAL_NOTE.md`](experiments/cross_arch_tolerance/TECHNICAL_NOTE.md)
