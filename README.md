# neuro-analog

The machine learning model zoo has diversified. State-space models (S4, Mamba), deep equilibrium networks (DEQs), normalizing flows, energy-based models, and diffusion models are now deployment candidates alongside transformers and CNNs. At the same time, analog in-memory computing (AIMC) hardware is maturing - IBM's PCM arrays, Mythic's flash-based compute, Celestial AI's photonic interconnects - all promising 10-100x energy reduction for inference.

Codesign of analog hardware and emerging neural architectures is a key enabler for both energy-efficient AI and scaling towards general distributed intelligence on the edge. The inherent computational structure of each architecture interacts with different analog substrates, and their nonidealities, in unique ways. Some architectures are more robust to certain types of noise than others, while others leverage that noise to perform their key computations, maximizing the compute efficiency of the analog substrate for accelerating deep learning.

This repository is a framework to help provide a basis and spur research in this niche but critical emerging area of digital/analog and neural network architecture co-design.

---

## What We're Doing

We train 7 neural architecture families - Transformer, Neural ODE, SSM (S4D), DEQ, Normalizing Flow, EBM, and Diffusion - from scratch on two real tasks at meaningful scale: CIFAR-10 image classification and WikiText-2 language modeling. After training, we inject physics-grounded analog nonidealities (fabrication mismatch, thermal noise, ADC quantization) into every weight-carrying operation and measure how task accuracy and perplexity degrade as noise increases. 50 Monte Carlo trials per noise level establish statistical bounds.

The result is a 14-model, dual-task benchmark: the first systematic characterization of analog tolerance across modern architecture families at both image and language tasks.

**Current Status:**
- **Pilot study (small-scale)**: complete - results available in `experiments/cross_arch_tolerance/`
- **Unified benchmark infrastructure**: complete - all 14 models implemented, training scripts ready
- **Unified benchmark training**: pending - infrastructure ready, awaiting compute resources

---

## Background

The closest prior work is IBM's demonstration of MoE-style transformer inference on 3D AIMC (Nature Computational Science, 2024) and the Nature Reviews Electrical Engineering (2025) survey of AIMC software stacks. 

Existing analog simulation tools - CrossSim (Sandia), AIHWKit (IBM), NeuroSim (Stanford), and the recently published XBTorch (2025) - excel at device-level modeling, answering 'how does this CNN perform on this crossbar.'

Both confirm that analog hardware is ready for transformers. 

What they don't answer is the architecture-level question: among the modern zoo of model families, which computational structures are inherently analog-compatible and which ones break, and why? What happens to architectures that rely on iterative fixed-point convergence (DEQ) or multi-step sampling pipelines (diffusion) under the same conditions.

We ask that question empirically, with a framework designed to be architecture-agnostic from the start.

---

## Preliminary Findings (small-scale pilot)

Before the unified benchmark, we ran a pilot study: 7 tiny models (1K-103K params) trained on low-dimensional tasks, swept over conductance mismatch sigma in [0%, 15%]. Three structural failure modes emerged:

**Single-pass architectures are broadly analog-tolerant.** Transformer, Neural ODE, SSM, Normalizing Flow, and EBM all maintain >=90% normalized quality at sigma=15% mismatch. No failure threshold within the tested range.

**Iterative convergence amplifies mismatch.** DEQ - the only architecture requiring fixed-point iteration z* = f(z*) - shows quality degradation at sigma approx 11%. The pilot study suggests fixed-point architectures may be more sensitive to analog noise, though further investigation is needed to isolate the mechanistic cause.

**Multi-step pipelines accumulate quantization error.** Diffusion models (DDIM, 20 steps) never reach 90% quality at any sigma, including sigma=0. ADC quantization of the score network output accumulates across inference steps - a structural incompatibility with per-layer ADC, not a mismatch problem.

These three degradation patterns are structurally distinct and may map to identifiable computational patterns. The unified benchmark will test whether these patterns hold at real scale and across both image and language tasks.

### Pilot Results Tables

**Conservative profile** (ADC at every layer boundary):

| Architecture | sigma threshold @ 10% degradation | Dominant noise source |
|---|---|---|
| Neural ODE | >= 15% | negligible |
| S4D (diagonal SSM) | >= 15% | negligible |
| EBM | >= 15% | negligible |
| Flow | >= 15% | mismatch |
| Transformer | >= 15% | negligible |
| **DEQ** | **11%** | mismatch |
| Diffusion | N/A 1 | quantization |

1 Diffusion's quality floor at approx 88% is present even at sigma=0. Structural quantization accumulation across 20 denoising steps - ADC bits don't help.

**Full-analog profile** (ADC at final readout only):

| Architecture | sigma threshold |
|---|---|
| Neural ODE | >= 15% |
| S4D (diagonal SSM) | >= 15% |
| EBM | >= 15% |
| Flow | >= 15% |
| Transformer | >= 15% |
| **DEQ** | **12%** |
| Diffusion | N/A |

Thermal noise is negligible across all families at demo-scale widths.

50-trial Monte Carlo sweep, 1K-103K parameter models, small-scale tasks. Results available in `experiments/cross_arch_tolerance/results/` (136 JSON files with mismatch, ADC, ablation, thermal, and layer sensitivity data).

Note: These findings are based on small-scale models on low-dimensional tasks. The unified benchmark infrastructure is complete, but training on CIFAR-10 and WikiText-2 awaits compute resources. When the unified benchmark completes, we will validate whether these patterns hold at meaningful scale.

---

## The Unified Benchmark (infrastructure complete, training pending)

The pilot used toy models on toy tasks. The unified benchmark uses real architectures on real benchmarks, trains them properly, and sweeps analog noise afterward.

**Current status:** All 14 model implementations are complete, training scripts are ready, but no training has been executed yet due to compute constraints. The infrastructure is ready to run on cloud GPUs (RunPod, etc.) when resources become available.

### Why two tasks

CIFAR-10 classification (32x32 images, 10 classes) and WikiText-2 language modeling (50K vocabulary, sequence length 256) stress different computational paths. Classification models collapse a spatial representation to a single prediction; language models maintain token-level predictions across a full sequence. An architecture's analog sensitivity on one may not predict its sensitivity on the other - SSMs benefit from their sequential inductive bias in LM, Diffusion LMs diffuse over embedding space, and DEQ LM uses causal attention inside each fixed-point step. Jointly measuring both tasks reveals whether the pilot findings generalize across modalities. 

Further, these are more representative of real-world applications and provide a more comprehensive evaluation of analog hardware compatibility. This is meant to show the potential of analog hardware and it's drawbacks in real-world applications.

### Architecture matrix

| Architecture | CIFAR-10 variant | LM variant | CIFAR-10 params | WikiText-2 params |
|---|---|---|---:|---:|
| Transformer | ViT (patch=4, 6 layers) | GPT-style decoder | 4.8M | 44.6M |
| Neural ODE | Conv features + ODE depth | Encoder-style ODE depth | 1.8M | 27.8M |
| S4D | Diagonal SSM on pixel sequence | Diagonal SSM LM | 2.5M | 38.7M |
| DEQ | Fixed-point conv features | Fixed-point attention LM | 1.8M | 28.9M |
| Flow | Class-conditional coupling flows | Coupling layers on embeddings | ~450M | 28.9M |
| EBM | Energy classification (Langevin) | SGLD embedding refinement | 1.7M | 26.3M |
| Diffusion | Denoising score classifier | Diffusion over embeddings | 22.4M | 27.8M |

Flow CIFAR-10's ~450M params (10 class-conditional flows x 8 coupling layers over 4096-dim features) is the outlier. It fits on A100 80GB but trains slowly. All other models are under 45M params.

### What the sweep measures

After training each model to convergence:
- Mismatch sigma swept over {0%, 3%, 5%, 7%, 10%, 12%, 15%} - 50 Monte Carlo trials per sigma
- ADC bits swept over {2, 4, 6, 8} bits at a fixed representative sigma
- Two analog profiles: conservative (ADC at every layer) and full-analog (ADC at final readout only)
- Metric: normalized quality - (analog metric) / (digital baseline), so 1.0 = no degradation

The degradation threshold is the interpolated sigma where normalized quality crosses 0.90 (10% loss). This is the sigma at which a hardware designer would need to worry.

### Design choices and known limitations

**Optimizer:** Transformer, S4D, and DEQ use Nesterov SGD (momentum=0.95, lr=3e-4). Neural ODE, Flow, EBM, Diffusion use AdamW (lr=3e-4). This reflects the known preference of attention-based architectures for high-momentum schedules versus adaptive rates for continuous/generative models.

**NeuralODE LM is encoder-style.** The ODE integrates over depth with the full [batch, seq_len, hidden_dim] tensor - all positions update in parallel with no causal masking. Absolute perplexity is not directly comparable to autoregressive models. Relative mismatch degradation is still valid for the analog tolerance comparison.

**Neural ODE failure mode prediction.** Neural ODEs use an adaptive ODE solver that iterates to convergence - structurally similar to DEQ's fixed-point iteration, but with different contraction properties. Our prior expectation is that Neural ODE will land in the single-pass tolerant category (>=15% sigma) because the ODE solver treats the network as a black-box function rather than a contraction map: errors introduced by weight mismatch shift the trajectory but don't compound the way DEQ's fixed-point residuals do. The unified benchmark will confirm or refute this.

**S4D only, not Mamba.** The benchmark uses diagonal S4D with fixed A, B, C matrices. Mamba's input-dependent selective scan introduces per-token projections that are distinct failure modes from the recurrence structure. S4D results characterize the recurrence structure shared by all diagonal SSMs; Mamba-specific tolerance is a separate study.

**Demo-to-medium scale.** This is not ResNet-50 or BERT-base. The unified benchmark establishes whether the pilot findings hold at a scale where the models are actually doing meaningful computation (not memorization of small datasets). Production-scale validation is future work.

---

## Framework

### System Requirements

- **Python**: 3.10 or later
- **PyTorch**: 2.1 or later with CUDA support
- **CUDA**: 11.8 or 12.1+ (tested on CUDA 12.1)
- **GPU Memory**:
  - Quick start examples: CPU-only or any GPU
  - Pilot study (cross_arch_tolerance): ~8GB VRAM
  - Unified benchmark training: 40-80GB VRAM (A100 recommended)
- **OS**: Linux (Ubuntu 20.04+ tested), macOS (limited CUDA support), Windows (WSL2 recommended)
- **Disk Space**: ~10GB for datasets (CIFAR-10, WikiText-2) + model checkpoints

### Install

```bash
git clone https://github.com/apumutyala/neuro-analog
cd neuro-analog
pip install -e ".[dev]"
```

Optional extras: `[ssm]` for Mamba extraction, `[jax]` for Diffrax/Ark evaluation, `[full]` for everything.

**Troubleshooting installation:**

If you encounter CUDA version mismatches:
```bash
# Check your CUDA version
nvcc --version
# Install PyTorch with matching CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If `torchdiffeq` fails to install:
```bash
# Install from source if precompiled wheel unavailable
pip install git+https://github.com/rtqichen/torchdiffeq.git
```

If Ark export fails (JAX/Diffrax issues):
```bash
# Install JAX with CUDA support
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Quick Start

```python
from neuro_analog.simulator import analogize, mismatch_sweep

# Drop analog noise onto any PyTorch model - no refactoring needed
analog_model = analogize(model, sigma_mismatch=0.05, n_adc_bits=8)

# Monte Carlo sweep over noise levels
result = mismatch_sweep(
    model, eval_fn,
    sigma_values=[0.0, 0.05, 0.10, 0.15],
    n_trials=50
)
threshold = result.degradation_threshold(max_relative_loss=0.10)
print(f"sigma at 10% degradation: {threshold:.3f}")
```

**Verification:** Test your installation with the quick start script:
```bash
python examples/01_quickstart.py
```
Expected output: Training completes in ~2 minutes on CPU, prints mismatch sweep results with degradation threshold.

### Examples Catalog

| Script | Description | Runtime |
|---|---|---|
| `01_quickstart.py` | Analogize a tiny MLP, run mismatch/ADC/ablation sweeps | ~2 min CPU |
| `02_seven_arch_sweep.py` | Sweep all 7 architecture families on pilot-scale tasks | ~90 min GPU |
| `03_ark_pipeline.py` | Neural ODE - Ark BaseAnalogCkt export walkthrough | ~1 min |
| `05_cdg_bridge.py` | Cohen-Grossberg dynamics - Ark CDG compilation | ~1 min |
| `06_ebm_ark.py` | EBM (Hopfield) - HopfieldAnalogCkt export | ~1 min |
| `07_flow_ark.py` | Normalizing Flow - FlowODE analysis export | ~1 min |
| `08_diffusion_ark.py` | Diffusion - DiffusionAnalogCkt export | ~1 min |
| `09_transformer_ffn_ark.py` | Transformer FFN - crossbar partition export | ~1 min |
| `10_deq_ark.py` | DEQ - gradient flow fixed-point ODE export | ~1 min |
| `11_ssm_ark.py` | SSM - S4D real/imag split dynamics export | ~1 min |

Ark export examples require `pip install -e ".[jax]"` and Ark installation (see Ark Export section below).

`analogize()` does a deep copy of your model and recursively replaces:
- `nn.Linear` - `AnalogLinear` (crossbar MVM with mismatch + thermal + ADC)
- `nn.Conv{1,2,3}d` - `AnalogConv` (receptive-field noise scaling)
- `nn.MultiheadAttention` - `AnalogMultiheadAttention` (fused QKV weight handling)
- `nn.Tanh/Sigmoid/ReLU/GELU/SiLU` - differential-pair / current-mirror equivalents

Operations without efficient analog implementations - `LayerNorm`, `Softmax`, dynamic Q.K^T, `Embedding` - stay digital. The original model is unchanged.

### Noise model

Three independent sources, applied in order per forward pass:

| Source | Model | Notes |
|---|---|---|
| Fabrication mismatch | delta ~ N(1, sigma^2) per weight, static per device instance | Wang & Achour (arXiv:2411.03557) section 4.1 |
| Thermal noise | epsilon ~ N(0, kT/C * N_in) per readout, dynamic | Johnson-Nyquist; sqrt(N_in) scaling is conservative upper bound |
| ADC quantization | Hard uniform quantization; swept over 2-16 bits | Per-layer V_ref calibration via `calibrate_analog_model()` |

Not modeled: 1/f noise, settling time, conductance drift (PCM/RRAM), inter-layer noise correlation.

**Planned: per-operation SNR profiling.** The next framework addition will log SNR degradation per analog op and correlate with end-to-end metric drop, providing mechanistic explanations for the failure mode taxonomy beyond the structural arguments - quantifying exactly how much each layer contributes to the degradation curve.

### Typed IR

```python
from neuro_analog.ir import AnalogGraph, extract_analog_graph

graph = extract_analog_graph(model, sample_input)
print(graph.analog_flop_fraction())   # fraction of compute that's analog-native
print(graph.da_boundary_count())      # number of ADC/DAC crossings per inference
```

The IR classifies 30 primitive operation types into `ANALOG` (crossbar MVM, RC integration, differential-pair activation), `DIGITAL` (Softmax, LayerNorm, dynamic matmul), and `HYBRID` (approximation-possible with accuracy tradeoff). D/A boundary count is a proxy for ADC overhead and correlates with ADC bit-width requirements in the sweep results.

---

## Running the Benchmark

### Data Handling

Datasets are downloaded automatically by PyTorch/HuggingFace datasets on first use:

- **CIFAR-10**: Downloaded to `~/.torch/datasets/` or `data/` (if configured)
- **WikiText-2**: Downloaded via HuggingFace `datasets` library to `~/.cache/huggingface/datasets/`

No manual download required. Training scripts handle dataset loading automatically. If you need to specify a custom data directory, set the `HF_DATASETS_CACHE` environment variable for WikiText-2 or modify the data path in training scripts.

### Full training + sweep (RunPod)

```bash
# Build zip and upload to RunPod
python make_zip.py   # produces ../neuro-analog.zip

# On RunPod - single A100 80GB (spot recommended, ~$0.79-1.10/hr):
unzip neuro-analog.zip && cd neuro-analog
pip install -e .
cd experiments/unified_benchmark
python run_sweeps.py --task both --gpus 0

# Two GPUs (2x A100 80GB, ~2x faster):
python run_sweeps.py --task both --gpus 0,1
```

**Note:** Training has not been executed yet. The infrastructure is ready but requires compute resources to run.

Monitor progress:
```bash
ls run_logs/*.log | xargs tail -f
```

After all 14 training jobs complete:
```bash
python sweep_all_cifar10.py \
  --checkpoint-dir checkpoints/cifar10 \
  --output-dir results/cifar10 \
  --sigma-values 0.0,0.03,0.05,0.07,0.10,0.12,0.15 \
  --n-trials 50 --device cuda

python sweep_all_wikitext2.py \
  --checkpoint-dir checkpoints/wikitext2 \
  --output-dir results/wikitext2 \
  --sigma-values 0.0,0.03,0.05,0.07,0.10,0.12,0.15 \
  --n-trials 50 --device cuda
```

See [RUNPOD_INSTRUCTIONS.md](RUNPOD_INSTRUCTIONS.md) for GPU configuration and monitoring.

### Train/sweep a single architecture

```bash
cd experiments/unified_benchmark

python train_cifar10.py --arch transformer --device cuda
python train_wikitext2.py --arch s4d --device cuda

python sweep_all_cifar10.py \
  --arch transformer \
  --checkpoint-dir checkpoints/cifar10 \
  --output-dir results/cifar10 \
  --n-trials 50 --device cuda
```

### Reproduce the pilot study

```bash
cd experiments/cross_arch_tolerance
python train_all.py      # ~3 min on GPU
python sweep_all.py      # ~90 min, 50 trials
python plot_results.py   # generates figures/
```

### Expected Outputs

After running experiments, the following directory structure is created:

```
experiments/
├── cross_arch_tolerance/
│   ├── checkpoints/          # Trained pilot models (7 architecture families) ✅ EXISTS
│   ├── results/              # JSON files with sweep results per architecture ✅ EXISTS
│   │   ├── transformer_mismatch.json
│   │   ├── neural_ode_mismatch.json
│   │   └── ... (136 JSON files total)
│   └── figures/              # Generated plots (heatmaps, degradation curves) ✅ EXISTS
│       ├── degradation_curves.png
│       └── failure_modes.png
└── unified_benchmark/
    ├── checkpoints/          # CIFAR-10 and WikiText-2 trained models ⏳ PENDING
    │   ├── cifar10/
    │   │   ├── transformer/
    │   │   │   └── checkpoint.pt
    │   │   └── ...
    │   └── wikitext2/
    │       └── ...
    ├── results/              # Analog sweep results for large-scale models ⏳ PENDING
    │   ├── cifar10/
    │   └── wikitext2/
    └── run_logs/             # Training logs per architecture ⏳ PENDING
```

**Current state:**
- Pilot study results are complete and available in `experiments/cross_arch_tolerance/`
- Unified benchmark infrastructure is ready but no training has been executed yet
- CIFAR-10 dataset is downloaded and available in `experiments/unified_benchmark/data/`

---

## Ark Export (Systems Bridge)

neuro-analog connects to [Ark](https://github.com/WangYuNeng/Ark) (Wang & Achour, ASPLOS '24), an analog circuit compiler that takes `BaseAnalogCkt` subclasses through `OptCompiler` to produce circuit-level schedules.

```bash
python examples/03_ark_pipeline.py       # Neural ODE - NeuralODEAnalogCkt
python examples/06_ebm_ark.py           # EBM (Hopfield) - HopfieldAnalogCkt
python examples/07_flow_ark.py          # Flow - FlowAnalogCkt
python examples/08_diffusion_ark.py     # Diffusion - DiffusionAnalogCkt
python examples/09_transformer_ffn_ark.py  # Transformer FFN - crossbar partition
python examples/10_deq_ark.py          # DEQ - gradient flow fixed-point ODE
python examples/11_ssm_ark.py          # SSM - S4D real/imag split dynamics
```

**CDG bridge** (`neural_ode_cdg.py`): Cohen-Grossberg normal form - `CDGSpec` - `CDG` - `OptCompiler`. The full compilation chain validated end-to-end.

| Architecture | Status | Generated class |
|---|---|---|
| Neural ODE | Runnable | `NeuralODEAnalogCkt` |
| SSM (S4D) | Runnable | `SSMAnalogCkt` |
| DEQ | Runnable | `DEQAnalogCkt` |
| Diffusion | Runnable | `DiffusionAnalogCkt` |
| EBM | Runnable | `HopfieldAnalogCkt` |
| Flow | Runnable | `FlowAnalogCkt` |
| Transformer | Partial - FFN only | attention stays digital |

Ark install order matters (torch before JAX):
```bash
apt-get install -y graphviz
git clone https://github.com/WangYuNeng/Ark.git
cd Ark && pip install -r requirement_torch.txt && pip install -r requirement.txt && pip install -e .
```

---

## Testing

Run the test suite to verify installation and framework functionality:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ark_export.py

# Run with verbose output
pytest tests/ -v
```

**Test coverage:**
- `test_ark_export.py`: Validates Ark BaseAnalogCkt code generation for Neural ODE, SSM, DEQ, Diffusion, and Flow architectures (requires JAX/Diffrax)
- `test_backend_extension.py`: Tests analog layer replacements and noise injection
- Additional tests verify IR extraction and analog amenability scoring

Note: Ark export tests require `pip install -e ".[jax]"` and are skipped if JAX/Diffrax are not installed.

---

## Repository Layout

```
neuro-analog/
├── README.md
├── RESEARCH_NOTES.md          # Theoretical grounding, analog-native design principles
├── RUNPOD_INSTRUCTIONS.md     # Full deployment guide
├── neuro_analog/
│   ├── simulator/
│   │   ├── analog_linear.py   # AnalogLinear - crossbar MVM, all 3 noise sources
│   │   ├── analog_conv.py     # AnalogConv{1,2,3}d
│   │   ├── analog_attention.py
│   │   ├── analog_activation.py
│   │   ├── analog_ode_solver.py   # ODE solver with per-step weight noise
│   │   ├── analog_ssm_solver.py   # SSM A_bar mismatch modeling
│   │   ├── analog_model.py    # analogize(), configure_analog_profile(), calibrate_analog_model()
│   │   └── sweep.py           # mismatch_sweep(), adc_sweep(), ablation_sweep()
│   ├── ir/                    # AnalogGraph, OpType taxonomy, D/A boundary detection
│   └── ark_bridge/            # PyTorch - Ark BaseAnalogCkt export
├── experiments/
│   ├── cross_arch_tolerance/  # Pilot study (small-scale, results complete)
│   └── unified_benchmark/     # Main benchmark (in training)
│       ├── models/            # 14 implementations - 7 arch x 2 tasks
│       ├── train_cifar10.py
│       ├── train_wikitext2.py
│       ├── sweep_all_cifar10.py
│       ├── sweep_all_wikitext2.py
│       └── run_sweeps.py      # Multi-GPU parallel job runner
├── examples/                  # Quickstart, Ark export walkthroughs
├── make_zip.py
└── pyproject.toml
```

---

## Common Issues and Troubleshooting

**CUDA out of memory during training:**
- Reduce batch size in training scripts: `--batch-size 64` (default is 128 for CIFAR-10, 32 for WikiText-2)
- Use gradient checkpointing: add `--gradient-checkpointing` flag if supported by the architecture
- Use a smaller model variant: adjust hidden dimensions in model configs

**Dataset download fails:**
- Check internet connection; CIFAR-10 and WikiText-2 are downloaded automatically
- For WikiText-2, ensure HuggingFace datasets can access `~/.cache/huggingface/datasets/`
- Set `HF_DATASETS_OFFLINE=1` to use cached datasets if available

**Ark export fails with "module not found":**
- Ensure Ark is installed: `pip install -e .` in the Ark repository directory
- Install JAX/Diffrax: `pip install -e ".[jax]"`
- Check that PyTorch is installed before JAX (install order matters)

**Training is too slow:**
- Use GPU acceleration: ensure `--device cuda` is specified
- For pilot study, use a single GPU (scales linearly with more GPUs)
- For unified benchmark, consider using RunPod or cloud GPUs with A100/H100

**Results not reproducible:**
- Set random seeds in training scripts: `--seed 42`
- Ensure deterministic CUDA operations: `export CUDA_LAUNCH_BLOCKING=1`
- Check that noise sweeps use the same `--n-trials` and `--sigma-values`

---

## Citation

If you use this framework or reproduce the experiments, please cite:

```bibtex
@software{neuro_analog,
  title={neuro-analog: Cross-architecture neural-to-analog compilation framework},
  author={Mutyala, Apuroop},
  year={2025},
  url={https://github.com/apumutyala/neuro-analog},
  license={MIT}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions, bug reports, or collaboration inquiries:
- Open an issue on GitHub: https://github.com/apumutyala/neuro-analog/issues
- Email: [author email to be added]

---

## Limitations

- **Simulated nonidealities only.** No silicon validation. The noise model follows Wang & Achour (arXiv:2411.03557) and is grounded in published device physics, but real hardware introduces effects we don't model: settling time, 1/f noise, inter-cell coupling, conductance drift.
- **Medium scale.** 1.7M-45M params for most models; 450M for Flow CIFAR. Larger models with more redundancy typically degrade more gracefully - these results are plausibly conservative bounds, not ceilings.
- **Relative degradation, not absolute quality.** Thresholds measure when a model degrades relative to its own digital baseline, not whether the baseline itself is good. A weak baseline that degrades gracefully may still be useless. Task results (CIFAR-10 and WikiText-2) are reported as separate strata - not pooled - since activation dynamic ranges and degradation definitions differ across modalities.
- **Diffusion sigma=0 failure is structural, not a training artifact.** Diffusion's quality floor at sigma=0 reflects ADC quantization accumulating across 20 denoising steps even before fabrication mismatch is applied. This is a property of multi-step inference pipelines with per-layer ADC, not a sign the model failed to train.
- **S4D, not Mamba.** Diagonal SSM with fixed parameters. Mamba's selective scan with input-dependent A, B, C is a distinct case.
- **NeuralODE LM is bidirectional.** Encoder-style, not autoregressive. Perplexity numbers are not directly comparable to GPT-style models.
