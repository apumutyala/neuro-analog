# Unified Benchmark — Notes

Consolidated reference for the `experiments/unified_benchmark/` training infrastructure.
Replaces: `BUILD_COMPLETE.md`, `DUAL_TASK_EXECUTION.md`, `TRAINING_OPTIMIZATIONS.md`, `README.md`, `README_WIKITEXT2.md`, `VERIFICATION_RESULTS.md`.

---

## Status (April 14, 2026)

Both benchmarks are fully built and **ready for RunPod**. WikiText-2 training loop has been updated with all optimizations (mixed precision, Muon optimizer, comprehensive checkpointing) — parity with CIFAR-10 is complete.

| Infrastructure | Status |
|----------------|--------|
| CIFAR-10: 7 model architectures | ✅ |
| WikiText-2: 7 model architectures | ✅ |
| `train_cifar10.py` + `train_wikitext2.py` | ✅ |
| `sweep_all_cifar10.py` + `sweep_all_wikitext2.py` | ✅ |
| `launch_training.sh` + `launch_training_lm.sh` | ✅ |
| `run_sweeps.py` (multi-GPU auto-job-runner) | ✅ |
| Mixed precision (AMP) — both tasks | ✅ |
| Muon-style optimizer — both tasks | ✅ |
| Comprehensive checkpointing + JSON summary | ✅ |

---

## Architecture Matrix (14 Models)

| # | Architecture | Core Mechanism | CIFAR-10 Params | WikiText-2 Params |
|---|--------------|----------------|----------------:|------------------:|
| 1 | Transformer | Self-attention | 4.8M | 44.6M |
| 2 | Neural ODE | Continuous-depth ODE | 1.8M | 27.8M |
| 3 | S4D (SSM) | Diagonal state-space | 2.5M | 38.7M |
| 4 | DEQ | Fixed-point iteration (30 steps) | 1.8M | 28.9M |
| 5 | Flow | Normalizing flow (8 coupling layers) | 357M* | 28.9M |
| 6 | EBM | Energy minimization (Langevin) | 1.7M | 26.3M |
| 7 | Diffusion | Denoising score matching (20 steps) | 22.4M | 27.8M |

\* Flow CIFAR model is oversized (357M params due to `n_flows=8`). Optionally reduce to `n_flows=4` (~180M) before training.

---

## Training Optimizations Applied

### Both Tasks

| Optimization | Details |
|---|---|
| **cuDNN Benchmark** | `torch.backends.cudnn.benchmark = True` — auto-selects best conv algorithms |
| **TF32** | `allow_tf32 = True` — 8× matmul speedup on A100/RTX 30+, negligible accuracy loss |
| **Mixed Precision (AMP)** | `GradScaler` + `autocast` — 2-3× speedup, 50% VRAM reduction |
| **Non-blocking transfers** | `to(device, non_blocking=True)` + `pin_memory=True` |
| **Efficient zero_grad** | `set_to_none=True` |
| **Gradient clipping** | `clip_grad_norm_(1.0)` — critical for ODE/DEQ/Diffusion stability |

### Optimizer Selection (per architecture)

| Architecture | Optimizer | Rationale |
|---|---|---|
| Transformer | Muon (SGD + Nesterov momentum=0.95) | Native for attention |
| S4D | Muon | State-space benefits from momentum |
| DEQ | Muon | Attention-based fixed-point |
| Neural ODE | AdamW | Adaptive for continuous dynamics |
| Flow | AdamW | Stable for invertible transforms |
| EBM | AdamW | Better for energy landscapes |
| Diffusion | AdamW | Stable for iterative denoising |

### Training Config

| Setting | CIFAR-10 | WikiText-2 |
|---|---|---|
| `max_epochs` | 300 | 150 |
| `patience` | 25 | 15 |
| `grad_clip` | 1.0 | 1.0 |
| `target` | 85% accuracy | PPL ≤ 100 |
| `min_acceptable` | 80% accuracy | PPL ≤ 150 |
| LR schedule | CosineAnnealingWarmRestarts (T_0=50) | CosineAnnealingLR |

### Expected Speedup (vs. unoptimized)
- CIFAR-10 per epoch: ~45s → ~22s (**51% faster**)
- WikiText-2 per epoch: ~8 min → ~5 min (**37% faster**)

---

## Expected Results

### CIFAR-10 Baseline Accuracy
- Transformer: ~86%, Neural ODE: ~85%, Flow: ~85%
- S4D: ~84%, DEQ: ~85%, EBM: ~82%, Diffusion: ~82%

### WikiText-2 Baseline Perplexity
- Transformer: ~95-105, S4D: ~100-110, Neural ODE: ~110-120
- Flow: ~115-125, DEQ: ~120-130, EBM: ~130-140, Diffusion: ~140-150

### Analog Tolerance Hypotheses (σ @ 10% degradation)
- **CIFAR-10:** ODE/Flow (σ≥15%) > S4D (σ~13%) > Transformer (σ~12%) > DEQ (σ~11%) > EBM (σ~10%) > Diffusion (structural floor)
- **WikiText-2:** S4D (σ≥14%) > ODE (σ~13%) > Transformer (σ~12%) > Flow (σ~11%) > DEQ (σ~10%) > EBM/Diffusion (lower)

### Cross-Task Insight
S4D should outperform Neural ODE on sequences (architecture-task fit). If confirmed, this is a novel finding: native architecture-task alignment affects analog robustness.

---

## File Reference

### Launch Scripts
```bash
# Sequential single GPU
python train_cifar10.py --arch transformer --device cuda
python train_wikitext2.py --arch transformer --device cuda

# Multi-GPU parallel sweep (recommended on RunPod)
python run_sweeps.py --task both --gpus 0,1,2,3,4,5,6,7
```

### Monitoring
```bash
# Tail all logs simultaneously
watch -n 60 'tail -n 2 logs/*_train.log'

# Individual experiment
tail -f logs/transformer_train.log
tail -f run_logs/cifar10_transformer_runner.log
```

### Output Structure
```
checkpoints/
├── cifar10/
│   ├── {arch}_cifar10_best.pt        # Model weights + training history
│   └── {arch}_cifar10_summary.json   # Results summary
└── wikitext2/
    ├── {arch}_lm_best.pt
    └── {arch}_lm_summary.json

logs/
├── {arch}_train.log        # CIFAR-10 logs
└── {arch}_lm_train.log     # WikiText-2 logs
```

---

## Known Issues

### Flow Model (CIFAR-10)
- 357M parameters with `n_flows=8` — may be too slow to train efficiently
- Fix: Change to `n_flows=4` in `models/flow_cifar.py` (~180M params, 2× faster)

### DEQ (cross_arch_tolerance experiment)
- Sweep baseline stored in JSON doesn't match current checkpoint (21% discrepancy on deterministic metric)
- The σ≈11% threshold finding is internally consistent within sweep data but preliminary
- Fix: Re-run full DEQ sweep from current checkpoint

### EBM (cross_arch_tolerance experiment)
- Evaluation varies ±6% due to Gibbs chain stochasticity
- Fig 5 test sample uses near-empty image (3 non-zero pixels)
- Fix: Average over multiple chains, fix test sample seed

### Diffusion (cross_arch_tolerance experiment)
- Structural quantization floor is real, but demo model too small
- σ=0 baseline already poor (score network limited capacity)
- Fix: Replace 3-layer MLP with U-Net score network

### Transformer Warning (WikiText-2)
```
enable_nested_tensor is True, but self.use_nested_tensor is False 
because encoder_layer.norm_first was True
```
- Cosmetic warning only, no effect on training. Can suppress with `warnings.filterwarnings`.

---

## Dependencies

```bash
# Core (usually pre-installed on RunPod PyTorch images)
pip install torch torchvision

# For WikiText-2
pip install transformers datasets

# For Neural ODE
pip install torchdiffeq
```
