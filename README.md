# neuro-analog

A framework for co-designing analog hardware with modern neural network architectures. This codebase bridges hardware and neural architects by decomposing neural networks into circuit-level primitives. Hardware architects learn what computational patterns their substrates can support; neural architects learn how their models will behave on analog hardware.

## Quick Start

Install and run a basic analog sweep in 2 minutes:

```bash
pip install -e .
python examples/01_quickstart.py
```

This trains a tiny MLP, converts it to analog, runs a mismatch sweep, and reports:
- Accuracy degradation vs analog noise
- Energy savings vs digital baseline
- Speedup vs digital baseline

For a deeper walkthrough, see `notebooks/quickstart_tour.ipynb`.

For specialized deep-dives on specific topics:
- `notebooks/architecture_families.ipynb` - Computational patterns and analog amenability across 7 architecture families
- `notebooks/intermediate_representation.ipynb` - IR system, graph construction, and hardware annotation
- `notebooks/amenability_analysis.ipynb` - Amenability scoring, failure modes, and design recommendations

For compiling trained models to analog circuits, see the [Analog Circuit Export (Ark Bridge)](#analog-circuit-export-ark-bridge) section.

## What This Does

Modern neural architectures go beyond transformers and CNNs. State-space models (S4, Mamba), deep equilibrium networks (DEQs), normalizing flows, energy-based models, and diffusion models each have different computational structures. Analog in-memory computing (AIMC) hardware offers 10-100x energy reduction for inference, but not all architectures tolerate analog noise equally.

This framework answers two questions:

1. For hardware architects: What circuit primitives can implement neural computations? What in-memory-compute technologies/components (RRAM, PCM, SRAM-IMC) are best suited for which computational patterns? How do I map neural operations to my hardware efficiently?

2. For neural architects: How will my model behave on analog hardware? Which operations in my architecture are analog-compatible? What energy/latency gains can I expect from analog deployment?

We simulate physics-grounded analog effects (conductance mismatch, thermal noise, ADC quantization) across 7 architecture families on real tasks (CIFAR-10, WikiText-2). The framework decomposes models into about 20 circuit-level primitives (crossbar MVM, integrator, RC decay, Gibbs sampler) and maps each to analog-native, digital-required, or hybrid domains.

## Current Status

- Pilot study (small-scale): complete, results in `experiments/cross_arch_tolerance/`
- Unified benchmark infrastructure: complete, all 14 models implemented
- Unified benchmark training: pending, awaiting compute resources
- Circuit-mode defaults: sweeps default to rc_integrator/hopfield for true analog measurement
- Energy/latency modeling: included in all sweep results
- Ark circuit export: ground-up bridge in `neuro_analog/revised_ark_bridge/` compiles six model families to analog circuits, each verified against its PyTorch oracle

## How It Works

### Two-Layer Architecture

The codebase separates concerns into two layers:

**Layer 1: Analog Primitives (physics simulation)**
- `AnalogLinear`, `AnalogConv`, `AnalogMultiheadAttention` model crossbar physics
- Four noise sources: DAC quantization, conductance mismatch, thermal read noise, ADC quantization
- `AnalogODEIntegrator` models Johnson-Nyquist noise on integration capacitors
- `AnalogSSMSolver` models RC time constant mismatch in state-space models

**Layer 2: Intermediate Representation (cost modeling)**
- `AnalogGraph` IR represents neural network operations with domain annotations
- `HardwareProfile` defines energy/latency constants (5 pJ/MAC analog, 100 pJ/MAC digital)
- `estimate_node_cost()` maps each operation to hardware estimates
- Amenability scoring evaluates analog compatibility

See `notebooks/intermediate_representation.ipynb` for a deep dive into the IR system.

### Circuit-Level Computation Modes

The framework models different analog substrates:

- rc_integrator: For Neural ODEs and flows. Models RC circuit ODE solver with Johnson-Nyquist noise on integration capacitors.
- hopfield: For DEQs. Models continuous-time analog feedback relaxation with thermal noise.
- classic: For diffusion. Standard DDIM sampling.
- cld: For diffusion. Critically-damped Langevin dynamics mapping to RLC circuits.

By default, sweeps use circuit modes (rc_integrator, hopfield) to measure true analog hyperefficiency rather than digital approximations.

### Energy and Latency Estimation

Sweeps report energy/latency metrics using parameter counting as a proxy for MAC operations:

```python
from neuro_analog.ir.energy_model import HardwareProfile
from neuro_analog.simulator import mismatch_sweep

profile = HardwareProfile()
result = mismatch_sweep(model, eval_fn, hardware_profile=profile)

print(f"Energy saving vs digital: {result.energy_saving_vs_digital*100:.1f}%")
print(f"Speedup vs digital: {result.speedup_vs_digital:.1f}x")
```

Hardware constants are sourced from IBM PCM modeling, AIMC surveys, and SRAM IMC benchmarks.

## Analog Circuit Export (Ark Bridge)

`neuro_analog/revised_ark_bridge/` compiles a trained PyTorch dynamics model into an Ark analog circuit (`BaseAnalogCkt`) for circuit-level co-design. Each family is lowered to a Circuit Dependency Graph (CDG) and compiled to a differentiable JAX/diffrax solver, with per-weight conductance mismatch applied at the device level.

Two lowering paths cover six families:
- CDG-native (`cdg_native/`): DEQ, EBM, and SSM map onto reusable CDGSpec factories (e.g. the additive-recurrent Hopfield form).
- Plain fallback (`plain_fallback/`): Neural ODE, Flow, and Diffusion lower through a direct MLP vector-field circuit.

Every export is checked against the original PyTorch model as the oracle (`verify.py`): circuit trajectories are compared to the model's own dynamics (DEQ fixed point, ODE integration, EBM mean-field relaxation, or CLD step).

```python
from neuro_analog.revised_ark_bridge import build_deq, verify_family, solve_with_mismatch

ckt = build_deq(deq_model, mismatch_sigma=0.05)        # trained DEQ -> BaseAnalogCkt
err = verify_family("deq", ckt, deq_model, x_input=x, z0=z0)   # max trajectory error vs oracle
traj = solve_with_mismatch(ckt, z0, sigma=0.05, seed=0)        # one device-mismatch realization
```

This path requires the Ark framework (JAX, diffrax, equinox). Notebook walkthroughs live in `neuro_analog/revised_ark_bridge/notebooks/` (CDG-native families, plain-fallback families, the CDG-to-circuit compiler, and why softmax attention is analog-hostile). Validate the environment and full compile/solve/mismatch/gradient path with:

```bash
python neuro_analog/revised_ark_bridge/smoke_test.py
```

## Usage

### Basic Sweep

```python
from neuro_analog.simulator import analogize, mismatch_sweep
from neuro_analog.ir.energy_model import HardwareProfile

# Convert model to analog
analog_model = analogize(model, sigma_mismatch=0.05, n_adc_bits=8)

# Run sweep with energy/latency estimation
profile = HardwareProfile()
result = mismatch_sweep(
    model, eval_fn,
    sigma_values=[0.0, 0.05, 0.10, 0.15],
    n_trials=50,
    hardware_profile=profile,
)

# Access results
print(f"Threshold @ 10% loss: {result.degradation_threshold(0.10):.3f}")
print(f"Energy saving: {result.energy_saving_vs_digital*100:.1f}%")
```

### Running the Unified Benchmark

```bash
# CIFAR-10
python experiments/unified_benchmark/train_cifar10.py --arch neural_ode
python experiments/unified_benchmark/sweep_all_cifar10.py --arch neural_ode

# WikiText-2
python experiments/unified_benchmark/train_wikitext2.py --arch transformer
python experiments/unified_benchmark/sweep_all_wikitext2.py --arch transformer
```

### Running the Pilot Study

```bash
python experiments/cross_arch_tolerance/sweep_all.py --only neural_ode
python experiments/cross_arch_tolerance/sweep_all.py --analog-substrate all
```

## Preliminary Findings

From the pilot study (7 tiny models, 1K-103K params, low-dimensional tasks):

Single-pass architectures are broadly analog-tolerant. Transformer, Neural ODE, SSM, Flow, and EBM maintain at least 90% quality at 15% mismatch.

Iterative convergence amplifies mismatch. DEQ degrades at about 11% mismatch, suggesting fixed-point architectures are more sensitive to analog noise.

Multi-step pipelines accumulate quantization error. Diffusion never reaches 90% quality even at sigma=0 due to ADC quantization accumulating across 20 denoising steps.

These patterns need validation at real scale. The unified benchmark will test whether they hold on CIFAR-10 and WikiText-2.

## Background and Novel Contributions

Existing analog simulation tools (CrossSim, AIHWKit, NeuroSim, XBTorch) focus on device-level modeling. They answer "how does this CNN perform on this crossbar" by modeling crossbar physics and device nonidealities at the tile or layer level.

These tools confirm that analog hardware works for transformers. But they do not address the architecture-level question: among the modern zoo of model families, which computational structures are inherently analog-compatible and which break?

neuro-analog fills this gap by:

1. Architecture-agnostic IR: Decomposes models into about 30 circuit-level primitives (MVM, integration, decay, Gibbs sampling) with domain annotations. This enables D/A boundary detection across diverse architectures.

2. Systematic benchmark: The first systematic characterization of analog tolerance across 7 modern architecture families (Transformer, Neural ODE, SSM, DEQ, Flow, EBM, Diffusion) on real tasks.

3. Physics-grounded noise models: Each primitive has appropriate noise sources (kT/C for integrators, shot noise for samplers, ADC quantization for crossbars).

4. Energy/latency modeling: Hardware-aware cost estimation alongside accuracy degradation.

Key primitive mappings:
- S4D / Neural ODE / DEQ: INTEGRATION, DECAY, ANALOG_FIR
- EBM / Hopfield: GIBBS_STEP, SAMPLE, NOISE_INJECTION
- Diffusion: MVM + NOISE_INJECTION + SAMPLE

## Project Structure

```
neuro_analog/
  simulator/          # Analog primitives (AnalogLinear, AnalogConv, etc.)
  ir/                 # Intermediate representation and energy modeling
  extractors/         # Architecture-specific IR builders
  analysis/           # Precision and profiling tools
  revised_ark_bridge/ # Compile trained models to Ark analog circuits (CDG-native + plain fallback)
experiments/
  cross_arch_tolerance/  # Pilot study
  unified_benchmark/     # CIFAR-10 and WikiText-2 benchmark
examples/
  01_quickstart.py    # Basic usage example
notebooks/
  quickstart_tour.ipynb          # Interactive walkthrough (15-20 min)
  architecture_families.ipynb   # Deep dive on 7 architecture families
  intermediate_representation.ipynb  # Deep dive on IR system
  amenability_analysis.ipynb    # Deep dive on amenability scoring
```

The Ark circuit-export bridge and its walkthrough notebooks live under `neuro_analog/revised_ark_bridge/` (see the Analog Circuit Export section above).

## For Hardware Architects

Use this codebase to:
- Understand which circuit primitives (crossbar, integrator, RC decay) can implement neural computations (see `notebooks/intermediate_representation.ipynb`)
- Learn what computational patterns your hardware should support efficiently (see `notebooks/architecture_families.ipynb`)
- Map neural operations to your substrate (RRAM, PCM, SRAM-IMC)
- Estimate energy/latency gains for different computational patterns
- Validate that your target patterns tolerate your device noise levels (see `notebooks/amenability_analysis.ipynb`)

## For Neural Architects

Use this codebase to:
- Understand how your model will behave on analog hardware (see `notebooks/quickstart_tour.ipynb`)
- Identify analog-incompatible operations in your architecture (see `notebooks/intermediate_representation.ipynb`)
- Estimate energy/latency gains from analog deployment
- Compare analog tolerance across architecture families (see `notebooks/architecture_families.ipynb`)
- Export trained models to analog circuits (see the [Analog Circuit Export (Ark Bridge)](#analog-circuit-export-ark-bridge) section)

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{neuro_analog,
  title = {neuro-analog: A Framework for Analog-Aware Neural Architecture Co-Design},
  author = {Mutyala, Apuroop},
  year = {2026},
  url = {https://github.com/apumutyala/neuro-analog}
}
```

## Installation

```bash
git clone https://github.com/apumutyala/neuro-analog
cd neuro-analog
pip install -e ".[dev]"
```

Optional extras: `[jax]` for Ark circuit export.

Requirements:
- Python 3.10+
- PyTorch 2.1+ with CUDA support
- GPU: 8GB VRAM for pilot study, 40-80GB for unified benchmark

## Testing

```bash
pytest tests/
```

## License

MIT
