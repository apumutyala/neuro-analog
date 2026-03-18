# Shem / Ark Export

neuro-analog measures the analog gap. Ark and Shem close it.

- **Ark** ([arXiv:2309.08774](https://arxiv.org/abs/2309.08774)) — DSL for specifying and validating analog compute paradigms using Constrained Dynamical Graphs (CDGs)
- **Shem** ([arXiv:2411.03557](https://arxiv.org/abs/2411.03557)) — hardware-aware optimization layer on top of Ark; trains δ◦θ models that compensate for fabrication variance via adjoint-based gradient computation

neuro-analog sits upstream of both: it answers *which architectures survive fabrication noise and at what level* before you commit to a CDG design.

---

## Setup

The core neuro-analog library is PyTorch-only. The export pipeline requires additional dependencies that aren't on PyPI and must be installed manually:

```bash
# JAX stack
pip install jax diffrax equinox lineax

# Ark (no PyPI package — install from source)
git clone https://github.com/WangYuNeng/Ark
pip install -e ./Ark
```

Verify:

```python
from ark.optimization.base_module import BaseAnalogCkt  # should import cleanly
```

---

## The pipeline

```
PyTorch model
    ↓  analogize()
analog degradation measurement (this library)
    ↓  NeuralODEExtractor / export_neural_ode_to_shem()
Shem ODE specification (JAX/Diffrax)
    ↓  Shem.compile()
adjoint-optimized δ◦θ model
    ↓  analogize() again
verify quality recovery
```

Step 2 (measurement) and step 3 (export) are what neuro-analog provides. Steps 4–5 require a working Shem installation and the Ark repo ([github.com/WangYuNeng/Ark](https://github.com/WangYuNeng/Ark)).

---

## Neural ODE export (working today)

Neural ODE is the most analog-native architecture because `dx/dt = f_θ(x,t)` is literally Ark/Shem's input format. See `examples/03_shem_pipeline.py` for a full walkthrough.

```python
from neuro_analog.extractors.neural_ode import NeuralODEExtractor, export_neural_ode_to_shem

extractor = NeuralODEExtractor.from_module(
    model, state_dim=2, t_span=(0.0, 1.0),
    model_name="neural_ode_make_circles",
)
extractor.load_model()
profile = extractor.run()

code = export_neural_ode_to_shem(extractor, "outputs/neural_ode_shem.py", mismatch_sigma=0.10)
```

The generated file subclasses `BaseAnalogCkt` (from `ark.optimization.base_module`) and implements the required interface: `make_args`, `ode_fn`, `noise_fn`, `readout`. Then:

```python
from shem import Shem
from outputs.neural_ode_shem import NeuralODEAnalog

compiled = Shem.compile(NeuralODEAnalog())
grad = compiled.gradient(your_nll_loss)
```

Why Neural ODE maps cleanly:

- Only 2 D/A boundaries (1 DAC input + 1 ADC output)
- f_θ is a small MLP (~1K params) — fits within a 128×128 crossbar
- Adjoint training and Diffrax ODE solver are already what Shem uses internally
- No approximation needed — the export is exact

---

## Per-architecture status

| Architecture | Export status | Notes |
|---|---|---|
| **Neural ODE** | ✓ Working | `export_neural_ode_to_shem()` in `extractors/neural_ode.py` |
| **SSM** | In progress | S4D recurrence maps to `dx/dt = Ax + Bu` — very close to CDG format. Discretization step needs JAX equivalent. |
| **DEQ** | Partial | Fixed-point `z* = f(z*,x)` → continuous relaxation `dz/dt = f(z,x) - z` is valid Ark ODE format. Template in `ir/shem_export.py`. |
| **Flow** | Partial | `dx/dt = v_θ(x,t)` maps directly. Gap: v_θ is a 12B-param network in production (FLUX.1), not a compact CDG. Demo-scale flow model works. |
| **Transformer** | Not applicable | Not an ODE system — softmax, LayerNorm, dynamic attention matmuls have no CDG analog. Crossbar MVM share (~82%) is exportable but the architecture isn't a dynamics compiler target. |
| **Diffusion** | Not applicable | Score network ∇log p(x_t) is a U-Net/MLP, not a compact ODE. CLD substrate (Langevin SDE) is conceptually analog-native but export infrastructure not built. |
| **EBM** | Partial | Boltzmann sampling is analog-native (p-bit arrays) but energy landscape minimization doesn't map to CDG format. Different hardware primitive than crossbar MVM. |

---

## Ark base interface

The generated export must implement `BaseAnalogCkt` from `ark.optimization.base_module`:

```python
from ark.optimization.base_module import BaseAnalogCkt

class MyAnalogModel(BaseAnalogCkt):
    a_trainable: list  # analog (mismatch-perturbed) parameters
    d_trainable: list  # digital (exact) parameters

    def make_args(self):
        """Return initial state and time span."""
        ...

    def ode_fn(self, t, y, args):
        """Dynamics: dy/dt = f(t, y, args)"""
        ...

    def noise_fn(self, t, y, args):
        """Diffusion term for SDE: g(t, y, args)"""
        ...

    def readout(self, sol):
        """Extract task output from ODE solution."""
        ...
```

Shem adds `AnalogTrainable` parameter wrappers and `mismatch(param, sigma)` perturbation on top of this interface.

---

## Key hardware numbers

From the Shem paper and Ark CDG targets:

| Parameter | Value | Source |
|---|---|---|
| Mismatch model | δ ~ N(1, σ²·I), multiplicative | Shem §4.1 |
| Thermal noise | √(kT/C) per sense node | Shem §4.2 |
| Integration capacitor | C = 1 pF | HCDCv2 reference design |
| Crossbar scale | 128×128 demonstrated | Ark paper |
| ADC model | Gumbel-Softmax (training), hard round (inference) | Shem §4.3 |
| ODE solver | Diffrax (Heun/Dopri5) | Ark/Shem |

neuro-analog uses the same physical constants.

---

## What Shem optimization is expected to recover

From Achour & Wang (2024) Table 2: CNN at σ=0.10 achieves 93% quality recovery after Shem optimization. Our measurements give Neural ODE ≥2 D/A boundaries vs. CNN's 6-8 — fewer domain crossings means fewer sources of unrecoverable quantization error, so better recovery is expected. EBM and SSM (already ≥98% at σ=15% without optimization) are strong candidates for the easiest full-recovery demonstrations.
