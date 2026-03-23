# Ark Export

neuro-analog measures the analog gap and exports compatible architectures to [Ark](https://arxiv.org/abs/2309.08774) (Wang & Achour, ASPLOS '24).

- **Ark** — DSL for specifying and compiling analog compute paradigms using Constrained Dynamical Graphs (CDGs). Open source: [github.com/WangYuNeng/Ark](https://github.com/WangYuNeng/Ark)
- **Wang & Achour (arXiv:2411.03557)** — hardware-aware optimization framework (closed source) describing the mismatch model (§4.1), transient noise SDE (§4.2), and Gumbel-Softmax discrete optimization (§4.3) that neuro-analog's physics model is based on.

neuro-analog sits upstream of Ark: it answers *which architectures survive fabrication noise and at what level* before you commit to a CDG design or Ark optimization run.

---

## Setup

```bash
# JAX stack
pip install jax diffrax equinox lineax

# Ark (no PyPI package — install from source)
git clone https://github.com/WangYuNeng/Ark
pip install -e ./Ark
```

Verify:

```python
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo  # should import cleanly
```

---

## Two export paths

### Path 1 — Direct code generation

`export_neural_ode_to_ark`, `export_ssm_to_ark`, `export_deq_to_ark` read weights from the extracted `ODESystem` / `AnalogGraph` and emit a standalone Python/JAX file containing a `BaseAnalogCkt` subclass with all weights embedded as `jnp.array(...)` literals.

```python
from neuro_analog.extractors.neural_ode import NeuralODEExtractor, export_neural_ode_to_ark

extractor = NeuralODEExtractor.from_module(
    model, state_dim=2, t_span=(0.0, 1.0),
    model_name="neural_ode_make_circles",
)
extractor.load_model()

code = export_neural_ode_to_ark(extractor, "outputs/neural_ode_ark.py", mismatch_sigma=0.10)
```

The generated file is runnable without neuro-analog installed — only Ark + JAX + Diffrax required:

```python
from outputs.neural_ode_ark import NeuralODEAnalogCkt
from ark.optimization.base_module import TimeInfo
import jax.numpy as jnp

ckt = NeuralODEAnalogCkt()
ti = TimeInfo(t0=0.0, t1=1.0, dt0=0.1, saveat=jnp.array([1.0]))
result = ckt(ti, jnp.zeros(2), switch=jnp.array([]), args_seed=0, noise_seed=0)
```

See `examples/04_ark_export.py` and `examples/03_ark_pipeline.py` for full walkthroughs.

### Path 2 — CDG bridge (Hopfield / Cohen-Grossberg form only)

`neuro_analog/ark_bridge/neural_ode_cdg.py` converts Neural ODE weights in the form:

```
dx/dt = −x + J·tanh(x) + b + K·u
```

to a proper Ark `CDGSpec` → `CDG` → `OptCompiler` → `BaseAnalogCkt` subclass using Ark's full compiler pipeline (typed node/edge production rules, `TrainableMgr` registration, optional per-weight mismatch tagging).

```python
from neuro_analog.ark_bridge.neural_ode_cdg import neural_ode_to_cdg
from ark.optimization.opt_compiler import OptCompiler

cdg, spec, trainable_mgr, state_nodes, inp_nodes = neural_ode_to_cdg(J, b, K, mismatch_sigma=0.05)
opt_compiler = OptCompiler()
CktClass = opt_compiler.compile("HopfieldCkt", cdg, spec, trainable_mgr)
ckt = CktClass(init_trainable=trainable_mgr.get_initial_vals(), is_stochastic=True, solver=diffrax.Heun())
```

See `examples/05_cdg_bridge.py` for a complete example (`--n 4 --sigma 0.10`).

---

## Per-architecture export status

| Architecture | Export status | Generated class | Notes |
|---|---|---|---|
| **Neural ODE** | ✓ Working (both paths) | `NeuralODEAnalogCkt(BaseAnalogCkt)` | Direct path + CDG bridge (Hopfield form) |
| **SSM / Mamba** | ✓ Working (direct) | `SSMAnalogCkt(BaseAnalogCkt)` | Diagonal A → RC bank; B/C/D as crossbar MVM |
| **DEQ** | ✓ Working (direct) | `DEQAnalogCkt(BaseAnalogCkt)` | Fixed-point → ODE form: `dz/dt = f(z,x) − z` |
| **Flow** | Analysis document | `FlowODE` (plain class) | v_θ is 12B params in FLUX.1; not a fixed-weight CDG |
| **Diffusion** | Analysis document | `DiffusionDynamics` (plain class) | Score network is transformer-scale; 100-step SDE schedule |
| **Transformer** | Not applicable | — | No native ODE; Softmax/LayerNorm are digital |
| **EBM** | Not applicable | — | Gibbs sampling → p-bit, not RC integrator; wrong hardware primitive |

---

## The BaseAnalogCkt interface (Ark)

```python
@dataclass
class TimeInfo:
    t0: float; t1: float; dt0: float; saveat: list[float]

class BaseAnalogCkt(eqx.Module):
    a_trainable: jax.Array        # continuous trainable parameters (mismatch-perturbed)
    d_trainable: list[jax.Array]  # discrete trainable parameters (Gumbel-Softmax)
    is_stochastic: bool
    solver: diffrax.AbstractSolver

    def __call__(
        self,
        time_info: TimeInfo,
        initial_state: jax.Array,
        switch: jax.Array,
        args_seed: int,
        noise_seed: int,
        gumbel_temp: float = 1.0,
        hard_gumbel: bool = False,
    ) -> jax.Array: ...

    # Abstract methods (implemented by subclass / OptCompiler):
    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel) -> tuple: ...
    def ode_fn(self, t, y, args) -> jax.Array: ...
    def noise_fn(self, t, y, args) -> jax.Array: ...
    def readout(self, y) -> jax.Array: ...
```

The `make_args` method samples mismatch via `jrandom.normal(mismatch_seed) * sigma`, applies Gumbel-Softmax for discrete trainables, and packs everything into the `args` tuple that `ode_fn` and `noise_fn` receive.

---

## Key hardware numbers

| Parameter | Value | Source |
|---|---|---|
| Mismatch model | δ ~ N(1, σ²·I), multiplicative per weight | Wang & Achour §4.1 |
| Thermal noise | σ = √(kT/C) · √N_in per readout | Johnson-Nyquist / §4.2 |
| Integration capacitor | C = 1 pF | HCDCv2 reference design |
| Crossbar scale | 128×128 demonstrated | Ark paper |
| ADC model | Gumbel-Softmax relaxation (optimization), hard round (inference) | §4.3 |
| ODE solver | Diffrax (default: Tsit5 for non-stochastic, Heun for SDE) | Ark runtime |

---

## Why Neural ODE maps cleanest

- `dx/dt = f_θ(x,t)` is Ark's native input format — zero structural translation
- Only 2 D/A boundaries (input DAC + output ADC)
- f_θ is a small MLP (~1K params) — fits within a single 128×128 crossbar
- Adjoint training via JAX/Diffrax is exactly what Ark uses for optimization
- No approximation: the exported `ode_fn` is the exact learned dynamics

For DEQ and SSM, the mapping requires one additional step (fixed-point → relaxation ODE; diagonal scan → coupled RC ODE), but the resulting `BaseAnalogCkt` is equally valid.
