"""
Auto-generated Shem ODE specification — neuro-analog.
Model      : test_ssm
Architecture: SSM (Mamba)

Usage:
    from shem import Shem
    model = <ClassName>()
    compiled = Shem.compile(model)
    sol = compiled.diffeq_solve(mismatch_sample_seed=42)
    grad = compiled.gradient(your_loss)
"""

import jax
import jax.numpy as jnp
import diffrax
# from shem import AnalogTrainable, mismatch, Shem

# No time constants extracted — run MambaExtractor with a loaded model.
# Placeholder class below uses default values.

class SSMAnalogODE:
    """Continuous-time SSM for test_ssm.

    State dimension N = 16
    Time constant spread: 1.0× (max/min ratio)
    Wider spread → more diverse RC component values needed.

    Analog circuit mapping:
      A_i → RC time constant τ_i = 1/|a_i|
      B·u → current injection into RC node
      C·h → weighted current summation (output projection)

    REAL WEIGHTS: Extracted from test_ssm
    """

    def __init__(self):
        # Diagonal state matrix entries (extracted from A_log via a_i = -exp(A_log_i))
        # Each a_i corresponds to one RC circuit with τ_i = 1/|a_i|
        _A = jnp.array([-1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0, -1000.0])
        self.A = mismatch(AnalogTrainable(init=_A), sigma=0.05)

        # Input coupling B: extracted from layer 0 x_proj (first 4 shown)
        _B = jnp.array([0.044257, -0.039832, 0.053285, 0.036911, 0.068157, -0.064478, -0.054186, 7.5e-05, -0.034051, 0.014658, 0.052679, 0.000268, 0.003292, -0.047835, -0.014097, -0.043539, -0.009045, 0.024541, -0.083362, -0.081864, 0.036129, 0.038006, -0.052534, -0.035325, -0.012247, -0.075283, -0.068904, 0.029665, 0.030347, -0.077677, -0.01537, 0.045054, -0.043024, 0.084797, 0.052822, 0.003511, -0.067637, -0.063889, -0.062847, -0.064478, 0.033794, 0.055388, -0.032374, 0.087348, 0.052269, -0.009546, -0.040135, 0.007019, 0.059364, 0.022157, -0.042023, -0.084338, 0.003563, -0.074102, 0.010659, -0.04118, 0.021556, -0.08366, -0.048619, -0.028736, 0.077842, -0.000868, 0.050037, 0.053448])
        self.B = mismatch(AnalogTrainable(init=_B), sigma=0.05)

        # Output coupling C: extracted from layer 0 x_proj (first 4 shown)
        _C = jnp.array([-0.027097, -0.006622, -0.068293, -0.063165, -0.031635, -0.054262, 0.057019, 0.050099, -0.043393, -0.043819, -0.079163, -0.03542, 0.037975, -0.087796, -0.067052, 0.034397, 0.011833, 0.048863, 0.043723, 0.028045, 0.040131, -0.053017, 0.058471, -0.034844, -0.023845, -0.068208, 0.036705, 0.071283, 0.080179, 0.048436, 0.048899, 0.000565, 0.057244, -0.047683, 0.051004, 0.005646, -0.073558, -0.022405, -0.036214, -0.00968, -0.056112, -0.040692, -0.011281, 0.005186, 0.027956, 0.057447, -0.013903, -0.03936, -0.016954, -0.044178, -0.026718, 0.035383, -0.009699, 0.00094, 0.023589, -0.028902, -0.065238, -0.063167, -0.073257, 0.027939, -0.021298, 0.06268, 0.032417, -0.013119])
        self.C = mismatch(AnalogTrainable(init=_C), sigma=0.05)

        # Readout time: Shem evaluates cost at each token boundary
        # For sequence length L: readout_times = jnp.linspace(0, 1, L+1)[1:]
        self.readout_times = jnp.linspace(0.0, 1.0, 32)  # Example: 32 tokens

    def dynamics(self, t, x, u=None):
        """dx/dt = A·x + B·u — diagonal SSM ODE.

        Analog: A·x via RC decay (element-wise), B·u via current injection.
        u is the input signal; for autonomous evolution set u=0.
        """
        if u is None:
            u = jnp.zeros_like(x)
        return self.A * x + self.B * u  # element-wise: N independent ODEs

    def solve(self, y0: jnp.ndarray) -> jnp.ndarray:
        """Integrate ODE using Diffrax (Shem's internal solver).

        Solver: Tsit5 (Runge-Kutta 4/5 with error control).
        Readout times: [1.0]
        For Shem: cost is evaluated at these time points.
        """
        term = diffrax.ODETerm(self.dynamics)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=jnp.array([1.0]))
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=1.0, dt0=0.001,
            y0=y0, saveat=saveat,
        )
        return sol.ys  # shape: (len(readout_times), state_dim)

# ── Analog Circuit Mapping ──────────────────────────────────────────────
# State variable → RC integrator
# Required component values (extracted from pretrained test_ssm):
#
#   i   a_i (s⁻¹)     τ_i = 1/|a_i|   R (10kΩ)  C (pF)
# ─────────────────────────────────────────────────────────
#   0      -1000.0000         0.001000s     10kΩ   100000.000 pF
#   1      -1000.0000         0.001000s     10kΩ   100000.000 pF
#   2      -1000.0000         0.001000s     10kΩ   100000.000 pF
#   3      -1000.0000         0.001000s     10kΩ   100000.000 pF
#   4      -1000.0000         0.001000s     10kΩ   100000.000 pF
#   5      -1000.0000         0.001000s     10kΩ   100000.000 pF
#   6      -1000.0000         0.001000s     10kΩ   100000.000 pF
#   7      -1000.0000         0.001000s     10kΩ   100000.000 pF

