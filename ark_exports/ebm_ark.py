"""
Ark-compatible Hopfield network — HopfieldAnalogCkt.
ODE: dx/dt = -x + tanh(W_sym @ x + b)
n=4, mismatch_sigma=0.05

Usage:
    from ark.optimization.base_module import TimeInfo
    ckt = HopfieldAnalogCkt()
    time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.01, saveat=jnp.array([5.0]))
    x0 = jnp.zeros((4,))
    result = ckt(time_info, x0, switch=jnp.array([]), args_seed=42, noise_seed=43)
"""

import jax.numpy as jnp
import jax.random as jrandom
import diffrax
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo

class HopfieldAnalogCkt(BaseAnalogCkt):
    """Hopfield analog circuit.

    ODE: dx/dt = -x + tanh(W_sym @ x + b)
    n=4, mismatch_sigma=0.05

    a_trainable = [W.flatten() | b] — mismatch applied in make_args().
    """

    def __init__(self):
        _W = jnp.array([[0.0, -0.7477548237235829, 0.18341250957554212, 0.2516488534881075], [-0.7477548237235829, 0.0, -0.18130088110157366, 0.20274965365611267], [0.18341250957554212, -0.18130088110157366, 0.0, 0.3113253194202485], [0.2516488534881075, 0.20274965365611267, 0.3113253194202485, 0.0]])
        _b = jnp.array([0.0, 0.0, 0.0, 0.0])
        a_trainable = jnp.concatenate([_W.flatten(), _b])
        super().__init__(
            init_trainable=a_trainable,
            is_stochastic=False,
            solver=diffrax.Heun(),
        )

    def make_args(self, switch, mismatch_seed, gumbel_temp, hard_gumbel):
        key = jrandom.PRNGKey(mismatch_seed)
        keys = jrandom.split(key, 2)
        sigma = 0.05
        W = (self.a_trainable[:16] * (1.0 + sigma * jrandom.normal(keys[0], (16,)))).reshape((4, 4))
        b = (self.a_trainable[16:20] * (1.0 + sigma * jrandom.normal(keys[1], (4,))))
        return (W, b)

    def ode_fn(self, t, x, args):
        W, b = args
        return -x + jnp.tanh(W @ x + b)

    def noise_fn(self, t, x, args):
        # Zero by default — Hopfield settles deterministically to energy minimum.
        # For Langevin sampling add:  jnp.ones_like(x) * jnp.sqrt(2.0 / beta)
        return jnp.zeros_like(x)

    def readout(self, y):
        # y shape: (len(saveat), n) — return state at final time
        return y[-1]

if __name__ == '__main__':
    ckt = HopfieldAnalogCkt()
    time_info = TimeInfo(t0=0.0, t1=5.0, dt0=0.01, saveat=jnp.array([5.0]))
    x0 = jnp.zeros((4,))
    switch = jnp.array([])
    result = ckt(time_info, x0, switch, args_seed=42, noise_seed=43)
    print(f'Result shape: {result.shape}')

