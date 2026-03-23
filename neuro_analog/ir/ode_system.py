"""
ODESystem: extracted dynamical system ready for analog simulation.

This is the central data structure that bridges:
  - Extraction (neural_ode.py, ssm.py)  → populates parameters, bounds, dynamics
  - Simulation (analog_ode_solver.py)    → reads parameters, applies mismatch + noise
  - Export (ark_export.py)               → serializes to JAX/Diffrax code

Based on the noise/mismatch model described in Wang & Achour (arXiv:2411.03557):
  - Mismatch is multiplicative δ ~ N(1, σ²) per parameter (§4.1)
  - Transient noise is additive SDE diffusion term g(x,θ,t)·dW (§4.2)
  - Parameter bounds enforce physical constraints (§4.3)
  - Readout times specify when cost is evaluated along the trajectory

Unlike AnalogGraph (which catalogs operations as nodes), ODESystem represents
the actual mathematical system: dx/dt = f(x, θ, t) + g(x, θ, t)·ξ(t).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn


@dataclass
class ParameterSpec:
    """Specification for one parameter in the dynamical system.

    Maps to Ark's Trainable (via TrainableMgr.new_analog) and AttrDefMismatch
    when mismatch_sigma > 0 (see neuro_analog.ark_bridge).

    Attributes:
        name: Human-readable name (e.g., "f_theta.W0", "ssm.A_diag").
        value: Nominal parameter tensor (extracted from pretrained model).
        bounds: Physical bounds (lo, hi). None = unbounded.
                Parameters clipped to feasible range (Wang & Achour §4.3).
        mismatch_sigma: Per-parameter mismatch σ. δ ~ N(1, σ²).
                        0.0 = ideal (no fabrication variation).
        trainable: Whether to include in Ark's TrainableMgr.
                   True for all weights; False for fixed hardware constants.
        analog_primitive: What this maps to in hardware
                          (e.g., "crossbar_conductance", "RC_time_constant").
    """
    name: str
    value: torch.Tensor
    bounds: Optional[tuple[float, float]] = None
    mismatch_sigma: float = 0.05
    trainable: bool = True
    analog_primitive: str = ""


@dataclass
class NoiseProfile:
    """Transient noise specification for the SDE diffusion term.

    Wang & Achour §4.2: dx = f(x,θ,t)dt + g(x,θ,t)·dW
    The noise amplitude g can be state-dependent or constant.

    For analog hardware:
      - Thermal noise: g = sqrt(kT/C) per integration capacitor
      - Shot noise: g = sqrt(2qI) per current source
      - Combined: σ_transient covers the dominant source

    Attributes:
        sigma: Global transient noise std dev. Applied as:
               dx += sigma * sqrt(dt) * N(0,I)  [Euler-Maruyama]
        per_dim_sigma: Per-state-dimension noise (if different RC caps).
                       None = use global sigma for all dimensions.
        bandwidth_hz: Noise bandwidth (for power spectral density context).
    """
    sigma: float = 0.0
    per_dim_sigma: Optional[list[float]] = None
    bandwidth_hz: Optional[float] = None

    @property
    def is_active(self) -> bool:
        if self.sigma > 0:
            return True
        if self.per_dim_sigma and any(s > 0 for s in self.per_dim_sigma):
            return True
        return False


@dataclass
class ODESystem:
    """A complete dynamical system extracted from a neural network.

    This is the output of extraction and the input to simulation + export.

    The system is: dx/dt = f(x, θ, t) + g(x, θ, t) · ξ(t)
    where θ are the parameters, each subject to mismatch δ ~ N(1, σ²).

    Usage:
        # Extract from a Neural ODE
        ode_sys = extractor.extract_ode_system()

        # Apply mismatch and simulate
        perturbed = ode_sys.sample_mismatch(sigma=0.05)
        trajectory = analog_odeint(perturbed.dynamics_fn, y0, t_span, ...)

        # Export to Ark-compatible JAX
        export_neural_ode_to_ark(ode_sys, "output.py")

    Attributes:
        name: Model name for identification.
        family: Architecture family ("neural_ode", "ssm", etc.).
        state_dim: Dimension of the state vector x.
        parameters: Named parameter specs (the θ vector).
        dynamics_fn: Callable (t, x) → dx/dt using nominal parameters.
                     This is the PyTorch module's forward pass.
        dynamics_module: The nn.Module implementing f(t, x).
                         Kept for weight access and analogize() compatibility.
        readout_times: Time points where cost/output is evaluated.
                       Cost is defined over trajectory at these t's (matches Ark's saveat).
        t_span: Integration interval (t0, t1).
        noise: Transient noise profile for SDE simulation.
        metadata: Extra info (stiffness, NFE, architecture details).
    """
    name: str = ""
    family: str = "neural_ode"
    state_dim: int = 0
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)
    dynamics_fn: Optional[Callable] = None
    dynamics_module: Optional[nn.Module] = None
    readout_times: list[float] = field(default_factory=lambda: [1.0])
    t_span: tuple[float, float] = (0.0, 1.0)
    noise: NoiseProfile = field(default_factory=NoiseProfile)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def parameter_count(self) -> int:
        """Total number of scalar parameters in the system."""
        return sum(p.value.numel() for p in self.parameters.values())

    @property
    def bounded_parameter_count(self) -> int:
        """Number of parameters with physical bounds."""
        return sum(1 for p in self.parameters.values() if p.bounds is not None)

    def sample_mismatch(self, sigma: Optional[float] = None) -> ODESystem:
        """Return a new ODESystem with mismatch applied to all parameters.

        Mismatch model (Wang & Achour arXiv:2411.03557 §4.1):
            θ' = δ ⊙ θ,  δ ~ N(1, σ²·I)

        Each parameter gets an independent multiplicative perturbation.
        If sigma is provided, it overrides per-parameter mismatch_sigma.

        After mismatch, parameters are clamped to their bounds (§4.3).

        Returns a new ODESystem with perturbed parameter values and an updated
        dynamics_module whose weights reflect the mismatch.
        """
        perturbed = ODESystem(
            name=self.name,
            family=self.family,
            state_dim=self.state_dim,
            parameters={},
            dynamics_fn=self.dynamics_fn,
            dynamics_module=copy.deepcopy(self.dynamics_module) if self.dynamics_module else None,
            readout_times=list(self.readout_times),
            t_span=self.t_span,
            noise=self.noise,
            metadata=dict(self.metadata),
        )

        for pname, pspec in self.parameters.items():
            s = sigma if sigma is not None else pspec.mismatch_sigma
            if s > 0:
                delta = 1.0 + s * torch.randn_like(pspec.value)
                new_value = pspec.value * delta
            else:
                new_value = pspec.value.clone()

            # Enforce physical bounds (§4.3)
            if pspec.bounds is not None:
                lo, hi = pspec.bounds
                new_value = torch.clamp(new_value, lo, hi)

            perturbed.parameters[pname] = ParameterSpec(
                name=pspec.name,
                value=new_value,
                bounds=pspec.bounds,
                mismatch_sigma=s,
                trainable=pspec.trainable,
                analog_primitive=pspec.analog_primitive,
            )

        # If we have a dynamics module, update its weights to match
        if perturbed.dynamics_module is not None:
            _apply_parameters_to_module(perturbed.dynamics_module, perturbed.parameters)
            perturbed.dynamics_fn = perturbed.dynamics_module

        return perturbed

    def resample_mismatch_inplace(self, sigma: Optional[float] = None) -> None:
        """Apply fresh mismatch to parameters in-place (avoids deepcopy overhead).

        For Monte Carlo sweeps where we need many mismatch samples.
        """
        for pname, pspec in self.parameters.items():
            s = sigma if sigma is not None else pspec.mismatch_sigma
            if s > 0:
                delta = 1.0 + s * torch.randn_like(pspec.value)
                new_value = pspec.value * delta
            else:
                new_value = pspec.value.clone()

            if pspec.bounds is not None:
                lo, hi = pspec.bounds
                new_value = torch.clamp(new_value, lo, hi)

            pspec.value = new_value

        if self.dynamics_module is not None:
            _apply_parameters_to_module(self.dynamics_module, self.parameters)

    def summary(self) -> str:
        """One-paragraph summary of the ODE system."""
        n_params = self.parameter_count
        n_bounded = self.bounded_parameter_count
        noise_str = f"σ_transient={self.noise.sigma:.2e}" if self.noise.is_active else "no transient noise"
        return (
            f"ODESystem '{self.name}' ({self.family}): "
            f"state_dim={self.state_dim}, {n_params:,} parameters "
            f"({n_bounded} bounded), t∈{self.t_span}, "
            f"readout@{self.readout_times}, {noise_str}"
        )


def _apply_parameters_to_module(
    module: nn.Module,
    parameters: dict[str, ParameterSpec],
) -> None:
    """Write perturbed parameter values back into the nn.Module's buffers/params.

    This bridges the ODESystem parameter dict back to PyTorch so we can run
    forward passes through the (now-mismatched) dynamics module.
    """
    param_lookup = {}
    for pname, pspec in parameters.items():
        # Parameter names use dots: "net.0.weight" → module path
        param_lookup[pspec.name] = pspec.value

    for name, param in module.named_parameters():
        if name in param_lookup:
            param.data.copy_(param_lookup[name])

    for name, buf in module.named_buffers():
        if name in param_lookup:
            buf.copy_(param_lookup[name])
