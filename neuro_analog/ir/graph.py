"""
AnalogGraph: container for a complete neuro-analog intermediate representation.

Holds all AnalogNodes for a model and provides graph-level analysis:
- Analog/digital FLOP fractions
- D/A boundary detection and counting
- Critical path analysis
- Partition optimization
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterator  # noqa: F401 — kept for any callers that import it
import math

from .types import (
    ArchitectureFamily, AnalogAmenabilityProfile, Domain, DynamicsProfile, OpType, TargetBackend,
)
from .node import AnalogNode
from .energy_model import HardwareProfile, estimate_node_cost


@dataclass
class DABoundary:
    """A digital-analog boundary where domain conversion (ADC/DAC) is required."""
    source_node_id: str
    target_node_id: str
    source_domain: Domain
    target_domain: Domain

    @property
    def direction(self) -> str:
        if self.source_domain == Domain.ANALOG and self.target_domain == Domain.DIGITAL:
            return "ADC"
        elif self.source_domain == Domain.DIGITAL and self.target_domain == Domain.ANALOG:
            return "DAC"
        return "HYBRID"


class AnalogGraph:
    """Complete analog IR graph for a neural network model."""

    def __init__(self, name: str = "", family: ArchitectureFamily = ArchitectureFamily.TRANSFORMER, model_params: int = 0):
        self.name = name
        self.family = family
        self.model_params = model_params
        self._nodes: dict[str, AnalogNode] = {}
        self._edges: list[tuple[str, str]] = []
        self._dynamics: DynamicsProfile = DynamicsProfile()

    def add_node(self, node: AnalogNode) -> str:
        node_id = node.name if node.name and node.name not in self._nodes else node.node_id
        node.node_id = node_id
        self._nodes[node_id] = node
        return node_id

    def get_node(self, node_id: str) -> AnalogNode | None:
        return self._nodes.get(node_id)

    def add_edge(self, source_id: str, target_id: str):
        assert source_id in self._nodes, f"Source {source_id} not in graph"
        assert target_id in self._nodes, f"Target {target_id} not in graph"
        self._edges.append((source_id, target_id))
        self._nodes[source_id].outputs.append(target_id)
        self._nodes[target_id].inputs.append(source_id)

    def set_dynamics(self, dynamics: DynamicsProfile):
        self._dynamics = dynamics

    def validate(self) -> tuple[bool, list[str]]:
        """Validate graph integrity before analysis.
        
        Checks that all nodes have valid attributes needed for hardware analysis:
        - seq_len is not None (needed for energy/latency calculations)
        - flops >= 0 (non-negative computation count)
        - param_count >= 0 (non-negative parameter count)
        
        Returns:
            (is_valid, errors): Tuple of validation result and list of error messages
        """
        errors = []
        
        for node_id, node in self._nodes.items():
            # Check seq_len is set
            if node.seq_len is None:
                errors.append(
                    f"Node '{node_id}' has seq_len=None. "
                    f"seq_len must be set (use 1 for non-sequential ops)"
                )
            
            # Check flops is non-negative
            if node.flops < 0:
                errors.append(
                    f"Node '{node_id}' has negative flops={node.flops}. "
                    f"FLOPs must be non-negative."
                )
            
            # Check param_count is non-negative
            if node.param_count < 0:
                errors.append(
                    f"Node '{node_id}' has negative param_count={node.param_count}. "
                    f"Parameter count must be non-negative."
                )
        
        is_valid = len(errors) == 0
        return is_valid, errors

    @property
    def nodes(self) -> list[AnalogNode]:
        return list(self._nodes.values())

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def flop_breakdown(self) -> dict[Domain, int]:
        breakdown: dict[Domain, int] = defaultdict(int)
        for node in self._nodes.values():
            breakdown[node.domain] += node.flops
        return dict(breakdown)

    def flop_fractions(self) -> dict[Domain, float]:
        breakdown = self.flop_breakdown()
        total = sum(breakdown.values())
        if total == 0:
            return {d: 0.0 for d in Domain}
        return {d: breakdown.get(d, 0) / total for d in Domain}

    def find_da_boundaries(self) -> list[DABoundary]:
        boundaries = []
        for src_id, tgt_id in self._edges:
            src, tgt = self._nodes[src_id], self._nodes[tgt_id]
            src_d = Domain.ANALOG if src.domain == Domain.HYBRID else src.domain
            tgt_d = Domain.ANALOG if tgt.domain == Domain.HYBRID else tgt.domain
            if src_d != tgt_d:
                boundaries.append(DABoundary(src_id, tgt_id, src.domain, tgt.domain))
        return boundaries

    def compute_stability_bounds(self) -> 'StabilityBounds':
        """Compute analytical stability bounds for analog weight perturbations.

        Uses matrix perturbation theory (Bauer-Fike, Weilandt-Hoffmann) and
        dynamical systems analysis to estimate the maximum σ before the system
        becomes unstable or output quality degrades beyond 10%.

        The bounds are architecture-family-specific:
        - DEQ / implicit equilibrium: spectral radius of the fixed-point Jacobian
        - Neural ODE / Flow: Lipschitz constant vs. integrator stability region
        - SSM: pole location sensitivity to state-matrix perturbation
        - Transformer / EBM / Diffusion: output-variance bound via matrix sensitivity

        Returns:
            StabilityBounds dataclass with all computed limits.
        """
        from .types import StabilityBounds
        import math

        bounds = StabilityBounds()

        # Gather analog-native MVM nodes (the ones that will be implemented on crossbars)
        mvm_nodes = [
            n for n in self._nodes.values()
            if n.op_type == OpType.MVM and n.domain in (Domain.ANALOG, Domain.HYBRID)
        ]

        # Estimate nominal spectral radius for iterative architectures
        # For DEQ/SSM, the state transition matrix A or implicit Jacobian J
        # is approximated from the weight statistics of MVM nodes.
        if self.family in (ArchitectureFamily.DEQ, ArchitectureFamily.SSM):
            # Collect spectral radii estimates from weight statistics
            rho_estimates = []
            for node in mvm_nodes:
                if node.param_count > 0:
                    # Approximate spectral radius from matrix dimensions and weight std
                    # For a random matrix W ∈ ℝ^(n×n) with std σ_w, E[ρ] ≈ σ_w·√n
                    # We use a conservative estimate: ρ ≈ 0.9 (trained DEQs are tuned to be stable)
                    # and then compute how much perturbation pushes it over 1.0
                    n = int(math.sqrt(node.param_count)) if node.param_count > 0 else 1
                    # Nominal spectral radius — trained systems are typically ρ ≈ 0.5–0.9
                    rho_nom = 0.75  # conservative midpoint
                    rho_estimates.append(rho_nom)

            if rho_estimates:
                rho_nom = max(rho_estimates)  # worst-case loop
            else:
                rho_nom = 0.75

            bounds.spectral_radius_nominal = rho_nom
            # Bauer-Fike bound: |λ_perturbed - λ| ≤ ‖ΔW‖_2
            # For Gaussian ΔW with std σ, E[‖ΔW‖_2] ≈ σ·(√m + √n) ≈ 2σ·√n for square n×n
            # We want max eigenvalue < 1, so need σ < (1 - ρ_nom) / (2·√n)
            # Use the largest MVM dimension
            max_n = max(
                (int(math.sqrt(n.param_count)) for n in mvm_nodes if n.param_count > 0),
                default=64
            )
            bounds.max_sigma_spectral = (1.0 - rho_nom) / (2.0 * math.sqrt(max_n)) if rho_nom < 1.0 else 0.0
            bounds.stability_margin_dB = 20.0 * math.log10(1.0 / rho_nom) if rho_nom > 0 else float('inf')

            # SSM time-constant bound
            if self.family == ArchitectureFamily.SSM and self._dynamics.time_constants:
                tcs = self._dynamics.time_constants
                if len(tcs) >= 2 and min(tcs) > 0:
                    bounds.time_constant_spread_dB = 20.0 * math.log10(max(tcs) / min(tcs))
                    # Perturbation shifts poles: Δa ≈ σ·|a|. Stability requires Re(a) < 0.
                    # For diagonal SSM, a_i = -1/τ_i. Margin = min(|a_i|) / max(|a_i|) = min(τ)/max(τ)
                    # Max σ before pole crosses to RHP: σ < min(|a_i|) / (2·max(|a_i|))
                    bounds.max_sigma_timeconstant = min(tcs) / (2.0 * max(tcs))

        # Neural ODE / Flow: Lipschitz-based bound
        if self.family in (ArchitectureFamily.NEURAL_ODE, ArchitectureFamily.FLOW):
            L = self._dynamics.lipschitz_constant
            if L is not None and L > 0:
                bounds.lipschitz_constant = L
                # For explicit Euler with step h, stability requires h·L < 2 (real axis)
                # Analog integrator bandwidth B implies effective h ≈ 1/B.
                # Typical analog bandwidth = 250 kHz → h ≈ 4 μs.
                h_eff = 4e-6  # 4 microseconds effective step (250 kHz bandwidth)
                # Perturbation increases effective L by ~σ·L·√n (relative perturbation)
                max_n = max(
                    (int(math.sqrt(n.param_count)) for n in mvm_nodes if n.param_count > 0),
                    default=64
                )
                # Critical condition: h·L·(1 + σ·√n) < 2
                # → σ < (2/(h·L) - 1) / √n
                crit = (2.0 / (h_eff * L) - 1.0) / math.sqrt(max_n)
                bounds.max_sigma_lipschitz = max(0.0, crit)
            else:
                # Default L = 10 for small toy models
                bounds.lipschitz_constant = 10.0
                h_eff = 4e-6
                max_n = max(
                    (int(math.sqrt(n.param_count)) for n in mvm_nodes if n.param_count > 0),
                    default=64
                )
                crit = (2.0 / (h_eff * 10.0) - 1.0) / math.sqrt(max_n)
                bounds.max_sigma_lipschitz = max(0.0, crit)

        # Output variance bound (applies to all architectures)
        # For a linear system y = Wx, var(y) ≈ σ²·‖x‖²·tr(WWᵀ) / m
        # Simplified: output MSE grows as σ²·E[‖W‖_F²]·E[‖x‖²]
        # We estimate the σ at which output variance = 0.1² (10% relative error)
        total_params = sum(n.param_count for n in mvm_nodes)
        if total_params > 0:
            # Heuristic: for a single layer, σ_output ≈ 0.1 / √(fan_in)
            # For deep networks, variance compounds: multiply by √depth
            depth = max(len(mvm_nodes), 1)
            avg_fan_in = total_params / depth
            bounds.output_sensitivity = 1.0 / math.sqrt(avg_fan_in) if avg_fan_in > 0 else 0.0
            bounds.max_sigma_output_10pct = 0.1 / (math.sqrt(avg_fan_in) * math.sqrt(depth))
        else:
            bounds.max_sigma_output_10pct = 0.05  # default fallback

        return bounds

    def analyze(self, target_backend: TargetBackend | None = None, hardware_profile: HardwareProfile | None = None) -> AnalogAmenabilityProfile:
        """Compute analog amenability profile for this graph.
        
        Args:
            target_backend: Target hardware backend. If specified and not ANALOG_CIRCUIT,
                compute_scores() will raise an error (analog-physics scoring not valid).
                If None (default), assumes analog-circuit backend (legacy behavior).
            hardware_profile: Optional HardwareProfile for energy/latency estimation.
                If provided, computes energy and latency metrics for analog vs digital.
        
        Returns:
            AnalogAmenabilityProfile with computed scores and optionally energy/latency metrics.
        """
        fractions = self.flop_fractions()
        boundaries = self.find_da_boundaries()

        # Derive min precision from node PrecisionSpec (weight_bits field).
        # Falls back to 8 (float32 → 8-bit for analog deployment).
        node_bits = [
            n.precision.weight_bits
            for n in self.nodes
            if n.precision is not None
        ]
        min_bits = min(node_bits) if node_bits else 8

        profile = AnalogAmenabilityProfile(
            architecture=self.family, model_name=self.name, model_params=self.model_params,
            analog_flop_fraction=fractions.get(Domain.ANALOG, 0.0),
            digital_flop_fraction=fractions.get(Domain.DIGITAL, 0.0),
            hybrid_flop_fraction=fractions.get(Domain.HYBRID, 0.0),
            da_boundary_count=len(boundaries), dynamics=self._dynamics,
            min_weight_precision_bits=min_bits,
            layer_count=self.node_count,
            stability_bounds=self.compute_stability_bounds(),
        )
        
        # Compute energy/latency metrics if hardware profile provided
        if hardware_profile is not None:
            analog_energy_pJ = 0.0
            digital_energy_pJ = 0.0
            analog_latency_ns = 0.0
            digital_latency_ns = 0.0
            
            # Estimate cost for each node
            for node in self.nodes:
                # Cost on native hardware (analog for analog nodes, digital for digital nodes)
                native_estimate = estimate_node_cost(node, hardware_profile)
                
                # Cost on digital hardware — this is the true digital baseline for ALL nodes,
                # regardless of their analog-friendliness. We temporarily override domain
                # to DIGITAL so estimate_node_cost uses digital MAC/memory formulas.
                digital_node = replace(node, domain=Domain.DIGITAL)
                digital_estimate = estimate_node_cost(digital_node, hardware_profile)
                
                # Analog deployment cost: analog/hybrid nodes run on analog hardware,
                # digital nodes stay on digital hardware (no analog implementation)
                if node.domain == Domain.ANALOG or node.domain == Domain.HYBRID:
                    node.analog_estimate = native_estimate
                    analog_energy_pJ += native_estimate.energy_pJ
                    analog_latency_ns += native_estimate.latency_ns
                else:
                    node.digital_estimate = digital_estimate
                    analog_energy_pJ += digital_estimate.energy_pJ
                    analog_latency_ns += digital_estimate.latency_ns
                
                # Digital baseline: sum cost of ALL nodes running on digital hardware
                digital_energy_pJ += digital_estimate.energy_pJ
                digital_latency_ns += digital_estimate.latency_ns
            
            # Add ADC/DAC costs for D/A boundaries
            for b in boundaries:
                src_node = self._nodes[b.source_node_id]
                elements_per_token = math.prod(src_node.output_shape) if src_node.output_shape else 1
                # FIX: Handle None seq_len - default to 1 for non-sequential operations
                seq_len = src_node.seq_len if src_node.seq_len is not None else 1
                volume = elements_per_token * seq_len
                
                if b.direction == "ADC":
                    analog_energy_pJ += volume * hardware_profile.adc_energy_pJ
                elif b.direction == "DAC":
                    analog_energy_pJ += volume * hardware_profile.dac_energy_pJ
                
                if hardware_profile.num_parallel_converters > 0:
                    parallel_factor = math.ceil(elements_per_token / hardware_profile.num_parallel_converters)
                    lat_factor = seq_len * parallel_factor
                else:
                    lat_factor = seq_len
                    
                if b.direction == "ADC":
                    analog_latency_ns += lat_factor * hardware_profile.adc_latency_ns
                elif b.direction == "DAC":
                    analog_latency_ns += lat_factor * hardware_profile.dac_latency_ns
            
            # Apply architecture-specific iteration multiplier to digital baseline
            # Iterative architectures (DEQ, Neural ODE, Diffusion, Flow, EBM) execute
            # many digital solver steps per inference. Analog hardware replaces these
            # with physical settling (O(1) per step). Multiply digital energy/latency
            # by the number of digital iterations to get fair per-inference comparison.
            iteration_multiplier = 1.0
            if self._dynamics.has_dynamics:
                if self._dynamics.num_diffusion_steps and self._dynamics.num_diffusion_steps > 1:
                    iteration_multiplier = float(self._dynamics.num_diffusion_steps)
                elif self._dynamics.num_function_evaluations and self._dynamics.num_function_evaluations > 1:
                    iteration_multiplier = float(self._dynamics.num_function_evaluations)
                elif self._dynamics.dynamics_type == "implicit_equilibrium":
                    iteration_multiplier = 30.0  # DEQ default fixed-point iterations
                elif self._dynamics.dynamics_type == "energy_minimization":
                    iteration_multiplier = 500.0  # EBM default Gibbs sweeps
                elif self._dynamics.dynamics_type == "LTI_ODE":
                    iteration_multiplier = 1.0  # SSM: recurrent, no iteration reduction
            digital_energy_pJ *= iteration_multiplier
            digital_latency_ns *= iteration_multiplier
            
            # Store metrics in profile (now consistent: digital values include multiplier)
            profile.analog_energy_pJ = analog_energy_pJ
            profile.digital_energy_pJ = digital_energy_pJ
            profile.analog_latency_ns = analog_latency_ns
            profile.digital_latency_ns = digital_latency_ns
             # Compute speedup and energy savings vs digital baseline
            if digital_latency_ns > 0 and analog_latency_ns > 0:
                profile.analog_speedup_vs_digital = digital_latency_ns / analog_latency_ns
            elif digital_latency_ns > 0 and analog_latency_ns <= 0:
                # Analog latency is zero (no analog nodes detected) — mark as invalid for fallback
                profile.analog_speedup_vs_digital = 0.0
            if digital_energy_pJ > 0 and analog_energy_pJ >= 0:
                raw_saving = 1.0 - (analog_energy_pJ / digital_energy_pJ)
                # Store raw saving WITHOUT clamping.
                # Values > 1.0 mean analog uses <1% of digital energy (common for
                # iterative architectures with large iteration multipliers).
                # Values < 0 mean analog uses MORE energy than digital (possible
                # for attention-heavy models with many ADC/DAC boundaries).
                profile.analog_energy_saving_vs_digital = raw_saving
                # Also store the reduction factor (digital/analog) for unambiguous
                # multiplicative interpretation on posters.
                profile.analog_energy_reduction_factor = (
                    digital_energy_pJ / analog_energy_pJ if analog_energy_pJ > 0 else float("inf")
                )
        
        profile.compute_scores(target_backend=target_backend)
        return profile

    def summary_table(self) -> str:
        fractions = self.flop_fractions()
        boundaries = self.find_da_boundaries()
        lines = [
            f"AnalogGraph: {self.name} ({self.family.value})",
            f"{'='*60}",
            f"Nodes: {self.node_count}  |  Params: {self.model_params:,}",
            "\nFLOP Breakdown:",
        ]
        for domain in Domain:
            frac = fractions.get(domain, 0.0)
            bar = "█" * int(frac * 40)
            lines.append(f"  {domain.name:8s}: {frac:6.1%} {bar}")
        adc = sum(1 for b in boundaries if b.direction == "ADC")
        dac = sum(1 for b in boundaries if b.direction == "DAC")
        lines.append(f"\nD/A Boundaries: {len(boundaries)} ({adc} ADC, {dac} DAC)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize graph to dict (JSON-compatible).
        
        Includes target_backend and is_mixed_signal_boundary when set, but omits
        them when unset (backward compatibility with analog-only graphs).
        """
        nodes_data = []
        for n in self._nodes.values():
            node_dict = {
                "id": n.node_id, "name": n.name, "op_type": n.op_type.name,
                "domain": n.domain.name, "flops": n.flops, "param_count": n.param_count,
            }
            # Include target_backend if set
            if n.target_backend is not None:
                node_dict["target_backend"] = n.target_backend.name
            # Include is_mixed_signal_boundary if True (omit if False for compactness)
            if n.is_mixed_signal_boundary:
                node_dict["is_mixed_signal_boundary"] = True
            nodes_data.append(node_dict)
        
        return {
            "name": self.name, "family": self.family.value, "model_params": self.model_params,
            "nodes": nodes_data,
            "edges": self._edges,
        }
