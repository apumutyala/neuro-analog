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
from dataclasses import dataclass, field
from typing import Iterator  # noqa: F401 — kept for any callers that import it

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
        )
        
        # Compute energy/latency metrics if hardware profile provided
        if hardware_profile is not None:
            analog_energy_pJ = 0.0
            digital_energy_pJ = 0.0
            analog_latency_ns = 0.0
            digital_latency_ns = 0.0
            
            # Estimate cost for each node
            for node in self.nodes:
                estimate = estimate_node_cost(node, hardware_profile)
                
                # Store estimate on node
                if node.domain == Domain.ANALOG or node.domain == Domain.HYBRID:
                    node.analog_estimate = estimate
                    analog_energy_pJ += estimate.energy_pJ
                    analog_latency_ns += estimate.latency_ns
                else:
                    node.digital_estimate = estimate
                    digital_energy_pJ += estimate.energy_pJ
                    digital_latency_ns += estimate.latency_ns
            
            # Add ADC/DAC costs for D/A boundaries
            adc_count = sum(1 for b in boundaries if b.direction == "ADC")
            dac_count = sum(1 for b in boundaries if b.direction == "DAC")
            
            analog_energy_pJ += adc_count * hardware_profile.adc_energy_pJ
            analog_energy_pJ += dac_count * hardware_profile.dac_energy_pJ
            analog_latency_ns += adc_count * hardware_profile.adc_latency_ns
            analog_latency_ns += dac_count * hardware_profile.dac_latency_ns
            
            # Store metrics in profile
            profile.analog_energy_pJ = analog_energy_pJ
            profile.digital_energy_pJ = digital_energy_pJ
            profile.analog_latency_ns = analog_latency_ns
            profile.digital_latency_ns = digital_latency_ns
            
            # Compute speedup and energy savings vs digital baseline
            if digital_latency_ns > 0:
                profile.analog_speedup_vs_digital = digital_latency_ns / analog_latency_ns if analog_latency_ns > 0 else 0.0
            if digital_energy_pJ > 0:
                profile.analog_energy_saving_vs_digital = 1.0 - (analog_energy_pJ / digital_energy_pJ)
        
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
