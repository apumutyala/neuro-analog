"""
Example: Energy/Latency Modeling and Analog Amenability Scoring

This example demonstrates:
1. Loading a sweep result JSON file
2. Building an AnalogGraph for a model
3. Computing energy/latency metrics with HardwareProfile
4. Computing amenability score from sweep results
5. Classifying failure mode and printing design recommendations
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
import sys
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from neuro_analog.ir import AnalogGraph, AnalogNode, OpType, Domain, ArchitectureFamily, DynamicsProfile
from neuro_analog.ir.energy_model import HardwareProfile, compute_amenability_score, estimate_node_cost
from neuro_analog.analysis.design_heuristics import classify_failure_mode, print_heuristic_report


def create_simple_mlp_graph() -> AnalogGraph:
    """Create a simple AnalogGraph for an MLP model."""
    graph = AnalogGraph(name="simple_mlp", family=ArchitectureFamily.TRANSFORMER, model_params=1000)
    
    # Input layer (784 -> 256)
    input_layer = AnalogNode(
        name="layer1",
        op_type=OpType.MVM,
        domain=Domain.ANALOG,
        input_shape=(784,),
        output_shape=(256,),
        weight_shape=(784, 256),
        flops=2 * 784 * 256,
        param_count=784 * 256,
    )
    graph.add_node(input_layer)
    
    # Activation
    activation = AnalogNode(
        name="relu1",
        op_type=OpType.ANALOG_RELU,
        domain=Domain.ANALOG,
        input_shape=(256,),
        output_shape=(256,),
        flops=256,
        param_count=0,
    )
    graph.add_node(activation)
    graph.add_edge(input_layer.node_id, activation.node_id)
    
    # Hidden layer (256 -> 128)
    hidden_layer = AnalogNode(
        name="layer2",
        op_type=OpType.MVM,
        domain=Domain.ANALOG,
        input_shape=(256,),
        output_shape=(128,),
        weight_shape=(256, 128),
        flops=2 * 256 * 128,
        param_count=256 * 128,
    )
    graph.add_node(hidden_layer)
    graph.add_edge(activation.node_id, hidden_layer.node_id)
    
    # Output layer (128 -> 10)
    output_layer = AnalogNode(
        name="layer3",
        op_type=OpType.MVM,
        domain=Domain.ANALOG,
        input_shape=(128,),
        output_shape=(10,),
        weight_shape=(128, 10),
        flops=2 * 128 * 10,
        param_count=128 * 10,
    )
    graph.add_node(output_layer)
    graph.add_edge(hidden_layer.node_id, output_layer.node_id)
    
    # Softmax (digital)
    softmax = AnalogNode(
        name="softmax",
        op_type=OpType.SOFTMAX,
        domain=Domain.DIGITAL,
        input_shape=(10,),
        output_shape=(10,),
        flops=10,
        param_count=0,
    )
    graph.add_node(softmax)
    graph.add_edge(output_layer.node_id, softmax.node_id)
    
    return graph


def main():
    print("=" * 70)
    print("Energy/Latency Modeling and Analog Amenability Scoring Example")
    print("=" * 70)
    
    # 1. Create or load hardware profile
    print("\n1. Loading hardware profile...")
    profile = HardwareProfile()  # Use default values
    # Or load from config:
    # profile = HardwareProfile.from_config("configs/hardware_profile.yaml")
    print(f"   Analog MAC energy: {profile.analog_mac_energy_pJ} pJ")
    print(f"   Digital MAC energy: {profile.digital_mac_energy_pJ} pJ")
    print(f"   ADC energy: {profile.adc_energy_pJ} pJ")
    
    # 2. Build AnalogGraph
    print("\n2. Building AnalogGraph...")
    graph = create_simple_mlp_graph()
    print(f"   Graph: {graph.name}")
    print(f"   Nodes: {graph.node_count}")
    print(f"   FLOP breakdown: {graph.flop_fractions()}")
    
    # 3. Analyze with energy/latency estimation
    print("\n3. Analyzing with energy/latency estimation...")
    amenability_profile = graph.analyze(hardware_profile=profile)
    print(f"   Analog energy: {amenability_profile.analog_energy_pJ:.2e} pJ")
    print(f"   Digital energy: {amenability_profile.digital_energy_pJ:.2e} pJ")
    print(f"   Analog latency: {amenability_profile.analog_latency_ns:.2e} ns")
    print(f"   Digital latency: {amenability_profile.digital_latency_ns:.2e} ns")
    print(f"   Energy saving: {amenability_profile.analog_energy_saving_vs_digital:.1%}")
    print(f"   Speedup: {amenability_profile.analog_speedup_vs_digital:.2f}x")
    
    # 4. Simulate sweep result (in practice, load from JSON)
    print("\n4. Simulating sweep result...")
    # In practice, load from sweep results JSON:
    # with open("results/transformer_mismatch.json", "r") as f:
    #     sweep_result = json.load(f)
    # sigma_10pct = sweep_result.get("degradation_threshold_10pct", 0.0)
    
    # For this example, simulate a good result
    amenability_profile.sigma_10pct = 0.15  # 15% mismatch threshold
    print(f"   Sigma @ 10% degradation: {amenability_profile.sigma_10pct:.1%}")
    
    # 5. Compute amenability score
    print("\n5. Computing amenability score...")
    amenability_profile.amenability_score = compute_amenability_score(amenability_profile)
    print(f"   Amenability score: {amenability_profile.amenability_score:.3f}")
    
    # 6. Classify failure mode
    print("\n6. Classifying failure mode...")
    failure_mode = classify_failure_mode(amenability_profile)
    print(f"   Failure mode: {failure_mode}")
    
    # 7. Print detailed heuristic report
    print_heuristic_report(amenability_profile)
    
    # 8. Demonstrate per-node cost estimation
    print("\n8. Per-node cost estimation:")
    for node in graph.nodes:
        estimate = estimate_node_cost(node, profile)
        print(f"   {node.name:20s} {node.op_type.name:20s} E={estimate.energy_pJ:10.2e} pJ T={estimate.latency_ns:10.2e} ns")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
