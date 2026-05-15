"""
Tests for sequence-length-aware profiling accuracy.

Verifies:
1. Factory functions scale FLOPs by seq_len.
2. Physical latency model scales linearly with seq_len.
3. ADC/DAC boundary energy scales by per-element volume.
4. TransformerExtractor.reference() produces consistent scaling at 128 vs 256 tokens.
"""

import math
import pytest

from neuro_analog.ir import AnalogGraph, make_mvm_node, make_norm_node, make_activation_node, AnalogNode, OpType, Domain
from neuro_analog.ir.energy_model import estimate_node_cost, HardwareProfile


# ──────────────────────────────────────────────────────────────────────
# 1. Factory function FLOP scaling
# ──────────────────────────────────────────────────────────────────────

class TestFactoryFlopScaling:
    def test_make_mvm_node_scales_flops(self):
        node1 = make_mvm_node("test", 128, 256, seq_len=1)
        node10 = make_mvm_node("test", 128, 256, seq_len=10)
        assert node10.flops == node1.flops * 10

    def test_make_mvm_node_preserves_shape(self):
        node = make_mvm_node("test", 128, 256, seq_len=64)
        assert node.input_shape == (128,)
        assert node.output_shape == (256,)
        assert node.seq_len == 64

    def test_make_norm_node_scales_flops(self):
        node1 = make_norm_node("test", 768, "layer_norm", seq_len=1)
        node8 = make_norm_node("test", 768, "layer_norm", seq_len=8)
        assert node8.flops == node1.flops * 8

    def test_make_activation_node_scales_flops(self):
        node1 = make_activation_node("test", 1024, "gelu", seq_len=1)
        node4 = make_activation_node("test", 1024, "gelu", seq_len=4)
        assert node4.flops == node1.flops * 4

    def test_mvm_flop_count_is_correct(self):
        """2 * in * out * seq_len for multiply-accumulate."""
        node = make_mvm_node("test", 64, 128, seq_len=10)
        assert node.flops == 2 * 64 * 128 * 10


# ──────────────────────────────────────────────────────────────────────
# 2. Physical latency model
# ──────────────────────────────────────────────────────────────────────

class TestPhysicalLatencyModel:
    def test_mvm_latency_scales_with_seq_len(self):
        hw = HardwareProfile(crossbar_read_latency_ns=2.0, use_tops_latency=False)
        cost1 = estimate_node_cost(make_mvm_node("t", 128, 128, seq_len=1), hw)
        cost128 = estimate_node_cost(make_mvm_node("t", 128, 128, seq_len=128), hw)
        assert cost128.latency_ns == pytest.approx(cost1.latency_ns * 128)

    def test_mvm_energy_scales_with_seq_len(self):
        hw = HardwareProfile(analog_mac_energy_pJ=5.0)
        cost1 = estimate_node_cost(make_mvm_node("t", 64, 64, seq_len=1), hw)
        cost10 = estimate_node_cost(make_mvm_node("t", 64, 64, seq_len=10), hw)
        assert cost10.energy_pJ == pytest.approx(cost1.energy_pJ * 10)

    def test_tops_legacy_latency_flag(self):
        hw = HardwareProfile(use_tops_latency=True, analog_mac_throughput=1e12)
        node = make_mvm_node("t", 128, 128, seq_len=1)
        cost = estimate_node_cost(node, hw)
        expected_macs = node.flops // 2
        expected_latency = expected_macs / 1e12 * 1e9
        assert cost.latency_ns == pytest.approx(expected_latency)

    def test_digital_node_cost_scales(self):
        hw = HardwareProfile()
        node1 = AnalogNode(
            name="soft", op_type=OpType.SOFTMAX, domain=Domain.DIGITAL,
            input_shape=(768,), output_shape=(768,), seq_len=1, flops=768,
        )
        node10 = AnalogNode(
            name="soft", op_type=OpType.SOFTMAX, domain=Domain.DIGITAL,
            input_shape=(768,), output_shape=(768,), seq_len=10, flops=7680,
        )
        c1 = estimate_node_cost(node1, hw)
        c10 = estimate_node_cost(node10, hw)
        assert c10.energy_pJ == pytest.approx(c1.energy_pJ * 10)


# ──────────────────────────────────────────────────────────────────────
# 3. ADC/DAC boundary volume scaling
# ──────────────────────────────────────────────────────────────────────

class TestBoundaryScaling:
    def test_dac_boundary_energy_scales_by_volume(self):
        """Energy should be elements_per_token * seq_len * dac_energy_pJ."""
        graph = AnalogGraph("test_graph")
        hw = HardwareProfile(
            adc_energy_pJ=1.0, dac_energy_pJ=2.0,
            adc_latency_ns=1.0, dac_latency_ns=1.0,
            num_parallel_converters=0,  # fully parallel => latency = seq_len only
        )
        digital_node = AnalogNode(
            name="digi", op_type=OpType.GELU, domain=Domain.DIGITAL,
            input_shape=(128,), output_shape=(128,), seq_len=10, flops=1280,
        )
        analog_node = make_mvm_node("ana", 128, 128, seq_len=10)
        graph.add_node(digital_node)
        graph.add_node(analog_node)
        graph.add_edge("digi", "ana")  # DIGITAL -> ANALOG = DAC boundary

        profile = graph.analyze(hardware_profile=hw)

        # DAC volume = 128 elements * 10 seq = 1280
        # DAC energy = 1280 * 2.0 pJ = 2560 pJ
        assert profile.analog_energy_pJ >= 2560.0

    def test_parallel_converters_reduce_latency(self):
        """With parallel converters, latency should decrease."""
        graph1 = AnalogGraph("test1")
        graph2 = AnalogGraph("test2")
        
        for g in (graph1, graph2):
            d = AnalogNode(name="d", op_type=OpType.GELU, domain=Domain.DIGITAL,
                           input_shape=(128,), output_shape=(128,), seq_len=1, flops=128)
            a = make_mvm_node("a", 128, 128, seq_len=1)
            g.add_node(d)
            g.add_node(a)
            g.add_edge("d", "a")

        hw_serial = HardwareProfile(dac_latency_ns=10.0, num_parallel_converters=1)
        hw_parallel = HardwareProfile(dac_latency_ns=10.0, num_parallel_converters=128)

        p1 = graph1.analyze(hardware_profile=hw_serial)
        p2 = graph2.analyze(hardware_profile=hw_parallel)
        
        # Serial: ceil(128/1) * 1 * 10 = 1280 ns boundary latency
        # Parallel: ceil(128/128) * 1 * 10 = 10 ns boundary latency
        assert p1.analog_latency_ns > p2.analog_latency_ns


# ──────────────────────────────────────────────────────────────────────
# 4. TransformerExtractor scaling test
# ──────────────────────────────────────────────────────────────────────

class TestTransformerScaling:
    def test_reference_128_vs_256(self):
        from neuro_analog.extractors.transformer import TransformerExtractor
        
        ext128 = TransformerExtractor.reference(dim=256, n_layers=2, heads=4, seq_len=128)
        ext256 = TransformerExtractor.reference(dim=256, n_layers=2, heads=4, seq_len=256)
        
        g128 = ext128._graph
        g256 = ext256._graph

        flops_128 = sum(n.flops for n in g128.nodes)
        flops_256 = sum(n.flops for n in g256.nodes)

        # Total FLOPs should increase when seq_len doubles.
        # MVM FLOPs scale 2x, attention FLOPs scale 4x (quadratic in seq_len).
        assert flops_256 > flops_128 * 1.5  # Conservative: must be meaningfully larger

    def test_all_nodes_have_seq_len_set(self):
        from neuro_analog.extractors.transformer import TransformerExtractor
        ext = TransformerExtractor.reference(dim=128, n_layers=1, heads=2, seq_len=64)
        
        for node in ext._graph.nodes:
            assert node.seq_len == 64, f"Node {node.name} has seq_len={node.seq_len}, expected 64"

    def test_mvm_shapes_are_per_token(self):
        """Shapes should NOT contain seq_len — they represent per-token dimensions."""
        from neuro_analog.extractors.transformer import TransformerExtractor
        ext = TransformerExtractor.reference(dim=256, n_layers=1, heads=4, seq_len=512)
        
        for node in ext._graph.nodes:
            if node.op_type == OpType.MVM:
                # input_shape and output_shape should be small per-token shapes, not inflated by 512
                assert max(node.input_shape) <= 1024  # 4*dim at most for FFN
                assert max(node.output_shape) <= 1024
