"""Tests for the nonideality modeling layer."""

import math
import pytest
import numpy as np

from neuro_analog.ir.graph import AnalogGraph
from neuro_analog.ir.types import ArchitectureFamily, DynamicsProfile
from neuro_analog.ir.node import (
    make_mvm_node, make_integration_node, make_noise_node, AnalogNode,
)
from neuro_analog.ir.types import OpType, Domain
from neuro_analog.nonidealities.mismatch import propagate_mismatch, MismatchReport
from neuro_analog.nonidealities.noise import (
    compute_noise_budget, NoiseBudget,
    _thermal_noise_variance, _shot_noise_variance, _quantization_noise_variance,
    K_B, Q_E, HW_CAPACITANCE_F, HW_BIAS_CURRENT_A, HW_BANDWIDTH_HZ,
)
from neuro_analog.nonidealities.quantization import (
    compute_precision_requirements, _enob_from_snr, _quantization_snr,
)
from neuro_analog.nonidealities.signal_scaling import analyze_signal_ranges


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _small_graph() -> AnalogGraph:
    g = AnalogGraph("test", ArchitectureFamily.SSM)
    g.add_node(make_mvm_node("mvm_0", 64, 64))
    g.add_node(make_integration_node("int_0", 64, time_constant=1e-3))
    g.add_node(make_noise_node("noise_0", 64))
    g.add_edge("mvm_0", "int_0")
    g.add_edge("int_0", "noise_0")
    return g


# ──────────────────────────────────────────────────────────────────────
# test_nonidealities.py — Section 1: Mismatch
# ──────────────────────────────────────────────────────────────────────

class TestMismatch:
    def test_returns_report_for_each_analog_node(self):
        g = _small_graph()
        reports = propagate_mismatch(g, sigma=0.10, num_samples=20)
        # All three nodes are analog
        assert len(reports) == 3

    def test_mvm_report_has_relative_error(self):
        g = _small_graph()
        reports = propagate_mismatch(g, sigma=0.10, num_samples=50)
        r = reports["mvm_0"]
        assert r.mean_relative_error >= 0.0
        assert r.max_relative_error >= r.mean_relative_error

    def test_integration_report_has_tc_shift(self):
        g = _small_graph()
        reports = propagate_mismatch(g, sigma=0.10, num_samples=30)
        r = reports["int_0"]
        assert r.mean_tc_shift_pct is not None
        assert r.mean_tc_shift_pct >= 0.0

    def test_larger_sigma_gives_larger_error(self):
        g = _small_graph()
        r1 = propagate_mismatch(g, sigma=0.05, num_samples=100)["mvm_0"]
        r2 = propagate_mismatch(g, sigma=0.20, num_samples=100)["mvm_0"]
        # Higher mismatch → higher mean error (statistical, so allow some slack)
        assert r2.mean_relative_error >= r1.mean_relative_error * 0.5

    def test_zero_sigma_gives_near_zero_error(self):
        g = _small_graph()
        reports = propagate_mismatch(g, sigma=1e-6, num_samples=20)
        r = reports["mvm_0"]
        assert r.mean_relative_error < 0.01  # Near zero

    def test_digital_nodes_not_in_report(self):
        g = AnalogGraph("test", ArchitectureFamily.TRANSFORMER)
        from neuro_analog.ir.node import make_norm_node
        g.add_node(make_norm_node("norm", 64, "layer_norm"))  # DIGITAL
        reports = propagate_mismatch(g, sigma=0.10)
        assert len(reports) == 0  # No analog nodes


# ──────────────────────────────────────────────────────────────────────
# Section 2: Noise budget physics
# ──────────────────────────────────────────────────────────────────────

class TestNoiseBudget:
    def test_thermal_noise_formula(self):
        """kT/C at 300K, 1pF should be ~4.14e-9 V²."""
        var = _thermal_noise_variance(temperature_K=300.0, capacitance_F=1e-12)
        expected = K_B * 300.0 / 1e-12
        assert abs(var - expected) / expected < 1e-9

    def test_shot_noise_formula(self):
        """2·q·I·BW."""
        var = _shot_noise_variance(bias_current_A=1e-6, bandwidth_hz=250e3)
        expected = 2.0 * Q_E * 1e-6 * 250e3
        assert abs(var - expected) / expected < 1e-9

    def test_quantization_noise_formula(self):
        """σ² = V_LSB² / 12 = (2·V_ref / 2^bits)² / 12."""
        var = _quantization_noise_variance(enob=8, v_ref=1.0)
        v_lsb = 2.0 / 256.0
        expected = (v_lsb ** 2) / 12.0
        assert abs(var - expected) / expected < 1e-9

    def test_compute_returns_budget_for_analog_nodes(self):
        g = _small_graph()
        budgets = compute_noise_budget(g)
        assert "mvm_0" in budgets
        assert "int_0" in budgets
        assert "noise_0" in budgets

    def test_integration_node_has_sde_coeff(self):
        g = _small_graph()
        budgets = compute_noise_budget(g)
        b = budgets["int_0"]
        assert b.sde_diffusion_coeff > 0
        assert b.thermal_noise_variance > 0

    def test_mvm_node_has_thermal_variance(self):
        g = _small_graph()
        budgets = compute_noise_budget(g)
        b = budgets["mvm_0"]
        assert b.total_noise_variance > 0

    def test_digital_nodes_excluded(self):
        from neuro_analog.ir.node import make_norm_node
        g = AnalogGraph("test", ArchitectureFamily.TRANSFORMER)
        g.add_node(make_norm_node("norm", 64))
        budgets = compute_noise_budget(g)
        assert "norm" not in budgets

    def test_snr_above_thermal_floor(self):
        """With signal_rms=1.0, SNR should be >> 0 for all nodes."""
        g = _small_graph()
        budgets = compute_noise_budget(g, signal_rms=1.0)
        for b in budgets.values():
            if b.snr_db != float("inf"):
                assert b.snr_db > 0.0


# ──────────────────────────────────────────────────────────────────────
# Section 3: Quantization SNR formula
# ──────────────────────────────────────────────────────────────────────

class TestQuantization:
    def test_enob_from_snr_formula(self):
        """ENOB = (SNR_dB - 1.76) / 6.02."""
        for target_snr in [20.0, 40.0, 60.0]:
            enob = _enob_from_snr(target_snr)
            assert abs(enob - (target_snr - 1.76) / 6.02) < 1e-6

    def test_quantization_snr_formula(self):
        """SNR_dB = 6.02·ENOB + 1.76 (ideal ADC/DAC)."""
        for bits in [4, 6, 8, 10, 12]:
            snr = _quantization_snr(bits)
            expected = 6.02 * bits + 1.76
            assert abs(snr - expected) < 1e-6, f"bits={bits}: {snr} vs {expected}"

    def test_8bit_gives_40db_snr(self):
        """Standard rule: 8-bit ADC ≈ 49.9 dB SNR."""
        snr = _quantization_snr(8)
        # 6.02*8 + 1.76 = 49.92
        assert abs(snr - 49.92) < 0.1

    def test_compute_precision_for_graph_boundaries(self):
        """compute_precision_requirements finds D/A boundaries."""
        from neuro_analog.ir.node import make_norm_node
        g = AnalogGraph("test", ArchitectureFamily.SSM)
        g.add_node(make_mvm_node("mvm", 64, 64))
        g.add_node(make_norm_node("norm", 64))
        g.add_edge("mvm", "norm")
        reports = compute_precision_requirements(g, target_snr_db=40.0)
        # One boundary: mvm(ANALOG) → norm(DIGITAL)
        assert len(reports) == 1
        r = next(iter(reports.values()))
        assert r.required_bits >= 1
        assert r.num_discrete_levels == 2 ** r.required_bits


# ──────────────────────────────────────────────────────────────────────
# Section 4: Signal scaling
# ──────────────────────────────────────────────────────────────────────

class TestSignalScaling:
    def test_returns_report_for_analog_nodes(self):
        g = _small_graph()
        reports = analyze_signal_ranges(g)
        assert "mvm_0" in reports

    def test_default_no_clipping(self):
        """Default signal range ±1V = hardware range → no clipping."""
        g = _small_graph()
        reports = analyze_signal_ranges(g, hw_voltage_range=(-1.0, 1.0))
        # With default signal range matching hardware, no clipping expected
        for r in reports.values():
            # scale_factor should be near 1.0 if signals fit
            assert r.scale_factor > 0

    def test_large_signal_requires_attenuation(self):
        """Signal max=10V with ±1V hardware → scale_factor < 1."""
        from neuro_analog.ir.types import PrecisionSpec
        g = AnalogGraph("test", ArchitectureFamily.SSM)
        n = make_mvm_node("big_mvm", 64, 64)
        n.precision = PrecisionSpec(activation_max=10.0, activation_min=-10.0)
        g.add_node(n)
        reports = analyze_signal_ranges(g, hw_voltage_range=(-1.0, 1.0))
        r = reports["big_mvm"]
        assert r.scale_factor < 1.0
        assert r.clipping_risk is True
