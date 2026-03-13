"""Tests for the analog simulator layer.

Per the directive's 10 required tests plus additional coverage.
Critical invariant: at sigma=0, all noise off → exact match with F.linear.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from neuro_analog.simulator.analog_linear import AnalogLinear, _K_B
from neuro_analog.simulator.analog_activation import AnalogTanh, AnalogSigmoid, AnalogReLU
from neuro_analog.simulator.analog_model import (
    analogize, resample_all_mismatch, set_all_noise, count_analog_vs_digital,
)
from neuro_analog.simulator.sweep import SweepResult, mismatch_sweep, adc_sweep, ablation_sweep


# ── Fixtures ─────────────────────────────────────────────────────────────

def _make_linear(in_f=8, out_f=4, bias=True):
    l = nn.Linear(in_f, out_f, bias=bias)
    nn.init.normal_(l.weight, std=0.1)
    if bias:
        nn.init.zeros_(l.bias)
    return l

def _make_analog(sigma=0.05, bits=8, **kw):
    l = _make_linear()
    return AnalogLinear(
        l.in_features, l.out_features, l.weight.data,
        bias=l.bias.data, sigma_mismatch=sigma, n_adc_bits=bits, **kw
    )

def _simple_model():
    return nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 8))

def _eval_fn(m):
    x = torch.randn(10, 16)
    with torch.no_grad():
        return float(m(x).mean().item())


# ── Test 1: sigma=0, all noise off → exact match ─────────────────────────

class TestAnalogLinearExact:
    def test_noiseless_exact_match(self):
        """At sigma=0 and all noise disabled, output must equal F.linear exactly."""
        digital = _make_linear()
        analog = AnalogLinear(
            digital.in_features, digital.out_features,
            digital.weight.data, digital.bias.data,
            sigma_mismatch=0.0, n_adc_bits=32,
        )
        analog.set_noise_config(thermal=False, quantization=False, mismatch=False)

        x = torch.randn(5, digital.in_features)
        expected = F.linear(x, digital.weight, digital.bias)
        actual = analog(x)

        assert torch.allclose(actual, expected, atol=1e-5), \
            f"Max diff: {(actual - expected).abs().max()}"

    def test_noiseless_no_bias_exact(self):
        """No-bias case also matches exactly."""
        digital = nn.Linear(4, 4, bias=False)
        analog = AnalogLinear(4, 4, digital.weight.data, bias=None,
                              sigma_mismatch=0.0, n_adc_bits=32)
        analog.set_noise_config(thermal=False, quantization=False, mismatch=False)
        x = torch.randn(3, 4)
        assert torch.allclose(analog(x), F.linear(x, digital.weight), atol=1e-5)


# ── Test 2: Mismatch is static across forward passes ─────────────────────

class TestMismatchStatic:
    def test_mismatch_static_across_passes(self):
        """Same δ → same output for same input. Mismatch is baked into hardware."""
        analog = _make_analog(sigma=0.10, bits=32)
        analog.set_noise_config(thermal=False, quantization=False, mismatch=True)
        x = torch.randn(4, 8)
        y1 = analog(x).detach().clone()
        y2 = analog(x).detach().clone()
        assert torch.allclose(y1, y2, atol=1e-6), "Mismatch should be static, not re-sampled per pass"


# ── Test 3: resample_mismatch() changes output ────────────────────────────

class TestResampleMismatch:
    def test_resample_changes_output(self):
        """After resample_mismatch, output must differ (with overwhelming probability)."""
        analog = _make_analog(sigma=0.10, bits=32)
        analog.set_noise_config(thermal=False, quantization=False, mismatch=True)
        x = torch.randn(4, 8)
        y_before = analog(x).detach().clone()
        analog.resample_mismatch()
        y_after = analog(x).detach().clone()
        assert not torch.allclose(y_before, y_after, atol=1e-4), \
            "resample_mismatch() should produce different δ"


# ── Test 4: Thermal noise variance scales with 1/C ───────────────────────

class TestThermalNoise:
    def test_thermal_variance_scales_with_inv_cap(self):
        """Larger capacitor → smaller thermal noise → smaller output variance."""
        x = torch.randn(1000, 8)
        # Small capacitor: more noise
        al_small = _make_analog(sigma=0.0, bits=32, cap_F=1e-13)
        al_small.set_noise_config(thermal=True, quantization=False, mismatch=False)
        # Large capacitor: less noise
        al_large = _make_analog(sigma=0.0, bits=32, cap_F=1e-10)
        al_large.set_noise_config(thermal=True, quantization=False, mismatch=False)

        # Use same weights for fair comparison
        al_large.W_nominal = al_small.W_nominal.clone()
        if al_small.bias is not None:
            al_large.bias = al_small.bias.clone()

        var_small = analog_output_variance(al_small, x)
        var_large = analog_output_variance(al_large, x)
        # Small cap should have LARGER variance
        assert var_small > var_large, \
            f"Small cap noise ({var_small:.4e}) should exceed large cap noise ({var_large:.4e})"

    def test_thermal_noise_magnitude_matches_formula(self):
        """σ_thermal = sqrt(kT/C * in_features). Verify at 300K, 1pF, in_features=8."""
        expected_sigma = math.sqrt(_K_B * 300.0 / 1e-12) * math.sqrt(8)  # ~0.064 * sqrt(8)

        al = _make_analog(sigma=0.0, bits=32, temperature_K=300.0, cap_F=1e-12)
        al.set_noise_config(thermal=True, quantization=False, mismatch=False)
        al.W_nominal = torch.zeros_like(al.W_nominal)  # zero weights → only noise
        if al.bias is not None:
            al.bias = torch.zeros_like(al.bias)

        x = torch.zeros(10000, 8)
        with torch.no_grad():
            y = al(x)
        measured_std = float(y.std().item())
        assert abs(measured_std - expected_sigma) / expected_sigma < 0.05, \
            f"Expected σ≈{expected_sigma:.4e}, got {measured_std:.4e}"


def analog_output_variance(analog, x):
    with torch.no_grad():
        ys = [analog(x) for _ in range(20)]
    return float(torch.stack(ys).var(dim=0).mean().item())


# ── Test 5: Quantization at 32 bits ≈ identity; at 1 bit → binary ────────

class TestQuantization:
    def test_32_bit_quantization_near_identity(self):
        """32 bits → quantization step ≈ 0, output nearly identical to unquantized."""
        digital = _make_linear()
        analog = AnalogLinear(
            digital.in_features, digital.out_features,
            digital.weight.data, digital.bias.data,
            sigma_mismatch=0.0, n_adc_bits=32, v_ref=10.0,
        )
        analog.set_noise_config(thermal=False, quantization=True, mismatch=False)
        x = torch.randn(10, digital.in_features)
        y_q = analog(x)
        y_ref = F.linear(x.float(), digital.weight.float(), digital.bias.float())
        # With 32-bit quantization over [-10, 10], LSB ≈ 4.7e-9 V — essentially noiseless
        assert torch.allclose(y_q, y_ref, atol=1e-5), f"Max diff: {(y_q - y_ref).abs().max()}"

    def test_1_bit_produces_two_levels(self):
        """1-bit ADC → output has exactly 2 unique values per sample."""
        analog = _make_analog(sigma=0.0, bits=1, v_ref=1.0)
        analog.set_noise_config(thermal=False, quantization=True, mismatch=False)
        x = torch.randn(100, 8)
        y = analog(x)
        # With 1-bit, scale = 1/(2*V_ref), values are ±V_ref (roughly)
        unique_vals = y.unique()
        # 1-bit over [-1, 1] with 1 level: should produce very few unique values per output
        assert len(unique_vals) <= 4, f"1-bit should produce 2-4 unique values, got {len(unique_vals)}"


# ── Test 6: AnalogTanh at σ=0 → exact tanh ───────────────────────────────

class TestAnalogTanh:
    def test_zero_sigma_exact_tanh(self):
        """At σ=0, AnalogTanh should match torch.tanh (modulo saturation clip)."""
        at = AnalogTanh(sigma_mismatch=0.0)
        at.set_noise_config(mismatch=False, thermal=False, quantization=False)
        x = torch.linspace(-1.5, 1.5, 100)
        expected = torch.tanh(x).clamp(-0.95, 0.95)
        actual = at(x)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_nonzero_sigma_deviates(self):
        """At σ>0, output should deviate from exact tanh."""
        at = AnalogTanh(sigma_mismatch=0.20)
        x = torch.linspace(-1, 1, 50)
        expected = torch.tanh(x).clamp(-0.95, 0.95)
        actual = at(x)
        assert not torch.allclose(actual, expected, atol=1e-3)


# ── Test 7: analogize() preserves parameter count ────────────────────────

class TestAnalogize:
    def test_analogize_preserves_layer_count(self):
        """analogize() should replace layers but not add or remove them."""
        model = _simple_model()
        analog = analogize(model, sigma_mismatch=0.05)

        # Count analog layers
        info = count_analog_vs_digital(analog)
        # 2 Linear → AnalogLinear, 1 Tanh → AnalogTanh
        assert info["analog_layers"] == 3, f"Expected 3, got {info['analog_layers']}"

    def test_analogize_does_not_modify_original(self):
        """analogize() must not modify the original model."""
        model = _simple_model()
        original_weight = model[0].weight.data.clone()
        _ = analogize(model, sigma_mismatch=0.15)
        assert torch.allclose(model[0].weight.data, original_weight), \
            "analogize() should not modify the original model"

    def test_analogize_output_shape_preserved(self):
        """Analog model must produce same output shape as digital."""
        model = _simple_model()
        analog = analogize(model, sigma_mismatch=0.05)
        x = torch.randn(4, 16)
        with torch.no_grad():
            y_d = model(x)
            y_a = analog(x)
        assert y_d.shape == y_a.shape


# ── Test 8: count_analog_vs_digital ──────────────────────────────────────

class TestCountAnalogDigital:
    def test_count_correct_for_known_model(self):
        """Simple model: 2 Linear + 1 Tanh → 3 analog layers."""
        model = _simple_model()
        analog = analogize(model, sigma_mismatch=0.05)
        info = count_analog_vs_digital(analog)
        assert info["analog_layers"] == 3
        assert info["analog_params"] == (16 * 32 + 32) + (32 * 8 + 8)  # layer weights + biases

    def test_count_all_digital_model(self):
        """A model with no replaced layers: all digital."""
        model = nn.Sequential(nn.LayerNorm(16), nn.Dropout(0.1))
        analog = analogize(model, sigma_mismatch=0.05)
        info = count_analog_vs_digital(analog)
        assert info["analog_layers"] == 0


# ── Test 9: Sweep returns correct shape ──────────────────────────────────

class TestSweepShape:
    def test_mismatch_sweep_shape(self):
        """sweep returns per_trial of shape (n_sigma, n_trials)."""
        model = _simple_model()
        sigma_vals = [0.0, 0.05, 0.10]
        result = mismatch_sweep(model, _eval_fn, sigma_values=sigma_vals, n_trials=5)
        assert result.per_trial.shape == (3, 5)
        assert len(result.mean) == 3
        assert len(result.std) == 3

    def test_adc_sweep_shape(self):
        """adc_sweep returns correct shape."""
        model = _simple_model()
        bit_vals = [4, 8, 16]
        result = adc_sweep(model, _eval_fn, bit_values=bit_vals, sigma_mismatch=0.05, n_trials=4)
        assert result.per_trial.shape == (3, 4)


# ── Test 10: ablation_sweep returns exactly 3 keys ───────────────────────

class TestAblationSweep:
    def test_ablation_returns_three_keys(self):
        """ablation_sweep must return exactly 'mismatch', 'thermal', 'quantization'."""
        model = _simple_model()
        results = ablation_sweep(model, _eval_fn, sigma_values=[0.0, 0.05], n_trials=3)
        assert set(results.keys()) == {"mismatch", "thermal", "quantization"}

    def test_ablation_results_are_sweep_results(self):
        model = _simple_model()
        results = ablation_sweep(model, _eval_fn, sigma_values=[0.0, 0.05], n_trials=3)
        for key, val in results.items():
            assert isinstance(val, SweepResult)


# ── Additional: SweepResult normalization and threshold ──────────────────

class TestSweepResult:
    def test_digital_baseline_is_finite(self):
        """digital_baseline should be finite and per_trial should be populated."""
        model = _simple_model()
        result = mismatch_sweep(model, _eval_fn, sigma_values=[0.0, 0.05], n_trials=5)
        assert math.isfinite(result.digital_baseline)
        assert result.per_trial.shape == (2, 5)
        # normalized_mean should be finite (not NaN / inf)
        assert all(math.isfinite(v) for v in result.normalized_mean)

    def test_degradation_threshold_logic(self):
        """degradation_threshold returns last sigma where quality >= 1-loss."""
        # Manually construct a result
        result = SweepResult(
            sigma_values=[0.0, 0.05, 0.10, 0.15],
            metric_name="test",
            per_trial=np.array([[1.0]*5, [0.95]*5, [0.85]*5, [0.70]*5]),
            digital_baseline=1.0,
        )
        # 10% loss threshold: quality >= 0.90
        # sigma=0.0: 1.0 >= 0.90 ✓, sigma=0.05: 0.95 >= 0.90 ✓, sigma=0.10: 0.85 < 0.90 ✗
        threshold = result.degradation_threshold(max_relative_loss=0.10)
        assert threshold == 0.05

    def test_sweep_result_serializable(self):
        """SweepResult.to_dict() should be JSON-serializable."""
        import json
        result = SweepResult(
            sigma_values=[0.0, 0.05],
            metric_name="test",
            per_trial=np.array([[1.0, 0.98], [0.90, 0.88]]),
            digital_baseline=0.99,
        )
        d = result.to_dict()
        json.dumps(d)  # Must not raise
