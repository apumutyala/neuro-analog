#!/usr/bin/env python3
"""Diagnostic tests to understand why analog degradation is minimal."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from models import transformer, neural_ode
from neuro_analog.simulator import analogize, resample_all_mismatch, set_all_noise

print("=" * 70)
print("ANALOG HARDWARE DIAGNOSTIC TESTS")
print("=" * 70)

# Load a trained model (transformer for simplicity)
model = transformer.load_model("checkpoints/transformer.pt")
print(f"\n✓ Loaded model: {transformer.get_family_name()}")

# Test 1: Check activation magnitudes and V_ref
print("\n" + "=" * 70)
print("TEST 1: Activation Magnitudes vs V_ref")
print("=" * 70)

# Get some test data
X_train, y_train, X_test, y_test = transformer._get_data()
sample_input = X_test[:10]

# Run through digital model
model.eval()
with torch.no_grad():
    digital_out = model(sample_input)
    
print(f"Input range: [{sample_input.min():.4f}, {sample_input.max():.4f}]")
print(f"Output range: [{digital_out.min():.4f}, {digital_out.max():.4f}]")

# Analogize and check V_ref for each layer
analog_model = analogize(model, sigma_mismatch=0.05, n_adc_bits=8)
print(f"\nAnalog layers V_ref values:")
for name, module in analog_model.named_modules():
    if hasattr(module, 'v_ref'):
        print(f"  {name:20s}: V_ref = {module.v_ref:.4f}")
        print(f"    Quantization step (8-bit): {2 * module.v_ref / 255:.6f}")

# Test 2: Verify mismatch is actually sampled
print("\n" + "=" * 70)
print("TEST 2: Verify Conductance Mismatch")
print("=" * 70)

analog_model = analogize(model, sigma_mismatch=0.10, n_adc_bits=16)
for name, module in analog_model.named_modules():
    if hasattr(module, 'delta'):
        delta_vals = module.delta.flatten().numpy()
        print(f"\n{name}:")
        print(f"  Delta mean: {delta_vals.mean():.6f} (expect ~1.0)")
        print(f"  Delta std:  {delta_vals.std():.6f} (expect ~0.10)")
        print(f"  Delta min/max: [{delta_vals.min():.4f}, {delta_vals.max():.4f}]")
        break  # Just check first layer

# Test 3: Digital vs Analog Output Difference
print("\n" + "=" * 70)
print("TEST 3: Digital vs Analog Output Comparison")
print("=" * 70)

analog_model_light = analogize(model, sigma_mismatch=0.05, n_adc_bits=8)
analog_model_heavy = analogize(model, sigma_mismatch=0.30, n_adc_bits=2)

with torch.no_grad():
    digital_out = model(sample_input)
    analog_out_light = analog_model_light(sample_input)
    analog_out_heavy = analog_model_heavy(sample_input)
    
diff_light = (digital_out - analog_out_light).abs()
diff_heavy = (digital_out - analog_out_heavy).abs()

print(f"Digital output range: [{digital_out.min():.4f}, {digital_out.max():.4f}]")
print(f"\nσ=0.05, 8-bit ADC:")
print(f"  Analog output range: [{analog_out_light.min():.4f}, {analog_out_light.max():.4f}]")
print(f"  Mean absolute diff: {diff_light.mean():.6f}")
print(f"  Max absolute diff: {diff_light.max():.6f}")
print(f"  Relative error: {(diff_light.mean() / digital_out.abs().mean()):.4%}")

print(f"\nσ=0.30, 2-bit ADC (EXTREME):")
print(f"  Analog output range: [{analog_out_heavy.min():.4f}, {analog_out_heavy.max():.4f}]")
print(f"  Mean absolute diff: {diff_heavy.mean():.6f}")
print(f"  Max absolute diff: {diff_heavy.max():.6f}")
print(f"  Relative error: {(diff_heavy.mean() / digital_out.abs().mean()):.4%}")

# Test 4: Accuracy with extreme parameters
print("\n" + "=" * 70)
print("TEST 4: Accuracy Under Extreme Analog Noise")
print("=" * 70)

test_configs = [
    ("Digital", None, None),
    ("σ=0.05, 8-bit", 0.05, 8),
    ("σ=0.15, 4-bit", 0.15, 4),
    ("σ=0.30, 2-bit (EXTREME)", 0.30, 2),
    ("σ=0.50, 2-bit (BREAK)", 0.50, 2),
]

for name, sigma, bits in test_configs:
    if sigma is None:
        # Digital
        acc = transformer.evaluate(model)
    else:
        # Analog
        analog_model = analogize(model, sigma_mismatch=sigma, n_adc_bits=bits)
        acc = transformer.evaluate(analog_model)
    
    print(f"  {name:30s}: {acc:.4f}")

# Test 5: Check if thermal noise matters (for non-ODE models)
print("\n" + "=" * 70)
print("TEST 5: Thermal Noise Impact (Stochastic Check)")
print("=" * 70)

analog_model = analogize(model, sigma_mismatch=0.05, n_adc_bits=8)

# Run same input twice - should get different outputs due to thermal noise
with torch.no_grad():
    out1 = analog_model(sample_input)
    out2 = analog_model(sample_input)
    
thermal_diff = (out1 - out2).abs()
print(f"Same input, two forward passes:")
print(f"  Mean difference: {thermal_diff.mean():.6e}")
print(f"  Max difference: {thermal_diff.max():.6e}")
print(f"  Relative to output: {(thermal_diff.mean() / out1.abs().mean()):.6e}")

if thermal_diff.mean() < 1e-6:
    print("  ⚠️  WARNING: Outputs are identical - thermal noise may be OFF")
else:
    print("  ✓ Outputs differ - thermal noise is active")

# Test 6: Layer-wise noise analysis
print("\n" + "=" * 70)
print("TEST 6: Layer-wise Noise Magnitude Analysis")
print("=" * 70)

analog_model = analogize(model, sigma_mismatch=0.10, n_adc_bits=8)
print("\nExpected noise magnitudes:")
for name, module in analog_model.named_modules():
    if hasattr(module, 'in_features'):
        import math
        k_B = 1.380649e-23
        T = 300.0
        C = 1e-12
        sigma_th = math.sqrt(k_B * T / C) * math.sqrt(module.in_features)
        print(f"  {name:20s}: thermal σ = {sigma_th:.6e}, quant step = {2*module.v_ref/255:.6f}")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print("""
Key Questions:
1. Are V_ref values reasonable? (not too large)
2. Is mismatch actually sampled? (delta != 1.0)
3. Do analog outputs differ from digital?
4. Does accuracy drop with extreme parameters?
5. Is thermal noise active? (stochastic outputs)

If answers are YES but degradation is still minimal:
→ Tasks are genuinely easy and models are robust
→ Consider harder benchmarks or larger models

If answers are NO:
→ Analog simulator has bugs that need fixing
""")
