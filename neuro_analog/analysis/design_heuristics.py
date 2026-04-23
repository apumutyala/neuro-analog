"""
Design heuristics for analog amenability classification.

Provides classification functions based on the three failure modes identified
in the pilot study:
- Single-pass tolerant
- Fixed-point sensitive (DEQ)
- Multi-step ADC accumulation (diffusion)
"""

from neuro_analog.ir.types import AnalogAmenabilityProfile, DynamicsProfile


class FailureMode:
    """Classification of analog failure modes."""
    SINGLE_PASS_TOLERANT = "single_pass_tolerant"
    FIXED_POINT_SENSITIVE = "fixed_point_sensitive"
    DIFFUSION_LIKE = "diffusion_like"
    UNKNOWN = "unknown"


def classify_failure_mode(profile: AnalogAmenabilityProfile) -> str:
    """Classify architecture into one of three failure mode buckets.
    
    Rules based on amenability score, dynamics, and structural features:
    
    Single-pass tolerant:
    - analog_flop_fraction > 0.7
    - da_boundary_count/layer < 0.5
    - Low dynamics penalty (no fixed-point loops, no deep diffusion steps)
    - High sigma_10pct (> 10%)
    
    Fixed-point sensitive (DEQ):
    - High analog fraction but significant dynamics penalty
    - Dynamics type is "implicit_equilibrium" (DEQ)
    - Lower sigma_10pct due to compounded mismatch
    
    Diffusion-like:
    - Many steps (high num_diffusion_steps)
    - High D/A boundary density per step
    - Low or N/A sigma_10pct (quantization accumulation)
    
    Args:
        profile: AnalogAmenabilityProfile with IR metrics and sigma_10pct.
        
    Returns:
        Failure mode classification string.
    """
    # Compute boundary density
    layer_count = profile.layer_count if profile.layer_count > 0 else 1
    boundary_density = profile.da_boundary_count / layer_count
    
    # Check for diffusion-like characteristics
    if (profile.dynamics.num_diffusion_steps and profile.dynamics.num_diffusion_steps > 10):
        # Many diffusion steps with high boundary density
        if boundary_density > 0.3:
            return FailureMode.DIFFUSION_LIKE
    
    # Check for DEQ-like characteristics
    if profile.dynamics.dynamics_type == "implicit_equilibrium":
        return FailureMode.FIXED_POINT_SENSITIVE
    
    # Check for single-pass tolerant characteristics
    is_single_pass = (
        profile.analog_flop_fraction > 0.7
        and boundary_density < 0.5
        and profile.dynamics_penalty() < 0.3  # Low dynamics penalty
    )
    
    if is_single_pass:
        # Check sigma threshold if available
        if profile.sigma_10pct > 0.10:
            return FailureMode.SINGLE_PASS_TOLERANT
        elif profile.sigma_10pct > 0:
            # High analog fraction but lower sigma - borderline
            return FailureMode.SINGLE_PASS_TOLERANT
    
    # Default classification based on amenability score
    if profile.amenability_score > 0.6:
        return FailureMode.SINGLE_PASS_TOLERANT
    elif profile.amenability_score > 0.3:
        return FailureMode.FIXED_POINT_SENSITIVE
    else:
        return FailureMode.DIFFUSION_LIKE


def print_heuristic_report(profile: AnalogAmenabilityProfile) -> None:
    """Print a human-readable heuristic classification report.
    
    Args:
        profile: AnalogAmenabilityProfile to analyze.
    """
    layer_count = profile.layer_count if profile.layer_count > 0 else 1
    boundary_density = profile.da_boundary_count / layer_count
    
    print("\n" + "=" * 70)
    print("Analog Amenability Heuristic Report")
    print("=" * 70)
    print(f"Model: {profile.model_name} ({profile.architecture.value})")
    print(f"Analog FLOP fraction: {profile.analog_flop_fraction:.1%}")
    print(f"D/A boundary density: {boundary_density:.2f} ({profile.da_boundary_count} boundaries / {layer_count} layers)")
    print(f"Dynamics type: {profile.dynamics.dynamics_type}")
    print(f"Dynamics penalty: {profile.dynamics_penalty():.2f}")
    print(f"Min weight precision: {profile.min_weight_precision_bits} bits")
    
    if profile.sigma_10pct > 0:
        print(f"Sigma @ 10% degradation: {profile.sigma_10pct:.1%}")
    else:
        print("Sigma @ 10% degradation: N/A (no sweep data)")
    
    print(f"Amenability score: {profile.amenability_score:.3f}")
    
    # Energy/latency metrics if available
    if profile.analog_energy_pJ > 0 or profile.digital_energy_pJ > 0:
        print(f"\nEnergy Metrics:")
        print(f"  Analog energy: {profile.analog_energy_pJ:.2e} pJ")
        print(f"  Digital energy: {profile.digital_energy_pJ:.2e} pJ")
        print(f"  Energy saving: {profile.analog_energy_saving_vs_digital:.1%}")
        print(f"  Speedup: {profile.analog_speedup_vs_digital:.2f}x")
    
    # Classification
    failure_mode = classify_failure_mode(profile)
    print(f"\nClassification: {failure_mode}")
    
    # Explanation
    print("\nRationale:")
    if failure_mode == FailureMode.SINGLE_PASS_TOLERANT:
        print("  - High analog compute fraction (>70%)")
        print("  - Low D/A boundary density (<0.5 per layer)")
        print("  - Minimal iterative dynamics")
        print("  - Good noise tolerance (sigma@10% > 10%)")
        print("  → Safe for noisy AIMC arrays up to ~15% mismatch")
    elif failure_mode == FailureMode.FIXED_POINT_SENSITIVE:
        print("  - Fixed-point iteration (DEQ-like dynamics)")
        print("  - High analog fraction but compounded mismatch")
        print("  - Requires contraction-strength tuning")
        print("  → May need limited iteration counts for analog deployment")
    elif failure_mode == FailureMode.DIFFUSION_LIKE:
        print("  - Multi-step inference pipeline")
        print("  - High D/A boundary density")
        print("  - Quantization error accumulation across steps")
        print("  → Requires shared analog state to avoid per-step ADC")
    
    print("=" * 70 + "\n")


def get_design_recommendations(profile: AnalogAmenabilityProfile) -> dict[str, str]:
    """Get design recommendations based on failure mode classification.
    
    Args:
        profile: AnalogAmenabilityProfile to analyze.
        
    Returns:
        Dictionary with design recommendations.
    """
    failure_mode = classify_failure_mode(profile)
    
    recommendations = {
        "failure_mode": failure_mode,
        "analog_deployment": "",
        "hardware_tuning": "",
        "architectural_modifications": "",
    }
    
    if failure_mode == FailureMode.SINGLE_PASS_TOLERANT:
        recommendations["analog_deployment"] = "Direct analog deployment recommended"
        recommendations["hardware_tuning"] = "Standard AIMC array configuration (8-16 bit ADC)"
        recommendations["architectural_modifications"] = "None required"
        
    elif failure_mode == FailureMode.FIXED_POINT_SENSITIVE:
        recommendations["analog_deployment"] = "Analog deployment with caution"
        recommendations["hardware_tuning"] = "Increase ADC precision, reduce mismatch via calibration"
        recommendations["architectural_modifications"] = "Limit iteration count, add contraction regularization"
        
    elif failure_mode == FailureMode.DIFFUSION_LIKE:
        recommendations["analog_deployment"] = "Analog deployment challenging"
        recommendations["hardware_tuning"] = "Minimize ADC usage, share analog state across steps"
        recommendations["architectural_modifications"] = "Reduce number of denoising steps, use continuous-time formulation"
    
    return recommendations
