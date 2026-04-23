"""
Integration test for amenability scoring with pilot sweep data.

Tests:
- Load pilot sweep results
- Compute amenability scores for all 7 architectures
- Verify expected classification (single-pass vs fixed-point vs diffusion)
- Check score correlations with pilot findings
"""

import pytest
import json
from pathlib import Path

from neuro_analog.ir.types import AnalogAmenabilityProfile, ArchitectureFamily, DynamicsProfile
from neuro_analog.ir.energy_model import compute_amenability_score
from neuro_analog.analysis.design_heuristics import classify_failure_mode


class TestAmenabilityIntegration:
    """Integration tests with pilot sweep data."""
    
    @pytest.fixture
    def pilot_results_dir(self):
        """Path to pilot sweep results directory."""
        return Path(__file__).parent.parent / "experiments" / "cross_arch_tolerance" / "results"
    
    @pytest.fixture
    def load_sweep_result(self, pilot_results_dir, arch_name):
        """Load sweep result for a specific architecture."""
        result_file = pilot_results_dir / f"{arch_name}_mismatch.json"
        if not result_file.exists():
            pytest.skip(f"Sweep result not found: {result_file}")
        
        with open(result_file, "r") as f:
            return json.load(f)
    
    def test_transformer_amenability(self, load_sweep_result):
        """Test transformer amenability classification (single-pass tolerant)."""
        sweep_result = load_sweep_result("transformer")
        sigma_10pct = sweep_result.get("degradation_threshold_10pct", 0.0)
        
        # Create profile with pilot data
        profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.TRANSFORMER,
            model_name="transformer",
            model_params=1000,
            analog_flop_fraction=0.7,  # High analog fraction
            digital_flop_fraction=0.3,
            hybrid_flop_fraction=0.0,
            da_boundary_count=5,
            layer_count=10,
            sigma_10pct=sigma_10pct,
            min_weight_precision_bits=8,
        )
        
        # Compute amenability score
        profile.amenability_score = compute_amenability_score(profile)
        
        # Classify failure mode
        failure_mode = classify_failure_mode(profile)
        
        # Transformer should be single-pass tolerant
        assert failure_mode == "single_pass_tolerant"
        assert profile.amenability_score > 0.5
    
    def test_deq_amenability(self, load_sweep_result):
        """Test DEQ amenability classification (fixed-point sensitive)."""
        sweep_result = load_sweep_result("deq")
        sigma_10pct = sweep_result.get("degradation_threshold_10pct", 0.0)
        
        # Create profile with DEQ characteristics
        profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.DEQ,
            model_name="deq",
            model_params=1000,
            analog_flop_fraction=0.7,  # High analog fraction
            digital_flop_fraction=0.3,
            hybrid_flop_fraction=0.0,
            da_boundary_count=5,
            layer_count=10,
            sigma_10pct=sigma_10pct,
            min_weight_precision_bits=8,
            dynamics=DynamicsProfile(
                has_dynamics=True,
                dynamics_type="implicit_equilibrium"
            ),
        )
        
        # Compute amenability score
        profile.amenability_score = compute_amenability_score(profile)
        
        # Classify failure mode
        failure_mode = classify_failure_mode(profile)
        
        # DEQ should be fixed-point sensitive
        assert failure_mode == "fixed_point_sensitive"
    
    def test_diffusion_amenability(self, load_sweep_result):
        """Test diffusion amenability classification (diffusion-like)."""
        sweep_result = load_sweep_result("diffusion")
        sigma_10pct = sweep_result.get("degradation_threshold_10pct", 0.0)
        
        # Create profile with diffusion characteristics
        profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.DIFFUSION,
            model_name="diffusion",
            model_params=1000,
            analog_flop_fraction=0.6,
            digital_flop_fraction=0.4,
            hybrid_flop_fraction=0.0,
            da_boundary_count=20,  # High boundary density
            layer_count=10,
            sigma_10pct=sigma_10pct,
            min_weight_precision_bits=8,
            dynamics=DynamicsProfile(
                has_dynamics=True,
                num_diffusion_steps=20
            ),
        )
        
        # Compute amenability score
        profile.amenability_score = compute_amenability_score(profile)
        
        # Classify failure mode
        failure_mode = classify_failure_mode(profile)
        
        # Diffusion should be diffusion-like
        assert failure_mode == "diffusion_like"
    
    def test_all_architectures_classification(self, pilot_results_dir):
        """Test classification for all 7 architectures from pilot study."""
        arch_names = ["transformer", "neural_ode", "ssm", "ebm", "flow", "deq", "diffusion"]
        
        classifications = {}
        
        for arch_name in arch_names:
            result_file = pilot_results_dir / f"{arch_name}_mismatch.json"
            if not result_file.exists():
                continue
            
            with open(result_file, "r") as f:
                sweep_result = json.load(f)
            
            sigma_10pct = sweep_result.get("degradation_threshold_10pct", 0.0)
            
            # Create appropriate profile based on architecture
            if arch_name == "deq":
                dynamics = DynamicsProfile(has_dynamics=True, dynamics_type="implicit_equilibrium")
            elif arch_name == "diffusion":
                dynamics = DynamicsProfile(has_dynamics=True, num_diffusion_steps=20)
            else:
                dynamics = DynamicsProfile()
            
            profile = AnalogAmenabilityProfile(
                architecture=getattr(ArchitectureFamily, arch_name.upper(), ArchitectureFamily.TRANSFORMER),
                model_name=arch_name,
                model_params=1000,
                analog_flop_fraction=0.7,
                digital_flop_fraction=0.3,
                hybrid_flop_fraction=0.0,
                da_boundary_count=5 if arch_name != "diffusion" else 20,
                layer_count=10,
                sigma_10pct=sigma_10pct,
                min_weight_precision_bits=8,
                dynamics=dynamics,
            )
            
            profile.amenability_score = compute_amenability_score(profile)
            failure_mode = classify_failure_mode(profile)
            classifications[arch_name] = failure_mode
        
        # Verify expected classifications based on pilot findings
        # Single-pass tolerant: transformer, neural_ode, ssm, ebm, flow
        single_pass = ["transformer", "neural_ode", "ssm", "ebm", "flow"]
        for arch in single_pass:
            if arch in classifications:
                assert classifications[arch] == "single_pass_tolerant"
        
        # Fixed-point sensitive: deq
        if "deq" in classifications:
            assert classifications["deq"] == "fixed_point_sensitive"
        
        # Diffusion-like: diffusion
        if "diffusion" in classifications:
            assert classifications["diffusion"] == "diffusion_like"
    
    def test_score_correlation_with_sigma(self, pilot_results_dir):
        """Test that amenability score correlates with sigma_10pct."""
        arch_names = ["transformer", "neural_ode", "ssm", "ebm", "flow"]
        
        scores = []
        sigmas = []
        
        for arch_name in arch_names:
            result_file = pilot_results_dir / f"{arch_name}_mismatch.json"
            if not result_file.exists():
                continue
            
            with open(result_file, "r") as f:
                sweep_result = json.load(f)
            
            sigma_10pct = sweep_result.get("degradation_threshold_10pct", 0.0)
            
            profile = AnalogAmenabilityProfile(
                architecture=ArchitectureFamily.TRANSFORMER,
                model_name=arch_name,
                model_params=1000,
                analog_flop_fraction=0.7,
                digital_flop_fraction=0.3,
                hybrid_flop_fraction=0.0,
                da_boundary_count=5,
                layer_count=10,
                sigma_10pct=sigma_10pct,
                min_weight_precision_bits=8,
            )
            
            profile.amenability_score = compute_amenability_score(profile)
            
            scores.append(profile.amenability_score)
            sigmas.append(sigma_10pct)
        
        # For single-pass architectures, higher sigma should correlate with higher score
        # This is a weak correlation test - just verify the relationship exists
        if len(scores) > 1 and len(sigmas) > 1:
            # Sort by sigma and check that scores are generally higher for higher sigma
            sorted_pairs = sorted(zip(sigmas, scores), key=lambda x: x[0])
            scores_sorted = [s for _, s in sorted_pairs]
            
            # Check monotonic trend (allowing some noise)
            increasing_count = sum(1 for i in range(len(scores_sorted)-1) if scores_sorted[i+1] >= scores_sorted[i])
            # At least half should be increasing
            assert increasing_count >= len(scores_sorted) // 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
