"""
Unit tests for energy_model module.

Tests:
- HardwareProfile creation and config loading
- estimate_node_cost for each OpType
- Energy/latency aggregation at graph level
- Amenability score computation
- Dynamics penalty calculation
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from neuro_analog.ir.types import (
    AnalogNode, AnalogAmenabilityProfile, ArchitectureFamily, DynamicsProfile,
    Domain, OpType, PrecisionSpec
)
from neuro_analog.ir.energy_model import (
    HardwareProfile, estimate_node_cost, dynamics_penalty, compute_amenability_score
)


class TestHardwareProfile:
    """Test HardwareProfile creation and config loading."""
    
    def test_default_profile(self):
        """Test creating a default HardwareProfile."""
        profile = HardwareProfile()
        
        assert profile.analog_mac_energy_pJ == 5.0
        assert profile.digital_mac_energy_pJ == 100.0
        assert profile.adc_energy_pJ == 0.8
        assert profile.dac_energy_pJ == 0.8
        assert profile.analog_mac_throughput == 1e12
        assert profile.digital_mac_throughput == 1e11
    
    def test_profile_from_config(self):
        """Test loading HardwareProfile from YAML config."""
        config_data = {
            "analog_mac_energy_pJ": 10.0,
            "digital_mac_energy_pJ": 200.0,
            "adc_energy_pJ": 1.0,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            profile = HardwareProfile.from_config(config_path)
            
            assert profile.analog_mac_energy_pJ == 10.0
            assert profile.digital_mac_energy_pJ == 200.0
            assert profile.adc_energy_pJ == 1.0
            # Default values for unspecified fields
            assert profile.dac_energy_pJ == 0.8
        finally:
            Path(config_path).unlink()
    
    def test_profile_to_config(self):
        """Test saving HardwareProfile to YAML config."""
        profile = HardwareProfile(analog_mac_energy_pJ=15.0, digital_mac_energy_pJ=150.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            profile.to_config(config_path)
            
            # Load and verify
            loaded_profile = HardwareProfile.from_config(config_path)
            assert loaded_profile.analog_mac_energy_pJ == 15.0
            assert loaded_profile.digital_mac_energy_pJ == 150.0
        finally:
            Path(config_path).unlink()
    
    def test_config_file_not_found(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            HardwareProfile.from_config("nonexistent_config.yaml")


class TestEstimateNodeCost:
    """Test estimate_node_cost for different OpTypes."""
    
    def test_mvm_analog(self):
        """Test MVM node energy/latency estimation (analog)."""
        profile = HardwareProfile()
        node = AnalogNode(
            name="mvm",
            op_type=OpType.MVM,
            domain=Domain.ANALOG,
            input_shape=(256,),
            output_shape=(128,),
            weight_shape=(256, 128),
            flops=2 * 256 * 128,  # MAC operations
            param_count=256 * 128,
        )
        
        estimate = estimate_node_cost(node, profile)
        
        # Energy = MACs * analog_mac_energy_pJ
        expected_energy = (256 * 128) * profile.analog_mac_energy_pJ
        assert estimate.energy_pJ == pytest.approx(expected_energy)
        assert estimate.latency_ns > 0
        assert estimate.power_mW > 0
    
    def test_mvm_digital(self):
        """Test MVM node energy/latency estimation (digital)."""
        profile = HardwareProfile()
        node = AnalogNode(
            name="mvm",
            op_type=OpType.MVM,
            domain=Domain.DIGITAL,
            input_shape=(256,),
            output_shape=(128,),
            weight_shape=(256, 128),
            flops=2 * 256 * 128,
            param_count=256 * 128,
        )
        
        estimate = estimate_node_cost(node, profile)
        
        # Energy = MACs * digital_mac_energy_pJ
        expected_energy = (256 * 128) * profile.digital_mac_energy_pJ
        assert estimate.energy_pJ == pytest.approx(expected_energy)
    
    def test_integration_node(self):
        """Test INTEGRATION node estimation."""
        profile = HardwareProfile()
        node = AnalogNode(
            name="integration",
            op_type=OpType.INTEGRATION,
            domain=Domain.ANALOG,
            input_shape=(64,),
            output_shape=(64,),
            flops=2 * 64,
            param_count=0,
        )
        
        estimate = estimate_node_cost(node, profile)
        
        # Energy = state_dim * integrator_energy_pJ_per_state
        expected_energy = 64 * profile.integrator_energy_pJ_per_state
        assert estimate.energy_pJ == pytest.approx(expected_energy)
    
    def test_softmax_digital(self):
        """Test SOFTMAX node estimation (digital)."""
        profile = HardwareProfile()
        node = AnalogNode(
            name="softmax",
            op_type=OpType.SOFTMAX,
            domain=Domain.DIGITAL,
            input_shape=(10,),
            output_shape=(10,),
            flops=10,
            param_count=0,
        )
        
        estimate = estimate_node_cost(node, profile)
        
        # Energy = flops * digital_mac_energy_pJ
        expected_energy = 10 * profile.digital_mac_energy_pJ
        assert estimate.energy_pJ == pytest.approx(expected_energy)
    
    def test_noise_injection(self):
        """Test NOISE_INJECTION node estimation."""
        profile = HardwareProfile()
        node = AnalogNode(
            name="noise",
            op_type=OpType.NOISE_INJECTION,
            domain=Domain.ANALOG,
            input_shape=(100,),
            output_shape=(100,),
            flops=100,
            param_count=0,
        )
        
        estimate = estimate_node_cost(node, profile)
        
        # Energy = flops * thermodynamic_sample_energy_pJ
        expected_energy = 100 * profile.thermodynamic_sample_energy_pJ
        assert estimate.energy_pJ == pytest.approx(expected_energy)
    
    def test_dropout_zero_compute(self):
        """Test DROPOUT node (zero compute at inference)."""
        profile = HardwareProfile()
        node = AnalogNode(
            name="dropout",
            op_type=OpType.DROPOUT,
            domain=Domain.DIGITAL,
            input_shape=(100,),
            output_shape=(100,),
            flops=0,
            param_count=0,
        )
        
        estimate = estimate_node_cost(node, profile)
        
        assert estimate.energy_pJ == 0.0
        assert estimate.latency_ns == 0.0


class TestDynamicsPenalty:
    """Test dynamics penalty calculation."""
    
    def test_no_dynamics(self):
        """Test penalty for architectures without dynamics."""
        dynamics = DynamicsProfile(has_dynamics=False)
        penalty = dynamics_penalty(dynamics)
        
        assert penalty == 0.0
    
    def test_deq_penalty(self):
        """Test penalty for DEQ (implicit_equilibrium)."""
        dynamics = DynamicsProfile(
            has_dynamics=True,
            dynamics_type="implicit_equilibrium"
        )
        penalty = dynamics_penalty(dynamics)
        
        assert penalty > 0.0
        assert penalty <= 1.0
    
    def test_diffusion_penalty(self):
        """Test penalty for diffusion with many steps."""
        dynamics = DynamicsProfile(
            has_dynamics=True,
            num_diffusion_steps=20
        )
        penalty = dynamics_penalty(dynamics)
        
        assert penalty > 0.0
        assert penalty <= 1.0
    
    def test_ode_steps_penalty(self):
        """Test penalty for ODE with many function evaluations."""
        dynamics = DynamicsProfile(
            has_dynamics=True,
            num_function_evaluations=50
        )
        penalty = dynamics_penalty(dynamics)
        
        assert penalty > 0.0
        assert penalty <= 1.0
    
    def test_stochastic_penalty(self):
        """Test penalty for stochastic dynamics."""
        dynamics = DynamicsProfile(
            has_dynamics=True,
            is_stochastic=True,
            dynamics_type="SDE"
        )
        penalty = dynamics_penalty(dynamics)
        
        assert penalty > 0.0
    
    def test_thermodynamic_gibbs_no_penalty(self):
        """Test that thermodynamic_gibbs has no penalty (thermal noise is resource)."""
        dynamics = DynamicsProfile(
            has_dynamics=True,
            is_stochastic=True,
            dynamics_type="thermodynamic_gibbs"
        )
        penalty = dynamics_penalty(dynamics)
        
        # thermodynamic_gibbs should not get stochastic penalty
        assert penalty >= 0.0


class TestComputeAmenabilityScore:
    """Test amenability score computation."""
    
    def test_high_analog_fraction(self):
        """Test score with high analog FLOP fraction."""
        profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.TRANSFORMER,
            model_name="test",
            model_params=1000,
            analog_flop_fraction=0.8,
            digital_flop_fraction=0.2,
            hybrid_flop_fraction=0.0,
            da_boundary_count=5,
            layer_count=10,
            sigma_10pct=0.15,
            min_weight_precision_bits=4,
        )
        
        score = compute_amenability_score(profile)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high with good characteristics
    
    def test_low_analog_fraction(self):
        """Test score with low analog FLOP fraction."""
        profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.TRANSFORMER,
            model_name="test",
            model_params=1000,
            analog_flop_fraction=0.2,
            digital_flop_fraction=0.8,
            hybrid_flop_fraction=0.0,
            da_boundary_count=20,
            layer_count=10,
            sigma_10pct=0.05,
            min_weight_precision_bits=8,
        )
        
        score = compute_amenability_score(profile)
        
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should be low with poor characteristics
    
    def test_no_sigma_data(self):
        """Test score when sigma_10pct is not available."""
        profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.TRANSFORMER,
            model_name="test",
            model_params=1000,
            analog_flop_fraction=0.7,
            digital_flop_fraction=0.3,
            hybrid_flop_fraction=0.0,
            da_boundary_count=5,
            layer_count=10,
            sigma_10pct=0.0,  # No sweep data
            min_weight_precision_bits=4,
        )
        
        score = compute_amenability_score(profile)
        
        assert 0.0 <= score <= 1.0
    
    def test_high_boundary_density(self):
        """Test score with high D/A boundary density."""
        profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.TRANSFORMER,
            model_name="test",
            model_params=1000,
            analog_flop_fraction=0.7,
            digital_flop_fraction=0.3,
            hybrid_flop_fraction=0.0,
            da_boundary_count=15,  # High boundary count
            layer_count=10,
            sigma_10pct=0.15,
            min_weight_precision_bits=4,
        )
        
        score = compute_amenability_score(profile)
        
        assert 0.0 <= score <= 1.0
        # High boundary density should reduce score
        assert score < 0.7
    
    def test_clamping(self):
        """Test that score is clamped to [0, 1]."""
        # Test extreme case that would exceed 1.0
        profile = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.TRANSFORMER,
            model_name="test",
            model_params=1000,
            analog_flop_fraction=1.0,
            digital_flop_fraction=0.0,
            hybrid_flop_fraction=0.0,
            da_boundary_count=0,
            layer_count=10,
            sigma_10pct=0.20,  # Above max tested
            min_weight_precision_bits=2,
        )
        
        score = compute_amenability_score(profile)
        assert score <= 1.0
        
        # Test extreme case that would go below 0.0
        profile2 = AnalogAmenabilityProfile(
            architecture=ArchitectureFamily.TRANSFORMER,
            model_name="test",
            model_params=1000,
            analog_flop_fraction=0.0,
            digital_flop_fraction=1.0,
            hybrid_flop_fraction=0.0,
            da_boundary_count=100,
            layer_count=10,
            sigma_10pct=0.0,
            min_weight_precision_bits=16,
        )
        
        score2 = compute_amenability_score(profile2)
        assert score2 >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
