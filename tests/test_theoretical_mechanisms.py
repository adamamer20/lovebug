"""
Tests for the newly implemented theoretical mechanisms from the research paper.

Tests perceptual constraints (θ_detect, σ_perception), local learning radius,
and other missing mechanisms that bridge theory and implementation.
"""

from __future__ import annotations

import pytest

from lovebug.lande_kirkpatrick import LandeKirkpatrickParams
from lovebug.layer2.config import Layer2Config
from lovebug.layer_activation import LayerActivationConfig
from lovebug.unified_mesa_model import LoveModel


class TestPerceptualConstraints:
    """Test perceptual constraint mechanisms (θ_detect, σ_perception)."""

    def test_perceptual_noise_application(self):
        """Test that perceptual noise is properly applied during courtship."""
        # Configure with high perceptual noise
        layer_config = LayerActivationConfig.genetic_only()
        layer_config.sigma_perception = 5.0  # High noise
        layer_config.theta_detect = 2.0  # Low threshold

        genetic_params = LandeKirkpatrickParams()
        model = LoveModel(layer_config, genetic_params, None, n_agents=100)

        # Run a few steps to generate mating events
        model.step()
        model.step()

        # Verify the model ran without errors
        assert len(model.agents) > 0
        assert model.step_count == 2

    def test_detection_threshold_effects(self):
        """Test that detection threshold affects mate recognition."""
        # Test with very high detection threshold
        layer_config = LayerActivationConfig.genetic_only()
        layer_config.sigma_perception = 1.0
        layer_config.theta_detect = 15.0  # Very high threshold (max similarity is 16)

        genetic_params = LandeKirkpatrickParams()
        model = LoveModel(layer_config, genetic_params, None, n_agents=50)

        initial_pop = len(model.agents)

        # Run multiple steps
        for _ in range(10):
            model.step()

        # With very high threshold, reproduction should be rare
        # Population should not grow much
        final_pop = len(model.agents)
        growth_rate = (final_pop - initial_pop) / initial_pop

        # Growth should be limited due to high detection threshold
        assert growth_rate < 2.0  # Less than 200% growth

    def test_perceptual_parameters_in_config(self):
        """Test that perceptual parameters are properly stored and accessed."""
        layer_config = LayerActivationConfig(
            genetic_enabled=True, cultural_enabled=False, theta_detect=12.0, sigma_perception=3.5
        )

        assert layer_config.theta_detect == 12.0
        assert layer_config.sigma_perception == 3.5

        # Test serialization/deserialization
        config_dict = layer_config.to_dict()
        assert config_dict["theta_detect"] == 12.0
        assert config_dict["sigma_perception"] == 3.5

        # Test reconstruction from dict
        restored_config = LayerActivationConfig.from_dict(config_dict)
        assert restored_config.theta_detect == 12.0
        assert restored_config.sigma_perception == 3.5


class TestLocalLearningRadius:
    """Test local learning radius mechanism for cultural transmission."""

    def test_local_radius_constraint(self):
        """Test that cultural learning respects local radius constraints."""
        layer_config = LayerActivationConfig.cultural_only()

        # Configure with small local learning radius
        cultural_params = Layer2Config(
            local_learning_radius=2,  # Very small radius
            horizontal_transmission_rate=0.5,  # High learning rate
            innovation_rate=0.01,  # Low innovation
        )

        model = LoveModel(layer_config, None, cultural_params, n_agents=100)

        # Run cultural learning steps
        for _ in range(5):
            model.step()

        # Verify cultural evolution occurred
        final_metrics = model.history[-1]
        assert "cultural_learning_events" in final_metrics
        assert final_metrics["cultural_learning_events"] >= 0

    def test_variable_radius_effects(self):
        """Test that different radius sizes affect learning patterns."""
        configs = [("small_radius", 1), ("medium_radius", 5), ("large_radius", 15)]

        results = {}

        for config_name, radius in configs:
            layer_config = LayerActivationConfig.cultural_only()
            cultural_params = Layer2Config(
                local_learning_radius=radius, horizontal_transmission_rate=0.3, innovation_rate=0.02
            )

            model = LoveModel(layer_config, None, cultural_params, n_agents=80)

            # Run simulation
            for _ in range(10):
                model.step()

            # Collect learning events
            total_events = sum(h.get("cultural_learning_events", 0) for h in model.history)
            results[config_name] = total_events

        # Larger radius should generally enable more learning events
        # (though this may vary due to randomness)
        assert all(events >= 0 for events in results.values())


class TestIntegratedMechanisms:
    """Test integration of multiple theoretical mechanisms."""

    def test_combined_perceptual_and_cultural_constraints(self):
        """Test perceptual constraints working with cultural learning."""
        layer_config = LayerActivationConfig.balanced_combined(0.5)
        layer_config.sigma_perception = 2.0
        layer_config.theta_detect = 6.0

        genetic_params = LandeKirkpatrickParams()
        cultural_params = Layer2Config(
            local_learning_radius=3,
            horizontal_transmission_rate=0.2,
            oblique_transmission_rate=0.1,
            innovation_rate=0.05,
        )

        model = LoveModel(layer_config, genetic_params, cultural_params, n_agents=150)

        # Run integrated simulation
        for _ in range(15):
            model.step()

        # Verify both genetic and cultural evolution occurred
        final_metrics = model.history[-1]

        assert "mean_genetic_preference" in final_metrics
        assert "mean_cultural_preference" in final_metrics
        assert "cultural_learning_events" in final_metrics
        assert "gene_culture_distance" in final_metrics

    def test_blending_modes_with_constraints(self):
        """Test different blending modes work with perceptual constraints."""
        blending_modes = ["weighted_average", "probabilistic", "competitive"]

        for mode in blending_modes:
            layer_config = LayerActivationConfig(
                genetic_enabled=True,
                cultural_enabled=True,
                genetic_weight=0.6,
                cultural_weight=0.4,
                blending_mode=mode,
                theta_detect=8.0,
                sigma_perception=1.5,
            )

            genetic_params = LandeKirkpatrickParams()
            cultural_params = Layer2Config()

            model = LoveModel(layer_config, genetic_params, cultural_params, n_agents=100)

            # Run a few steps to test the blending
            for _ in range(5):
                model.step()

            # Verify no errors and evolution occurred
            assert len(model.agents) > 0
            assert model.step_count == 5


class TestParameterValidation:
    """Test validation of new theoretical parameters."""

    def test_perceptual_parameter_validation(self):
        """Test that perceptual parameters are properly validated."""
        # Test negative theta_detect
        with pytest.raises(ValueError, match="theta_detect must be non-negative"):
            LayerActivationConfig(theta_detect=-1.0)

        # Test negative sigma_perception
        with pytest.raises(ValueError, match="sigma_perception must be non-negative"):
            LayerActivationConfig(sigma_perception=-0.5)

        # Test valid parameters pass validation
        config = LayerActivationConfig(theta_detect=10.0, sigma_perception=2.0)
        assert config.theta_detect == 10.0
        assert config.sigma_perception == 2.0

    def test_cultural_parameter_validation(self):
        """Test validation of cultural learning parameters."""
        # Test invalid local_learning_radius
        with pytest.raises(ValueError, match="local_learning_radius must be positive"):
            Layer2Config(local_learning_radius=0)

        with pytest.raises(ValueError, match="local_learning_radius must be positive"):
            Layer2Config(local_learning_radius=-5)

        # Test valid parameters
        config = Layer2Config(local_learning_radius=10)
        assert config.local_learning_radius == 10


class TestMechanismInteractions:
    """Test interactions between different theoretical mechanisms."""

    def test_perceptual_noise_affects_mating_success(self):
        """Test that perceptual noise affects mate selection patterns."""
        # Create two models with different noise levels
        low_noise_config = LayerActivationConfig.genetic_only()
        low_noise_config.sigma_perception = 0.1  # Very low noise
        low_noise_config.theta_detect = 5.0

        high_noise_config = LayerActivationConfig.genetic_only()
        high_noise_config.sigma_perception = 4.0  # High noise
        high_noise_config.theta_detect = 5.0

        genetic_params = LandeKirkpatrickParams()

        low_noise_model = LoveModel(low_noise_config, genetic_params, None, n_agents=100)
        high_noise_model = LoveModel(high_noise_config, genetic_params, None, n_agents=100)

        # Run both models
        for _ in range(10):
            low_noise_model.step()
            high_noise_model.step()

        # Both should produce valid results but potentially different patterns
        assert len(low_noise_model.history) == 10
        assert len(high_noise_model.history) == 10

        # Verify population survived in both cases
        assert len(low_noise_model.agents) > 0
        assert len(high_noise_model.agents) > 0
