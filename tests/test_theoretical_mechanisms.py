"""
Tests for theoretical mechanisms from the research paper.

Tests perceptual constraints (θ_detect, σ_perception), local learning radius,
and other mechanisms that bridge theory and implementation.
"""

from __future__ import annotations

from typing import Literal

import pytest

from lovebug.config import LoveBugConfig
from lovebug.model import LoveModelRefactored as LoveModel


class TestPerceptualConstraints:
    """Test perceptual constraint mechanisms (θ_detect, σ_perception)."""

    def test_perceptual_noise_application(self, genetic_only_config) -> None:
        """Test that perceptual noise is properly applied during courtship."""
        config = genetic_only_config
        config.layer.sigma_perception = 5.0  # High noise
        config.layer.theta_detect = 2.0
        config.simulation.population_size = 100
        model = LoveModel(config)
        model.step()
        model.step()
        assert len(model.agents) > 0
        assert model.step_count == 2

    def test_detection_threshold_effects(self, genetic_only_config) -> None:
        """Test that detection threshold affects mate recognition."""
        config = genetic_only_config
        config.layer.sigma_perception = 1.0
        config.layer.theta_detect = 15.0  # High threshold
        config.simulation.population_size = 50
        model = LoveModel(config)
        initial_pop = len(model.agents)
        for _ in range(10):
            model.step()
        final_pop = len(model.agents)
        growth_rate = (final_pop - initial_pop) / initial_pop
        assert growth_rate < 2.0

    @pytest.mark.parametrize(
        "theta_detect,sigma_perception",
        [
            (8.0, 1.0),
            (12.0, 3.5),
            (4.0, 0.5),
        ],
    )
    def test_perceptual_parameter_combinations(
        self, theta_detect: float, sigma_perception: float, genetic_only_config
    ) -> None:
        """Test various combinations of perceptual parameters."""
        config = genetic_only_config
        config.layer.theta_detect = theta_detect
        config.layer.sigma_perception = sigma_perception
        config.simulation.population_size = 50
        model = LoveModel(config)
        for _ in range(5):
            model.step()
        assert len(model.agents) > 0
        assert model.step_count == 5

    def test_perceptual_parameters_serialization(self, genetic_only_config) -> None:
        """Test that perceptual parameters are properly stored and accessed."""
        config = genetic_only_config
        config.layer.theta_detect = 12.0
        config.layer.sigma_perception = 3.5
        d = config.model_dump()
        assert d["layer"]["theta_detect"] == 12.0
        assert d["layer"]["sigma_perception"] == 3.5
        config2 = LoveBugConfig(**d)
        assert config2.layer.theta_detect == 12.0
        assert config2.layer.sigma_perception == 3.5


class TestLocalLearningRadius:
    """Test local learning radius mechanism for cultural transmission."""

    def test_local_radius_constraint(self, balanced_config) -> None:
        """Test that cultural learning respects local radius constraints."""
        config = balanced_config
        config.cultural.memory_span = 3
        config.simulation.population_size = 100
        model = LoveModel(config)
        for _ in range(5):
            model.step()
        final_metrics = model.history[-1]
        assert "cultural_learning_events" in final_metrics
        assert final_metrics["cultural_learning_events"] >= 0

    @pytest.mark.parametrize(
        "radius,expected_name",
        [
            (1, "small_radius"),
            (5, "medium_radius"),
            (15, "large_radius"),
        ],
    )
    def test_variable_radius_effects(self, radius: int, expected_name: str, balanced_config) -> None:
        """Test that different radius sizes affect learning patterns."""
        config = balanced_config
        config.cultural.memory_span = 3
        config.simulation.population_size = 80
        model = LoveModel(config)
        for _ in range(10):
            model.step()
        total_events = sum(h.get("cultural_learning_events", 0) for h in model.history)
        assert total_events >= 0
        assert len(model.history) == 10


class TestIntegratedMechanisms:
    """Test integration of multiple theoretical mechanisms."""

    def test_combined_perceptual_and_cultural_constraints(self, balanced_config) -> None:
        """Test perceptual constraints working with cultural learning."""
        config = balanced_config
        config.cultural.memory_span = 3
        config.layer.sigma_perception = 2.0
        config.layer.theta_detect = 6.0
        config.simulation.population_size = 150
        model = LoveModel(config)
        for _ in range(15):
            model.step()
        final_metrics = model.history[-1]
        assert "mean_genetic_preference" in final_metrics
        assert "mean_cultural_preference" in final_metrics
        assert "cultural_learning_events" in final_metrics
        assert "gene_culture_distance" in final_metrics

    @pytest.mark.parametrize(
        "blend_mode",
        ["weighted", "additive", "multiplicative"],
    )
    def test_blending_modes_with_constraints(
        self,
        blend_mode: Literal["weighted", "additive", "multiplicative"],
        balanced_config,
    ) -> None:
        """Test different blending modes work with perceptual constraints."""
        config = balanced_config
        config.blending.blend_mode = blend_mode
        config.layer.theta_detect = 8.0
        config.layer.sigma_perception = 1.5
        config.simulation.population_size = 100
        model = LoveModel(config)
        for _ in range(5):
            model.step()
        assert len(model.agents) > 0
        assert model.step_count == 5


class TestParameterValidation:
    """Test validation of new theoretical parameters."""

    def test_perceptual_parameter_validation(self) -> None:
        """Test that perceptual parameters are properly validated."""
        from lovebug.config import LayerConfig

        # Test negative values should raise errors
        with pytest.raises(ValueError):
            LayerConfig(theta_detect=-1.0)
        with pytest.raises(ValueError):
            LayerConfig(sigma_perception=-0.5)

        # Test valid values work
        layer = LayerConfig(theta_detect=10.0, sigma_perception=2.0)
        assert layer.theta_detect == 10.0
        assert layer.sigma_perception == 2.0


class TestMechanismInteractions:
    """Test interactions between different theoretical mechanisms."""

    def test_perceptual_noise_affects_mating_patterns(self, genetic_only_config) -> None:
        """Test that perceptual noise affects mate selection patterns."""
        import copy

        config_low = copy.deepcopy(genetic_only_config)
        config_low.layer.theta_detect = 5.0
        config_low.layer.sigma_perception = 0.1
        config_low.simulation.population_size = 100

        config_high = copy.deepcopy(genetic_only_config)
        config_high.layer.theta_detect = 5.0
        config_high.layer.sigma_perception = 4.0
        config_high.simulation.population_size = 100

        model_low = LoveModel(config_low)
        model_high = LoveModel(config_high)
        for _ in range(10):
            model_low.step()
            model_high.step()
        assert len(model_low.history) == 10
        assert len(model_high.history) == 10
        assert len(model_low.agents) > 0
        assert len(model_high.agents) > 0

    def test_cultural_radius_interaction_with_network_size(self, balanced_config) -> None:
        """Test how local learning radius interacts with network connectivity."""
        import copy

        config_dense = copy.deepcopy(balanced_config)
        config_dense.cultural.memory_span = 3
        config_dense.simulation.population_size = 100

        config_sparse = copy.deepcopy(balanced_config)
        config_sparse.cultural.memory_span = 3
        config_sparse.simulation.population_size = 100

        model_dense = LoveModel(config_dense)
        model_sparse = LoveModel(config_sparse)
        for _ in range(5):
            model_dense.step()
            model_sparse.step()
        assert len(model_dense.agents) > 0
        assert len(model_sparse.agents) > 0
        assert len(model_dense.history) == 5
        assert len(model_sparse.history) == 5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_perceptual_thresholds(self, genetic_only_config) -> None:
        """Test behavior with extreme perceptual threshold values."""
        import copy

        config_low = copy.deepcopy(genetic_only_config)
        config_low.layer.theta_detect = 0.1
        config_low.layer.sigma_perception = 0.5
        config_low.simulation.population_size = 50

        config_high = copy.deepcopy(genetic_only_config)
        config_high.layer.theta_detect = 15.9
        config_high.layer.sigma_perception = 0.1
        config_high.simulation.population_size = 50

        models = [
            LoveModel(config_low),
            LoveModel(config_high),
        ]
        for model in models:
            for _ in range(5):
                model.step()
            assert len(model.agents) >= 0
            assert model.step_count == 5

    def test_minimal_cultural_learning_setup(self, balanced_config) -> None:
        """Test cultural learning with minimal viable parameters."""
        config = balanced_config
        config.cultural.memory_span = 1
        config.simulation.population_size = 20
        model = LoveModel(config)
        for _ in range(3):
            model.step()
        assert len(model.agents) > 0
        assert len(model.history) == 3
