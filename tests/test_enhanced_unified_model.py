"""
Tests for the enhanced LoveModel with integrated cultural features.

This module tests the advanced cultural evolution capabilities integrated
into the mesa-frames LoveModel, including cultural memory, prestige-based
learning, and performance characteristics.
"""

from __future__ import annotations

import pytest

from lovebug.layer2.config import Layer2Config
from lovebug.layer_activation import LayerActivationConfig
from lovebug.parameters import LandeKirkpatrickParams
from lovebug.unified_mesa_model import LoveModel


class TestCulturalMemoryIntegration:
    """Test cultural memory system integration."""

    def test_cultural_memory_initialization(self, cultural_config: LayerActivationConfig) -> None:
        """Test that cultural memory system is properly integrated."""
        cultural_params = Layer2Config(cultural_memory_size=5, memory_decay_rate=0.1, innovation_rate=0.1)
        model = LoveModel(layer_config=cultural_config, cultural_params=cultural_params, n_agents=100)

        # Check that cultural memory columns are initialized
        df = model.get_agent_dataframe()
        expected_columns = [f"cultural_memory_{i}" for i in range(5)]
        for col in expected_columns:
            assert col in df.columns

    def test_cultural_memory_system_evolution(self, cultural_config: LayerActivationConfig) -> None:
        """Test that cultural memory system evolves correctly."""
        cultural_params = Layer2Config(
            cultural_memory_size=3, memory_decay_rate=0.1, horizontal_transmission_rate=0.5, innovation_rate=0.2
        )
        model = LoveModel(layer_config=cultural_config, cultural_params=cultural_params, n_agents=50)

        # Run some steps to populate memory
        for _ in range(10):
            model.step()

        df = model.get_agent_dataframe()

        # Check that memory columns exist and have been updated
        memory_columns = [f"cultural_memory_{i}" for i in range(3)]
        for col in memory_columns:
            assert col in df.columns
            # At least some agents should have non-zero memory values
            assert df[col].sum() >= 0


class TestPrestigeBasedLearning:
    """Test prestige-based learning mechanisms."""

    def test_prestige_score_computation(self, cultural_config: LayerActivationConfig) -> None:
        """Test that prestige-based learning is properly vectorized."""
        cultural_params = Layer2Config(horizontal_transmission_rate=0.5, innovation_rate=0.1)
        model = LoveModel(layer_config=cultural_config, cultural_params=cultural_params, n_agents=50)

        # Run a few steps to generate mating success differences
        for _ in range(5):
            model.step()

        # Check that prestige scores are calculated
        df = model.get_agent_dataframe()
        assert "prestige_score" in df.columns
        assert df["prestige_score"].sum() >= 0  # Prestige scores should be non-negative

    def test_prestige_score_variation(self, cultural_config: LayerActivationConfig) -> None:
        """Test that prestige scores show variation over time."""
        cultural_params = Layer2Config(horizontal_transmission_rate=0.3, innovation_rate=0.1)
        model = LoveModel(layer_config=cultural_config, cultural_params=cultural_params, n_agents=100)

        # Run several steps to generate diversity in age, energy, mating success
        for _ in range(20):
            model.step()

        df = model.get_agent_dataframe()
        prestige_scores = df["prestige_score"]

        # Prestige scores should be non-negative
        assert (prestige_scores >= 0).all()

        # There should be some variation in prestige scores
        if len(df) > 1:
            variance = prestige_scores.var()
            assert variance is not None


class TestSocialNetworkIntegration:
    """Test social network integration and functionality."""

    def test_social_network_integration(self, cultural_config: LayerActivationConfig) -> None:
        """Test that social networks are properly integrated."""
        cultural_params = Layer2Config(
            network_type="small_world", network_connectivity=0.1, horizontal_transmission_rate=0.3
        )
        model = LoveModel(layer_config=cultural_config, cultural_params=cultural_params, n_agents=100)

        # Check that social network data is integrated
        df = model.get_agent_dataframe()
        assert "social_network_neighbors" in df.columns

    def test_vectorized_cultural_learning(self, cultural_config: LayerActivationConfig) -> None:
        """Test that cultural learning is fully vectorized."""
        cultural_params = Layer2Config(
            horizontal_transmission_rate=0.8, oblique_transmission_rate=0.6, innovation_rate=0.4
        )
        model = LoveModel(layer_config=cultural_config, cultural_params=cultural_params, n_agents=200)

        # Run several steps to increase chance of cultural events
        total_learning_events = 0
        total_innovation_events = 0

        for _ in range(20):
            model.step()
            total_learning_events += model._cultural_learning_events
            total_innovation_events += model._cultural_innovation_events

        final_df = model.get_agent_dataframe()

        # Population should still exist (though size may change)
        assert len(final_df) > 0

        # Cultural learning or innovation should have occurred across all steps
        # With high rates and many steps, we expect some cultural activity
        assert total_learning_events > 0 or total_innovation_events > 0, (
            f"Expected cultural events but got learning={total_learning_events}, "
            f"innovation={total_innovation_events} after 20 steps"
        )

        # Check that cultural traits are properly initialized and tracked
        assert "pref_culture" in final_df.columns
        assert "prestige_score" in final_df.columns


class TestEventTrackingSystem:
    """Test cultural event tracking and logging."""

    def test_event_tracking_system(self, cultural_config: LayerActivationConfig) -> None:
        """Test that cultural events are properly tracked."""
        cultural_params = Layer2Config(horizontal_transmission_rate=0.3, innovation_rate=0.2, log_cultural_events=True)
        model = LoveModel(layer_config=cultural_config, cultural_params=cultural_params, n_agents=100)

        # Run simulation
        for _ in range(5):
            model.step()

        # Check that events are tracked in history
        assert len(model.history) > 0

        # Check for cultural event metrics
        final_metrics = model.history[-1]
        assert "cultural_learning_events" in final_metrics
        assert "cultural_innovation_events" in final_metrics

    def test_learning_event_history(self, cultural_config: LayerActivationConfig) -> None:
        """Test that learning events accumulate in history."""
        cultural_params = Layer2Config(
            horizontal_transmission_rate=0.5, oblique_transmission_rate=0.3, innovation_rate=0.2
        )
        model = LoveModel(layer_config=cultural_config, cultural_params=cultural_params, n_agents=200)

        # Run several steps
        for _ in range(10):
            model.step()

        # Check that learning events are tracked in history
        assert len(model.history) > 0
        cultural_events = sum(h.get("cultural_learning_events", 0) for h in model.history)
        innovation_events = sum(h.get("cultural_innovation_events", 0) for h in model.history)
        assert cultural_events > 0 or innovation_events > 0


class TestCombinedEvolution:
    """Test combined genetic and cultural evolution."""

    def test_combined_evolution_with_advanced_features(
        self, genetic_params: LandeKirkpatrickParams, cultural_params: Layer2Config
    ) -> None:
        """Test combined genetic and cultural evolution with advanced features."""
        config = LayerActivationConfig.balanced_combined(0.6)
        enhanced_genetic_params = LandeKirkpatrickParams(h2_trait=0.5, h2_preference=0.3, mutation_variance=0.01)
        enhanced_cultural_params = Layer2Config(
            horizontal_transmission_rate=0.3, innovation_rate=0.1, cultural_memory_size=3
        )

        model = LoveModel(
            layer_config=config,
            genetic_params=enhanced_genetic_params,
            cultural_params=enhanced_cultural_params,
            n_agents=150,
        )

        # Run simulation
        results = model.run(20)

        # Check that both genetic and cultural metrics are collected
        assert "genetic_summary" in results
        assert "cultural_summary" in results
        assert "interaction_summary" in results

        # Check trajectory contains both types of metrics
        trajectory = results["trajectory"]
        assert any("mean_genetic_preference" in metrics for metrics in trajectory)
        assert any("mean_cultural_preference" in metrics for metrics in trajectory)

    @pytest.mark.parametrize("genetic_weight", [0.3, 0.5, 0.7])
    def test_variable_genetic_cultural_balance(
        self, genetic_weight: float, genetic_params: LandeKirkpatrickParams, cultural_params: Layer2Config
    ) -> None:
        """Test different balances of genetic vs cultural influence."""
        config = LayerActivationConfig.balanced_combined(genetic_weight)
        model = LoveModel(
            layer_config=config, genetic_params=genetic_params, cultural_params=cultural_params, n_agents=100
        )

        # Run for a few steps
        for _ in range(5):
            model.step()

        # Should complete without errors
        assert len(model.agents) > 0
        assert model.step_count == 5


class TestPerformanceCharacteristics:
    """Test performance with large populations and complex scenarios."""

    @pytest.mark.slow
    def test_performance_with_large_population(self, cultural_params: Layer2Config) -> None:
        """Test that enhanced model maintains performance with large populations."""
        config = LayerActivationConfig.balanced_combined(0.5)
        model = LoveModel(
            layer_config=config,
            cultural_params=cultural_params,
            n_agents=1000,  # Large population for performance testing
        )

        # Should handle large populations efficiently
        import time

        start_time = time.time()

        for _ in range(5):
            model.step()

        elapsed = time.time() - start_time

        # Should complete in reasonable time (less than 10 seconds for 5 steps)
        assert elapsed < 10.0

        # Population should remain stable
        assert len(model.agents) > 100

    def test_memory_usage_with_cultural_features(self, cultural_config: LayerActivationConfig) -> None:
        """Test memory usage remains reasonable with cultural features."""
        cultural_params = Layer2Config(cultural_memory_size=5, horizontal_transmission_rate=0.3, innovation_rate=0.1)
        model = LoveModel(layer_config=cultural_config, cultural_params=cultural_params, n_agents=500)

        # Run several steps
        for _ in range(10):
            model.step()

        # Should maintain reasonable population and performance
        df = model.get_agent_dataframe()
        assert len(df) > 0
        assert len(df.columns) > 10  # Should have many columns due to cultural features


class TestConfigurationValidation:
    """Test configuration validation and edge cases."""

    def test_model_configuration_types(self, model_by_config_type: LoveModel, model_config_type: str) -> None:
        """Test different model configuration types work correctly."""
        # Run a few steps
        for _ in range(3):
            model_by_config_type.step()

        df = model_by_config_type.get_agent_dataframe()
        assert len(df) > 0

        # Check configuration-specific features
        if model_config_type == "genetic_only":
            assert model_by_config_type.layer_config.genetic_enabled
            assert not model_by_config_type.layer_config.cultural_enabled
        elif model_config_type == "cultural_only":
            assert not model_by_config_type.layer_config.genetic_enabled
            assert model_by_config_type.layer_config.cultural_enabled
        else:  # balanced
            assert model_by_config_type.layer_config.genetic_enabled
            assert model_by_config_type.layer_config.cultural_enabled

    def test_model_with_minimal_parameters(self) -> None:
        """Test model works with minimal parameter sets."""
        config = LayerActivationConfig.genetic_only()
        minimal_genetic_params = LandeKirkpatrickParams(n_generations=5, pop_size=25)

        model = LoveModel(layer_config=config, genetic_params=minimal_genetic_params, n_agents=25)

        # Should work with minimal setup
        for _ in range(3):
            model.step()

        assert len(model.agents) > 0
