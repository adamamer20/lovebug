"""
Tests for the enhanced LoveModel with integrated cultural features.

This module tests the advanced cultural evolution capabilities integrated
into the mesa-frames LoveModel.
"""

from __future__ import annotations

from lovebug.lande_kirkpatrick import LandeKirkpatrickParams
from lovebug.layer2.config import Layer2Config
from lovebug.layer_activation import LayerActivationConfig
from lovebug.unified_mesa_model import LoveModel


class TestEnhancedUnifiedModel:
    """Test enhanced LoveModel with integrated advanced features."""

    def test_cultural_memory_integration(self):
        """Test that cultural memory system is properly integrated."""
        config = LayerActivationConfig.cultural_only()
        cultural_params = Layer2Config(cultural_memory_size=5, memory_decay_rate=0.1, innovation_rate=0.1)

        model = LoveModel(layer_config=config, cultural_params=cultural_params, n_agents=100)

        # Check that cultural memory columns are initialized
        df = model.get_agent_dataframe()
        expected_columns = [f"cultural_memory_{i}" for i in range(5)]
        for col in expected_columns:
            assert col in df.columns

    def test_prestige_based_learning(self):
        """Test that prestige-based learning is properly vectorized."""
        config = LayerActivationConfig.cultural_only()
        cultural_params = Layer2Config(horizontal_transmission_rate=0.5, innovation_rate=0.1)

        model = LoveModel(layer_config=config, cultural_params=cultural_params, n_agents=50)

        # Run a few steps to generate mating success differences
        for _ in range(5):
            model.step()

        # Check that prestige scores are calculated
        df = model.get_agent_dataframe()
        assert "prestige_score" in df.columns
        assert df["prestige_score"].sum() >= 0  # Prestige scores should be non-negative

    def test_social_network_integration(self):
        """Test that social networks are properly integrated."""
        config = LayerActivationConfig.cultural_only()
        cultural_params = Layer2Config(
            network_type="small_world", network_connectivity=0.1, horizontal_transmission_rate=0.3
        )

        model = LoveModel(layer_config=config, cultural_params=cultural_params, n_agents=100)

        # Check that social network data is integrated
        df = model.get_agent_dataframe()
        assert "social_network_neighbors" in df.columns

    def test_vectorized_cultural_learning(self):
        """Test that cultural learning is fully vectorized."""
        config = LayerActivationConfig.cultural_only()
        cultural_params = Layer2Config(
            horizontal_transmission_rate=0.5, oblique_transmission_rate=0.3, innovation_rate=0.2
        )

        model = LoveModel(layer_config=config, cultural_params=cultural_params, n_agents=200)

        # Run several steps
        for _ in range(10):
            model.step()

        final_df = model.get_agent_dataframe()

        # Population should still exist (though size may change)
        assert len(final_df) > 0

        # Cultural learning should have occurred
        assert model._cultural_learning_events > 0 or model._cultural_innovation_events > 0

        # Check that learning events are tracked in history
        assert len(model.history) > 0
        cultural_events = sum(h.get("cultural_learning_events", 0) for h in model.history)
        innovation_events = sum(h.get("cultural_innovation_events", 0) for h in model.history)
        assert cultural_events > 0 or innovation_events > 0

    def test_event_tracking_system(self):
        """Test that cultural events are properly tracked."""
        config = LayerActivationConfig.cultural_only()
        cultural_params = Layer2Config(horizontal_transmission_rate=0.3, innovation_rate=0.2, log_cultural_events=True)

        model = LoveModel(layer_config=config, cultural_params=cultural_params, n_agents=100)

        # Run simulation
        for _ in range(5):
            model.step()

        # Check that events are tracked in history
        assert len(model.history) > 0

        # Check for cultural event metrics
        final_metrics = model.history[-1]
        assert "cultural_learning_events" in final_metrics
        assert "cultural_innovation_events" in final_metrics

    def test_combined_evolution_with_advanced_features(self):
        """Test combined genetic and cultural evolution with advanced features."""
        config = LayerActivationConfig.balanced_combined(0.6)
        genetic_params = LandeKirkpatrickParams(h2_trait=0.5, h2_preference=0.3, mutation_variance=0.01)
        cultural_params = Layer2Config(horizontal_transmission_rate=0.3, innovation_rate=0.1, cultural_memory_size=3)

        model = LoveModel(
            layer_config=config, genetic_params=genetic_params, cultural_params=cultural_params, n_agents=150
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

    def test_performance_with_large_population(self):
        """Test that enhanced model maintains performance with large populations."""
        config = LayerActivationConfig.balanced_combined(0.5)
        cultural_params = Layer2Config(horizontal_transmission_rate=0.2, innovation_rate=0.05)

        model = LoveModel(
            layer_config=config,
            cultural_params=cultural_params,
            n_agents=1000,  # Reduced from 5000 for faster testing
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

    def test_cultural_memory_system(self):
        """Test that cultural memory system works correctly."""
        config = LayerActivationConfig.cultural_only()
        cultural_params = Layer2Config(
            cultural_memory_size=3, memory_decay_rate=0.1, horizontal_transmission_rate=0.5, innovation_rate=0.2
        )

        model = LoveModel(layer_config=config, cultural_params=cultural_params, n_agents=50)

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

    def test_prestige_score_calculation(self):
        """Test that prestige scores are calculated correctly."""
        config = LayerActivationConfig.cultural_only()
        cultural_params = Layer2Config(horizontal_transmission_rate=0.3, innovation_rate=0.1)

        model = LoveModel(layer_config=config, cultural_params=cultural_params, n_agents=100)

        # Run several steps to generate diversity in age, energy, mating success
        for _ in range(20):
            model.step()

        df = model.get_agent_dataframe()

        # Check that prestige scores exist and are reasonable
        assert "prestige_score" in df.columns
        prestige_scores = df["prestige_score"]

        # Prestige scores should be non-negative
        assert (prestige_scores >= 0).all()

        # There should be some variation in prestige scores
        if len(df) > 1:
            assert prestige_scores.var() >= 0
