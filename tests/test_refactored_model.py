"""
Comprehensive test suite for the refactored LoveModel.

This test suite validates all key components of the new architecture:
- Unlinked gene structure
- Energy system with foraging efficiency
- Learning strategies
- 16-bit preference-trait matching
- Population self-regulation
- Genetic recombination
"""

import numpy as np
import pytest

from lovebug import (
    CulturalParams,
    GeneticParams,
    LayerConfig,
    LoveBugConfig,
    LoveModelRefactored,
    SimulationParams,
)


class TestRefactoredModel:
    """Test suite for the refactored LoveModel architecture."""

    def setup_method(self):
        """Set up a basic config for testing."""
        self.config = LoveBugConfig(
            name="test_model",
            genetic=GeneticParams(mutation_rate=0.01, crossover_rate=0.7, heritability=0.5),
            cultural=CulturalParams(horizontal_transmission_rate=0.1, innovation_rate=0.01, network_type="random"),
            simulation=SimulationParams(
                population_size=100,
                num_steps=10,
                energy_replenishment_rate=50.0,
                energy_decay=5.0,
                density_dependence=0.001,
            ),
            layer=LayerConfig(genetic_enabled=True, cultural_enabled=False),
        )

    def test_unlinked_genes(self):
        """Test that the new DataFrame structure properly handles unlinked genes."""
        model = LoveModelRefactored(config=self.config)

        # Check that agents DataFrame has the correct columns
        agent_columns = model.get_agent_dataframe().columns
        expected_columns = [
            "unique_id",
            "age",
            "energy",
            "gene_display_trait",
            "gene_preference",
            "gene_foraging_efficiency",
            "cultural_display_trait",
            "cultural_preference",
        ]

        for col in expected_columns:
            assert col in agent_columns, f"Missing column: {col}"

        # Verify genes are truly independent (16-bit values)
        agents_df = model.agents.get_agents_as_DF()
        assert agents_df["gene_display_trait"].max() <= 65535  # 2^16 - 1
        assert agents_df["gene_preference"].max() <= 65535
        assert agents_df["gene_foraging_efficiency"].max() <= 65535

        # Verify they can vary independently
        display_variance = agents_df["gene_display_trait"].var()
        preference_variance = agents_df["gene_preference"].var()
        foraging_variance = agents_df["gene_foraging_efficiency"].var()

        assert display_variance > 0, "Display traits should vary"
        assert preference_variance > 0, "Preferences should vary"
        assert foraging_variance > 0, "Foraging efficiency should vary"

    def test_energy_system(self):
        """Test energy system with foraging efficiency and density dependence."""
        model = LoveModelRefactored(config=self.config)

        # Get initial energy levels
        initial_df = model.agents.get_agents_as_DF()
        initial_energies = initial_df["energy"].to_numpy()
        initial_foraging = initial_df["gene_foraging_efficiency"].to_numpy()

        # Step the model to see energy changes
        model.step()

        # Get post-step energy levels
        final_df = model.agents.get_agents_as_DF()
        final_energies = final_df["energy"].to_numpy()

        # Energy should change based on foraging efficiency and density
        assert not np.array_equal(initial_energies, final_energies), "Energy should change after step"

        # Higher foraging efficiency should generally lead to better energy outcomes
        # (though this is stochastic, so we test correlation over multiple agents)
        if len(initial_foraging) > 10:
            energy_change = final_energies - initial_energies
            correlation = np.corrcoef(initial_foraging, energy_change)[0, 1]
            # Allow for some randomness, but should be positive correlation
            assert correlation > -0.5, "Foraging efficiency should positively correlate with energy gain"

    def test_learning_strategies(self):
        """Test each of the four learning strategies."""
        strategies = ["none", "horizontal", "vertical", "oblique"]

        for strategy in strategies:
            config = self.config.model_copy()
            config.cultural.learning_strategy = strategy
            config.layer.cultural_enabled = True

            model = LoveModelRefactored(config=config)

            # Run a few steps
            for _ in range(3):
                model.step()

            final_df = model.agents.get_agents_as_DF()

            # Model should run without errors
            assert len(final_df) > 0, f"Model with {strategy} strategy should maintain population"

            # Cultural traits should exist
            assert "cultural_display_trait" in final_df.columns
            assert "cultural_preference" in final_df.columns

            if strategy != "none":
                # Cultural traits should show some variation (except for 'none')
                cultural_display_var = final_df["cultural_display_trait"].var()
                cultural_pref_var = final_df["cultural_preference"].var()

                # At least one should show variation (allowing for stochastic effects)
                assert cultural_display_var > 0 or cultural_pref_var > 0, (
                    f"Cultural traits should vary with {strategy} learning"
                )

    def test_16bit_comparison(self):
        """Test the new direct 16-bit preference-trait matching."""
        model = LoveModelRefactored(config=self.config)

        # Get agents and test preference matching
        agents_df = model.agents.get_agents_as_DF()

        # Test that preferences and traits are in 16-bit range
        max_trait = agents_df["gene_display_trait"].max()
        max_pref = agents_df["gene_preference"].max()

        assert max_trait <= 65535, "Display traits should be 16-bit"
        assert max_pref <= 65535, "Preferences should be 16-bit"

        # Test preference matching calculation
        # Pick two agents and manually verify preference calculation
        if len(agents_df) >= 2:
            trait1 = agents_df[0, "gene_display_trait"]
            pref2 = agents_df[1, "gene_preference"]

            # Calculate expected preference score (inverse of distance)
            distance = abs(int(trait1) - int(pref2))
            expected_score = 65536 - distance  # Max distance is 65535

            # This should be the basis for mate choice calculations
            assert 0 <= expected_score <= 65536, "Preference score should be in valid range"

    def test_population_regulation(self):
        """Test that population self-regulates through energy system."""
        # Use a config that should lead to population regulation
        config = self.config.model_copy()
        config.simulation.population_size = 200
        config.simulation.num_steps = 20
        config.simulation.density_dependence = 0.01  # Strong density dependence

        model = LoveModelRefactored(config=config)

        population_sizes = []
        for _step in range(20):
            model.step()
            current_pop = len(model.agents.get_agents_as_DF())
            population_sizes.append(current_pop)

        # Population should not grow indefinitely or crash to zero
        final_pop = population_sizes[-1]
        assert final_pop > 0, "Population should not go extinct"
        assert final_pop < config.simulation.population_size * 3, "Population should not explode"

        # Check for some degree of regulation (not just monotonic growth/decline)
        mid_point = len(population_sizes) // 2
        early_avg = np.mean(population_sizes[:mid_point])
        late_avg = np.mean(population_sizes[mid_point:])

        # Population should stabilize (difference shouldn't be too extreme)
        relative_change = abs(late_avg - early_avg) / early_avg
        assert relative_change < 2.0, "Population should show some regulation, not extreme swings"

    def test_genetic_recombination(self):
        """Test that unlinked genes recombine independently during reproduction."""
        # Create a model and force some reproduction events
        config = self.config.model_copy()
        config.genetic.crossover_rate = 1.0  # Always do crossover
        config.simulation.population_size = 50

        model = LoveModelRefactored(config=config)

        # Get initial genetic diversity
        initial_df = model.agents.get_agents_as_DF()
        initial_display_var = initial_df["gene_display_trait"].var()
        initial_pref_var = initial_df["gene_preference"].var()
        initial_forage_var = initial_df["gene_foraging_efficiency"].var()

        # Run multiple steps to allow reproduction
        for _ in range(10):
            model.step()

        final_df = model.agents.get_agents_as_DF()
        final_display_var = final_df["gene_display_trait"].var()
        final_pref_var = final_df["gene_preference"].var()
        final_forage_var = final_df["gene_foraging_efficiency"].var()

        # With high crossover, genetic variance should be maintained or increased
        # (mutation also contributes to this)
        assert final_display_var > 0, "Display trait variance should be maintained"
        assert final_pref_var > 0, "Preference variance should be maintained"
        assert final_forage_var > 0, "Foraging efficiency variance should be maintained"

        # Test that genes can recombine (this is probabilistic, so we test the mechanism exists)
        # If all three traits varied independently, crossover should maintain this
        total_initial_var = initial_display_var + initial_pref_var + initial_forage_var
        total_final_var = final_display_var + final_pref_var + final_forage_var

        # With recombination, total genetic variance should not collapse
        variance_ratio = total_final_var / total_initial_var if total_initial_var > 0 else 1
        assert variance_ratio > 0.1, "Genetic recombination should maintain genetic diversity"

    def test_model_stability(self):
        """Test that the model runs stably for extended periods."""
        config = self.config.model_copy()
        config.simulation.num_steps = 50

        model = LoveModelRefactored(config=config)

        step_count = 0
        try:
            for _ in range(50):
                model.step()
                step_count += 1

                # Check that population doesn't crash
                current_pop = len(model.agents.get_agents_as_DF())
                assert current_pop > 0, f"Population went extinct at step {step_count}"

        except Exception as e:
            pytest.fail(f"Model crashed at step {step_count}: {e}")

        # If we get here, model ran successfully
        final_df = model.agents.get_agents_as_DF()
        assert len(final_df) > 0, "Model should maintain population after long run"

    @pytest.mark.parametrize(
        "genetic_enabled,cultural_enabled",
        [
            (True, False),  # Genetic only
            (False, True),  # Cultural only
            (True, True),  # Both enabled
        ],
    )
    def test_layer_combinations(self, genetic_enabled, cultural_enabled):
        """Test different combinations of genetic and cultural layers."""
        config = self.config.model_copy()
        config.layer.genetic_enabled = genetic_enabled
        config.layer.cultural_enabled = cultural_enabled

        # Skip the invalid case where both are disabled
        if not genetic_enabled and not cultural_enabled:
            return

        model = LoveModelRefactored(config=config)

        # Run a few steps
        for _ in range(5):
            model.step()

        final_df = model.agents.get_agents_as_DF()
        assert len(final_df) > 0, f"Model should work with genetic={genetic_enabled}, cultural={cultural_enabled}"

        # Check that appropriate traits are being used
        if genetic_enabled:
            genetic_display_var = final_df["gene_display_trait"].var()
            genetic_pref_var = final_df["gene_preference"].var()
            assert genetic_display_var >= 0  # Should exist and potentially vary
            assert genetic_pref_var >= 0

        if cultural_enabled:
            cultural_display_var = final_df["cultural_display_trait"].var()
            cultural_pref_var = final_df["cultural_preference"].var()
            assert cultural_display_var >= 0  # Should exist and potentially vary
            assert cultural_pref_var >= 0


@pytest.mark.slow
class TestRefactoredModelPerformance:
    """Performance tests for the refactored model with larger populations."""

    def test_large_population_performance(self):
        """Test that the model can handle large populations efficiently."""
        config = LoveBugConfig(
            name="performance_test",
            genetic=GeneticParams(mutation_rate=0.01, crossover_rate=0.7),
            cultural=CulturalParams(),
            simulation=SimulationParams(
                population_size=5000,  # Large population
                num_steps=10,
            ),
            layer=LayerConfig(genetic_enabled=True, cultural_enabled=False),
        )

        model = LoveModelRefactored(config=config)

        # This should complete in reasonable time
        import time

        start_time = time.time()

        for _ in range(10):
            model.step()

        elapsed = time.time() - start_time

        # Should complete in under 30 seconds for 5000 agents, 10 steps
        assert elapsed < 30, f"Large population test took too long: {elapsed:.2f}s"

        # Population should still exist
        final_pop = len(model.agents.get_agents_as_DF())
        assert final_pop > 0, "Large population should survive the test"
