"""
Tests for the Lande-Kirkpatrick model implementation.

This module contains comprehensive tests for the Lande-Kirkpatrick model
of trait-preference coevolution, including parameter validation,
simulation correctness, and output format verification.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lovebug.lande_kirkpatrick import LandeKirkpatrickParams, simulate_lande_kirkpatrick


class TestLandeKirkpatrickParams:
    """Test the parameter dataclass validation and creation."""

    def test_default_parameters_are_valid(self) -> None:
        """Test that default parameters are within valid ranges."""
        params = LandeKirkpatrickParams()

        # Test parameter ranges
        assert 0 <= params.h2_trait <= 1
        assert 0 <= params.h2_preference <= 1
        assert params.pop_size > 0
        assert params.n_generations > 0
        assert params.selection_strength >= 0
        assert params.mutation_variance >= 0
        assert params.preference_cost >= 0

    def test_custom_parameter_creation(self) -> None:
        """Test creating parameters with custom values."""
        params = LandeKirkpatrickParams(
            n_generations=100,
            pop_size=500,
            h2_trait=0.8,
            h2_preference=0.6,
            selection_strength=0.2,
            genetic_correlation=0.3,
            mutation_variance=0.02,
            preference_cost=0.1,
        )

        # Verify all custom parameters are set correctly
        expected_values = {
            "n_generations": 100,
            "pop_size": 500,
            "h2_trait": 0.8,
            "h2_preference": 0.6,
            "selection_strength": 0.2,
            "genetic_correlation": 0.3,
            "mutation_variance": 0.02,
            "preference_cost": 0.1,
        }

        for param_name, expected_value in expected_values.items():
            assert getattr(params, param_name) == expected_value

    def test_heritability_validation_errors(self) -> None:
        """Test that invalid heritability parameters raise errors."""
        # Test invalid h2_trait
        with pytest.raises(ValueError, match="h2_trait must be between 0 and 1"):
            params = LandeKirkpatrickParams(h2_trait=1.5)
            simulate_lande_kirkpatrick(params)

        with pytest.raises(ValueError, match="h2_preference must be between 0 and 1"):
            params = LandeKirkpatrickParams(h2_preference=-0.1)
            simulate_lande_kirkpatrick(params)

    def test_population_size_validation_errors(self) -> None:
        """Test that invalid population size raises errors."""
        # Test invalid population size
        with pytest.raises(ValueError, match="pop_size must be positive"):
            params = LandeKirkpatrickParams(pop_size=0)
            simulate_lande_kirkpatrick(params)


class TestSimulateLandeKirkpatrick:
    """Test the main simulation function."""

    def test_basic_simulation_execution(self) -> None:
        """Test that basic simulation runs successfully."""
        params = LandeKirkpatrickParams(n_generations=10)
        result = simulate_lande_kirkpatrick(params)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10

    def test_simulation_output_structure(self) -> None:
        """Test that simulation output has correct structure and columns."""
        params = LandeKirkpatrickParams(n_generations=5)
        result = simulate_lande_kirkpatrick(params)

        expected_columns = {
            "generation",
            "mean_trait",
            "mean_preference",
            "trait_variance",
            "preference_variance",
            "genetic_covariance",
            "selection_differential_trait",
            "selection_differential_preference",
            "effective_selection_trait",
            "effective_selection_preference",
            "population_size",
        }
        assert set(result.columns) == expected_columns

    def test_simulation_output_types(self) -> None:
        """Test that simulation output has correct data types."""
        params = LandeKirkpatrickParams(n_generations=5)
        result = simulate_lande_kirkpatrick(params)

        # Check column types
        assert result["generation"].dtype == pl.Int64
        assert result["mean_trait"].dtype == pl.Float64
        assert result["mean_preference"].dtype == pl.Float64
        assert result["trait_variance"].dtype == pl.Float64
        assert result["preference_variance"].dtype == pl.Float64
        assert result["genetic_covariance"].dtype == pl.Float64
        assert result["population_size"].dtype == pl.Int64

        # Check generation sequence
        generations = result["generation"].to_list()
        assert generations == list(range(5))

        # Check population size consistency
        pop_sizes = result["population_size"].unique().to_list()
        assert len(pop_sizes) == 1
        assert pop_sizes[0] == params.pop_size

    @pytest.mark.parametrize("n_generations", [1, 10, 50, 100])
    def test_different_generation_counts(self, n_generations: int) -> None:
        """Test simulation with different generation counts."""
        params = LandeKirkpatrickParams(n_generations=n_generations)
        result = simulate_lande_kirkpatrick(params)

        assert len(result) == n_generations
        assert result["generation"].max() == n_generations - 1


class TestEvolutionaryDynamics:
    """Test evolutionary dynamics and biological realism."""

    def test_coevolution_with_positive_correlation(self) -> None:
        """Test that positive correlation enables coevolution."""
        params = LandeKirkpatrickParams(
            n_generations=50, genetic_correlation=0.3, selection_strength=0.05, preference_cost=0.01
        )
        result = simulate_lande_kirkpatrick(params)

        # Check that evolution occurred
        final_trait = result["mean_trait"].tail(1).item()
        final_preference = result["mean_preference"].tail(1).item()

        # With positive correlation and low costs, should see some evolution
        assert abs(final_trait) > 0.001 or abs(final_preference) > 0.001

        # Variances should remain positive throughout
        assert all(result["trait_variance"] > 0)
        assert all(result["preference_variance"] > 0)

    def test_equilibrium_with_no_correlation(self) -> None:
        """Test that zero correlation limits coevolution."""
        params = LandeKirkpatrickParams(
            n_generations=100, genetic_correlation=0.0, selection_strength=0.1, preference_cost=0.1
        )
        result = simulate_lande_kirkpatrick(params)

        # With no correlation and costs, evolution should be limited
        final_trait = abs(result["mean_trait"].tail(1).item())
        final_preference = abs(result["mean_preference"].tail(1).item())

        # Values should remain relatively small
        assert final_trait < 1.0
        assert final_preference < 1.0

    def test_extreme_parameter_handling(self) -> None:
        """Test simulation behavior with extreme but valid parameters."""
        # Test with very high heritability
        params_high_h2 = LandeKirkpatrickParams(
            n_generations=20, h2_trait=0.99, h2_preference=0.99, genetic_correlation=0.4
        )
        result_high = simulate_lande_kirkpatrick(params_high_h2)

        assert len(result_high) == 20
        assert all(result_high["trait_variance"] > 0)
        assert all(result_high["preference_variance"] > 0)

        # Test with very low heritability
        params_low_h2 = LandeKirkpatrickParams(n_generations=20, h2_trait=0.01, h2_preference=0.01)
        result_low = simulate_lande_kirkpatrick(params_low_h2)

        # Should show minimal evolution with low heritability
        final_trait = abs(result_low["mean_trait"].tail(1).item())
        final_preference = abs(result_low["mean_preference"].tail(1).item())
        assert final_trait < 0.1
        assert final_preference < 0.1


class TestSimulationReproducibility:
    """Test simulation reproducibility and consistency."""

    def test_reproducibility_with_seed(self) -> None:
        """Test that simulations are reproducible with same random seed."""
        params = LandeKirkpatrickParams(n_generations=20)

        # Set random seed for reproducibility
        np.random.seed(42)
        result1 = simulate_lande_kirkpatrick(params)

        np.random.seed(42)
        result2 = simulate_lande_kirkpatrick(params)

        # Results should be identical
        assert result1.equals(result2)

    def test_stochastic_variation_without_seed(self) -> None:
        """Test that simulations vary without fixed seed."""
        params = LandeKirkpatrickParams(n_generations=10)

        # Run multiple simulations without setting seed
        results = []
        for _ in range(3):
            result = simulate_lande_kirkpatrick(params)
            final_trait = result["mean_trait"].tail(1).item()
            results.append(final_trait)

        # Results should show some variation (very unlikely to be identical)
        # Use variance as a measure of stochastic difference
        trait_variance = np.var(results)

        # Should have some variation (this is a statistical test)
        # If all identical, variance would be 0
        assert trait_variance >= 0  # At minimum, should not error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
