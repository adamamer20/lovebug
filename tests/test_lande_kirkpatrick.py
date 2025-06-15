"""
Tests for the Lande-Kirkpatrick model implementation.

This module contains comprehensive tests for the Lande-Kirkpatrick model
of trait-preference coevolution, including parameter validation,
simulation correctness, and output format verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add notebooks to path for importing
sys.path.append(str(Path(__file__).parent.parent / "notebooks"))

# Import the model components from the standalone module
from lovebug.lande_kirkpatrick import LandeKirkpatrickParams, simulate_lande_kirkpatrick


class TestLandeKirkpatrickParams:
    """Test the parameter dataclass."""

    def test_default_parameters(self):
        """Test that default parameters are within valid ranges."""
        params = LandeKirkpatrickParams()

        assert 0 <= params.h2_trait <= 1
        assert 0 <= params.h2_preference <= 1
        assert params.pop_size > 0
        assert params.n_generations > 0
        assert params.selection_strength >= 0
        assert params.mutation_variance >= 0
        assert params.preference_cost >= 0

    def test_custom_parameters(self):
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

        assert params.n_generations == 100
        assert params.pop_size == 500
        assert params.h2_trait == 0.8
        assert params.h2_preference == 0.6
        assert params.selection_strength == 0.2
        assert params.genetic_correlation == 0.3
        assert params.mutation_variance == 0.02
        assert params.preference_cost == 0.1


class TestSimulateLandeKirkpatrick:
    """Test the main simulation function."""

    def test_basic_simulation(self):
        """Test that basic simulation runs successfully."""
        params = LandeKirkpatrickParams(n_generations=10)
        result = simulate_lande_kirkpatrick(params)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10

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

    def test_parameter_validation(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid heritability
        with pytest.raises(ValueError, match="h2_trait must be between 0 and 1"):
            params = LandeKirkpatrickParams(h2_trait=1.5)
            simulate_lande_kirkpatrick(params)

        with pytest.raises(ValueError, match="h2_preference must be between 0 and 1"):
            params = LandeKirkpatrickParams(h2_preference=-0.1)
            simulate_lande_kirkpatrick(params)

        # Test invalid population size
        with pytest.raises(ValueError, match="pop_size must be positive"):
            params = LandeKirkpatrickParams(pop_size=0)
            simulate_lande_kirkpatrick(params)

    def test_simulation_output_format(self):
        """Test that simulation output has correct format and types."""
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

    def test_evolution_dynamics(self):
        """Test that evolution produces reasonable dynamics."""
        # Test with strong positive correlation - should show coevolution
        params = LandeKirkpatrickParams(
            n_generations=50, genetic_correlation=0.3, selection_strength=0.05, preference_cost=0.01
        )
        result = simulate_lande_kirkpatrick(params)

        # Check that values change over time (not stuck at zero)
        final_trait = result["mean_trait"].tail(1).item()
        final_preference = result["mean_preference"].tail(1).item()

        # With positive correlation and low costs, should see some evolution
        assert abs(final_trait) > 0.01 or abs(final_preference) > 0.01

        # Variances should remain positive
        assert result["trait_variance"].min() > 0
        assert result["preference_variance"].min() > 0

    def test_equilibrium_conditions(self):
        """Test that certain parameter combinations lead to equilibrium."""
        # Test with zero correlation - should lead to minimal coevolution
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

    @pytest.mark.parametrize("n_generations", [1, 10, 50, 100])
    def test_different_generation_counts(self, n_generations):
        """Test simulation with different generation counts."""
        params = LandeKirkpatrickParams(n_generations=n_generations)
        result = simulate_lande_kirkpatrick(params)

        assert len(result) == n_generations
        assert result["generation"].max() == n_generations - 1

    def test_reproducibility(self):
        """Test that simulations are reproducible with same random seed."""
        params = LandeKirkpatrickParams(n_generations=20)

        # Set random seed for reproducibility
        np.random.seed(42)
        result1 = simulate_lande_kirkpatrick(params)

        np.random.seed(42)
        result2 = simulate_lande_kirkpatrick(params)

        # Results should be identical
        assert result1.equals(result2)

    def test_extreme_parameters(self):
        """Test simulation with extreme but valid parameters."""
        # Test with very high heritability
        params = LandeKirkpatrickParams(n_generations=20, h2_trait=0.99, h2_preference=0.99, genetic_correlation=0.4)
        result = simulate_lande_kirkpatrick(params)

        # Should still produce valid output
        assert len(result) == 20
        assert all(result["trait_variance"] > 0)
        assert all(result["preference_variance"] > 0)

        # Test with very low heritability
        params = LandeKirkpatrickParams(n_generations=20, h2_trait=0.01, h2_preference=0.01)
        result = simulate_lande_kirkpatrick(params)

        # Should show minimal evolution
        final_trait = abs(result["mean_trait"].tail(1).item())
        final_preference = abs(result["mean_preference"].tail(1).item())
        assert final_trait < 0.1
        assert final_preference < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
