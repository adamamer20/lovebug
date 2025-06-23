"""
Tests for the unified LoveModel.

This module contains tests for the core LoveModel functionality,
including basic operations, population dynamics, and genetic evolution.
"""

from __future__ import annotations

import numpy as np
import pytest

from lovebug.model import LoveModelRefactored as LoveModel


class TestLoveModelBasics:
    """Test basic LoveModel functionality."""

    def test_hamming_similarity_all_bits_different(self) -> None:
        """Test Hamming similarity calculation with completely different bit patterns."""
        a = np.array([0], dtype=np.uint32)
        b = np.array([0xFFFF], dtype=np.uint32)

        # Simple Hamming similarity: 16 - number of different bits
        xor_result = a ^ b
        hamming_distance = bin(xor_result[0]).count("1")
        similarity = 16 - hamming_distance

        assert similarity == 0  # All 16 bits are different

    def test_model_initialization(self, genetic_only_config) -> None:
        """Test that LoveModel initializes correctly."""
        model = LoveModel(config=genetic_only_config)

        assert len(model.agents) == genetic_only_config.simulation.population_size
        assert model.config.layer.genetic_enabled is True
        assert model.config.layer.cultural_enabled is False

    def test_model_step_execution(self, small_genetic_model: LoveModel) -> None:
        """Test that model can execute steps without errors."""
        # Test that model can step
        small_genetic_model.step()

        # Population should still exist after one step
        assert len(small_genetic_model.agents) > 0
        assert small_genetic_model.step_count == 1

    def test_population_growth_over_time(self, small_genetic_model: LoveModel) -> None:
        """Test that population can grow through reproduction."""
        # Run for several steps to allow reproduction
        for _ in range(10):
            small_genetic_model.step()

        # Population should potentially grow or at least remain viable
        assert len(small_genetic_model.agents) > 0

        # Check that some evolution occurred
        assert small_genetic_model.step_count == 10

    @pytest.mark.parametrize("n_agents", [25, 50, 100])
    def test_model_with_different_population_sizes(self, genetic_only_config, n_agents: int) -> None:
        """Test model works with different initial population sizes."""
        config = genetic_only_config
        config.simulation.population_size = n_agents
        model = LoveModel(config=config)

        assert len(model.agents) == n_agents

        # Run a few steps to ensure stability
        for _ in range(3):
            model.step()

        # Should maintain some population
        assert len(model.agents) > 0
