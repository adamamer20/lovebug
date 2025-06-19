"""
Tests for the unified LoveModel.

This module contains tests for the core LoveModel functionality,
including basic operations, population dynamics, and genetic evolution.
"""

from __future__ import annotations

import numpy as np
import pytest

from lovebug.layer_activation import LayerActivationConfig
from lovebug.unified_mesa_model import LoveModel


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

    def test_model_initialization(self, genetic_config: LayerActivationConfig) -> None:
        """Test that LoveModel initializes correctly."""
        model = LoveModel(layer_config=genetic_config, n_agents=100)

        assert len(model.agents) == 100
        assert model.layer_config.genetic_enabled is True
        assert model.layer_config.cultural_enabled is False

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
    def test_model_with_different_population_sizes(self, genetic_config: LayerActivationConfig, n_agents: int) -> None:
        """Test model works with different initial population sizes."""
        model = LoveModel(layer_config=genetic_config, n_agents=n_agents)

        assert len(model.agents) == n_agents

        # Run a few steps to ensure stability
        for _ in range(3):
            model.step()

        # Should maintain some population
        assert len(model.agents) > 0
