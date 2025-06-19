"""
Test configuration and fixtures.

This module provides shared fixtures and utilities for the LoveBug test suite,
eliminating repetition and ensuring consistency across all test modules.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest

from lovebug.lande_kirkpatrick import LandeKirkpatrickParams
from lovebug.layer2.config import Layer2Config
from lovebug.layer_activation import LayerActivationConfig
from lovebug.unified_mesa_model import LoveModel

# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow (can be skipped)")


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def genetic_config() -> LayerActivationConfig:
    """Genetic-only layer activation configuration."""
    return LayerActivationConfig.genetic_only()


@pytest.fixture
def cultural_config() -> LayerActivationConfig:
    """Cultural-only layer activation configuration."""
    return LayerActivationConfig.cultural_only()


@pytest.fixture
def balanced_config() -> LayerActivationConfig:
    """Balanced genetic and cultural configuration."""
    return LayerActivationConfig.balanced_combined(0.5)


@pytest.fixture
def genetic_params() -> LandeKirkpatrickParams:
    """Standard genetic parameters for testing."""
    return LandeKirkpatrickParams(
        n_generations=20,
        pop_size=100,
        h2_trait=0.5,
        h2_preference=0.3,
        selection_strength=0.1,
        mutation_variance=0.01,
    )


@pytest.fixture
def cultural_params() -> Layer2Config:
    """Standard cultural parameters for testing."""
    return Layer2Config(
        horizontal_transmission_rate=0.3,
        oblique_transmission_rate=0.2,
        innovation_rate=0.1,
        cultural_memory_size=5,
        local_learning_radius=3,
    )


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
def small_genetic_model(genetic_config: LayerActivationConfig, genetic_params: LandeKirkpatrickParams) -> LoveModel:
    """Small genetic-only model for quick testing."""
    return LoveModel(layer_config=genetic_config, genetic_params=genetic_params, n_agents=50)


@pytest.fixture
def small_cultural_model(cultural_config: LayerActivationConfig, cultural_params: Layer2Config) -> LoveModel:
    """Small cultural-only model for quick testing."""
    return LoveModel(layer_config=cultural_config, cultural_params=cultural_params, n_agents=50)


@pytest.fixture
def medium_balanced_model(
    balanced_config: LayerActivationConfig, genetic_params: LandeKirkpatrickParams, cultural_params: Layer2Config
) -> LoveModel:
    """Medium-sized balanced model for integration testing."""
    return LoveModel(
        layer_config=balanced_config, genetic_params=genetic_params, cultural_params=cultural_params, n_agents=100
    )


@pytest.fixture
def large_performance_model(
    balanced_config: LayerActivationConfig, genetic_params: LandeKirkpatrickParams, cultural_params: Layer2Config
) -> LoveModel:
    """Large model for performance testing."""
    return LoveModel(
        layer_config=balanced_config, genetic_params=genetic_params, cultural_params=cultural_params, n_agents=500
    )


# ============================================================================
# Mock Objects
# ============================================================================


@pytest.fixture
def mock_agent_set() -> Mock:
    """Mock agent set for testing Layer2 components."""
    mock_agents = Mock()
    mock_agents.agents = pl.DataFrame(
        {
            "agent_id": range(10),
            "genome": [123456] * 10,
            "energy": [5.0] * 10,
            "age": [2] * 10,
            "mating_success": [1] * 10,
            "pref_culture": [128] * 10,
            "cultural_innovation_count": [0] * 10,
            "prestige_score": [0.5] * 10,
            "last_learning_event": [0] * 10,
            "learning_eligibility": [True] * 10,
            "effective_preference": [128] * 10,
            "neighbors": [[] for _ in range(10)],
            "network_degree": [0] * 10,
        }
    )
    mock_agents.__len__ = lambda self: 10
    return mock_agents


@pytest.fixture
def sample_agents_df() -> pl.DataFrame:
    """Sample agent DataFrame for testing."""
    return pl.DataFrame(
        {
            "agent_id": range(10),
            "age": [0, 1, 2, 5, 10, 1, 2, 5, 10, 15],
            "energy": [0.5, 1.5, 2.0, 3.0, 5.0, 1.0, 2.5, 3.5, 4.5, 6.0],
            "network_degree": [0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
            "last_learning_event": [0, 0, 0, 2, 5, 0, 1, 2, 3, 4],
            "pref_culture": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
            "prestige_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "mating_success": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
        }
    )


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_trajectory_data() -> pl.DataFrame:
    """Sample trajectory data for visualization testing."""
    np.random.seed(42)  # Reproducible test data
    n_steps = 10

    return pl.DataFrame(
        {
            "step": range(n_steps),
            "mean_display": np.random.uniform(0, 65535, n_steps),
            "mean_preference": np.random.uniform(0, 255, n_steps),
            "mean_threshold": np.random.uniform(0, 255, n_steps),
            "display_variance": np.random.uniform(0, 1000, n_steps),
            "preference_variance": np.random.uniform(0, 100, n_steps),
            "threshold_variance": np.random.uniform(0, 100, n_steps),
            "trait_preference_covariance": np.random.uniform(-500, 500, n_steps),
            "population_size": np.random.randint(80, 120, n_steps),
            "mean_age": np.random.uniform(5, 50, n_steps),
            "mean_energy": np.random.uniform(5, 15, n_steps),
            "cultural_genetic_distance": np.random.uniform(0, 8, n_steps),
            "mating_success_rate": np.random.uniform(0.1, 0.9, n_steps),
        }
    )


@pytest.fixture
def test_data_file(tmp_path: Path, sample_trajectory_data: pl.DataFrame) -> Path:
    """Create a temporary test data file with metadata."""
    data_file = tmp_path / "test_data.parquet"
    sample_trajectory_data.write_parquet(data_file)

    # Create metadata file
    metadata = {
        "population_size": 100,
        "test_run": True,
        "experiment_id": "test_viz_001",
        "total_steps": len(sample_trajectory_data),
    }
    metadata_file = data_file.with_suffix(".metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return data_file


# ============================================================================
# Utility Functions
# ============================================================================


def run_model_steps(model: LoveModel, n_steps: int) -> list[dict[str, Any]]:
    """Run model for specified steps and return history.

    Parameters
    ----------
    model : LoveModel
        Model to run
    n_steps : int
        Number of steps to run

    Returns
    -------
    list[dict[str, Any]]
        Model history after running
    """
    for _ in range(n_steps):
        model.step()
    return model.history


def measure_execution_time(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Measure execution time of a function.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to measure
    *args : Any
        Positional arguments for function
    **kwargs : Any
        Keyword arguments for function

    Returns
    -------
    tuple[Any, float]
        Function result and execution time in seconds
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time


def assert_model_evolution(model: LoveModel, min_steps: int = 5) -> None:
    """Assert that model evolution has occurred properly.

    Parameters
    ----------
    model : LoveModel
        Model to check
    min_steps : int, default=5
        Minimum number of steps to verify
    """
    assert len(model.history) >= min_steps
    assert len(model.agents) > 0
    assert model.step_count >= min_steps


def assert_valid_config_serialization(config: LayerActivationConfig | Layer2Config) -> None:
    """Assert that configuration can be serialized and deserialized.

    Parameters
    ----------
    config : LayerActivationConfig | Layer2Config
        Configuration to test
    """
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)

    # Test reconstruction
    if isinstance(config, LayerActivationConfig):
        restored_config = LayerActivationConfig.from_dict(config_dict)
        assert restored_config.to_dict() == config_dict
    else:
        # Layer2Config doesn't have from_dict method, so just verify to_dict works
        assert config_dict is not None


def create_agents_with_memory(n_agents: int, memory_size: int) -> pl.DataFrame:
    """Create agent DataFrame with cultural memory columns.

    Parameters
    ----------
    n_agents : int
        Number of agents to create
    memory_size : int
        Size of cultural memory

    Returns
    -------
    pl.DataFrame
        DataFrame with agents and memory columns
    """
    df = pl.DataFrame(
        {
            "agent_id": range(n_agents),
            "pref_culture": [100 + i for i in range(n_agents)],
        }
    )

    # Add memory columns
    for i in range(memory_size):
        df = df.with_columns(
            [
                pl.lit(90.0 + i * 5).alias(f"cultural_memory_{i}"),
                pl.lit(0.8 - i * 0.1).alias(f"memory_weights_{i}"),
            ]
        )

    return df


# ============================================================================
# Session-scoped fixtures for expensive operations
# ============================================================================


@pytest.fixture(scope="session")
def session_config() -> dict[str, Any]:
    """Session-scoped configuration fixture."""
    return {"test_mode": True, "random_seed": 42}


@pytest.fixture(scope="session")
def temporary_directory() -> Generator[Path, None, None]:
    """Session-scoped temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Parametrized fixtures for testing multiple scenarios
# ============================================================================


@pytest.fixture(params=["genetic_only", "cultural_only", "balanced"])
def model_config_type(request: pytest.FixtureRequest) -> str:
    """Parametrized fixture for different model configuration types."""
    return request.param


@pytest.fixture
def model_by_config_type(
    model_config_type: str, genetic_params: LandeKirkpatrickParams, cultural_params: Layer2Config
) -> LoveModel:
    """Create model based on configuration type parameter."""
    if model_config_type == "genetic_only":
        config = LayerActivationConfig.genetic_only()
        return LoveModel(layer_config=config, genetic_params=genetic_params, n_agents=50)
    elif model_config_type == "cultural_only":
        config = LayerActivationConfig.cultural_only()
        return LoveModel(layer_config=config, cultural_params=cultural_params, n_agents=50)
    else:  # balanced
        config = LayerActivationConfig.balanced_combined(0.5)
        return LoveModel(
            layer_config=config, genetic_params=genetic_params, cultural_params=cultural_params, n_agents=50
        )
