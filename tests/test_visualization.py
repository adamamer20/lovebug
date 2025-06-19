"""
Comprehensive pytest suite for the LoveBug visualization system.

Tests the visualization data collection, chart factory, and engine components
with proper fixtures, assertions, and cleanup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from lovebug.layer_activation import LayerActivationConfig
from lovebug.unified_mesa_model import LoveModel
from lovebug.visualization.core import ChartFactory, VisualizationEngine
from lovebug.visualization.data import DataCollector, DataLoader

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_model() -> LoveModel:
    """Create a small LoveModel for testing.

    Returns:
        LoveModel instance with 50 agents
    """
    config = LayerActivationConfig.genetic_only()
    return LoveModel(layer_config=config, n_agents=50)


@pytest.fixture
def sample_data_collector() -> DataCollector:
    """Create a DataCollector with test metadata.

    Returns:
        Configured DataCollector instance
    """
    collector = DataCollector()
    collector.set_metadata(population_size=50, test_run=True, experiment_id="test_viz_001")
    return collector


@pytest.fixture
def sample_trajectory_data() -> pl.DataFrame:
    """Create sample trajectory data for testing.

    Returns:
        Polars DataFrame with complete trajectory metrics
    """
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
    """Create a temporary test data file.

    Args:
        tmp_path: Pytest temporary directory fixture
        sample_trajectory_data: Test trajectory data

    Returns:
        Path to the created test data file
    """
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


@pytest.mark.unit
class TestDataCollector:
    """Test suite for DataCollector functionality."""

    def test_initialization(self) -> None:
        """Test DataCollector initialization."""
        collector = DataCollector()

        assert collector.data_history == []
        assert collector.metadata == {}

    def test_set_metadata(self, sample_data_collector: DataCollector) -> None:
        """Test metadata setting functionality."""
        additional_metadata = {"researcher": "test_user", "version": "1.0"}
        sample_data_collector.set_metadata(**additional_metadata)

        expected_metadata = {
            "population_size": 50,
            "test_run": True,
            "experiment_id": "test_viz_001",
            "researcher": "test_user",
            "version": "1.0",
        }

        assert sample_data_collector.metadata == expected_metadata

    def test_collect_step_data_with_empty_model(self, sample_data_collector: DataCollector) -> None:
        """Test data collection with empty/extinct population."""
        # Mock an empty model
        empty_model = Mock()
        empty_model.agents = []

        step_data = sample_data_collector.collect_step_data(empty_model, 0)

        assert step_data["step"] == 0
        assert step_data["population_size"] == 0
        assert np.isnan(step_data["mean_display"])
        assert np.isnan(step_data["mean_preference"])
        assert step_data["mating_success_rate"] == 0.0
        assert len(sample_data_collector.data_history) == 1

    @pytest.mark.integration
    def test_collect_step_data_with_live_model(
        self, sample_data_collector: DataCollector, sample_model: LoveModel
    ) -> None:
        """Test data collection with a live model."""
        # Run model for a few steps
        for step in range(3):
            sample_model.step()
            step_data = sample_data_collector.collect_step_data(sample_model, step)

            # Verify data structure
            assert isinstance(step_data, dict)
            assert step_data["step"] == step
            assert isinstance(step_data["population_size"], int)
            assert step_data["population_size"] > 0

            # Verify genetic metrics are present
            required_keys = [
                "mean_display",
                "mean_preference",
                "mean_threshold",
                "display_variance",
                "preference_variance",
                "threshold_variance",
                "trait_preference_covariance",
                "mating_success_rate",
                "mean_age",
                "mean_energy",
            ]

            for key in required_keys:
                assert key in step_data
                assert not np.isnan(step_data[key]) or key == "cultural_genetic_distance"

        assert len(sample_data_collector.data_history) == 3

    def test_save_and_clear_data(
        self, sample_data_collector: DataCollector, sample_model: LoveModel, tmp_path: Path
    ) -> None:
        """Test saving data to file and clearing memory."""
        # Collect some data
        for step in range(2):
            sample_model.step()
            sample_data_collector.collect_step_data(sample_model, step)

        # Save data
        test_file = tmp_path / "test_save.parquet"
        sample_data_collector.save_run_data(test_file)

        # Verify files exist
        assert test_file.exists()
        assert test_file.with_suffix(".metadata.json").exists()

        # Verify data can be loaded
        loaded_data = pl.read_parquet(test_file)
        assert len(loaded_data) == 2
        assert "step" in loaded_data.columns
        assert "population_size" in loaded_data.columns

        # Test clear functionality
        sample_data_collector.clear()
        assert sample_data_collector.data_history == []
        assert sample_data_collector.metadata == {}

    def test_save_empty_data_warning(self, sample_data_collector: DataCollector, tmp_path: Path) -> None:
        """Test that saving empty data issues a warning."""
        test_file = tmp_path / "empty_test.parquet"

        with pytest.warns(UserWarning, match="No data collected yet"):
            sample_data_collector.save_run_data(test_file)

        assert not test_file.exists()


@pytest.mark.unit
class TestDataLoader:
    """Test suite for DataLoader functionality."""

    def test_initialization_with_valid_file(self, test_data_file: Path) -> None:
        """Test DataLoader initialization with valid file."""
        loader = DataLoader(test_data_file)

        assert loader.filepath == test_data_file
        assert loader._data is None  # Lazy loading not triggered yet
        assert loader._metadata is None

    def test_initialization_with_invalid_file(self, tmp_path: Path) -> None:
        """Test DataLoader initialization with non-existent file."""
        invalid_file = tmp_path / "nonexistent.parquet"

        with pytest.raises(FileNotFoundError, match="Data file not found"):
            DataLoader(invalid_file)

    def test_lazy_data_loading(self, test_data_file: Path) -> None:
        """Test lazy loading of data."""
        loader = DataLoader(test_data_file)

        # First access should load data
        data = loader.data
        assert isinstance(data, pl.DataFrame)
        assert len(data) == 10
        assert "step" in data.columns

        # Second access should use cached data
        data2 = loader.data
        assert data is data2  # Same object reference

    def test_metadata_loading(self, test_data_file: Path) -> None:
        """Test metadata loading functionality."""
        loader = DataLoader(test_data_file)

        metadata = loader.metadata
        assert isinstance(metadata, dict)
        assert metadata["population_size"] == 100
        assert metadata["test_run"] is True
        assert metadata["experiment_id"] == "test_viz_001"

    def test_get_time_range(self, test_data_file: Path) -> None:
        """Test time range filtering."""
        loader = DataLoader(test_data_file)

        # Test partial range
        partial_data = loader.get_time_range(2, 5)
        assert len(partial_data) == 4  # Steps 2, 3, 4, 5
        assert partial_data["step"].min() == 2
        assert partial_data["step"].max() == 5

        # Test open-ended range
        tail_data = loader.get_time_range(7)
        assert len(tail_data) == 3  # Steps 7, 8, 9
        assert tail_data["step"].min() == 7

    def test_get_final_state(self, test_data_file: Path) -> None:
        """Test final state extraction."""
        loader = DataLoader(test_data_file)

        final_state = loader.get_final_state()
        assert len(final_state) == 1
        assert final_state["step"].item() == 9  # Last step

    def test_get_summary_stats(self, test_data_file: Path) -> None:
        """Test summary statistics computation."""
        loader = DataLoader(test_data_file)

        summary = loader.get_summary_stats()

        assert isinstance(summary, dict)
        assert summary["total_steps"] == 10
        assert isinstance(summary["final_population"], int)
        assert isinstance(summary["max_population"], int)
        assert isinstance(summary["min_population"], int)
        assert isinstance(summary["mean_covariance"], float)
        assert isinstance(summary["final_covariance"], float)
        assert summary["extinction_step"] is None  # No extinction in test data

    def test_get_trajectory_data(self, test_data_file: Path) -> None:
        """Test trajectory data extraction."""
        loader = DataLoader(test_data_file)

        # Test valid metrics
        metrics = ["mean_display", "mean_preference", "trait_preference_covariance"]
        trajectory = loader.get_trajectory_data(metrics)

        assert len(trajectory) == 10
        assert set(trajectory.columns) == {"step"} | set(metrics)

        # Test invalid metrics
        with pytest.raises(ValueError, match="Metrics not found in data"):
            loader.get_trajectory_data(["invalid_metric"])


@pytest.mark.unit
class TestChartFactory:
    """Test suite for ChartFactory functionality."""

    def test_initialization(self) -> None:
        """Test ChartFactory initialization."""
        factory = ChartFactory()

        # Should have at least trajectory chart registered
        chart_types = factory.list_chart_types()
        assert isinstance(chart_types, list)
        # Note: Available charts depend on imports, so we just test the interface

    def test_list_chart_types(self) -> None:
        """Test chart type listing."""
        factory = ChartFactory()
        chart_types = factory.list_chart_types()

        assert isinstance(chart_types, list)
        # All chart types should be strings
        for chart_type in chart_types:
            assert isinstance(chart_type, str)

    @patch("lovebug.visualization.charts.trajectory.TrajectoryChart")
    def test_create_chart_success(self, mock_chart_class, test_data_file: Path) -> None:
        """Test successful chart creation."""
        factory = ChartFactory()
        loader = DataLoader(test_data_file)
        config = {"title": "Test Chart", "trajectory_type": "trait_preference"}

        # Register mock chart
        factory.register_chart("test_chart", mock_chart_class)

        # Create chart
        chart = factory.create_chart("test_chart", loader, config)

        # Verify mock was called correctly
        mock_chart_class.assert_called_once_with(loader, config)
        assert chart is mock_chart_class.return_value

    def test_create_unknown_chart_type(self, test_data_file: Path) -> None:
        """Test error handling for unknown chart types."""
        factory = ChartFactory()
        loader = DataLoader(test_data_file)

        with pytest.raises(ValueError, match="Unknown chart type: nonexistent"):
            factory.create_chart("nonexistent", loader, {})


@pytest.mark.unit
class TestVisualizationEngine:
    """Test suite for VisualizationEngine functionality."""

    def test_initialization(self, test_data_file: Path) -> None:
        """Test VisualizationEngine initialization."""
        engine = VisualizationEngine(test_data_file)

        assert isinstance(engine.data_loader, DataLoader)
        assert isinstance(engine.chart_factory, ChartFactory)
        assert engine.data_loader.filepath == test_data_file

    def test_list_available_charts(self, test_data_file: Path) -> None:
        """Test available charts listing."""
        engine = VisualizationEngine(test_data_file)

        charts = engine.list_available_charts()
        assert isinstance(charts, list)
        # Interface test - actual availability depends on imports

    def test_list_available_backends(self, test_data_file: Path) -> None:
        """Test available backends listing."""
        engine = VisualizationEngine(test_data_file)

        backends = engine.list_available_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0

        # Should include at least the static backend
        assert "static" in backends

        # Should include placeholder backends for future development
        future_backends = {"animation", "web"}
        assert future_backends.issubset(set(backends))

        # All backends should be strings
        for backend in backends:
            assert isinstance(backend, str)
            assert len(backend) > 0

    def test_get_data_summary(self, test_data_file: Path) -> None:
        """Test data summary retrieval."""
        engine = VisualizationEngine(test_data_file)

        summary = engine.get_data_summary()
        assert isinstance(summary, dict)
        assert "total_steps" in summary
        assert summary["total_steps"] == 10

    def test_unimplemented_backend_error(self, test_data_file: Path) -> None:
        """Test that unimplemented backends raise appropriate errors."""
        engine = VisualizationEngine(test_data_file)

        # Test backends that are registered but not implemented
        unimplemented_backends = ["animation", "web"]

        for backend_name in unimplemented_backends:
            if backend_name in engine.list_available_backends():
                with pytest.raises(ValueError, match=f"Backend '{backend_name}' is not yet implemented"):
                    engine.backend_registry.get_backend(backend_name)

        # Interactive backend should warn but be registered (if plotly available)
        if "interactive" in engine.list_available_backends():
            # Should either work or raise NotImplementedError
            try:
                backend = engine.backend_registry.get_backend("interactive")
                # If it succeeds, verify it's a valid backend instance
                assert hasattr(backend, "render_chart")
            except ValueError as e:
                # Should indicate it's not yet implemented
                assert "not yet implemented" in str(e)


@pytest.mark.integration
class TestVisualizationIntegration:
    """Integration tests for the complete visualization pipeline."""

    def test_end_to_end_data_collection_and_loading(self, sample_model: LoveModel, tmp_path: Path) -> None:
        """Test complete pipeline from model to visualization data."""
        # Step 1: Collect data
        collector = DataCollector()
        collector.set_metadata(test_integration=True)

        for step in range(5):
            sample_model.step()
            collector.collect_step_data(sample_model, step)

        # Step 2: Save data
        data_file = tmp_path / "integration_test.parquet"
        collector.save_run_data(data_file)

        # Step 3: Load and verify
        loader = DataLoader(data_file)

        assert len(loader.data) == 5
        assert loader.metadata["test_integration"] is True

        # Step 4: Create engine
        engine = VisualizationEngine(data_file)
        summary = engine.get_data_summary()

        assert summary["total_steps"] == 5
        assert summary["final_population"] > 0

    @pytest.mark.slow
    def test_performance_with_large_dataset(self, tmp_path: Path) -> None:
        """Test performance with larger dataset."""
        # Create larger test dataset
        n_steps = 1000
        np.random.seed(42)

        large_data = pl.DataFrame(
            {
                "step": range(n_steps),
                "population_size": np.random.randint(50, 200, n_steps),
                "mean_display": np.random.uniform(0, 65535, n_steps),
                "trait_preference_covariance": np.random.uniform(-1000, 1000, n_steps),
            }
        )

        data_file = tmp_path / "large_test.parquet"
        large_data.write_parquet(data_file)

        # Test loading performance
        import time

        start_time = time.time()

        loader = DataLoader(data_file)
        _ = loader.data  # Trigger loading
        summary = loader.get_summary_stats()

        load_time = time.time() - start_time

        # Performance assertion - should load 1000 steps quickly
        assert load_time < 1.0  # Less than 1 second
        assert summary["total_steps"] == n_steps
