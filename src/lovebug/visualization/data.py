"""
Data Collection and Storage System for LoveBug Visualizations

This module provides efficient data collection and storage for post-hoc analysis
of LoveBug agent-based model simulations. It integrates with Mesa-Frames and
uses Polars for high-performance data processing.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..unified_mesa_model import LoveModel


class DataCollector:
    """
    Collects essential metrics during LoveBug model execution for visualization.

    Integrates seamlessly with Mesa-Frames and captures key sexual selection
    metrics including population statistics, genetic variance, mating success,
    and cultural evolution dynamics.

    Example:
        collector = DataCollector()
        from lovebug.config import LayerBlendingParams
        config = LayerBlendingParams(blend_mode="weighted", blend_weight=1.0)
        model = LoveModel(layer_config=config, n_agents=1000)

        for step in range(100):
            model.step()
            collector.collect_step_data(model, step)

        collector.save_run_data('simulation_results.parquet')
    """

    def __init__(self):
        self.data_history: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

    def collect_step_data(self, model: LoveModel, step: int) -> dict[str, Any]:
        """
        Collect key metrics for one time step.

        Args:
            model: The LoveBug model instance
            step: Current simulation step number

        Returns:
            Dictionary containing all collected metrics for this step
        """
        if not hasattr(model, "agents") or len(model.agents) == 0:
            # Handle empty population case
            step_data = self._get_empty_step_data(step)
        else:
            agents_df = model.agents._agentsets[0].agents
            step_data = self._compute_step_metrics(agents_df, step)

        self.data_history.append(step_data)
        return step_data

    def _get_empty_step_data(self, step: int) -> dict[str, Any]:
        """Handle case where population is empty or extinct."""
        return {
            "step": step,
            "population_size": 0,
            "mean_display": np.nan,
            "mean_preference": np.nan,
            "mean_threshold": np.nan,
            "display_variance": np.nan,
            "preference_variance": np.nan,
            "threshold_variance": np.nan,
            "trait_preference_covariance": np.nan,
            "mating_success_rate": 0.0,
            "mean_age": np.nan,
            "mean_energy": np.nan,
            "cultural_genetic_distance": np.nan,
        }

    def _compute_step_metrics(self, agents_df: pl.DataFrame, step: int) -> dict[str, Any]:
        """Compute all metrics for a single step from agents DataFrame."""

        # Basic population statistics
        population_size = len(agents_df)

        # Extract genome components using bitwise operations
        genomes = agents_df["genome"].to_numpy().astype(np.uint32)

        # Display traits (bits 0-15)
        display_traits = genomes & 0x0000_FFFF
        mean_display = float(np.mean(display_traits))
        display_variance = float(np.var(display_traits))

        # Preferences (bits 16-23)
        preferences = (genomes & 0x00FF_0000) >> 16
        mean_preference = float(np.mean(preferences))
        preference_variance = float(np.var(preferences))

        # Thresholds (bits 24-31)
        thresholds = (genomes & 0xFF00_0000) >> 24
        mean_threshold = float(np.mean(thresholds))
        threshold_variance = float(np.var(thresholds))

        # Trait-preference genetic covariance (key metric for sexual selection)
        trait_preference_covariance = float(np.cov(display_traits, preferences)[0, 1])

        # Life history metrics
        ages = agents_df["age"].to_numpy()
        energies = agents_df["energy"].to_numpy()
        mean_age = float(np.mean(ages))
        mean_energy = float(np.mean(energies))

        # Cultural evolution metrics (if available)
        cultural_genetic_distance = self._compute_cultural_distance(agents_df)

        # Mating success (approximated by recent reproduction)
        # This is a placeholder - would need integration with courtship tracking
        mating_success_rate = self._estimate_mating_success(agents_df)

        return {
            "step": step,
            "population_size": population_size,
            "mean_display": mean_display,
            "mean_preference": mean_preference,
            "mean_threshold": mean_threshold,
            "display_variance": display_variance,
            "preference_variance": preference_variance,
            "threshold_variance": threshold_variance,
            "trait_preference_covariance": trait_preference_covariance,
            "mating_success_rate": mating_success_rate,
            "mean_age": mean_age,
            "mean_energy": mean_energy,
            "cultural_genetic_distance": cultural_genetic_distance,
        }

    def _compute_cultural_distance(self, agents_df: pl.DataFrame) -> float:
        """Compute mean Hamming distance between genetic and cultural preferences."""
        if "pref_culture" not in agents_df.columns:
            return np.nan

        genomes = agents_df["genome"].to_numpy().astype(np.uint32)
        genetic_prefs = (genomes & 0x00FF_0000) >> 16
        cultural_prefs = agents_df["pref_culture"].to_numpy().astype(np.uint32)

        # Hamming distance between 8-bit preferences
        xor_result = genetic_prefs ^ cultural_prefs
        hamming_distances = np.array([bin(x).count("1") for x in xor_result])

        return float(np.mean(hamming_distances))

    def _estimate_mating_success(self, agents_df: pl.DataFrame) -> float:
        """Estimate mating success rate based on age distribution."""
        # Placeholder implementation - ideally would track actual mating events
        ages = agents_df["age"].to_numpy()
        # Assume agents with age < 5 are recent offspring, indicating mating success
        recent_offspring = np.sum(ages < 5)
        total_adults = np.sum(ages >= 5)

        if total_adults == 0:
            return 0.0
        return float(recent_offspring / (total_adults + recent_offspring))

    def set_metadata(self, **kwargs) -> None:
        """Set metadata for the simulation run."""
        self.metadata.update(kwargs)

    def save_run_data(self, filepath: str | Path) -> None:
        """
        Save complete run data to Parquet format for efficient storage and loading.

        Args:
            filepath: Path to save the Parquet file
        """
        if not self.data_history:
            warnings.warn("No data collected yet. Nothing to save.", stacklevel=2)
            return

        # Convert to Polars DataFrame for efficient storage
        df = pl.DataFrame(self.data_history)

        # Ensure output directory exists
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save data with metadata as custom properties
        df.write_parquet(filepath)

        # Save metadata separately (Parquet metadata support is limited)
        if self.metadata:
            metadata_path = filepath.with_suffix(".metadata.json")
            import json

            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

        print(f"Saved simulation data: {len(df)} steps to {filepath}")

    def clear(self) -> None:
        """Clear collected data and metadata."""
        self.data_history.clear()
        self.metadata.clear()


class DataLoader:
    """
    Efficient loading and preprocessing of saved simulation data.

    Provides lazy loading, filtering, and preprocessing capabilities
    for post-hoc visualization analysis.
    """

    def __init__(self, filepath: str | Path):
        """
        Initialize with path to saved simulation data.

        Args:
            filepath: Path to Parquet file with simulation data
        """
        self.filepath = Path(filepath)
        self._data: pl.DataFrame | None = None
        self._metadata: dict[str, Any] | None = None

        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

    @property
    def data(self) -> pl.DataFrame:
        """Lazy-loaded simulation data."""
        if self._data is None:
            self._data = pl.read_parquet(self.filepath)
        return self._data

    @property
    def metadata(self) -> dict[str, Any]:
        """Lazy-loaded metadata."""
        if self._metadata is None:
            metadata_path = self.filepath.with_suffix(".metadata.json")
            if metadata_path.exists():
                import json

                with open(metadata_path) as f:
                    self._metadata = json.load(f)
            else:
                self._metadata = {}
        return self._metadata if self._metadata is not None else {}

    def get_time_range(self, start_step: int = 0, end_step: int | None = None) -> pl.DataFrame:
        """
        Get data for a specific time range.

        Args:
            start_step: Starting step (inclusive)
            end_step: Ending step (inclusive), None for end of data

        Returns:
            Filtered DataFrame for the specified time range
        """
        df = self.data

        if end_step is None:
            return df.filter(pl.col("step") >= start_step)
        else:
            return df.filter((pl.col("step") >= start_step) & (pl.col("step") <= end_step))

    def get_final_state(self) -> pl.DataFrame:
        """Get the final time step data."""
        return self.data.filter(pl.col("step") == pl.col("step").max())

    def get_summary_stats(self) -> dict[str, Any]:
        """Compute summary statistics across the entire run."""
        df = self.data

        return {
            "total_steps": len(df),
            "final_population": df.select(pl.col("population_size").last()).item(),
            "max_population": df.select(pl.col("population_size").max()).item(),
            "min_population": df.select(pl.col("population_size").min()).item(),
            "mean_covariance": df.select(pl.col("trait_preference_covariance").mean()).item(),
            "final_covariance": df.select(pl.col("trait_preference_covariance").last()).item(),
            "extinction_step": self._find_extinction_step(df),
        }

    def _find_extinction_step(self, df: pl.DataFrame) -> int | None:
        """Find the step where population went extinct, if any."""
        extinct_steps = df.filter(pl.col("population_size") == 0)
        if len(extinct_steps) > 0:
            return extinct_steps.select(pl.col("step").min()).item()
        return None

    def get_trajectory_data(self, metrics: list[str]) -> pl.DataFrame:
        """
        Get trajectory data for specific metrics.

        Args:
            metrics: List of metric names to extract

        Returns:
            DataFrame with step and requested metrics
        """
        available_metrics = set(self.data.columns)
        requested_metrics = set(metrics)

        missing_metrics = requested_metrics - available_metrics
        if missing_metrics:
            raise ValueError(f"Metrics not found in data: {missing_metrics}")

        return self.data.select(["step"] + metrics)
