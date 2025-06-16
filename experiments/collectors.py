#!/usr/bin/env python3
"""
Type-Safe Result Collectors for Experiment Data

This module provides collectors that manage experiment results with proper
type safety and efficient storage, eliminating null pollution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import polars as pl
from beartype import beartype

from experiments.models import (
    CulturalExperimentResult,
    ExperimentMetadata,
    GeneticExperimentResult,
    IntegratedExperimentResult,
)

__all__ = [
    "ResultCollector",
    "GeneticResultCollector",
    "CulturalResultCollector",
    "IntegratedResultCollector",
    "ExperimentStorage",
]

logger = logging.getLogger(__name__)


class ResultCollector(Protocol):
    """Interface for experiment result collectors.

    All result collectors must implement these methods for consistent
    data handling across different experiment types.
    """

    def add_result(self, result: Any) -> None:
        """Add a single experiment result."""
        ...

    def to_dataframe(self) -> pl.DataFrame:
        """Convert collected results to a clean Polars DataFrame."""
        ...

    def save_to_file(self, path: Path) -> None:
        """Save results to a Parquet file."""
        ...

    def get_summary_stats(self) -> dict[str, Any]:
        """Generate summary statistics for collected results."""
        ...

    def clear(self) -> None:
        """Clear all collected results."""
        ...


class GeneticResultCollector:
    """Collects and manages genetic experiment results.

    Provides type-safe collection of genetic evolution experiment results
    with efficient conversion to analysis-ready DataFrames.

    Examples
    --------
    >>> collector = GeneticResultCollector()
    >>> # Add results from experiments
    >>> collector.add_result(genetic_result)
    >>> df = collector.to_dataframe()
    >>> len(df) > 0
    True
    """

    def __init__(self) -> None:
        self.results: list[GeneticExperimentResult] = []
        logger.info("Initialized genetic result collector")

    @beartype
    def add_result(self, result: GeneticExperimentResult) -> None:
        """Add a genetic experiment result.

        Parameters
        ----------
        result : GeneticExperimentResult
            Validated genetic experiment result
        """
        self.results.append(result)
        logger.debug(f"Added genetic result: {result.metadata.experiment_id}")

    def to_dataframe(self) -> pl.DataFrame:
        """Convert genetic results to clean Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            Clean DataFrame with no null values, optimized for analysis

        Raises
        ------
        ValueError
            If no results have been collected
        """
        if not self.results:
            raise ValueError("No genetic results to convert")

        # Flatten nested dataclasses into flat dictionary structure
        rows = []
        for result in self.results:
            row = {
                # Metadata fields
                "experiment_id": result.metadata.experiment_id,
                "name": result.metadata.name,
                "experiment_type": result.metadata.experiment_type,
                "start_time": result.metadata.start_time,
                "duration_seconds": result.metadata.duration_seconds,
                "success": result.metadata.success,
                "process_id": result.metadata.process_id,
                # Common parameters
                "n_generations": result.common_params.n_generations,
                "population_size": result.common_params.population_size,
                "random_seed": result.common_params.random_seed,
                # Genetic results
                "final_trait": result.final_trait,
                "final_preference": result.final_preference,
                "final_covariance": result.final_covariance,
                "outcome": result.outcome,
                "generations_completed": result.generations_completed,
                # Genetic parameters
                "h2_trait": result.h2_trait,
                "h2_preference": result.h2_preference,
                "genetic_correlation": result.genetic_correlation,
                "selection_strength": result.selection_strength,
                "preference_cost": result.preference_cost,
                "mutation_variance": result.mutation_variance,
            }
            rows.append(row)

        df = pl.DataFrame(rows)
        logger.info(f"Converted {len(self.results)} genetic results to DataFrame")
        return df

    @beartype
    def save_to_file(self, path: Path) -> None:
        """Save genetic results to Parquet file.

        Parameters
        ----------
        path : Path
            Output file path (will create parent directories)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.write_parquet(path)
        logger.info(f"Saved {len(self.results)} genetic results to {path}")

    def get_summary_stats(self) -> dict[str, Any]:
        """Generate summary statistics for genetic results.

        Returns
        -------
        dict[str, Any]
            Summary statistics including outcome distribution and parameter ranges
        """
        if not self.results:
            return {"total_experiments": 0}

        df = self.to_dataframe()

        return {
            "total_experiments": len(self.results),
            "success_rate": df["success"].mean(),
            "avg_duration_seconds": df["duration_seconds"].mean(),
            "outcome_distribution": df["outcome"].value_counts().to_dict(),
            "parameter_ranges": {
                "final_covariance": {
                    "min": df["final_covariance"].min(),
                    "max": df["final_covariance"].max(),
                    "mean": df["final_covariance"].mean(),
                },
                "population_size": {
                    "min": df["population_size"].min(),
                    "max": df["population_size"].max(),
                },
            },
        }

    def clear(self) -> None:
        """Clear all collected genetic results."""
        count = len(self.results)
        self.results.clear()
        logger.info(f"Cleared {count} genetic results")


class CulturalResultCollector:
    """Collects and manages cultural experiment results.

    Provides type-safe collection of cultural evolution experiment results
    with efficient conversion to analysis-ready DataFrames.

    Examples
    --------
    >>> collector = CulturalResultCollector()
    >>> # Add results from experiments
    >>> collector.add_result(cultural_result)
    >>> df = collector.to_dataframe()
    >>> len(df) > 0
    True
    """

    def __init__(self) -> None:
        self.results: list[CulturalExperimentResult] = []
        logger.info("Initialized cultural result collector")

    @beartype
    def add_result(self, result: CulturalExperimentResult) -> None:
        """Add a cultural experiment result.

        Parameters
        ----------
        result : CulturalExperimentResult
            Validated cultural experiment result
        """
        self.results.append(result)
        logger.debug(f"Added cultural result: {result.metadata.experiment_id}")

    def to_dataframe(self) -> pl.DataFrame:
        """Convert cultural results to clean Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            Clean DataFrame with no null values, optimized for analysis

        Raises
        ------
        ValueError
            If no results have been collected
        """
        if not self.results:
            raise ValueError("No cultural results to convert")

        # Flatten nested dataclasses into flat dictionary structure
        rows = []
        for result in self.results:
            row = {
                # Metadata fields
                "experiment_id": result.metadata.experiment_id,
                "name": result.metadata.name,
                "experiment_type": result.metadata.experiment_type,
                "start_time": result.metadata.start_time,
                "duration_seconds": result.metadata.duration_seconds,
                "success": result.metadata.success,
                "process_id": result.metadata.process_id,
                # Common parameters
                "n_generations": result.common_params.n_generations,
                "population_size": result.common_params.population_size,
                "random_seed": result.common_params.random_seed,
                # Cultural results
                "final_diversity": result.final_diversity,
                "diversity_trend": result.diversity_trend,
                "total_events": result.total_events,
                "cultural_outcome": result.cultural_outcome,
                "generations_completed": result.generations_completed,
                # Cultural parameters
                "innovation_rate": result.innovation_rate,
                "horizontal_transmission_rate": result.horizontal_transmission_rate,
                "oblique_transmission_rate": result.oblique_transmission_rate,
                "network_type": result.network_type,
                "network_connectivity": result.network_connectivity,
                "cultural_memory_size": result.cultural_memory_size,
            }
            rows.append(row)

        df = pl.DataFrame(rows)
        logger.info(f"Converted {len(self.results)} cultural results to DataFrame")
        return df

    @beartype
    def save_to_file(self, path: Path) -> None:
        """Save cultural results to Parquet file.

        Parameters
        ----------
        path : Path
            Output file path (will create parent directories)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.write_parquet(path)
        logger.info(f"Saved {len(self.results)} cultural results to {path}")

    def get_summary_stats(self) -> dict[str, Any]:
        """Generate summary statistics for cultural results.

        Returns
        -------
        dict[str, Any]
            Summary statistics including outcome distribution and diversity metrics
        """
        if not self.results:
            return {"total_experiments": 0}

        df = self.to_dataframe()

        return {
            "total_experiments": len(self.results),
            "success_rate": df["success"].mean(),
            "avg_duration_seconds": df["duration_seconds"].mean(),
            "cultural_outcome_distribution": df["cultural_outcome"].value_counts().to_dict(),
            "diversity_metrics": {
                "final_diversity": {
                    "min": df["final_diversity"].min(),
                    "max": df["final_diversity"].max(),
                    "mean": df["final_diversity"].mean(),
                },
                "avg_total_events": df["total_events"].mean(),
            },
            "parameter_ranges": {
                "population_size": {
                    "min": df["population_size"].min(),
                    "max": df["population_size"].max(),
                },
                "innovation_rate": {
                    "min": df["innovation_rate"].min(),
                    "max": df["innovation_rate"].max(),
                },
            },
        }

    def clear(self) -> None:
        """Clear all collected cultural results."""
        count = len(self.results)
        self.results.clear()
        logger.info(f"Cleared {count} cultural results")


class IntegratedResultCollector:
    """Collects and manages integrated experiment results.

    Placeholder for future integrated genetic+cultural experiments.

    Examples
    --------
    >>> collector = IntegratedResultCollector()
    >>> # Future: collector.add_result(integrated_result)
    """

    def __init__(self) -> None:
        self.results: list[IntegratedExperimentResult] = []
        logger.info("Initialized integrated result collector (placeholder)")

    @beartype
    def add_result(self, result: IntegratedExperimentResult) -> None:
        """Add an integrated experiment result (future implementation)."""
        self.results.append(result)
        logger.debug(f"Added integrated result: {result.metadata.experiment_id}")

    def to_dataframe(self) -> pl.DataFrame:
        """Convert integrated results to clean Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            Clean DataFrame with no null values, optimized for analysis

        Raises
        ------
        ValueError
            If no results have been collected
        """
        if not self.results:
            raise ValueError("No integrated results to convert")

        # Flatten nested dataclasses into flat dictionary structure
        rows = []
        for result in self.results:
            row = {
                # Metadata fields
                "experiment_id": result.metadata.experiment_id,
                "name": result.metadata.name,
                "experiment_type": result.metadata.experiment_type,
                "start_time": result.metadata.start_time,
                "duration_seconds": result.metadata.duration_seconds,
                "success": result.metadata.success,
                "process_id": result.metadata.process_id,
                # Common parameters
                "n_generations": result.common_params.n_generations,
                "population_size": result.common_params.population_size,
                "random_seed": result.common_params.random_seed,
                # Integration-specific results
                "gene_culture_correlation": result.gene_culture_correlation,
                "interaction_strength": result.interaction_strength,
                # Genetic component results
                "genetic_final_trait": result.genetic_component.final_trait,
                "genetic_final_preference": result.genetic_component.final_preference,
                "genetic_final_covariance": result.genetic_component.final_covariance,
                "genetic_outcome": result.genetic_component.outcome,
                "genetic_h2_trait": result.genetic_component.h2_trait,
                "genetic_h2_preference": result.genetic_component.h2_preference,
                "genetic_correlation": result.genetic_component.genetic_correlation,
                "selection_strength": result.genetic_component.selection_strength,
                "preference_cost": result.genetic_component.preference_cost,
                "mutation_variance": result.genetic_component.mutation_variance,
                # Cultural component results
                "cultural_final_diversity": result.cultural_component.final_diversity,
                "cultural_diversity_trend": result.cultural_component.diversity_trend,
                "cultural_total_events": result.cultural_component.total_events,
                "cultural_outcome": result.cultural_component.cultural_outcome,
                "innovation_rate": result.cultural_component.innovation_rate,
                "horizontal_transmission_rate": result.cultural_component.horizontal_transmission_rate,
                "oblique_transmission_rate": result.cultural_component.oblique_transmission_rate,
                "network_type": result.cultural_component.network_type,
                "network_connectivity": result.cultural_component.network_connectivity,
                "cultural_memory_size": result.cultural_component.cultural_memory_size,
                # Emergent properties
                **{f"emergent_{k}": v for k, v in result.emergent_properties.items()},
            }
            rows.append(row)

        df = pl.DataFrame(rows)
        logger.info(f"Converted {len(self.results)} integrated results to DataFrame")
        return df

    def save_to_file(self, path: Path) -> None:
        """Save integrated results to Parquet file.

        Parameters
        ----------
        path : Path
            Output file path (will create parent directories)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.write_parquet(path)
        logger.info(f"Saved {len(self.results)} integrated results to {path}")

    def get_summary_stats(self) -> dict[str, Any]:
        """Generate summary statistics for integrated results.

        Returns
        -------
        dict[str, Any]
            Summary statistics including interaction metrics and component summaries
        """
        if not self.results:
            return {"total_experiments": 0}

        df = self.to_dataframe()

        return {
            "total_experiments": len(self.results),
            "success_rate": df["success"].mean(),
            "avg_duration_seconds": df["duration_seconds"].mean(),
            "interaction_metrics": {
                "mean_gene_culture_correlation": df["gene_culture_correlation"].mean(),
                "mean_interaction_strength": df["interaction_strength"].mean(),
                "correlation_range": {
                    "min": df["gene_culture_correlation"].min(),
                    "max": df["gene_culture_correlation"].max(),
                },
            },
            "genetic_component_summary": {
                "outcome_distribution": df["genetic_outcome"].value_counts().to_dict(),
                "mean_final_covariance": df["genetic_final_covariance"].mean(),
            },
            "cultural_component_summary": {
                "outcome_distribution": df["cultural_outcome"].value_counts().to_dict(),
                "mean_final_diversity": df["cultural_final_diversity"].mean(),
                "total_cultural_events": df["cultural_total_events"].sum(),
            },
            "emergent_properties": {
                col.replace("emergent_", ""): df[col].mean()
                for col in df.columns
                if col.startswith("emergent_") and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            },
            "parameter_ranges": {
                "population_size": {
                    "min": df["population_size"].min(),
                    "max": df["population_size"].max(),
                },
                "n_generations": {
                    "min": df["n_generations"].min(),
                    "max": df["n_generations"].max(),
                },
            },
        }

    def clear(self) -> None:
        """Clear integrated results."""
        count = len(self.results)
        self.results.clear()
        logger.info(f"Cleared {count} integrated results")


class ExperimentStorage:
    """Handles type-safe storage of all experiment results.

    Manages separate collectors for each experiment type and provides
    unified interface for saving results to appropriate clean files.

    Parameters
    ----------
    base_path : Path
        Base directory for experiment results

    Examples
    --------
    >>> storage = ExperimentStorage(Path("experiments/results"))
    >>> storage.store_genetic_result(genetic_result)
    >>> storage.store_cultural_result(cultural_result)
    >>> storage.save_all("20231216_143022")
    """

    @beartype
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.genetic_collector = GeneticResultCollector()
        self.cultural_collector = CulturalResultCollector()
        self.integrated_collector = IntegratedResultCollector()

        # Initialize directory structure
        self._initialize_directories()
        logger.info(f"Initialized experiment storage at {base_path}")

    def _initialize_directories(self) -> None:
        """Create clean directory structure."""
        directories = ["genetic", "cultural", "integrated", "run_logs"]
        for dir_name in directories:
            (self.base_path / dir_name).mkdir(parents=True, exist_ok=True)
        logger.debug("Initialized clean directory structure")

    @beartype
    def store_genetic_result(self, result: GeneticExperimentResult) -> None:
        """Store a genetic experiment result.

        Parameters
        ----------
        result : GeneticExperimentResult
            Validated genetic experiment result
        """
        self.genetic_collector.add_result(result)

    @beartype
    def store_cultural_result(self, result: CulturalExperimentResult) -> None:
        """Store a cultural experiment result.

        Parameters
        ----------
        result : CulturalExperimentResult
            Validated cultural experiment result
        """
        self.cultural_collector.add_result(result)

    @beartype
    def store_integrated_result(self, result: IntegratedExperimentResult) -> None:
        """Store an integrated experiment result (future implementation).

        Parameters
        ----------
        result : IntegratedExperimentResult
            Validated integrated experiment result
        """
        self.integrated_collector.add_result(result)

    @beartype
    def save_all(self, timestamp: str) -> dict[str, Path]:
        """Save all collected results to clean separate files.

        Parameters
        ----------
        timestamp : str
            Timestamp for file naming (e.g., "20231216_143022")

        Returns
        -------
        dict[str, Path]
            Mapping of experiment type to saved file path
        """
        saved_files = {}

        # Save genetic results if any
        if self.genetic_collector.results:
            genetic_path = self.base_path / "genetic" / f"experiments_{timestamp}.parquet"
            self.genetic_collector.save_to_file(genetic_path)
            saved_files["genetic"] = genetic_path

            # Save metadata
            metadata_path = genetic_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(self.genetic_collector.get_summary_stats(), f, indent=2, default=str)

        # Save cultural results if any
        if self.cultural_collector.results:
            cultural_path = self.base_path / "cultural" / f"experiments_{timestamp}.parquet"
            self.cultural_collector.save_to_file(cultural_path)
            saved_files["cultural"] = cultural_path

            # Save metadata
            metadata_path = cultural_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(self.cultural_collector.get_summary_stats(), f, indent=2, default=str)

        # Save integrated results if any (future)
        if self.integrated_collector.results:
            integrated_path = self.base_path / "integrated" / f"experiments_{timestamp}.parquet"
            self.integrated_collector.save_to_file(integrated_path)
            saved_files["integrated"] = integrated_path

        # Save run log
        run_log = {
            "timestamp": timestamp,
            "genetic_experiments": len(self.genetic_collector.results),
            "cultural_experiments": len(self.cultural_collector.results),
            "integrated_experiments": len(self.integrated_collector.results),
            "saved_files": {k: str(v) for k, v in saved_files.items()},
        }

        run_log_path = self.base_path / "run_logs" / f"run_{timestamp}.json"
        with open(run_log_path, "w") as f:
            json.dump(run_log, f, indent=2, default=str)

        logger.info(f"Saved all results to {len(saved_files)} clean files")
        return saved_files

    def get_total_results(self) -> dict[str, int]:
        """Get count of collected results by type.

        Returns
        -------
        dict[str, int]
            Count of results for each experiment type
        """
        return {
            "genetic": len(self.genetic_collector.results),
            "cultural": len(self.cultural_collector.results),
            "integrated": len(self.integrated_collector.results),
        }

    def clear_all(self) -> None:
        """Clear all collected results."""
        self.genetic_collector.clear()
        self.cultural_collector.clear()
        self.integrated_collector.clear()
        logger.info("Cleared all collected results")


if __name__ == "__main__":
    # Simple validation that collectors work
    from datetime import datetime

    from experiments.models import CommonParameters, ExperimentMetadata, GeneticExperimentResult

    # Create test data
    metadata = ExperimentMetadata(
        experiment_id="test_001",
        name="validation_test",
        experiment_type="genetic",
        start_time=datetime.now(),
        duration_seconds=0.1,
        success=True,
        process_id=12345,
    )

    params = CommonParameters(n_generations=100, population_size=1000, random_seed=42)

    genetic_result = GeneticExperimentResult(
        metadata=metadata,
        common_params=params,
        final_trait=0.1,
        final_preference=0.05,
        final_covariance=0.02,
        outcome="equilibrium",
        generations_completed=100,
        h2_trait=0.5,
        h2_preference=0.5,
        genetic_correlation=0.2,
        selection_strength=0.1,
        preference_cost=0.05,
        mutation_variance=0.01,
    )

    # Test collector
    collector = GeneticResultCollector()
    collector.add_result(genetic_result)

    logger.info("✅ Collectors validated")
    print("✅ Clean result collectors created successfully!")
    print(f"   Collected: {len(collector.results)} genetic results")

    # Test DataFrame conversion
    df = collector.to_dataframe()
    print(f"   DataFrame: {len(df)} rows, {len(df.columns)} columns")
    print(f"   No nulls: {df.null_count().sum_horizontal().item() == 0}")
