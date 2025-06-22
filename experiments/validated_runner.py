#!/usr/bin/env python3
"""
Validated experiment runner using Pydantic models.

This module replaces the fragile dictionary-based experiment runner with
a type-safe version that uses Pydantic models for validation and clear
interfaces for model initialization.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from pydantic import ValidationError

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config_models import (
    CombinedExperimentConfig,
    CulturalExperimentConfig,
    GeneticExperimentConfig,
)
from experiments.model_factory import create_love_model_from_config
from experiments.models import (
    CommonParameters,
    CulturalExperimentResult,
    ExperimentMetadata,
    GeneticExperimentResult,
    IntegratedExperimentResult,
)

__all__ = ["run_validated_experiment", "ValidatedExperimentRunner"]

logger = logging.getLogger(__name__)


@beartype
def run_validated_genetic_experiment(config: GeneticExperimentConfig) -> GeneticExperimentResult:
    """
    Execute a genetic evolution experiment with validated configuration.

    Parameters
    ----------
    config : GeneticExperimentConfig
        Validated experiment configuration

    Returns
    -------
    GeneticExperimentResult
        Type-safe genetic experiment result

    Raises
    ------
    ValueError
        If model initialization or execution fails
    """
    experiment_start = time.time()
    start_time = datetime.now()

    try:
        logger.info(f"Starting validated genetic experiment: {config.name}")
        logger.debug(f"Configuration: {config.model_dump_json(indent=2)}")

        # Create model using validated factory
        model = create_love_model_from_config(config)

        # Run simulation
        model_results = model.run(n_steps=config.n_generations)

        # Extract results
        trajectory = model_results.get("trajectory", [])
        final_metrics = model_results.get("final_metrics", {})

        # Calculate final values
        if trajectory:
            final_metrics_trajectory = trajectory[-1]
            final_trait = final_metrics_trajectory.get("mean_genetic_trait", 0.0)
            final_preference = final_metrics_trajectory.get("mean_genetic_preference", 128.0)
            final_covariance = final_metrics.get("genetic_covariance", config.genetic_correlation)
        else:
            final_trait = 0.0
            final_preference = 128.0
            final_covariance = config.genetic_correlation

        # Classify outcome using configuration's computed field
        scenario_type = config.scenario_type
        if scenario_type in ["runaway", "stasis", "equilibrium"]:
            outcome = scenario_type  # type: ignore[assignment]
        else:
            outcome = "equilibrium"  # Default fallback

        # Create metadata
        experiment_id = str(uuid.uuid4())[:8]
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=config.name,
            experiment_type="genetic",
            start_time=start_time,
            duration_seconds=time.time() - experiment_start,
            success=True,
            process_id=0,  # Will be set by process manager if needed
        )

        common_params = CommonParameters(
            n_generations=config.n_generations,
            population_size=config.population_size,
            random_seed=config.random_seed,
        )

        result = GeneticExperimentResult(
            metadata=metadata,
            common_params=common_params,
            final_trait=final_trait,
            final_preference=final_preference,
            final_covariance=final_covariance,
            outcome=outcome,
            generations_completed=len(trajectory),
            h2_trait=config.h2_trait,
            h2_preference=config.h2_preference,
            genetic_correlation=config.genetic_correlation,
            selection_strength=config.selection_strength,
            preference_cost=config.preference_cost,
            mutation_variance=config.mutation_variance,
        )

        logger.info(
            f"Genetic experiment completed: {config.name} → {outcome} (final population: {final_metrics.get('population_size', 'unknown')})"
        )
        return result

    except Exception as e:
        logger.exception(f"Genetic experiment failed: {config.name}")
        raise ValueError(f"Genetic experiment failed: {e}") from e


@beartype
def run_validated_cultural_experiment(config: CulturalExperimentConfig) -> CulturalExperimentResult:
    """
    Execute a cultural evolution experiment with validated configuration.

    Parameters
    ----------
    config : CulturalExperimentConfig
        Validated experiment configuration

    Returns
    -------
    CulturalExperimentResult
        Type-safe cultural experiment result

    Raises
    ------
    ValueError
        If model initialization or execution fails
    """
    experiment_start = time.time()
    start_time = datetime.now()

    try:
        logger.info(f"Starting validated cultural experiment: {config.name}")
        logger.debug(f"Configuration: {config.model_dump_json(indent=2)}")

        # Create model using validated factory
        model = create_love_model_from_config(config)

        # Run simulation
        model_results = model.run(n_steps=config.n_generations)

        # Extract results
        trajectory = model_results.get("trajectory", [])
        final_metrics = model_results.get("final_metrics", {})

        # Calculate cultural metrics
        final_diversity = final_metrics.get("cultural_diversity", 0.5)
        total_events = sum(h.get("cultural_learning_events", 0) for h in trajectory)

        # Calculate diversity trend
        diversity_samples = [h.get("cultural_diversity", 0.5) for h in trajectory[::10]]  # Sample every 10 steps
        diversity_trend = 0.0
        if len(diversity_samples) > 1:
            diversity_trend = np.polyfit(range(len(diversity_samples)), diversity_samples, 1)[0]

        # Classify cultural outcome
        if final_diversity > 0.7:
            cultural_outcome = "high_diversity"
        elif final_diversity < 0.2:
            cultural_outcome = "low_diversity"
        else:
            cultural_outcome = "moderate_diversity"

        # Create metadata
        experiment_id = str(uuid.uuid4())[:8]
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=config.name,
            experiment_type="cultural",
            start_time=start_time,
            duration_seconds=time.time() - experiment_start,
            success=True,
            process_id=0,
        )

        common_params = CommonParameters(
            n_generations=config.n_generations,
            population_size=config.population_size,
            random_seed=config.random_seed,
        )

        result = CulturalExperimentResult(
            metadata=metadata,
            common_params=common_params,
            final_diversity=final_diversity,
            diversity_trend=diversity_trend,
            total_events=total_events,
            cultural_outcome=cultural_outcome,
            generations_completed=len(trajectory),
            innovation_rate=config.innovation_rate,
            horizontal_transmission_rate=config.horizontal_transmission_rate,
            oblique_transmission_rate=config.oblique_transmission_rate,
            network_type=config.network_type,
            network_connectivity=config.network_connectivity,
            cultural_memory_size=config.cultural_memory_size,
        )

        logger.info(
            f"Cultural experiment completed: {config.name} → {cultural_outcome} (diversity: {final_diversity:.3f})"
        )
        return result

    except Exception as e:
        logger.exception(f"Cultural experiment failed: {config.name}")
        raise ValueError(f"Cultural experiment failed: {e}") from e


@beartype
def run_validated_combined_experiment(config: CombinedExperimentConfig) -> IntegratedExperimentResult:
    """
    Execute a combined genetic+cultural experiment with validated configuration.

    Parameters
    ----------
    config : CombinedExperimentConfig
        Validated experiment configuration

    Returns
    -------
    IntegratedExperimentResult
        Type-safe integrated experiment result

    Raises
    ------
    ValueError
        If model initialization or execution fails
    """
    experiment_start = time.time()
    start_time = datetime.now()

    try:
        logger.info(f"Starting validated combined experiment: {config.name}")
        logger.debug(f"Configuration: {config.model_dump_json(indent=2)}")

        # Create model using validated factory
        model = create_love_model_from_config(config)

        # Run simulation
        model_results = model.run(n_steps=config.n_generations)

        # Extract results
        trajectory = model_results.get("trajectory", [])
        final_metrics = model_results.get("final_metrics", {})

        # Create genetic component result
        genetic_component = GeneticExperimentResult(
            metadata=ExperimentMetadata(
                experiment_id=f"genetic_{str(uuid.uuid4())[:8]}",
                name=f"genetic_component_{config.name}",
                experiment_type="genetic",
                start_time=start_time,
                duration_seconds=time.time() - experiment_start,
                success=True,
                process_id=0,
            ),
            common_params=CommonParameters(
                n_generations=config.n_generations,
                population_size=config.population_size,
                random_seed=config.random_seed,
            ),
            final_trait=final_metrics.get("mean_genetic_trait", 0.0),
            final_preference=final_metrics.get("mean_genetic_preference", 128.0),
            final_covariance=final_metrics.get("genetic_covariance", 0.0),
            outcome="equilibrium",  # Could be determined from trajectory analysis
            generations_completed=len(trajectory),
            h2_trait=config.h2_trait,
            h2_preference=config.h2_preference,
            genetic_correlation=config.genetic_correlation,
            selection_strength=config.selection_strength,
            preference_cost=config.preference_cost,
            mutation_variance=config.mutation_variance,
        )

        # Create cultural component result
        final_diversity = final_metrics.get("cultural_diversity", 0.5)
        total_events = sum(h.get("cultural_learning_events", 0) for h in trajectory)

        cultural_outcome = "moderate_diversity"
        if final_diversity > 0.7:
            cultural_outcome = "high_diversity"
        elif final_diversity < 0.2:
            cultural_outcome = "low_diversity"

        cultural_component = CulturalExperimentResult(
            metadata=ExperimentMetadata(
                experiment_id=f"cultural_{str(uuid.uuid4())[:8]}",
                name=f"cultural_component_{config.name}",
                experiment_type="cultural",
                start_time=start_time,
                duration_seconds=time.time() - experiment_start,
                success=True,
                process_id=0,
            ),
            common_params=CommonParameters(
                n_generations=config.n_generations,
                population_size=config.population_size,
                random_seed=config.random_seed,
            ),
            final_diversity=final_diversity,
            diversity_trend=0.0,  # Could be calculated from trajectory
            total_events=total_events,
            cultural_outcome=cultural_outcome,
            generations_completed=len(trajectory),
            innovation_rate=config.innovation_rate,
            horizontal_transmission_rate=config.horizontal_transmission_rate,
            oblique_transmission_rate=config.oblique_transmission_rate,
            network_type=config.network_type,
            network_connectivity=config.network_connectivity,
            cultural_memory_size=config.cultural_memory_size,
        )

        # Calculate interaction metrics
        gene_culture_correlation = final_metrics.get("gene_culture_correlation", 0.0)
        interaction_strength = config.normalized_genetic_weight * config.normalized_cultural_weight

        # Collect emergent properties
        emergent_properties = {
            "effective_preference_variance": final_metrics.get("var_effective_preference", 0.0),
            "gene_culture_distance": final_metrics.get("gene_culture_distance", 0.0),
            "population_stability": 1.0
            if final_metrics.get("population_size", 0) > config.population_size * 0.1
            else 0.0,
            "blending_efficiency": interaction_strength,
            "perceptual_constraint_effect": config.theta_detect / 16.0,
        }

        # Create integrated result
        experiment_id = str(uuid.uuid4())[:8]
        integrated_result = IntegratedExperimentResult(
            metadata=ExperimentMetadata(
                experiment_id=experiment_id,
                name=config.name,
                experiment_type="integrated",
                start_time=start_time,
                duration_seconds=time.time() - experiment_start,
                success=True,
                process_id=0,
            ),
            common_params=CommonParameters(
                n_generations=config.n_generations,
                population_size=config.population_size,
                random_seed=config.random_seed,
            ),
            genetic_component=genetic_component,
            cultural_component=cultural_component,
            gene_culture_correlation=gene_culture_correlation,
            interaction_strength=interaction_strength,
            emergent_properties=emergent_properties,
        )

        logger.info(
            f"Combined experiment completed: {config.name} (final population: {final_metrics.get('population_size', 'unknown')})"
        )
        return integrated_result

    except Exception as e:
        logger.exception(f"Combined experiment failed: {config.name}")
        raise ValueError(f"Combined experiment failed: {e}") from e


@beartype
def run_validated_experiment(
    config: GeneticExperimentConfig | CulturalExperimentConfig | CombinedExperimentConfig,
) -> GeneticExperimentResult | CulturalExperimentResult | IntegratedExperimentResult:
    """
    Execute any type of experiment with validated configuration.

    This is the main entry point for the validated experiment system.
    It dispatches to the appropriate experiment function based on the
    configuration type.

    Parameters
    ----------
    config : GeneticExperimentConfig | CulturalExperimentConfig | CombinedExperimentConfig
        Validated experiment configuration

    Returns
    -------
    GeneticExperimentResult | CulturalExperimentResult | IntegratedExperimentResult
        Type-safe experiment result

    Raises
    ------
    ValueError
        If configuration type is not recognized or experiment fails
    """
    if isinstance(config, GeneticExperimentConfig):
        return run_validated_genetic_experiment(config)
    elif isinstance(config, CulturalExperimentConfig):
        return run_validated_cultural_experiment(config)
    elif isinstance(config, CombinedExperimentConfig):
        return run_validated_combined_experiment(config)
    else:
        raise ValueError(f"Unknown experiment configuration type: {type(config)}")


class ValidatedExperimentRunner:
    """
    Experiment runner that uses validated Pydantic configurations.

    This class replaces the fragile dictionary-based experiment runner
    with a type-safe version that validates all parameters before
    model initialization.
    """

    def __init__(self, validate_on_init: bool = True):
        """
        Initialize the validated experiment runner.

        Parameters
        ----------
        validate_on_init : bool
            Whether to validate configurations immediately when added
        """
        self.validate_on_init = validate_on_init
        self.experiments_run = 0
        self.experiments_failed = 0

        logger.info("Initialized ValidatedExperimentRunner with Pydantic validation")

    @beartype
    def run_from_dict(
        self,
        experiment_type: str,
        parameters: dict[str, Any],
    ) -> GeneticExperimentResult | CulturalExperimentResult | IntegratedExperimentResult:
        """
        Run experiment from dictionary parameters (with validation).

        This method provides backward compatibility with the dictionary-based
        interface while adding Pydantic validation.

        Parameters
        ----------
        experiment_type : str
            Type of experiment ("genetic", "cultural", or "combined")
        parameters : dict[str, Any]
            Raw experiment parameters

        Returns
        -------
        GeneticExperimentResult | CulturalExperimentResult | IntegratedExperimentResult
            Type-safe experiment result

        Raises
        ------
        ValidationError
            If parameters are invalid
        ValueError
            If experiment type is unknown or execution fails
        """
        try:
            # Validate parameters using appropriate Pydantic model
            if experiment_type == "genetic":
                config = GeneticExperimentConfig(**parameters)
            elif experiment_type == "cultural":
                config = CulturalExperimentConfig(**parameters)
            elif experiment_type == "combined":
                config = CombinedExperimentConfig(**parameters)
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")

            # Run validated experiment
            result = run_validated_experiment(config)
            self.experiments_run += 1
            return result

        except ValidationError as e:
            logger.error(f"Parameter validation failed for {experiment_type} experiment: {e}")
            self.experiments_failed += 1
            raise
        except Exception:
            logger.exception(f"Experiment execution failed: {experiment_type}")
            self.experiments_failed += 1
            raise

    @beartype
    def run_from_config(
        self,
        config: GeneticExperimentConfig | CulturalExperimentConfig | CombinedExperimentConfig,
    ) -> GeneticExperimentResult | CulturalExperimentResult | IntegratedExperimentResult:
        """
        Run experiment from validated Pydantic configuration.

        Parameters
        ----------
        config : GeneticExperimentConfig | CulturalExperimentConfig | CombinedExperimentConfig
            Already validated experiment configuration

        Returns
        -------
        GeneticExperimentResult | CulturalExperimentResult | IntegratedExperimentResult
            Type-safe experiment result
        """
        try:
            result = run_validated_experiment(config)
            self.experiments_run += 1
            return result
        except Exception:
            logger.exception(f"Experiment execution failed: {config.name}")
            self.experiments_failed += 1
            raise

    def get_stats(self) -> dict[str, int | float]:
        """Get experiment execution statistics."""
        return {
            "experiments_run": self.experiments_run,
            "experiments_failed": self.experiments_failed,
            "success_rate": self.experiments_run / max(1, self.experiments_run + self.experiments_failed),
        }
