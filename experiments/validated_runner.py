#!/usr/bin/env python3
"""
Simplified experiment runner using unified LoveBugConfig.

This module provides a clean, type-safe experiment runner that accepts
only the unified LoveBugConfig and determines experiment behavior based
on the configuration contents.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from beartype import beartype

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.models import (
    ExperimentMetadata,
)
from lovebug.config import LoveBugConfig
from lovebug.model import LoveModelRefactored

__all__ = ["run_validated_experiment", "ValidatedExperimentRunner"]

logger = logging.getLogger(__name__)


@beartype
def run_validated_experiment(config: LoveBugConfig) -> dict[str, Any]:
    """
    Execute an experiment with unified configuration.

    The experiment type is determined automatically based on the layer configuration:
    - Genetic-only: layer.genetic_enabled=True, layer.cultural_enabled=False
    - Cultural-only: layer.genetic_enabled=False, layer.cultural_enabled=True
    - Combined: layer.genetic_enabled=True, layer.cultural_enabled=True

    Parameters
    ----------
    config : LoveBugConfig
        Unified experiment configuration

    Returns
    -------
    dict[str, Any]
        Experiment results with metadata

    Raises
    ------
    ValueError
        If model initialization or execution fails
    """
    experiment_start = time.time()
    start_time = datetime.now()
    experiment_id = str(uuid.uuid4())[:8]

    try:
        logger.info(f"Starting validated experiment: {config.name} (ID: {experiment_id})")
        logger.info(f"Layer config: genetic={config.layer.genetic_enabled}, cultural={config.layer.cultural_enabled}")
        logger.debug(f"Configuration: {config.model_dump_json(indent=2)}")

        # Create and run model
        model = LoveModelRefactored(config=config)

        # Use simulation parameters for steps and replications
        n_steps = config.simulation.steps

        # Run simulation
        model_results = model.run(n_steps=n_steps)

        # Determine experiment type for metadata
        if config.layer.is_genetic_only():
            experiment_type = "genetic"
        elif config.layer.is_cultural_only():
            experiment_type = "cultural"
        elif config.layer.is_combined():
            experiment_type = "integrated"
        else:
            experiment_type = "genetic"  # Default fallback
            logger.warning("Neither genetic nor cultural layers enabled, defaulting to genetic")

        experiment_duration = time.time() - experiment_start
        end_time = datetime.now()

        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=config.name,
            experiment_type=experiment_type,
            start_time=start_time,
            duration_seconds=experiment_duration,
            success=True,
            process_id=os.getpid(),
        )

        # Combine results
        results = {
            "metadata": {
                "experiment_id": metadata.experiment_id,
                "name": metadata.name,
                "experiment_type": metadata.experiment_type,
                "start_time": metadata.start_time,
                "end_time": end_time,
                "duration_seconds": metadata.duration_seconds,
                "population_size": config.simulation.population_size,
                "n_generations": n_steps,
                "random_seed": config.simulation.seed,
                "success": metadata.success,
                "process_id": metadata.process_id,
            },
            "model_results": model_results,
            "config": config.model_dump(),
            "final_population_size": len(model.agents),
            "total_steps": model.step_count,
            "success": True,
        }

        logger.info(f"Experiment {experiment_id} completed successfully in {experiment_duration:.2f}s")
        return results

    except Exception as e:
        experiment_duration = time.time() - experiment_start
        logger.error(f"Experiment {experiment_id} failed after {experiment_duration:.2f}s: {e}")

        # Return failure result
        return {
            "metadata": {
                "experiment_id": experiment_id,
                "experiment_name": config.name,
                "start_time": start_time,
                "end_time": datetime.now(),
                "duration_seconds": experiment_duration,
                "experiment_type": "failed",
                "error": str(e),
            },
            "success": False,
            "error": str(e),
        }


class ValidatedExperimentRunner:
    """
    Unified experiment runner for LoveBug simulations.

    This class provides a clean interface for running experiments with
    automatic type detection and consistent result formatting.
    """

    def __init__(self, base_output_dir: Path | str = "experiments/results"):
        """
        Initialize the experiment runner.

        Parameters
        ----------
        base_output_dir : Path | str
            Base directory for experiment outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_output_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Track experiment statistics
        self.experiments_run = 0
        self.experiments_failed = 0

        logger.info(f"ValidatedExperimentRunner initialized with output dir: {self.base_output_dir}")
        logger.info(f"Session results will be saved to: {self.session_dir}")

    @beartype
    def run_experiment(self, config: LoveBugConfig, save_results: bool = True) -> dict[str, Any]:
        """
        Run a single experiment with the given configuration.

        Parameters
        ----------
        config : LoveBugConfig
            Experiment configuration
        save_results : bool, default True
            Whether to save results to disk

        Returns
        -------
        dict[str, Any]
            Experiment results
        """
        results = run_validated_experiment(config)

        # Track statistics
        self.experiments_run += 1
        if not results.get("success", False):
            self.experiments_failed += 1

        if save_results and results.get("success", False):
            self._save_results(results)

        return results

    @beartype
    def run_batch(self, configs: list[LoveBugConfig], save_results: bool = True) -> list[dict[str, Any]]:
        """
        Run a batch of experiments.

        Parameters
        ----------
        configs : list[LoveBugConfig]
            List of experiment configurations
        save_results : bool, default True
            Whether to save results to disk

        Returns
        -------
        list[dict[str, Any]]
            List of experiment results
        """
        results = []
        total_configs = len(configs)

        logger.info(f"Starting batch run of {total_configs} experiments")

        for i, config in enumerate(configs, 1):
            logger.info(f"Running experiment {i}/{total_configs}: {config.name}")

            try:
                result = self.run_experiment(config, save_results=save_results)
                results.append(result)

                if result.get("success", False):
                    logger.info(f"Experiment {i}/{total_configs} completed successfully")
                else:
                    logger.error(f"Experiment {i}/{total_configs} failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Exception in experiment {i}/{total_configs}: {e}")
                results.append(
                    {
                        "success": False,
                        "error": str(e),
                        "config": config.model_dump(),
                    }
                )

        successful = sum(1 for r in results if r.get("success", False))
        logger.info(f"Batch completed: {successful}/{total_configs} experiments successful")

        return results

    def get_stats(self) -> dict[str, Any]:
        """
        Get runner statistics.

        Returns
        -------
        dict[str, Any]
            Runner statistics
        """
        success_rate = 0.0
        if self.experiments_run > 0:
            success_rate = (self.experiments_run - self.experiments_failed) / self.experiments_run

        return {
            "output_directory": str(self.base_output_dir),
            "runner_type": "ValidatedExperimentRunner",
            "experiments_run": self.experiments_run,
            "experiments_failed": self.experiments_failed,
            "success_rate": success_rate,
        }

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save experiment results to disk."""
        try:
            metadata = results.get("metadata", {})
            experiment_id = metadata.get("experiment_id", "unknown")
            experiment_name = metadata.get("name", metadata.get("experiment_name", "unnamed"))

            # Create safe filename with proper experiment name
            safe_name = "".join(c for c in experiment_name if c.isalnum() or c in ("-", "_"))
            if safe_name == "unnamed" or not safe_name:
                safe_name = f"experiment_{experiment_id}"
            filename = f"{safe_name}_{experiment_id}.json"
            filepath = self.session_dir / filename

            import json

            with open(filepath, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.debug(f"Results saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")


# Backwards compatibility functions (deprecated)
@beartype
def run_validated_genetic_experiment(config: LoveBugConfig) -> dict[str, Any]:
    """
    DEPRECATED: Use run_validated_experiment() instead.

    Execute a genetic evolution experiment with validated configuration.
    """
    logger.warning("run_validated_genetic_experiment is deprecated. Use run_validated_experiment instead.")
    return run_validated_experiment(config)


@beartype
def run_validated_cultural_experiment(config: LoveBugConfig) -> dict[str, Any]:
    """
    DEPRECATED: Use run_validated_experiment() instead.

    Execute a cultural evolution experiment with validated configuration.
    """
    logger.warning("run_validated_cultural_experiment is deprecated. Use run_validated_experiment instead.")
    return run_validated_experiment(config)


@beartype
def run_validated_integrated_experiment(config: LoveBugConfig) -> dict[str, Any]:
    """
    DEPRECATED: Use run_validated_experiment() instead.

    Execute an integrated genetic-cultural experiment with validated configuration.
    """
    logger.warning("run_validated_integrated_experiment is deprecated. Use run_validated_experiment instead.")
    return run_validated_experiment(config)
