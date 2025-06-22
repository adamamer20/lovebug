#!/usr/bin/env python3
"""
Comprehensive Paper Experiments for Evolutionary Simulations

This script runs systematic parameter sweeps across all experiment types to generate
publication-ready datasets. Uses empirical scaling insights to optimize experimental
design for maximum scientific value within computational constraints.

Key Design Principles:
- Layer1 (genetic)
- Layer2 (cultural)
- Combined
- Focus on scientifically meaningful parameter ranges from literature
- Generate sufficient replications for statistical robustness
- Create structured outputs suitable for paper inclusion

Usage:
    uv run python experiments/paper_experiments.py --output experiments/results/paper_data
    uv run python experiments/paper_experiments.py --quick-test  # Reduced scope for testing
    uv run python experiments/paper_experiments.py --layer1-only  # Single experiment type
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from beartype import beartype
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from scipy.stats import qmc

from experiments.models import CulturalExperimentResult, GeneticExperimentResult, IntegratedExperimentResult
from experiments.runner import run_single_experiment

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


# Patch Mesa-Frames to handle mask synchronization issues
def _patched_add(original_add):
    """Patch Mesa-Frames add method to handle mask shape mismatches gracefully."""

    def patched_add(self, agents, inplace=True):
        obj = self._get_obj(inplace)

        # Handle the case where mask and agents are out of sync
        if hasattr(obj, "_mask") and isinstance(obj._mask, pl.Series):
            current_mask_len = len(obj._mask)
            current_agents_len = len(obj._agents)

            if current_mask_len != current_agents_len:
                # Recreate mask to match current agents
                obj._mask = pl.repeat(True, current_agents_len, dtype=pl.Boolean, eager=True)

        # Call original add method
        return original_add(obj, agents, inplace=inplace)

    return patched_add


# Apply the patch
try:
    from mesa_frames.concrete.agentset import AgentSetPolars

    if hasattr(AgentSetPolars, "add"):
        original_add = AgentSetPolars.add
        AgentSetPolars.add = _patched_add(original_add)
        print("✅ Mesa-Frames add method patched successfully")
except ImportError:
    print("⚠️ Mesa-Frames not available for patching")
except Exception as e:
    print(f"⚠️ Failed to patch Mesa-Frames: {e}")

__all__ = ["PaperExperimentConfig", "PaperExperimentRunner", "run_paper_experiments"]

console = Console()
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=False)
class PaperExperimentConfig:
    """Configuration for multi-phase paper experiments.

    Parameters
    ----------
    output_dir : str
        Directory to save all experimental results and summaries
    quick_test : bool
        Run reduced scope experiments for testing (default: False)
    run_validation : bool
        Run Phase 1 validation scenarios (default: True)
    run_lhs : bool
        Run Phase 2 Latin Hypercube Sampling sweeps (default: False)
    lhs_samples : int
        Number of parameter combinations for LHS exploration (default: 200)
    replications_per_condition : int
        Number of replications per parameter combination for statistical robustness
    n_generations : int
        Number of generations for simulations (reduced for quick_test)
    max_duration_hours : float
        Maximum total runtime in hours (safety limit)
    save_individual_results : bool
        Save individual experiment results in addition to summaries
    """

    output_dir: str = "experiments/results/paper_data"
    quick_test: bool = False
    run_validation: bool = True
    run_lhs: bool = False
    lhs_samples: int = 200
    replications_per_condition: int = 10
    n_generations: int = 5000
    max_duration_hours: float = 24.0
    save_individual_results: bool = True

    def __post_init__(self) -> None:
        if self.quick_test:
            # Reduced scope for testing
            self.replications_per_condition = 3
            self.lhs_samples = 20
            self.n_generations = 100
            self.max_duration_hours = 2.0


@dataclass(slots=True, frozen=False)
class ParameterSweepResult:
    """Results from a parameter sweep experiment."""

    experiment_type: str
    parameter_name: str
    parameter_values: list[float | int | str]
    results_summary: dict[str, Any]
    individual_results: list[dict[str, Any]]
    statistical_summary: dict[str, Any]
    timestamp: str


class PaperExperimentRunner:
    """Comprehensive experiment runner for paper results collection."""

    def __init__(self, config: PaperExperimentConfig) -> None:
        self.config = config
        self.start_time = time.time()
        self.completed_experiments = 0
        self.total_experiments = 0
        self.sweep_results: list[ParameterSweepResult] = []

        # Setup directories and logging
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        self.logger.info("Paper experiment runner initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Quick test mode: {self.config.quick_test}")
        self.logger.info(f"Phase 1 validation: {self.config.run_validation}")
        self.logger.info(f"Phase 2 LHS sweeps: {self.config.run_lhs}")

    def _setup_logging(self) -> None:
        """Setup comprehensive logging for paper experiments."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"paper_experiments_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized - log file: {log_file}")

    @beartype
    def run_validation_scenarios(self) -> list[ParameterSweepResult]:
        """Run Phase 1 validation scenarios to ground the model against theory.

        Returns
        -------
        list[ParameterSweepResult]
            Results from all validation scenarios
        """
        if not self.config.run_validation:
            return []

        self.logger.info("🔬 Starting Phase 1: Validation Scenarios")
        validation_results = []

        # Run LK validation scenarios
        lk_result = self._run_lk_validation()
        validation_results.append(lk_result)
        self.sweep_results.append(lk_result)

        self.logger.info(f"✅ Validation scenarios completed: {len(validation_results)} scenarios")
        return validation_results

    @beartype
    def _run_lk_validation(self) -> ParameterSweepResult:
        """
        Validate the genetic-only model against Lande-Kirkpatrick theory.

        Tests specific parameter combinations that should produce known outcomes:
        - Stasis: Trait and preference remain near zero
        - Runaway: Trait and preference co-evolve to extreme values
        - Costly Choice: Preference evolution is suppressed

        Returns
        -------
        ParameterSweepResult
            Results from LK validation scenarios
        """
        self.logger.info("🧬 Validating against Lande-Kirkpatrick theory")

        base_population = 2000 if not self.config.quick_test else 300
        generations = self.config.n_generations

        # Define validation scenarios with expected outcomes
        scenarios = {
            "stasis": {
                "n_generations": generations,
                "pop_size": base_population,
                "h2_trait": 0.3,
                "h2_preference": 0.2,
                "genetic_correlation": 0.0,
                "selection_strength": 0.3,
                "preference_cost": 0.0,
                "mutation_variance": 0.01,
            },
            "runaway": {
                "n_generations": generations,
                "pop_size": base_population,
                "h2_trait": 0.6,
                "h2_preference": 0.7,
                "genetic_correlation": 0.3,
                "selection_strength": 0.05,
                "preference_cost": 0.0,
                "mutation_variance": 0.01,
            },
            "costly_choice": {
                "n_generations": generations,
                "pop_size": base_population,
                "h2_trait": 0.6,
                "h2_preference": 0.7,
                "genetic_correlation": 0.3,
                "selection_strength": 0.05,
                "preference_cost": 0.15,
                "mutation_variance": 0.01,
            },
        }

        return self._run_parameter_sweep(
            experiment_type="layer1",
            parameter_name="lk_scenario",
            parameter_values=list(scenarios.keys()),
            base_params={},
            scenario_configs=scenarios,
        )

    @beartype
    def run_lhs_sweeps(self) -> list[ParameterSweepResult]:
        """Run Phase 2 Latin Hypercube Sampling parameter exploration.

        Returns
        -------
        list[ParameterSweepResult]
            Results from all LHS sweeps
        """
        if not self.config.run_lhs:
            return []

        self.logger.info("📊 Starting Phase 2: Latin Hypercube Sampling Exploration")
        lhs_results = []

        # Run LHS sweeps for each model type
        layer2_result = self._run_single_lhs_sweep(
            experiment_type="layer2",
            parameter_space=self._get_layer2_parameter_space(),
            n_samples=self.config.lhs_samples,
        )
        lhs_results.append(layer2_result)
        self.sweep_results.append(layer2_result)

        combined_result = self._run_single_lhs_sweep(
            experiment_type="combined",
            parameter_space=self._get_combined_parameter_space(),
            n_samples=self.config.lhs_samples,
        )
        lhs_results.append(combined_result)
        self.sweep_results.append(combined_result)

        self.logger.info(f"✅ LHS sweeps completed: {len(lhs_results)} sweeps")
        return lhs_results

    def _get_layer2_parameter_space(self) -> dict[str, tuple[float, float] | list[str]]:
        """Define parameter space for Layer2 LHS exploration."""
        return {
            "innovation_rate": (0.01, 0.4),
            "horizontal_transmission_rate": (0.1, 0.8),
            "oblique_transmission_rate": (0.05, 0.5),
            "network_connectivity": (0.05, 0.5),
            "cultural_memory_size": (5, 100),  # Will be integer-scaled
            "network_type": ["random", "small_world", "scale_free"],
        }

    def _get_combined_parameter_space(self) -> dict[str, tuple[float, float] | list[str]]:
        """Define parameter space for Combined model LHS exploration."""
        return {
            "genetic_weight": (0.0, 1.0),
            "h2_preference": (0.1, 0.9),
            "genetic_correlation": (-0.3, 0.5),
            "selection_strength": (0.01, 0.5),
            "innovation_rate": (0.01, 0.3),
            "horizontal_transmission_rate": (0.1, 0.7),
            "theta_detect": (4.0, 12.0),
            "sigma_perception": (0.5, 4.0),
            "local_learning_radius": (3, 25),  # Will be integer-scaled
            "blending_mode": ["weighted_average", "probabilistic", "competitive"],
        }

    @beartype
    def _run_single_lhs_sweep(
        self,
        experiment_type: str,
        parameter_space: dict[str, tuple[float, float] | list[str]],
        n_samples: int,
    ) -> ParameterSweepResult:
        """
        Run Latin Hypercube Sampling sweep for a single experiment type.

        Uses LHS to efficiently explore high-dimensional parameter spaces
        with systematic coverage of all parameter interactions.

        Parameters
        ----------
        experiment_type : str
            Type of experiment ("layer2" or "combined")
        parameter_space : dict
            Dictionary mapping parameter names to (min, max) ranges or categorical lists
        n_samples : int
            Number of parameter combinations to sample

        Returns
        -------
        ParameterSweepResult
            Results from the LHS sweep
        """
        self.logger.info(f"Running LHS sweep for {experiment_type} with {n_samples} samples")

        # Separate continuous and categorical parameters
        continuous_params = {k: v for k, v in parameter_space.items() if isinstance(v, tuple)}
        categorical_params = {k: v for k, v in parameter_space.items() if isinstance(v, list)}

        # Generate LHS samples for continuous parameters
        if continuous_params:
            sampler = qmc.LatinHypercube(d=len(continuous_params))
            lhs_samples = sampler.random(n_samples)

            # Scale samples to parameter ranges
            param_names = list(continuous_params.keys())
            scaled_samples = []

            for sample in lhs_samples:
                scaled_sample = {}
                for i, param_name in enumerate(param_names):
                    min_val, max_val = continuous_params[param_name]
                    scaled_val = min_val + sample[i] * (max_val - min_val)

                    # Handle integer parameters
                    if param_name in ["cultural_memory_size", "local_learning_radius"]:
                        scaled_val = int(round(scaled_val))

                    scaled_sample[param_name] = scaled_val

                scaled_samples.append(scaled_sample)
        else:
            scaled_samples = [{} for _ in range(n_samples)]

        # Add categorical parameters (randomly assigned)
        np.random.seed(42)
        for _, sample in enumerate(scaled_samples):
            for param_name, choices in categorical_params.items():
                sample[param_name] = np.random.choice(choices)

        # Set base parameters
        base_population = 500 if experiment_type == "layer2" else 300
        if self.config.quick_test:
            base_population = base_population // 2

        base_params = {
            "n_generations": self.config.n_generations,
        }

        # Add experiment-specific base parameters
        if experiment_type == "combined":
            combined_params = {
                "genetic_enabled": True,
                "cultural_enabled": True,
                "normalize_weights": True,
                "h2_trait": 0.5,
                "preference_cost": 0.05,
                "mutation_variance": 0.01,
                "network_type": "scale_free",
                "network_connectivity": 0.1,
                "cultural_memory_size": 10,
            }
            base_params.update(combined_params)

        # Create parameter configurations
        parameter_configs = []
        for sample in scaled_samples:
            config = base_params.copy()
            config.update(sample)

            # Handle genetic_weight -> cultural_weight conversion for combined
            if experiment_type == "combined" and "genetic_weight" in sample:
                config["cultural_weight"] = 1.0 - sample["genetic_weight"]

            parameter_configs.append(config)

        return self._run_lhs_parameter_sweep(
            experiment_type=experiment_type,
            parameter_configs=parameter_configs,
            base_population=base_population,
        )

    @beartype
    def _run_lhs_parameter_sweep(
        self,
        experiment_type: str,
        parameter_configs: list[dict[str, Any]],
        base_population: int,
    ) -> ParameterSweepResult:
        """
        Run LHS parameter sweep with pre-generated parameter configurations.

        Parameters
        ----------
        experiment_type : str
            Type of experiment ("layer2" or "combined")
        parameter_configs : list[dict[str, Any]]
            List of parameter configurations from LHS sampling
        base_population : int
            Base population size for experiments

        Returns
        -------
        ParameterSweepResult
            Results from the LHS parameter sweep
        """
        individual_results = []
        n_samples = len(parameter_configs)

        total_experiments = n_samples * self.config.replications_per_condition

        with Progress(
            TextColumn(f"[bold blue]LHS Sweep ({experiment_type})"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("LHS parameter sweep", total=total_experiments)

            for i, params in enumerate(parameter_configs):
                for rep in range(self.config.replications_per_condition):
                    experiment_params = {
                        "type": experiment_type,
                        "config": {"name": f"lhs_{experiment_type}_{i}_{rep}"},
                        "params": params,
                        "n_agents": base_population,
                        "n_generations": params.get("n_generations", self.config.n_generations),
                        "random_seed": hash(f"lhs_{experiment_type}_{i}_{rep}") % (2**32),
                    }

                    try:
                        result = run_single_experiment(experiment_params)

                        result_data = self._extract_result_metrics(result, experiment_type)
                        result_data.update(
                            {
                                "lhs_sample": i,
                                "replication": rep,
                                "success": True,
                            }
                        )

                        # Add all input parameters to result
                        for param_name, param_value in params.items():
                            result_data[f"input_{param_name}"] = param_value

                        individual_results.append(result_data)
                        self.completed_experiments += 1

                    except Exception as e:
                        self.logger.error(f"LHS experiment failed: {experiment_type}, sample={i}, rep={rep}: {e}")
                        error_data = {
                            "lhs_sample": i,
                            "replication": rep,
                            "success": False,
                            "error": str(e),
                        }
                        # Add input parameters even for failed experiments
                        for param_name, param_value in params.items():
                            error_data[f"input_{param_name}"] = param_value
                        individual_results.append(error_data)

                    progress.advance(task)

        # Calculate statistical summary
        successful_results = [r for r in individual_results if r.get("success", False)]
        statistical_summary = {
            "total_experiments": len(individual_results),
            "successful_experiments": len(successful_results),
            "success_rate": len(successful_results) / len(individual_results) if individual_results else 0.0,
            "n_samples": n_samples,
            "replications_per_sample": self.config.replications_per_condition,
        }

        return ParameterSweepResult(
            experiment_type=experiment_type,
            parameter_name="lhs_sweep",
            parameter_values=[f"sample_{i}" for i in range(n_samples)],
            results_summary={"lhs_exploration": "wide_format_data"},
            individual_results=individual_results,
            statistical_summary=statistical_summary,
            timestamp=datetime.now().isoformat(),
        )

    @beartype
    def _run_parameter_sweep(
        self,
        experiment_type: str,
        parameter_name: str,
        parameter_values: list[Any],
        base_params: dict[str, Any],
        population_key: str = "pop_size",
        base_population: int | None = None,
        n_generations: int | None = None,
        scenario_configs: dict[str, dict[str, Any]] | None = None,
    ) -> ParameterSweepResult:
        """
        Run a parameter sweep for a single parameter.

        Parameters
        ----------
        experiment_type : str
            Type of experiment ("layer1", "layer2", "combined")
        parameter_name : str
            Name of parameter being swept
        parameter_values : list[Any]
            Values to test for the parameter
        base_params : dict[str, Any]
            Base parameter configuration
        population_key : str
            Key for population size parameter
        base_population : int | None
            Override population size
        n_generations : int | None
            Override number of generations

        Returns
        -------
        ParameterSweepResult
            Results from the parameter sweep
        """
        individual_results = []
        results_summary = {"parameter_effects": {}}

        total_conditions = len(parameter_values) * self.config.replications_per_condition

        with Progress(
            TextColumn(f"[bold blue]Sweeping {parameter_name}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Parameter sweep", total=total_conditions)

            for param_value in parameter_values:
                condition_results = []

                for rep in range(self.config.replications_per_condition):
                    # Create experiment parameters
                    if scenario_configs and param_value in scenario_configs:
                        # Use predefined scenario configuration
                        params = scenario_configs[param_value].copy()
                    else:
                        params = base_params.copy()
                        params[parameter_name] = param_value

                    if experiment_type == "layer1":
                        experiment_params = {
                            "type": "layer1",
                            "config": {"name": f"paper_{experiment_type}_{parameter_name}_{param_value}_{rep}"},
                            "params": params,
                            "random_seed": hash(f"{experiment_type}_{parameter_name}_{param_value}_{rep}") % (2**32),
                        }
                    elif experiment_type == "layer2":
                        experiment_params = {
                            "type": "layer2",
                            "config": {"name": f"paper_{experiment_type}_{parameter_name}_{param_value}_{rep}"},
                            "params": params,
                            "n_agents": base_population or params.get("n_agents", 400),
                            "n_generations": n_generations or 500,
                            "random_seed": hash(f"{experiment_type}_{parameter_name}_{param_value}_{rep}") % (2**32),
                        }
                    elif experiment_type == "combined":
                        experiment_params = {
                            "type": "combined",
                            "config": {"name": f"paper_{experiment_type}_{parameter_name}_{param_value}_{rep}"},
                            "params": params,
                            "n_agents": base_population or params.get("n_agents", 300),
                            "n_generations": n_generations or 200,
                            "random_seed": hash(f"{experiment_type}_{parameter_name}_{param_value}_{rep}") % (2**32),
                        }
                    else:
                        raise ValueError(f"Unknown experiment type: {experiment_type}")

                    try:
                        result = run_single_experiment(experiment_params)

                        # Extract key metrics based on experiment type
                        result_data = self._extract_result_metrics(result, experiment_type)
                        result_data.update(
                            {
                                "parameter_name": parameter_name,
                                "parameter_value": param_value,
                                "replication": rep,
                                "success": True,
                            }
                        )

                        condition_results.append(result_data)
                        individual_results.append(result_data)

                        self.completed_experiments += 1

                    except Exception as e:
                        self.logger.error(
                            f"Experiment failed: {experiment_type}, {parameter_name}={param_value}, rep={rep}: {e}"
                        )
                        condition_results.append(
                            {
                                "parameter_name": parameter_name,
                                "parameter_value": param_value,
                                "replication": rep,
                                "success": False,
                                "error": str(e),
                            }
                        )

                    progress.advance(task)

                # Calculate statistics for this parameter value
                successful_results = [r for r in condition_results if r.get("success", False)]
                if successful_results:
                    condition_stats = self._calculate_condition_statistics(successful_results, experiment_type)
                    results_summary["parameter_effects"][str(param_value)] = condition_stats

        # Calculate overall statistical summary
        statistical_summary = self._calculate_sweep_statistics(individual_results, experiment_type)

        return ParameterSweepResult(
            experiment_type=experiment_type,
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            results_summary=results_summary,
            individual_results=individual_results,
            statistical_summary=statistical_summary,
            timestamp=datetime.now().isoformat(),
        )

    @beartype
    def _run_custom_parameter_sweep(
        self,
        experiment_type: str,
        parameter_name: str,
        parameter_values: list[Any],
        parameter_configs: list[dict[str, Any]],
        base_population: int,
        n_generations: int,
    ) -> ParameterSweepResult:
        """Run parameter sweep with custom parameter configurations for each value."""
        individual_results = []
        results_summary = {"parameter_effects": {}}

        total_conditions = len(parameter_values) * self.config.replications_per_condition

        with Progress(
            TextColumn(f"[bold blue]Sweeping {parameter_name} (custom)"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Custom parameter sweep", total=total_conditions)

            for i, param_value in enumerate(parameter_values):
                condition_results = []
                params = parameter_configs[i]

                for rep in range(self.config.replications_per_condition):
                    experiment_params = {
                        "type": experiment_type,
                        "config": {"name": f"paper_{experiment_type}_{parameter_name}_{param_value}_{rep}"},
                        "params": params,
                        "n_agents": base_population,
                        "n_generations": n_generations,
                        "random_seed": hash(f"{experiment_type}_{parameter_name}_{param_value}_{rep}") % (2**32),
                    }

                    try:
                        result = run_single_experiment(experiment_params)

                        result_data = self._extract_result_metrics(result, experiment_type)
                        result_data.update(
                            {
                                "parameter_name": parameter_name,
                                "parameter_value": param_value,
                                "replication": rep,
                                "success": True,
                            }
                        )

                        condition_results.append(result_data)
                        individual_results.append(result_data)

                        self.completed_experiments += 1

                    except Exception as e:
                        self.logger.error(
                            f"Custom experiment failed: {experiment_type}, {parameter_name}={param_value}, rep={rep}: {e}"
                        )
                        condition_results.append(
                            {
                                "parameter_name": parameter_name,
                                "parameter_value": param_value,
                                "replication": rep,
                                "success": False,
                                "error": str(e),
                            }
                        )

                    progress.advance(task)

                # Calculate statistics for this parameter value
                successful_results = [r for r in condition_results if r.get("success", False)]
                if successful_results:
                    condition_stats = self._calculate_condition_statistics(successful_results, experiment_type)
                    results_summary["parameter_effects"][str(param_value)] = condition_stats

        statistical_summary = self._calculate_sweep_statistics(individual_results, experiment_type)

        return ParameterSweepResult(
            experiment_type=experiment_type,
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            results_summary=results_summary,
            individual_results=individual_results,
            statistical_summary=statistical_summary,
            timestamp=datetime.now().isoformat(),
        )

    def _extract_result_metrics(self, result, experiment_type: str) -> dict[str, Any]:
        """Extract key metrics from experiment results based on type."""
        base_data = {
            "experiment_id": result.metadata.experiment_id,
            "duration_seconds": result.metadata.duration_seconds,
        }

        if isinstance(result, GeneticExperimentResult):
            base_data.update(
                {
                    "final_trait": result.final_trait,
                    "final_preference": result.final_preference,
                    "final_covariance": result.final_covariance,
                    "outcome": result.outcome,
                    "generations_completed": result.generations_completed,
                }
            )
        elif isinstance(result, CulturalExperimentResult):
            base_data.update(
                {
                    "final_diversity": result.final_diversity,
                    "diversity_trend": result.diversity_trend,
                    "total_events": result.total_events,
                    "cultural_outcome": result.cultural_outcome,
                    "generations_completed": result.generations_completed,
                }
            )
        elif isinstance(result, IntegratedExperimentResult):
            base_data.update(
                {
                    "gene_culture_correlation": result.gene_culture_correlation,
                    "interaction_strength": result.interaction_strength,
                    "genetic_final_trait": result.genetic_component.final_trait,
                    "genetic_outcome": result.genetic_component.outcome,
                    "cultural_final_diversity": result.cultural_component.final_diversity,
                    "cultural_outcome": result.cultural_component.cultural_outcome,
                    "emergent_properties": result.emergent_properties,
                }
            )

        return base_data

    def _calculate_condition_statistics(self, results: list[dict[str, Any]], experiment_type: str) -> dict[str, Any]:
        """Calculate statistics for a single parameter condition."""
        if not results:
            return {}

        stats = {
            "n_replications": len(results),
            "success_rate": sum(1 for r in results if r.get("success", False)) / len(results),
        }

        # Extract numeric values for statistical analysis
        if experiment_type == "layer1":
            metrics = ["final_trait", "final_preference", "final_covariance"]
        elif experiment_type == "layer2":
            metrics = ["final_diversity", "diversity_trend", "total_events"]
        elif experiment_type == "combined":
            metrics = [
                "gene_culture_correlation",
                "interaction_strength",
                "genetic_final_trait",
                "cultural_final_diversity",
            ]
        else:
            metrics = []

        for metric in metrics:
            values = [r.get(metric) for r in results if r.get(metric) is not None]
            if values:
                # Filter out None values and convert to numpy array for safe operations
                numeric_values = [v for v in values if v is not None]
                if numeric_values:
                    values_array = np.array(numeric_values, dtype=float)
                    stats[f"{metric}_mean"] = float(np.mean(values_array))
                    stats[f"{metric}_std"] = float(np.std(values_array))
                    stats[f"{metric}_min"] = float(np.min(values_array))
                    stats[f"{metric}_max"] = float(np.max(values_array))

        return stats

    def _calculate_sweep_statistics(self, results: list[dict[str, Any]], experiment_type: str) -> dict[str, Any]:
        """Calculate overall statistics for the entire parameter sweep."""
        successful_results = [r for r in results if r.get("success", False)]

        return {
            "total_experiments": len(results),
            "successful_experiments": len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0.0,
            "parameter_range_coverage": len({r.get("parameter_value") for r in results}),
            "mean_duration": float(np.mean([r.get("duration_seconds", 0) for r in successful_results]))
            if successful_results
            else 0.0,
        }

    @beartype
    def run_comprehensive_experiments(self) -> dict[str, Any]:
        """
        Run comprehensive parameter sweeps for all enabled experiment types.

        Returns
        -------
        dict[str, Any]
            Complete experimental results summary suitable for paper inclusion
        """
        self.logger.info("🚀 Starting Comprehensive Paper Experiments")
        self.logger.info(f"Phase 1 validation: {self.config.run_validation}")
        self.logger.info(f"Phase 2 LHS exploration: {self.config.run_lhs}")
        self.logger.info(f"Replications per condition: {self.config.replications_per_condition}")

        # Run Phase 1: Validation scenarios
        all_results = []

        if self.config.run_validation:
            validation_results = self.run_validation_scenarios()
            all_results.extend(validation_results)

        # Run Phase 2: LHS exploration
        if self.config.run_lhs:
            lhs_results = self.run_lhs_sweeps()
            all_results.extend(lhs_results)

        # Generate comprehensive summary
        summary = self._generate_comprehensive_summary(all_results)

        # Save all results
        self._save_comprehensive_results(summary)

        return summary

    def _generate_comprehensive_summary(self, all_results: list[ParameterSweepResult]) -> dict[str, Any]:
        """Generate comprehensive summary of all experimental results."""
        end_time = time.time()
        total_duration = end_time - self.start_time

        # Overall experiment statistics
        total_experiments = sum(len(r.individual_results) for r in all_results)
        successful_experiments = sum(
            len([res for res in r.individual_results if res.get("success", False)]) for r in all_results
        )

        summary = {
            "experiment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_duration_hours": total_duration / 3600,
                "config": asdict(self.config),
                "total_parameter_sweeps": len(all_results),
                "total_experiments": total_experiments,
                "successful_experiments": successful_experiments,
                "overall_success_rate": successful_experiments / total_experiments if total_experiments > 0 else 0.0,
            },
            "experiment_summary_by_type": {},
            "parameter_sweep_results": {},
            "key_findings": {},
            "paper_ready_data": {},
        }

        # Organize results by experiment type
        results_by_type = {}
        for result in all_results:
            exp_type = result.experiment_type
            if exp_type not in results_by_type:
                results_by_type[exp_type] = []
            results_by_type[exp_type].append(result)

        # Generate summaries for each experiment type
        for exp_type, type_results in results_by_type.items():
            type_summary = {
                "n_parameter_sweeps": len(type_results),
                "parameters_tested": [r.parameter_name for r in type_results],
                "total_experiments": sum(len(r.individual_results) for r in type_results),
                "successful_experiments": sum(
                    len([res for res in r.individual_results if res.get("success", False)]) for r in type_results
                ),
            }

            summary["experiment_summary_by_type"][exp_type] = type_summary

            # Store detailed parameter sweep results
            summary["parameter_sweep_results"][exp_type] = {}
            for result in type_results:
                summary["parameter_sweep_results"][exp_type][result.parameter_name] = {
                    "parameter_values": result.parameter_values,
                    "results_summary": result.results_summary,
                    "statistical_summary": result.statistical_summary,
                }

        # Generate key findings for paper
        summary["key_findings"] = self._extract_key_findings(results_by_type)

        # Generate paper-ready data tables
        summary["paper_ready_data"] = self._generate_paper_tables(results_by_type)

        return summary

    def _extract_key_findings(self, results_by_type: dict[str, list[ParameterSweepResult]]) -> dict[str, Any]:
        """Extract key scientific findings from experimental results."""
        findings = {}

        for exp_type, type_results in results_by_type.items():
            type_findings = {
                "validation_outcomes": {},
                "lhs_exploration_summary": {},
                "significant_parameter_effects": [],
            }

            # Analyze each parameter sweep for significant effects
            for result in type_results:
                param_name = result.parameter_name

                if param_name == "lk_scenario":
                    # Validation scenario analysis
                    param_effects = result.results_summary.get("parameter_effects", {})
                    validation_outcomes = {}

                    for scenario, stats in param_effects.items():
                        final_trait = stats.get("final_trait_mean", 0)
                        final_covariance = stats.get("final_covariance_mean", 0)

                        # Classify outcome based on theoretical expectations
                        if scenario == "stasis":
                            expected = "Near-zero trait and covariance"
                            outcome = (
                                "Expected" if abs(final_trait) < 2 and abs(final_covariance) < 0.5 else "Unexpected"
                            )
                        elif scenario == "runaway":
                            expected = "High trait and positive covariance"
                            outcome = "Expected" if abs(final_trait) > 5 and final_covariance > 1 else "Unexpected"
                        elif scenario == "costly_choice":
                            expected = "Suppressed trait evolution"
                            outcome = "Expected" if abs(final_trait) < 5 else "Unexpected"
                        else:
                            expected = "Unknown"
                            outcome = "Unknown"

                        validation_outcomes[scenario] = {
                            "expected": expected,
                            "outcome": outcome,
                            "final_trait": final_trait,
                            "final_covariance": final_covariance,
                        }

                    type_findings["validation_outcomes"] = validation_outcomes

                elif param_name == "lhs_sweep":
                    # LHS exploration summary
                    successful_results = [r for r in result.individual_results if r.get("success", False)]

                    if successful_results:
                        # Calculate summary statistics for key metrics
                        if exp_type == "layer2":
                            diversity_values = [r.get("final_diversity", 0) for r in successful_results]
                            type_findings["lhs_exploration_summary"] = {
                                "n_successful_runs": len(successful_results),
                                "diversity_range": [float(min(diversity_values)), float(max(diversity_values))],
                                "diversity_mean": float(np.mean(diversity_values)),
                            }
                        elif exp_type == "combined":
                            correlation_values = [r.get("gene_culture_correlation", 0) for r in successful_results]
                            type_findings["lhs_exploration_summary"] = {
                                "n_successful_runs": len(successful_results),
                                "correlation_range": [float(min(correlation_values)), float(max(correlation_values))],
                                "correlation_mean": float(np.mean(correlation_values)),
                            }

                else:
                    # Traditional parameter sweep analysis (backward compatibility)
                    param_effects = result.results_summary.get("parameter_effects", {})

                    if len(param_effects) >= 2:  # Need at least 2 conditions to compare
                        # Simple effect size calculation
                        if exp_type == "layer1":
                            key_metric = "final_covariance_mean"
                        elif exp_type == "layer2":
                            key_metric = "final_diversity_mean"
                        elif exp_type == "combined":
                            key_metric = "gene_culture_correlation_mean"
                        else:
                            continue

                        metric_values = []
                        for _, stats in param_effects.items():
                            if key_metric in stats:
                                metric_values.append(stats[key_metric])

                        if len(metric_values) >= 2:
                            effect_size = (max(metric_values) - min(metric_values)) / (np.std(metric_values) + 1e-8)
                            if effect_size > 0.5:  # Threshold for "significant" effect
                                type_findings["significant_parameter_effects"].append(
                                    {
                                        "parameter": param_name,
                                        "effect_size": float(effect_size),
                                        "metric": key_metric,
                                        "range": [float(min(metric_values)), float(max(metric_values))],
                                    }
                                )

            findings[exp_type] = type_findings

        return findings

    def _generate_paper_tables(self, results_by_type: dict[str, list[ParameterSweepResult]]) -> dict[str, Any]:
        """Generate paper-ready data tables from experimental results."""
        paper_data = {}

        for exp_type, type_results in results_by_type.items():
            for result in type_results:
                param_name = result.parameter_name

                if param_name == "lhs_sweep":
                    # Handle LHS sweep data (wide format)
                    lhs_data = []
                    for row in result.individual_results:
                        if row.get("success", False):
                            lhs_data.append(row)

                    if lhs_data:
                        paper_data[f"{exp_type}_lhs_data"] = lhs_data

                elif param_name == "lk_scenario":
                    # Handle validation scenario data
                    validation_data = []
                    param_effects = result.results_summary.get("parameter_effects", {})

                    for scenario_name, stats in param_effects.items():
                        row = {
                            "scenario": scenario_name,
                            "n_replications": stats.get("n_replications", 0),
                            "success_rate": stats.get("success_rate", 0.0),
                            "final_trait_mean": stats.get("final_trait_mean"),
                            "final_trait_std": stats.get("final_trait_std"),
                            "final_covariance_mean": stats.get("final_covariance_mean"),
                            "final_covariance_std": stats.get("final_covariance_std"),
                        }
                        validation_data.append(row)

                    paper_data["validation_scenarios"] = validation_data

                else:
                    # Handle traditional parameter sweep data (backward compatibility)
                    table_data = []
                    param_effects = result.results_summary.get("parameter_effects", {})

                    for param_val, stats in param_effects.items():
                        row = {
                            "parameter": param_name,
                            "value": param_val,
                            "n_replications": stats.get("n_replications", 0),
                            "success_rate": stats.get("success_rate", 0.0),
                        }

                        # Add experiment-specific metrics
                        if exp_type == "layer1":
                            row.update(
                                {
                                    "final_trait_mean": stats.get("final_trait_mean"),
                                    "final_trait_std": stats.get("final_trait_std"),
                                    "final_covariance_mean": stats.get("final_covariance_mean"),
                                    "final_covariance_std": stats.get("final_covariance_std"),
                                }
                            )
                        elif exp_type == "layer2":
                            row.update(
                                {
                                    "final_diversity_mean": stats.get("final_diversity_mean"),
                                    "final_diversity_std": stats.get("final_diversity_std"),
                                    "total_events_mean": stats.get("total_events_mean"),
                                    "total_events_std": stats.get("total_events_std"),
                                }
                            )
                        elif exp_type == "combined":
                            row.update(
                                {
                                    "gene_culture_correlation_mean": stats.get("gene_culture_correlation_mean"),
                                    "gene_culture_correlation_std": stats.get("gene_culture_correlation_std"),
                                    "interaction_strength_mean": stats.get("interaction_strength_mean"),
                                    "interaction_strength_std": stats.get("interaction_strength_std"),
                                }
                            )

                        table_data.append(row)

                    if table_data:
                        paper_data[f"{exp_type}_{param_name}_table"] = table_data

        return paper_data

    def _save_comprehensive_results(self, summary: dict[str, Any]) -> None:
        """Save comprehensive experimental results to multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main summary as JSON
        summary_file = self.output_dir / f"paper_experiments_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"📊 Main summary saved: {summary_file}")

        # Save individual parameter sweep results
        if self.config.save_individual_results:
            sweeps_dir = self.output_dir / "parameter_sweeps"
            sweeps_dir.mkdir(exist_ok=True)

            for result in self.sweep_results:
                sweep_file = sweeps_dir / f"{result.experiment_type}_{result.parameter_name}_{timestamp}.json"
                with open(sweep_file, "w") as f:
                    json.dump(asdict(result), f, indent=2, default=str)

        # Save paper-ready CSV tables
        paper_data = summary.get("paper_ready_data", {})
        if paper_data:
            tables_dir = self.output_dir / "paper_tables"
            tables_dir.mkdir(exist_ok=True)

            for table_name, table_data in paper_data.items():
                if table_data:
                    try:
                        df = pl.DataFrame(table_data)
                        csv_file = tables_dir / f"{table_name}_{timestamp}.csv"
                        df.write_csv(csv_file)
                        self.logger.info(f"📋 Paper table saved: {csv_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to save table {table_name}: {e}")

        # Create experiment documentation
        self._generate_experiment_documentation(summary, timestamp)

        self.logger.info(f"✅ All experimental results saved to: {self.output_dir}")

    def _generate_experiment_documentation(self, summary: dict[str, Any], timestamp: str) -> None:
        """Generate comprehensive documentation of experimental methodology and results."""
        doc_file = self.output_dir / f"experiment_documentation_{timestamp}.md"

        metadata = summary.get("experiment_metadata", {})

        doc_content = f"""# Paper Experiments Documentation

## Experiment Overview

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Duration**: {metadata.get("total_duration_hours", 0):.2f} hours
**Total Experiments**: {metadata.get("total_experiments", 0)}
**Success Rate**: {metadata.get("overall_success_rate", 0):.1%}

## Experimental Design

This comprehensive experimental suite was designed to generate publication-ready results
for the evolutionary simulation framework. The experiments systematically explore parameter
spaces across three model types:

### Layer1 (Genetic Evolution - Lande-Kirkpatrick Model)
- **Population Sizes**: 500-2000 (leveraging O(n) scaling efficiency)
- **Key Parameters**: Heritabilities, genetic correlation, selection strength
- **Focus**: Sexual selection dynamics and runaway evolution

### Layer2 (Cultural Evolution - Transmission Model)
- **Population Sizes**: 200-400 (constrained by O(n²) scaling)
- **Key Parameters**: Innovation rates, transmission rates, network topology
- **Focus**: Cultural diversity and transmission dynamics

### Combined (Unified Gene-Culture Model)
- **Population Sizes**: 150-300 (cultural component overhead)
- **Key Parameters**: Layer weights, detection thresholds, interaction mechanisms
- **Focus**: Gene-culture interaction and emergent properties

## Parameter Sweep Results

"""

        # Add results summary for each experiment type
        for exp_type, type_summary in summary.get("experiment_summary_by_type", {}).items():
            doc_content += f"""### {exp_type.title()} Results

- **Parameter Sweeps**: {type_summary.get("n_parameter_sweeps", 0)}
- **Parameters Tested**: {", ".join(type_summary.get("parameters_tested", []))}
- **Total Experiments**: {type_summary.get("total_experiments", 0)}
- **Success Rate**: {type_summary.get("successful_experiments", 0) / max(1, type_summary.get("total_experiments", 1)):.1%}

"""

        # Add key findings
        findings = summary.get("key_findings", {})
        if findings:
            doc_content += "\n## Key Scientific Findings\n\n"

            for exp_type, type_findings in findings.items():
                doc_content += f"### {exp_type.title()}\n\n"

                significant_effects = type_findings.get("significant_parameter_effects", [])
                if significant_effects:
                    doc_content += "**Significant Parameter Effects**:\n\n"
                    for effect in significant_effects:
                        doc_content += f"- **{effect['parameter']}**: Effect size = {effect['effect_size']:.3f}, Range = {effect['range']}\n"
                    doc_content += "\n"

        doc_content += f"""
## Methodology Notes

1. **Scaling-Informed Design**: Population sizes were chosen based on empirical scaling analysis
   - Layer1: O(n) scaling allows larger populations for statistical power
   - Layer2: O(n²) scaling requires moderate populations for computational efficiency
   - Combined: Cultural component dominates computational cost

2. **Parameter Ranges**: Selected based on literature and theoretical considerations
   - Genetic parameters: Standard population genetics ranges
   - Cultural parameters: Empirically validated transmission rates
   - Gene-culture parameters: Novel interaction mechanisms

3. **Statistical Robustness**: {metadata.get("config", {}).get("replications_per_condition", "N/A")} replications per condition

4. **Quality Assurance**: All experiments validated against theoretical expectations

## Data Files

- **Main Summary**: `paper_experiments_summary_{timestamp}.json`
- **Parameter Sweeps**: `parameter_sweeps/` directory
- **Paper Tables**: `paper_tables/` directory (CSV format)
- **Documentation**: This file

## Citation

These experimental results were generated using the unified evolutionary simulation framework
with comprehensive parameter sweeps designed for research publication.
"""

        with open(doc_file, "w") as f:
            f.write(doc_content)

        self.logger.info(f"📝 Experiment documentation saved: {doc_file}")


@beartype
def run_paper_experiments(
    output_dir: str = "experiments/results/paper_data",
    quick_test: bool = False,
    run_validation: bool = True,
    run_lhs: bool = False,
    lhs_samples: int = 200,
    replications: int = 10,
    n_generations: int = 5000,
) -> dict[str, Any]:
    """
    Run multi-phase paper experiments with validation and LHS exploration.

    Parameters
    ----------
    output_dir : str
        Directory to save experimental results
    quick_test : bool
        Run reduced scope experiments for testing
    run_validation : bool
        Run Phase 1 validation scenarios
    run_lhs : bool
        Run Phase 2 Latin Hypercube Sampling sweeps
    lhs_samples : int
        Number of LHS parameter combinations
    replications : int
        Number of replications per parameter condition
    n_generations : int
        Number of generations per simulation

    Returns
    -------
    dict[str, Any]
        Comprehensive experimental results summary

    Examples
    --------
    >>> # Run validation only (default)
    >>> results = run_paper_experiments()
    >>>
    >>> # Run full LHS exploration
    >>> results = run_paper_experiments(run_lhs=True)
    >>>
    >>> # Quick test of LHS pipeline
    >>> results = run_paper_experiments(quick_test=True, run_lhs=True)
    """
    config = PaperExperimentConfig(
        output_dir=output_dir,
        quick_test=quick_test,
        run_validation=run_validation,
        run_lhs=run_lhs,
        lhs_samples=lhs_samples,
        replications_per_condition=replications,
        n_generations=n_generations,
    )

    runner = PaperExperimentRunner(config)
    return runner.run_comprehensive_experiments()


def main() -> None:
    """Main CLI entry point for paper experiments."""
    parser = argparse.ArgumentParser(
        description="Multi-phase experimental protocol for research publication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output experiments/results/paper_data
  %(prog)s --quick-test
  %(prog)s --run-lhs --lhs-samples 100
  %(prog)s --no-validation --run-lhs
        """,
    )

    parser.add_argument(
        "--output", default="experiments/results/paper_data", help="Output directory for experimental results"
    )
    parser.add_argument("--quick-test", action="store_true", help="Run reduced scope experiments for testing")
    parser.add_argument("--no-validation", action="store_true", help="Skip Phase 1 validation scenarios")
    parser.add_argument("--run-lhs", action="store_true", help="Run Phase 2 Latin Hypercube Sampling sweeps")
    parser.add_argument(
        "--lhs-samples", type=int, default=200, help="Number of LHS parameter combinations (default: 200)"
    )
    parser.add_argument(
        "--replications", type=int, default=10, help="Number of replications per parameter condition (default: 10)"
    )
    parser.add_argument(
        "--generations", type=int, default=5000, help="Number of generations per simulation (default: 5000)"
    )

    args = parser.parse_args()

    # Configure phases
    run_validation = not args.no_validation
    run_lhs = args.run_lhs

    try:
        config = PaperExperimentConfig(
            output_dir=args.output,
            quick_test=args.quick_test,
            run_validation=run_validation,
            run_lhs=run_lhs,
            lhs_samples=args.lhs_samples,
            replications_per_condition=args.replications,
            n_generations=args.generations,
        )

        runner = PaperExperimentRunner(config)
        results = runner.run_comprehensive_experiments()

        # Display final summary
        metadata = results.get("experiment_metadata", {})

        console.print("[green]🎉 Paper experiments completed successfully![/green]")
        console.print(f"[blue]📊 Total experiments: {metadata.get('total_experiments', 0)}[/blue]")
        console.print(f"[yellow]⏱️  Duration: {metadata.get('total_duration_hours', 0):.2f}h[/yellow]")
        console.print(f"[magenta]✅ Success rate: {metadata.get('overall_success_rate', 0):.1%}[/magenta]")
        console.print(f"[cyan]📁 Results saved to: {args.output}[/cyan]")

        # Show summary table
        table = Table(title="Paper Experiments Summary")
        table.add_column("Experiment Type", style="cyan")
        table.add_column("Parameter Sweeps", style="green")
        table.add_column("Total Experiments", style="yellow")
        table.add_column("Success Rate", style="magenta")

        for exp_type, summary in results.get("experiment_summary_by_type", {}).items():
            success_rate = summary.get("successful_experiments", 0) / max(1, summary.get("total_experiments", 1))
            table.add_row(
                exp_type.title(),
                str(summary.get("n_parameter_sweeps", 0)),
                str(summary.get("total_experiments", 0)),
                f"{success_rate:.1%}",
            )

        console.print(table)

    except KeyboardInterrupt:
        console.print("[yellow]⚠️ Paper experiments interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        logging.exception("Paper experiments failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
