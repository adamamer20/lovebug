#!/usr/bin/env python3
"""
Clean Experiment Runner for Evolutionary Simulations

Refactored to use type-safe data models and eliminate null pollution.
Supports Layer1 (Lande-Kirkpatrick), Layer2 (Cultural transmission), and combined experiments
with clean data storage and built-in analysis capabilities.

Usage:
    uv run python experiments/runner.py --config experiments/config.toml
    uv run python experiments/runner.py --config experiments/config.toml --type layer1
    uv run python experiments/runner.py --config experiments/config.toml --workers 16 --hours 6
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
from beartype import beartype
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Add parent directory to path so 'experiments' module can be imported
# This is needed when running the script directly from the experiments directory
experiments_parent = str(Path(__file__).parent.parent)
if experiments_parent not in sys.path:
    sys.path.insert(0, experiments_parent)

try:
    from scipy.stats import qmc

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    qmc = None

# Import our clean data models
from experiments.collectors import ExperimentStorage  # noqa: E402
from experiments.models import (  # noqa: E402
    CommonParameters,
    CulturalExperimentResult,
    ExperimentMetadata,
    GeneticExperimentResult,
    IntegratedExperimentResult,
)
from lovebug.lande_kirkpatrick import LandeKirkpatrickParams, simulate_lande_kirkpatrick  # noqa: E402
from lovebug.layer2.config import Layer2Config  # noqa: E402
from lovebug.layer2.social_learning.cultural_transmission import CulturalTransmissionManager  # noqa: E402
from lovebug.layer2.social_learning.social_networks import NetworkTopology, SocialNetwork  # noqa: E402
from lovebug.layer_activation import LayerActivationConfig  # noqa: E402
from lovebug.unified_mesa_model import UnifiedLoveModel  # noqa: E402

__all__ = ["UnifiedConfig", "CleanExperimentRunner", "run_experiments"]

console = Console()
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=False)
class UnifiedConfig:
    """Unified configuration for all experiment types with layer activation support."""

    # Runner configuration
    experiment_type: str = "combined"
    experiment_name: str = "unified_evolution_study"
    n_workers: int | None = None
    max_duration_hours: float = 12.0
    memory_limit_gb: float = 75.0
    stochastic_replications: int = 10
    checkpoint_interval_minutes: int = 5
    smart_sampling: bool = True
    sampling_method: str = "lhs"
    max_samples_per_sweep: int = 5000

    # Output configuration
    results_dir: str = "experiments/results"
    logs_dir: str = "experiments/logs/current"
    batch_size: int = 200

    # Layer activation configuration (new)
    genetic_enabled: bool = True
    cultural_enabled: bool = True
    genetic_weight: float = 0.5
    cultural_weight: float = 0.5
    blending_mode: str = "weighted_average"
    normalize_weights: bool = True

    # Parameter sweep configuration (new)
    enable_parameter_sweeps: bool = False
    sweep_genetic_weights: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    sweep_cultural_weights: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    sweep_replications_per_combo: int = 5

    # Layer1 (Lande-Kirkpatrick) parameters
    layer1_n_generations: int = 2000
    layer1_pop_size: int = 5000
    layer1_h2_trait: float = 0.5
    layer1_h2_preference: float = 0.5
    layer1_genetic_correlation: float = 0.2
    layer1_selection_strength: float = 0.1
    layer1_preference_cost: float = 0.05
    layer1_mutation_variance: float = 0.01

    # Layer2 (Cultural transmission) parameters
    layer2_innovation_rate: float = 0.1
    layer2_horizontal_transmission_rate: float = 0.3
    layer2_oblique_transmission_rate: float = 0.2
    layer2_network_type: str = "scale_free"
    layer2_network_connectivity: float = 0.1
    layer2_cultural_memory_size: int = 10
    layer2_n_agents: int = 2000

    def __post_init__(self) -> None:
        if self.n_workers is None:
            self.n_workers = max(1, int(mp.cpu_count() * 0.9))

        # Validate layer activation configuration
        self._validate_layer_activation()

    def _validate_layer_activation(self) -> None:
        """Validate layer activation parameters."""
        if not self.genetic_enabled and not self.cultural_enabled:
            raise ValueError("At least one layer must be enabled")

        if not (0.0 <= self.genetic_weight <= 1.0):
            raise ValueError(f"genetic_weight must be in [0.0, 1.0], got {self.genetic_weight}")
        if not (0.0 <= self.cultural_weight <= 1.0):
            raise ValueError(f"cultural_weight must be in [0.0, 1.0], got {self.cultural_weight}")

        valid_modes = {"weighted_average", "probabilistic", "competitive"}
        if self.blending_mode not in valid_modes:
            raise ValueError(f"blending_mode must be one of {valid_modes}, got {self.blending_mode}")

    def get_layer_activation_config(self) -> LayerActivationConfig:
        """
        Create LayerActivationConfig from current settings.

        Returns
        -------
        LayerActivationConfig
            Layer activation configuration instance
        """
        return LayerActivationConfig(
            genetic_enabled=self.genetic_enabled,
            cultural_enabled=self.cultural_enabled,
            genetic_weight=self.genetic_weight,
            cultural_weight=self.cultural_weight,
            blending_mode=self.blending_mode,  # type: ignore[arg-type]
            normalize_weights=self.normalize_weights,
        )


class MockAgentData:
    """Optimized mock agent data for large-scale experiments."""

    def __init__(self, n_agents: int = 2000, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.n_agents = n_agents
        self.cultural_preferences = np.random.randint(0, 256, n_agents, dtype=np.uint8)
        self.genetic_preferences = np.random.randint(0, 256, n_agents, dtype=np.uint8)

    def get_cultural_preferences(self) -> np.ndarray:
        return self.cultural_preferences

    def get_genetic_preferences(self) -> np.ndarray:
        return self.genetic_preferences

    def get_mating_success(self) -> np.ndarray:
        return np.random.exponential(1.0, self.n_agents).astype(np.float32)

    def get_ages(self) -> np.ndarray:
        return np.random.randint(0, 100, self.n_agents, dtype=np.int16)

    def get_agent_ids(self) -> np.ndarray:
        return np.arange(self.n_agents)

    def update_cultural_preference(self, agent_id: int, new_preference: int) -> None:
        if 0 <= agent_id < self.n_agents:
            self.cultural_preferences[agent_id] = new_preference


@beartype
def run_genetic_experiment(params_dict: dict[str, Any]) -> GeneticExperimentResult:
    """
    Execute a single genetic evolution experiment with clean results.

    Parameters
    ----------
    params_dict : dict[str, Any]
        Experiment parameters including configuration and genetic parameters

    Returns
    -------
    GeneticExperimentResult
        Type-safe genetic experiment result with no null fields

    Raises
    ------
    ValueError
        If required parameters are missing or invalid
    Exception
        If simulation fails for any reason
    """
    experiment_start = time.time()
    start_time = datetime.now()

    try:
        exp_config = params_dict.get("config", {})
        all_params = params_dict["params"]

        # Extract parameters for Lande-Kirkpatrick model
        lk_params = {
            k: v
            for k, v in all_params.items()
            if k
            in {
                "n_generations",
                "pop_size",
                "h2_trait",
                "h2_preference",
                "selection_strength",
                "genetic_correlation",
                "mutation_variance",
                "preference_cost",
            }
        }

        # Map pop_size to population_size for consistency
        if "pop_size" in lk_params:
            lk_params["pop_size"] = lk_params.pop("pop_size", 2000)

        # Run genetic simulation
        lk_model_params = LandeKirkpatrickParams(**lk_params)
        results = simulate_lande_kirkpatrick(lk_model_params)

        # Extract final values
        final_trait = float(results.select(pl.col("mean_trait").last()).item())
        final_preference = float(results.select(pl.col("mean_preference").last()).item())
        final_covariance = float(results.select(pl.col("genetic_covariance").last()).item())

        # Classify outcome
        if abs(final_covariance) > 0.3:
            outcome = "runaway"
        elif abs(final_covariance) < 0.01:
            outcome = "stasis"
        else:
            outcome = "equilibrium"

        # Create clean result object
        experiment_id = str(uuid.uuid4())[:8]

        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=exp_config.get("name", f"genetic_{experiment_id}"),
            experiment_type="genetic",
            start_time=start_time,
            duration_seconds=time.time() - experiment_start,
            success=True,
            process_id=os.getpid(),
        )

        common_params = CommonParameters(
            n_generations=lk_model_params.n_generations,
            population_size=lk_model_params.pop_size,
            random_seed=params_dict.get("random_seed"),
        )

        return GeneticExperimentResult(
            metadata=metadata,
            common_params=common_params,
            final_trait=final_trait,
            final_preference=final_preference,
            final_covariance=final_covariance,
            outcome=outcome,
            generations_completed=len(results),
            h2_trait=lk_model_params.h2_trait,
            h2_preference=lk_model_params.h2_preference,
            genetic_correlation=lk_model_params.genetic_correlation,
            selection_strength=lk_model_params.selection_strength,
            preference_cost=lk_model_params.preference_cost,
            mutation_variance=lk_model_params.mutation_variance,
        )

    except Exception as e:
        logger.exception(f"Genetic experiment failed: {e}")
        raise


@beartype
def run_cultural_experiment(params_dict: dict[str, Any]) -> CulturalExperimentResult:
    """
    Execute a single cultural evolution experiment with clean results.

    Parameters
    ----------
    params_dict : dict[str, Any]
        Experiment parameters including configuration and cultural parameters

    Returns
    -------
    CulturalExperimentResult
        Type-safe cultural experiment result with no null fields

    Raises
    ------
    ValueError
        If required parameters are missing or invalid
    Exception
        If simulation fails for any reason
    """
    experiment_start = time.time()
    start_time = datetime.now()

    try:
        exp_config = params_dict.get("config", {})
        all_params = params_dict["params"]

        # Extract Layer2 config parameters
        valid_config_fields = set(Layer2Config.__dataclass_fields__.keys())
        layer2_params = {k: v for k, v in all_params.items() if k in valid_config_fields}
        layer2_config = Layer2Config(**layer2_params)

        # Setup cultural simulation
        topology = NetworkTopology(
            network_type=layer2_config.network_type, connectivity=layer2_config.network_connectivity
        )

        n_agents = params_dict.get("n_agents", 2000)
        social_network = SocialNetwork(n_agents, topology)
        transmission_manager = CulturalTransmissionManager(layer2_config, social_network)

        # Agent data with process-specific seed
        seed = hash(f"{os.getpid()}_{time.time()}") % (2**32)
        agent_data = MockAgentData(n_agents, seed=seed)

        # Run cultural simulation
        n_generations = params_dict.get("n_generations", 1000)
        diversity_samples = []
        total_events = 0

        for generation in range(n_generations):
            events = transmission_manager.process_cultural_learning(agent_data, generation)
            total_events += len(events)

            if generation % 50 == 0:
                cultural_diversity = len(np.unique(agent_data.get_cultural_preferences())) / 256.0
                diversity_samples.append(cultural_diversity)

        # Final analysis
        final_diversity = diversity_samples[-1] if diversity_samples else 0.0
        diversity_trend = (
            np.polyfit(range(len(diversity_samples)), diversity_samples, 1)[0] if len(diversity_samples) > 1 else 0.0
        )

        # Classify cultural outcome
        if final_diversity > 0.7:
            cultural_outcome = "high_diversity"
        elif final_diversity < 0.2:
            cultural_outcome = "low_diversity"
        else:
            cultural_outcome = "moderate_diversity"

        # Create clean result object
        experiment_id = str(uuid.uuid4())[:8]

        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=exp_config.get("name", f"cultural_{experiment_id}"),
            experiment_type="cultural",
            start_time=start_time,
            duration_seconds=time.time() - experiment_start,
            success=True,
            process_id=os.getpid(),
        )

        common_params = CommonParameters(
            n_generations=n_generations, population_size=n_agents, random_seed=params_dict.get("random_seed")
        )

        return CulturalExperimentResult(
            metadata=metadata,
            common_params=common_params,
            final_diversity=final_diversity,
            diversity_trend=diversity_trend,
            total_events=total_events,
            cultural_outcome=cultural_outcome,
            generations_completed=n_generations,
            innovation_rate=layer2_config.innovation_rate,
            horizontal_transmission_rate=layer2_config.horizontal_transmission_rate,
            oblique_transmission_rate=layer2_config.oblique_transmission_rate,
            network_type=layer2_config.network_type,
            network_connectivity=layer2_config.network_connectivity,
            cultural_memory_size=layer2_config.cultural_memory_size,
        )

    except Exception as e:
        logger.exception(f"Cultural experiment failed: {e}")
        raise


@beartype
def run_combined_experiment(params_dict: dict[str, Any]) -> IntegratedExperimentResult:
    """
    Execute a combined genetic+cultural evolution experiment.

    Parameters
    ----------
    params_dict : dict[str, Any]
        Experiment parameters including layer activation config and both genetic/cultural parameters

    Returns
    -------
    IntegratedExperimentResult
        Type-safe integrated experiment result with both genetic and cultural components

    Raises
    ------
    ValueError
        If required parameters are missing or invalid
    Exception
        If simulation fails for any reason
    """
    experiment_start = time.time()
    start_time = datetime.now()

    try:
        exp_config = params_dict.get("config", {})
        all_params = params_dict["params"]

        # Create layer activation config from parameters
        layer_config = LayerActivationConfig(
            genetic_enabled=all_params.get("genetic_enabled", True),
            cultural_enabled=all_params.get("cultural_enabled", True),
            genetic_weight=all_params.get("genetic_weight", 0.5),
            cultural_weight=all_params.get("cultural_weight", 0.5),
            blending_mode=all_params.get("blending_mode", "weighted_average"),
            normalize_weights=all_params.get("normalize_weights", True),
            theta_detect=all_params.get("theta_detect", 8.0),
            sigma_perception=all_params.get("sigma_perception", 2.0),
        )

        # Extract genetic parameters
        genetic_params = None
        if layer_config.genetic_enabled:
            lk_params = {
                k: v
                for k, v in all_params.items()
                if k
                in {
                    "n_generations",
                    "pop_size",
                    "h2_trait",
                    "h2_preference",
                    "selection_strength",
                    "genetic_correlation",
                    "mutation_variance",
                    "preference_cost",
                }
            }
            if "pop_size" in lk_params:
                lk_params["pop_size"] = lk_params.get("pop_size", 2000)
            genetic_params = LandeKirkpatrickParams(**lk_params)

        # Extract cultural parameters
        cultural_params = None
        if layer_config.cultural_enabled:
            valid_config_fields = set(Layer2Config.__dataclass_fields__.keys())
            layer2_params = {k: v for k, v in all_params.items() if k in valid_config_fields}
            cultural_params = Layer2Config(**layer2_params)

        # Run unified simulation
        n_agents = params_dict.get("n_agents", 2000)
        n_generations = params_dict.get("n_generations", 1000)

        unified_model = UnifiedLoveModel(
            layer_config=layer_config,
            genetic_params=genetic_params,
            cultural_params=cultural_params,
            n_agents=n_agents,
        )

        # Execute simulation
        results = unified_model.run(n_generations)

        # Extract final metrics for creating component results
        final_metrics = results.get("final_metrics", {})
        trajectory = results.get("trajectory", [])

        # Create genetic component result
        genetic_component = GeneticExperimentResult(
            metadata=ExperimentMetadata(
                experiment_id=f"genetic_{str(uuid.uuid4())[:8]}",
                name=f"genetic_component_{exp_config.get('name', 'unknown')}",
                experiment_type="genetic",
                start_time=start_time,
                duration_seconds=time.time() - experiment_start,
                success=True,
                process_id=os.getpid(),
            ),
            common_params=CommonParameters(
                n_generations=n_generations,
                population_size=n_agents,
                random_seed=params_dict.get("random_seed"),
            ),
            final_trait=final_metrics.get("mean_genetic_trait", 0.0),
            final_preference=final_metrics.get("mean_genetic_preference", 128.0),
            final_covariance=final_metrics.get("genetic_covariance", 0.0),
            outcome="equilibrium",  # Could be determined from trajectory analysis
            generations_completed=n_generations,
            h2_trait=genetic_params.h2_trait if genetic_params else 0.5,
            h2_preference=genetic_params.h2_preference if genetic_params else 0.5,
            genetic_correlation=genetic_params.genetic_correlation if genetic_params else 0.2,
            selection_strength=genetic_params.selection_strength if genetic_params else 0.1,
            preference_cost=genetic_params.preference_cost if genetic_params else 0.05,
            mutation_variance=genetic_params.mutation_variance if genetic_params else 0.01,
        )

        # Create cultural component result
        cultural_component = CulturalExperimentResult(
            metadata=ExperimentMetadata(
                experiment_id=f"cultural_{str(uuid.uuid4())[:8]}",
                name=f"cultural_component_{exp_config.get('name', 'unknown')}",
                experiment_type="cultural",
                start_time=start_time,
                duration_seconds=time.time() - experiment_start,
                success=True,
                process_id=os.getpid(),
            ),
            common_params=CommonParameters(
                n_generations=n_generations,
                population_size=n_agents,
                random_seed=params_dict.get("random_seed"),
            ),
            final_diversity=final_metrics.get("cultural_diversity", 0.5),
            diversity_trend=0.0,  # Could be calculated from trajectory
            total_events=sum(h.get("cultural_learning_events", 0) for h in trajectory),
            cultural_outcome="moderate_diversity",  # Could be determined from final_diversity
            generations_completed=n_generations,
            innovation_rate=cultural_params.innovation_rate if cultural_params else 0.1,
            horizontal_transmission_rate=cultural_params.horizontal_transmission_rate if cultural_params else 0.3,
            oblique_transmission_rate=cultural_params.oblique_transmission_rate if cultural_params else 0.2,
            network_type=cultural_params.network_type if cultural_params else "scale_free",
            network_connectivity=cultural_params.network_connectivity if cultural_params else 0.1,
            cultural_memory_size=cultural_params.cultural_memory_size if cultural_params else 10,
        )

        # Calculate interaction metrics
        gene_culture_correlation = final_metrics.get("gene_culture_correlation", 0.0)
        interaction_strength = layer_config.genetic_weight * layer_config.cultural_weight

        # Collect emergent properties from simulation
        emergent_properties = {
            "effective_preference_variance": final_metrics.get("var_effective_preference", 0.0),
            "gene_culture_distance": final_metrics.get("gene_culture_distance", 0.0),
            "population_stability": 1.0 if final_metrics.get("population_size", 0) > n_agents * 0.1 else 0.0,
            "blending_efficiency": interaction_strength,
            "perceptual_constraint_effect": layer_config.theta_detect / 16.0,  # Normalized effect
        }

        # Create integrated result
        experiment_id = str(uuid.uuid4())[:8]

        integrated_result = IntegratedExperimentResult(
            metadata=ExperimentMetadata(
                experiment_id=experiment_id,
                name=exp_config.get("name", f"integrated_{experiment_id}"),
                experiment_type="integrated",
                start_time=start_time,
                duration_seconds=time.time() - experiment_start,
                success=True,
                process_id=os.getpid(),
            ),
            common_params=CommonParameters(
                n_generations=n_generations,
                population_size=n_agents,
                random_seed=params_dict.get("random_seed"),
            ),
            genetic_component=genetic_component,
            cultural_component=cultural_component,
            gene_culture_correlation=gene_culture_correlation,
            interaction_strength=interaction_strength,
            emergent_properties=emergent_properties,
        )

        return integrated_result

    except Exception as e:
        logger.exception(f"Combined experiment failed: {e}")
        raise


def run_single_experiment(
    params_dict: dict[str, Any],
) -> GeneticExperimentResult | CulturalExperimentResult | IntegratedExperimentResult:
    """
    Execute a single experiment and return properly typed result.

    This function routes to the appropriate experiment type and returns
    clean, type-safe results with no null pollution.

    Parameters
    ----------
    params_dict : dict[str, Any]
        Experiment parameters including type and configuration

    Returns
    -------
    GeneticExperimentResult | CulturalExperimentResult
        Type-safe experiment result

    Raises
    ------
    ValueError
        If experiment type is unknown
    """
    exp_type = params_dict["type"]

    if exp_type == "layer1":
        return run_genetic_experiment(params_dict)
    elif exp_type == "layer2":
        return run_cultural_experiment(params_dict)
    elif exp_type == "combined":
        return run_combined_experiment(params_dict)
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")


class CleanExperimentRunner:
    """
    Clean experiment runner using type-safe storage and eliminating null pollution.

    This runner uses separate collectors for each experiment type and provides
    clean, analysis-ready data storage.
    """

    def __init__(self, config: UnifiedConfig) -> None:
        self.config = config

        # Setup directories and clean storage
        self.results_dir = Path(config.results_dir)
        self.logs_dir = Path(config.logs_dir)
        self.storage = ExperimentStorage(self.results_dir)

        # Setup logging
        self.setup_logging()

        # Experiment state
        self.start_time = time.time()
        self.experiments_completed = 0
        self.experiments_failed = 0
        self.total_experiments = 0
        self.should_stop = False

        # Thread safety
        self.results_lock = threading.Lock()

        # Resource monitoring
        self.peak_memory_gb = 0.0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info(f"Clean experiment runner initialized: {config.experiment_name}")
        self.logger.info(f"Workers: {config.n_workers}, Memory limit: {config.memory_limit_gb}GB")
        self.logger.info("Using clean data architecture - zero null pollution!")

    def setup_logging(self) -> None:
        """Setup comprehensive logging to file and console."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"experiments_{timestamp}.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - [%(processName)s] - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized - log file: {log_file}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True

    def _check_resources(self) -> bool:
        """Monitor system resources."""
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.used / 1e9
        self.peak_memory_gb = max(self.peak_memory_gb, memory_gb)

        if memory_gb > self.config.memory_limit_gb:
            self.logger.warning(f"Memory limit exceeded: {memory_gb:.1f}GB")
            return False

        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours > self.config.max_duration_hours:
            self.logger.info(f"Time limit reached: {elapsed_hours:.1f}h")
            return False

        return True

    def _save_checkpoint(self) -> None:
        """Save experiment checkpoint with clean data."""
        checkpoint_path = self.results_dir / "checkpoint.json"

        with self.results_lock:
            result_counts = self.storage.get_total_results()
            checkpoint_data = {
                "experiment_name": self.config.experiment_name,
                "start_time": self.start_time,
                "experiments_completed": self.experiments_completed,
                "experiments_failed": self.experiments_failed,
                "total_experiments": self.total_experiments,
                "peak_memory_gb": self.peak_memory_gb,
                "results_by_type": result_counts,
                "total_results": sum(result_counts.values()),
            }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Save intermediate results using clean storage
        if sum(result_counts.values()) > 0:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_timestamp = f"checkpoint_{timestamp}"
                saved_files = self.storage.save_all(checkpoint_timestamp)
                self.logger.info(f"Checkpoint saved: {result_counts} -> {len(saved_files)} clean files")
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint results: {e}", exc_info=True)

    def generate_experiments(self) -> list[dict[str, Any]]:
        """Generate experiment tasks based on configuration."""
        tasks = []

        if self.config.experiment_type in ["layer1", "combined"]:
            # Layer 1 experiments
            for i in range(self.config.stochastic_replications):
                tasks.append(
                    {
                        "type": "layer1",
                        "config": {"name": f"layer1_exp_{i:04d}"},
                        "params": {
                            "n_generations": self.config.layer1_n_generations,
                            "pop_size": self.config.layer1_pop_size,
                            "h2_trait": self.config.layer1_h2_trait,
                            "h2_preference": self.config.layer1_h2_preference,
                            "genetic_correlation": self.config.layer1_genetic_correlation,
                            "selection_strength": self.config.layer1_selection_strength,
                            "preference_cost": self.config.layer1_preference_cost,
                            "mutation_variance": self.config.layer1_mutation_variance,
                        },
                    }
                )

        if self.config.experiment_type in ["layer2", "combined"]:
            # Layer 2 experiments
            for i in range(self.config.stochastic_replications):
                tasks.append(
                    {
                        "type": "layer2",
                        "config": {"name": f"layer2_exp_{i:04d}"},
                        "params": {
                            "innovation_rate": self.config.layer2_innovation_rate,
                            "horizontal_transmission_rate": self.config.layer2_horizontal_transmission_rate,
                            "oblique_transmission_rate": self.config.layer2_oblique_transmission_rate,
                            "network_type": self.config.layer2_network_type,
                            "network_connectivity": self.config.layer2_network_connectivity,
                            "cultural_memory_size": self.config.layer2_cultural_memory_size,
                        },
                        "n_agents": self.config.layer2_n_agents,
                        "n_generations": 1000,
                    }
                )

        # Combined experiments with parameter sweeps
        if self.config.experiment_type == "combined":
            if self.config.enable_parameter_sweeps:
                # Generate parameter sweep combinations
                for genetic_weight in self.config.sweep_genetic_weights:
                    for cultural_weight in self.config.sweep_cultural_weights:
                        # Normalize weights if needed
                        if self.config.normalize_weights:
                            total = genetic_weight + cultural_weight
                            if total > 0:
                                genetic_weight = genetic_weight / total
                                cultural_weight = cultural_weight / total

                        for _rep in range(self.config.sweep_replications_per_combo):
                            # Test different theoretical mechanism parameters
                            for theta_detect in [4.0, 8.0, 12.0]:  # Detection thresholds
                                for sigma_perception in [1.0, 2.0, 3.0]:  # Perceptual noise
                                    for local_radius in [3, 5, 10]:  # Local learning radius
                                        tasks.append(
                                            {
                                                "type": "combined",
                                                "config": {"name": f"combined_sweep_{len(tasks):05d}"},
                                                "params": {
                                                    # Layer activation parameters
                                                    "genetic_enabled": True,
                                                    "cultural_enabled": True,
                                                    "genetic_weight": genetic_weight,
                                                    "cultural_weight": cultural_weight,
                                                    "blending_mode": self.config.blending_mode,
                                                    "normalize_weights": False,  # Already normalized
                                                    "theta_detect": theta_detect,
                                                    "sigma_perception": sigma_perception,
                                                    # Genetic parameters
                                                    "n_generations": self.config.layer1_n_generations,
                                                    "pop_size": self.config.layer1_pop_size,
                                                    "h2_trait": self.config.layer1_h2_trait,
                                                    "h2_preference": self.config.layer1_h2_preference,
                                                    "genetic_correlation": self.config.layer1_genetic_correlation,
                                                    "selection_strength": self.config.layer1_selection_strength,
                                                    "preference_cost": self.config.layer1_preference_cost,
                                                    "mutation_variance": self.config.layer1_mutation_variance,
                                                    # Cultural parameters
                                                    "innovation_rate": self.config.layer2_innovation_rate,
                                                    "horizontal_transmission_rate": self.config.layer2_horizontal_transmission_rate,
                                                    "oblique_transmission_rate": self.config.layer2_oblique_transmission_rate,
                                                    "network_type": self.config.layer2_network_type,
                                                    "network_connectivity": self.config.layer2_network_connectivity,
                                                    "cultural_memory_size": self.config.layer2_cultural_memory_size,
                                                    "local_learning_radius": local_radius,
                                                },
                                                "n_agents": self.config.layer2_n_agents,
                                                "n_generations": min(self.config.layer1_n_generations, 1000),
                                            }
                                        )
            else:
                # Standard combined experiments without parameter sweeps
                for i in range(self.config.stochastic_replications):
                    tasks.append(
                        {
                            "type": "combined",
                            "config": {"name": f"combined_exp_{i:04d}"},
                            "params": {
                                # Layer activation parameters
                                "genetic_enabled": self.config.genetic_enabled,
                                "cultural_enabled": self.config.cultural_enabled,
                                "genetic_weight": self.config.genetic_weight,
                                "cultural_weight": self.config.cultural_weight,
                                "blending_mode": self.config.blending_mode,
                                "normalize_weights": self.config.normalize_weights,
                                "theta_detect": 8.0,  # Default theoretical mechanism parameters
                                "sigma_perception": 2.0,
                                # Genetic parameters
                                "n_generations": self.config.layer1_n_generations,
                                "pop_size": self.config.layer1_pop_size,
                                "h2_trait": self.config.layer1_h2_trait,
                                "h2_preference": self.config.layer1_h2_preference,
                                "genetic_correlation": self.config.layer1_genetic_correlation,
                                "selection_strength": self.config.layer1_selection_strength,
                                "preference_cost": self.config.layer1_preference_cost,
                                "mutation_variance": self.config.layer1_mutation_variance,
                                # Cultural parameters
                                "innovation_rate": self.config.layer2_innovation_rate,
                                "horizontal_transmission_rate": self.config.layer2_horizontal_transmission_rate,
                                "oblique_transmission_rate": self.config.layer2_oblique_transmission_rate,
                                "network_type": self.config.layer2_network_type,
                                "network_connectivity": self.config.layer2_network_connectivity,
                                "cultural_memory_size": self.config.layer2_cultural_memory_size,
                                "local_learning_radius": 5,  # Default local learning radius
                            },
                            "n_agents": self.config.layer2_n_agents,
                            "n_generations": min(self.config.layer1_n_generations, 1000),
                        }
                    )

        self.logger.info(f"Generated {len(tasks)} experiments")
        return tasks

    @beartype
    def run_experiments(self) -> dict[str, Any]:
        """Run experiments with clean data storage and progress tracking.

        Returns
        -------
        dict[str, Any]
            Summary of experiment run with statistics and saved file paths
        """

        # Generate experiments
        experiment_tasks = self.generate_experiments()
        self.total_experiments = len(experiment_tasks)

        self.logger.info(f"Starting {self.total_experiments} experiments with {self.config.n_workers} workers")
        self.logger.info("Using clean architecture - no null pollution!")

        # Progress tracking
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            main_task = progress.add_task("Clean Experiments", total=self.total_experiments)

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.n_workers, mp_context=mp.get_context("spawn")
            ) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(run_single_experiment, task): task for task in experiment_tasks}

                last_checkpoint = time.time()

                # Process results with type-safe storage
                for future in concurrent.futures.as_completed(future_to_task):
                    if self.should_stop:
                        break

                    try:
                        result = future.result()

                        # Store result in appropriate type-safe collector
                        with self.results_lock:
                            if isinstance(result, GeneticExperimentResult):
                                self.storage.store_genetic_result(result)
                                self.experiments_completed += 1
                            elif isinstance(result, CulturalExperimentResult):
                                self.storage.store_cultural_result(result)
                                self.experiments_completed += 1
                            elif isinstance(result, IntegratedExperimentResult):
                                self.storage.store_integrated_result(result)
                                self.experiments_completed += 1
                            else:
                                self.logger.error(f"Unknown result type: {type(result)}")
                                self.experiments_failed += 1

                        progress.advance(main_task)

                        # Checkpoint management
                        now = time.time()
                        if now - last_checkpoint > self.config.checkpoint_interval_minutes * 60:
                            self._save_checkpoint()
                            last_checkpoint = now

                        # Resource checks
                        if not self._check_resources():
                            self.logger.warning("Resource limits exceeded")
                            self.should_stop = True
                            break

                    except Exception as e:
                        self.logger.error(f"Task processing failed: {e}", exc_info=True)
                        self.experiments_failed += 1

        # Save final results
        return self._save_final_results()

    def _save_final_results(self) -> dict[str, Any]:
        """Save final results using clean data architecture and generate summary."""
        end_time = time.time()
        duration_hours = (end_time - self.start_time) / 3600

        # Get result counts from clean storage
        result_counts = self.storage.get_total_results()
        total_results = sum(result_counts.values())

        # Create comprehensive summary
        summary = {
            "experiment_name": self.config.experiment_name,
            "experiment_type": self.config.experiment_type,
            "start_time": self.start_time,
            "end_time": end_time,
            "total_duration_hours": duration_hours,
            "total_experiments": self.total_experiments,
            "completed_experiments": self.experiments_completed,
            "failed_experiments": self.experiments_failed,
            "success_rate": self.experiments_completed / max(1, self.total_experiments),
            "peak_memory_gb": self.peak_memory_gb,
            "results_by_type": result_counts,
            "total_clean_results": total_results,
            "config": {
                "n_workers": self.config.n_workers,
                "stochastic_replications": self.config.stochastic_replications,
            },
            "architecture": "clean_data_v1.0",  # Mark as clean architecture
        }

        # Save all results using clean storage
        if total_results > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = self.storage.save_all(timestamp)
            summary["saved_files"] = {k: str(v) for k, v in saved_files.items()}

            self.logger.info(f"✅ Clean results saved: {result_counts}")
            self.logger.info(f"📁 Files: {list(saved_files.keys())}")
        else:
            summary["saved_files"] = {}
            self.logger.warning("No results to save")

        self.logger.info(f"Experiments completed: {self.experiments_completed}/{self.total_experiments}")
        self.logger.info(f"Duration: {duration_hours:.2f}h, Peak memory: {self.peak_memory_gb:.1f}GB")
        self.logger.info("🎉 Clean data architecture - zero null pollution achieved!")

        return summary


@beartype
def load_config(config_path: str) -> UnifiedConfig:
    """Load configuration from TOML file."""
    import tomllib

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "rb") as f:
        config_dict = tomllib.load(f)

    # Flatten nested configuration
    flattened = {}

    # Runner section
    if "runner" in config_dict:
        flattened.update(config_dict["runner"])

    # Output section
    if "output" in config_dict:
        flattened.update(config_dict["output"])

    # Layer1 section (prefix with layer1_)
    if "layer1" in config_dict:
        for k, v in config_dict["layer1"].items():
            flattened[f"layer1_{k}"] = v

    # Layer2 section (prefix with layer2_)
    if "layer2" in config_dict:
        for k, v in config_dict["layer2"].items():
            flattened[f"layer2_{k}"] = v

    return UnifiedConfig(**flattened)


@beartype
def run_experiments(config_path: str, **overrides: Any) -> dict[str, Any]:
    """
    Run experiments from configuration with optional overrides.

    Parameters
    ----------
    config_path : str
        Path to configuration file
    **overrides : Any
        Configuration overrides

    Returns
    -------
    dict[str, Any]
        Experiment summary results
    """
    config = load_config(config_path)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    runner = CleanExperimentRunner(config)
    return runner.run_experiments()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for evolutionary simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config experiments/config.toml
  %(prog)s --config experiments/config.toml --type layer1
  %(prog)s --config experiments/config.toml --workers 16 --hours 6
        """,
    )

    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--type", choices=["layer1", "layer2", "combined"], help="Experiment type override")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--hours", type=float, help="Maximum duration in hours")
    parser.add_argument("--replications", type=int, help="Stochastic replications")

    args = parser.parse_args()

    try:
        # Build overrides
        overrides = {}
        if args.type:
            overrides["experiment_type"] = args.type
        if args.workers:
            overrides["n_workers"] = args.workers
        if args.hours:
            overrides["max_duration_hours"] = args.hours
        if args.replications:
            overrides["stochastic_replications"] = args.replications

        # Run experiments
        results = run_experiments(args.config, **overrides)

        # Display results
        success_rate = results["success_rate"]
        console.print("[green]✅ Experiments completed![/green]")
        console.print(
            f"[blue]📊 {results['completed_experiments']}/{results['total_experiments']} experiments ({success_rate:.1%} success)[/blue]"
        )
        console.print(f"[yellow]⏱️  Duration: {results['total_duration_hours']:.2f}h[/yellow]")
        console.print(f"[magenta]💾 Peak memory: {results['peak_memory_gb']:.1f}GB[/magenta]")

    except KeyboardInterrupt:
        console.print("[yellow]⚠️  Experiments interrupted - results saved[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        logging.exception("Experiment runner failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
