#!/usr/bin/env python3
"""
Unified Experiment Runner for Evolutionary Simulations

Supports Layer1 (Lande-Kirkpatrick), Layer2 (Cultural transmission), and combined experiments
with built-in logging, progress tracking, and result organization.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
from beartype import beartype
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

try:
    from scipy.stats import qmc

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    qmc = None

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lovebug.lande_kirkpatrick import LandeKirkpatrickParams, simulate_lande_kirkpatrick
from lovebug.layer2.config import Layer2Config
from lovebug.layer2.social_learning.cultural_transmission import CulturalTransmissionManager
from lovebug.layer2.social_learning.social_networks import NetworkTopology, SocialNetwork

__all__ = ["UnifiedConfig", "UnifiedExperimentRunner", "run_experiments"]

console = Console()


@dataclass(slots=True, frozen=False)
class UnifiedConfig:
    """Unified configuration for all experiment types."""

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


def run_single_experiment(params_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a single experiment (Layer1, Layer2, or combined).

    Parameters
    ----------
    params_dict : dict[str, Any]
        Experiment parameters including type and configuration

    Returns
    -------
    dict[str, Any]
        Experiment results with success status and metrics
    """
    experiment_start = time.time()

    try:
        exp_type = params_dict["type"]
        exp_config = params_dict.get("config", {})

        if exp_type == "layer1":
            # Layer 1 experiments
            lk_params = {
                k: v
                for k, v in params_dict["params"].items()
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
            params = LandeKirkpatrickParams(**lk_params)
            results = simulate_lande_kirkpatrick(params)

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

            return {
                "experiment_type": "layer1",
                "name": exp_config.get("name", "unnamed"),
                "duration_seconds": time.time() - experiment_start,
                "final_trait": final_trait,
                "final_preference": final_preference,
                "final_covariance": final_covariance,
                "outcome": outcome,
                "generations": len(results),
                "success": True,
                "process_id": os.getpid(),
                **{k: v for k, v in params_dict["params"].items() if k not in ["expected_outcome"]},
            }

        elif exp_type == "layer2":
            # Layer 2 experiments
            all_params = params_dict["params"]
            valid_config_fields = set(Layer2Config.__dataclass_fields__.keys())

            # Filter parameters for Layer2Config
            layer2_params = {k: v for k, v in all_params.items() if k in valid_config_fields}
            layer2_config = Layer2Config(**layer2_params)

            # Create network and manager
            topology = NetworkTopology(
                network_type=layer2_config.network_type, connectivity=layer2_config.network_connectivity
            )

            n_agents = params_dict.get("n_agents", 2000)
            social_network = SocialNetwork(n_agents, topology)
            transmission_manager = CulturalTransmissionManager(layer2_config, social_network)

            # Agent data with process-specific seed
            seed = hash(f"{os.getpid()}_{time.time()}") % (2**32)
            agent_data = MockAgentData(n_agents, seed=seed)

            # Run simulation
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
                np.polyfit(range(len(diversity_samples)), diversity_samples, 1)[0]
                if len(diversity_samples) > 1
                else 0.0
            )

            # Classify cultural outcome
            if final_diversity > 0.7:
                cultural_outcome = "high_diversity"
            elif final_diversity < 0.2:
                cultural_outcome = "low_diversity"
            else:
                cultural_outcome = "moderate_diversity"

            return {
                "experiment_type": "layer2",
                "name": exp_config.get("name", "unnamed"),
                "duration_seconds": time.time() - experiment_start,
                "final_diversity": final_diversity,
                "diversity_trend": diversity_trend,
                "total_events": total_events,
                "cultural_outcome": cultural_outcome,
                "n_agents": n_agents,
                "generations_completed": n_generations,
                "success": True,
                "process_id": os.getpid(),
                **layer2_params,
            }

        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")

    except Exception as e:
        exp_type_for_error = params_dict.get("type", "unknown")
        exp_name_for_error = params_dict.get("config", {}).get("name", "unnamed")
        logging.error(f"Experiment {exp_name_for_error} (type: {exp_type_for_error}) failed: {e}", exc_info=True)
        return {
            "experiment_type": exp_type_for_error,
            "name": exp_name_for_error,
            "error": str(e),
            "duration_seconds": time.time() - experiment_start,
            "success": False,
            "process_id": os.getpid(),
        }


class UnifiedExperimentRunner:
    """
    Unified experiment runner for all experiment types.
    Includes built-in logging, progress tracking, and result organization.
    """

    def __init__(self, config: UnifiedConfig) -> None:
        self.config = config

        # Setup directories
        self.results_dir = Path(config.results_dir)
        self.logs_dir = Path(config.logs_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Experiment state
        self.start_time = time.time()
        self.experiments_completed = 0
        self.experiments_failed = 0
        self.total_experiments = 0
        self.should_stop = False

        # Results storage
        self.all_results: list[dict[str, Any]] = []
        self.results_lock = threading.Lock()

        # Resource monitoring
        self.peak_memory_gb = 0.0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info(f"Unified experiment runner initialized: {config.experiment_name}")
        self.logger.info(f"Workers: {config.n_workers}, Memory limit: {config.memory_limit_gb}GB")

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
        """Save experiment checkpoint."""
        checkpoint_path = self.results_dir / "checkpoint.json"

        with self.results_lock:
            checkpoint_data = {
                "experiment_name": self.config.experiment_name,
                "start_time": self.start_time,
                "experiments_completed": self.experiments_completed,
                "experiments_failed": self.experiments_failed,
                "total_experiments": self.total_experiments,
                "peak_memory_gb": self.peak_memory_gb,
                "results_count": len(self.all_results),
            }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        if self.all_results:
            try:
                results_df = pl.DataFrame(self.all_results)
                results_path = self.results_dir / "checkpoint_results.parquet"
                results_df.write_parquet(results_path)
                self.logger.info(f"Checkpoint saved: {len(self.all_results)} results")
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint results: {e}")

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

        self.logger.info(f"Generated {len(tasks)} experiments")
        return tasks

    @beartype
    def run_experiments(self) -> dict[str, Any]:
        """Run experiments with progress tracking and result organization."""

        # Generate experiments
        experiment_tasks = self.generate_experiments()
        self.total_experiments = len(experiment_tasks)

        self.logger.info(f"Starting {self.total_experiments} experiments with {self.config.n_workers} workers")

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
            main_task = progress.add_task("Experiments", total=self.total_experiments)

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.n_workers, mp_context=mp.get_context("spawn")
            ) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(run_single_experiment, task): task for task in experiment_tasks}

                last_checkpoint = time.time()

                # Process results
                for future in concurrent.futures.as_completed(future_to_task):
                    if self.should_stop:
                        break

                    try:
                        result = future.result()

                        with self.results_lock:
                            self.all_results.append(result)

                            if result.get("success", False):
                                self.experiments_completed += 1
                            else:
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
                        self.logger.error(f"Task processing failed: {e}")
                        self.experiments_failed += 1

        # Save final results
        return self._save_final_results()

    def _save_final_results(self) -> dict[str, Any]:
        """Save final results and generate summary."""
        end_time = time.time()
        duration_hours = (end_time - self.start_time) / 3600

        # Create summary
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
            "config": {
                "n_workers": self.config.n_workers,
                "stochastic_replications": self.config.stochastic_replications,
            },
        }

        # Save results by experiment type
        if self.all_results:
            results_df = pl.DataFrame(self.all_results)

            # Save to appropriate subdirectory
            if self.config.experiment_type == "layer1":
                results_path = (
                    self.results_dir / "layer1" / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                )
            elif self.config.experiment_type == "layer2":
                results_path = (
                    self.results_dir / "layer2" / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                )
            else:  # combined
                results_path = (
                    self.results_dir / "combined" / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                )

            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.write_parquet(results_path)

            # Save summary
            summary_path = results_path.with_suffix(".json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Experiments completed: {self.experiments_completed}/{self.total_experiments}")
        self.logger.info(f"Duration: {duration_hours:.2f}h, Peak memory: {self.peak_memory_gb:.1f}GB")

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

    runner = UnifiedExperimentRunner(config)
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
