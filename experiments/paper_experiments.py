#!/usr/bin/env python3
"""
Validated Paper Experiments for Evolutionary Simulations

This script runs systematic parameter sweeps using the new Pydantic-based
validation system. It replaces the fragile dictionary-based approach with
type-safe, validated experiment configurations that prevent population
explosions and parameter errors.

Key improvements over the old system:
- Type-safe Pydantic models with comprehensive validation
- Guaranteed population control (carrying_capacity validation)
- Clear error messages for invalid configurations
- No more dictionary manipulation or manual parameter extraction
- Factory functions for common experiment scenarios

Usage:
    uv run python experiments/paper_experiments.py --output experiments/results/paper_data
    uv run python experiments/paper_experiments.py --quick-test  # Fast validation
    uv run python experiments/paper_experiments.py --run-lhs --lhs-samples 100  # LHS exploration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from beartype import beartype
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from scipy.stats import qmc

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the validated experiment system
from experiments.config_models import (
    CombinedExperimentConfig,
    CulturalExperimentConfig,
    GeneticExperimentConfig,
    create_lk_costly_choice_config,
    create_lk_runaway_config,
    create_lk_stasis_config,
)
from experiments.validated_runner import ValidatedExperimentRunner


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

__all__ = ["ValidatedPaperConfig", "ValidatedPaperRunner", "run_validated_paper_experiments"]

console = Console()
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=False)
class ValidatedPaperConfig:
    """Configuration for validated paper experiments."""

    output_dir: str = "experiments/results/paper_data"
    quick_test: bool = False
    run_validation: bool = True
    run_lhs: bool = False
    lhs_samples: int = 200
    replications_per_condition: int = 10
    n_generations: int = 5000
    max_duration_hours: float = 24.0

    def __post_init__(self) -> None:
        if self.quick_test:
            # Reduced scope for testing with safe parameters
            self.replications_per_condition = 3
            self.lhs_samples = 20
            self.n_generations = 60
            self.max_duration_hours = 1.0


class ValidatedPaperRunner:
    """
    Paper experiment runner using validated Pydantic configurations.

    This replaces the old fragile dictionary-based system with type-safe,
    validated experiment configurations that prevent population explosions
    and parameter errors.
    """

    def __init__(self, config: ValidatedPaperConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize validated experiment runner
        self.runner = ValidatedExperimentRunner()

        # Experiment tracking
        self.start_time = time.time()
        self.completed_experiments = 0
        self.failed_experiments = 0
        self.all_results: list[Any] = []

        # Setup logging
        self._setup_logging()

        logger.info("🔬 Validated Paper Experiment Runner initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Quick test mode: {self.config.quick_test}")
        logger.info(f"Phase 1 validation: {self.config.run_validation}")
        logger.info(f"Phase 2 LHS exploration: {self.config.run_lhs}")

        if self.config.quick_test:
            logger.info(
                f"Quick test parameters - replications: {self.config.replications_per_condition}, generations: {self.config.n_generations}"
            )

    def _setup_logging(self) -> None:
        """Setup comprehensive logging for paper experiments."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"validated_paper_experiments_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        logger.info(f"Logging initialized - log file: {log_file}")

    @beartype
    def run_lk_validation_scenarios(self) -> list[GeneticExperimentConfig]:
        """
        Generate validated Lande-Kirkpatrick validation scenarios.

        Returns
        -------
        list[GeneticExperimentConfig]
            List of validated experiment configurations
        """
        logger.info("🧬 Generating Lande-Kirkpatrick validation scenarios")

        # Determine population parameters based on test mode
        base_population = 2000 if not self.config.quick_test else 200
        carrying_capacity = 2000 if not self.config.quick_test else 200

        logger.info(
            f"LK validation - population: {base_population}, generations: {self.config.n_generations}, replications: {self.config.replications_per_condition}"
        )

        # Generate validated experiment configurations
        scenarios = []

        for rep in range(self.config.replications_per_condition):
            # Stasis scenario
            scenarios.append(
                create_lk_stasis_config(
                    name=f"lk_stasis_rep_{rep}",
                    population_size=base_population,
                    n_generations=self.config.n_generations,
                    carrying_capacity=carrying_capacity,
                )
            )

            # Runaway scenario
            scenarios.append(
                create_lk_runaway_config(
                    name=f"lk_runaway_rep_{rep}",
                    population_size=base_population,
                    n_generations=self.config.n_generations,
                    carrying_capacity=carrying_capacity,
                )
            )

            # Costly choice scenario
            scenarios.append(
                create_lk_costly_choice_config(
                    name=f"lk_costly_choice_rep_{rep}",
                    population_size=base_population,
                    n_generations=self.config.n_generations,
                    carrying_capacity=carrying_capacity,
                )
            )

        logger.info(f"Generated {len(scenarios)} validated LK scenarios")
        return scenarios

    @beartype
    def run_lhs_parameter_exploration(self) -> list[GeneticExperimentConfig]:
        """
        Generate Latin Hypercube Sampling parameter exploration.

        Returns
        -------
        list[GeneticExperimentConfig]
            List of validated experiment configurations for LHS exploration
        """
        logger.info("📊 Generating LHS parameter exploration")

        # Define parameter ranges for exploration
        param_ranges = {
            "h2_trait": (0.1, 0.9),
            "h2_preference": (0.1, 0.9),
            "genetic_correlation": (-0.5, 0.8),
            "selection_strength": (0.01, 0.5),
            "preference_cost": (0.0, 0.3),
            "mutation_variance": (0.001, 0.05),
        }

        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=len(param_ranges))
        lhs_samples = sampler.random(n=self.config.lhs_samples)

        # Scale samples to parameter ranges
        param_names = list(param_ranges.keys())
        scaled_samples = qmc.scale(
            lhs_samples,
            [param_ranges[name][0] for name in param_names],
            [param_ranges[name][1] for name in param_names],
        )

        # Generate validated configurations
        base_population = 1000 if not self.config.quick_test else 200
        carrying_capacity = 2000 if not self.config.quick_test else 200

        configurations = []
        for i, sample in enumerate(scaled_samples):
            param_dict = dict(zip(param_names, sample))

            config = GeneticExperimentConfig(
                name=f"lhs_exploration_{i:04d}",
                population_size=base_population,
                n_generations=self.config.n_generations,
                carrying_capacity=carrying_capacity,
                **param_dict,
            )
            configurations.append(config)

        logger.info(f"Generated {len(configurations)} validated LHS configurations")
        return configurations

    @beartype
    def run_cultural_experiments(self) -> list[CulturalExperimentConfig]:
        """
        Generate cultural evolution experiment configurations.

        Returns
        -------
        list[CulturalExperimentConfig]
            List of validated cultural experiment configurations
        """
        logger.info("🎭 Generating cultural evolution experiments")

        base_population = 1000 if not self.config.quick_test else 200

        configurations = []

        # Test different network topologies and transmission rates
        network_types: list[str] = ["scale_free", "small_world", "random"]
        transmission_rates = [0.1, 0.3, 0.5] if not self.config.quick_test else [0.3]

        for network_type in network_types:
            for transmission_rate in transmission_rates:
                for rep in range(self.config.replications_per_condition):
                    config = CulturalExperimentConfig(
                        name=f"cultural_{network_type}_tx{transmission_rate:.1f}_rep_{rep}",
                        population_size=base_population,
                        n_generations=self.config.n_generations,
                        carrying_capacity=base_population * 2,  # Safe default
                        network_type=network_type,  # type: ignore[arg-type]
                        horizontal_transmission_rate=transmission_rate,
                        oblique_transmission_rate=transmission_rate * 0.8,
                        innovation_rate=0.05,
                        network_connectivity=0.1,
                        cultural_memory_size=10,
                        local_learning_radius=5,
                    )
                    configurations.append(config)

        logger.info(f"Generated {len(configurations)} validated cultural configurations")
        return configurations

    @beartype
    def run_combined_experiments(self) -> list[CombinedExperimentConfig]:
        """
        Generate combined genetic+cultural experiment configurations.

        Returns
        -------
        list[CombinedExperimentConfig]
            List of validated combined experiment configurations
        """
        logger.info("🧬🎭 Generating combined evolution experiments")

        base_population = 1000 if not self.config.quick_test else 200

        configurations = []

        # Test different layer weight combinations
        weight_combinations = (
            [
                (0.8, 0.2),  # Genetic-dominated
                (0.5, 0.5),  # Balanced
                (0.2, 0.8),  # Cultural-dominated
            ]
            if not self.config.quick_test
            else [(0.5, 0.5)]
        )

        for genetic_weight, cultural_weight in weight_combinations:
            for rep in range(self.config.replications_per_condition):
                config = CombinedExperimentConfig(
                    name=f"combined_g{genetic_weight:.1f}_c{cultural_weight:.1f}_rep_{rep}",
                    population_size=base_population,
                    n_generations=self.config.n_generations,
                    carrying_capacity=base_population * 2,  # Safe default
                    genetic_enabled=True,
                    cultural_enabled=True,
                    genetic_weight=genetic_weight,
                    cultural_weight=cultural_weight,
                    normalize_weights=True,
                    # Genetic parameters (moderate values)
                    h2_trait=0.5,
                    h2_preference=0.5,
                    genetic_correlation=0.2,
                    selection_strength=0.1,
                    preference_cost=0.05,
                    mutation_variance=0.01,
                    # Cultural parameters (moderate values)
                    innovation_rate=0.1,
                    horizontal_transmission_rate=0.3,
                    oblique_transmission_rate=0.2,
                    network_type="scale_free",
                    network_connectivity=0.1,
                    cultural_memory_size=10,
                    local_learning_radius=5,
                )
                configurations.append(config)

        logger.info(f"Generated {len(configurations)} validated combined configurations")
        return configurations

    @beartype
    def execute_experiments(self, configurations: list[Any]) -> list[Any]:
        """
        Execute a list of validated experiment configurations.

        Parameters
        ----------
        configurations : list[GeneticExperimentConfig | CulturalExperimentConfig | CombinedExperimentConfig]
            List of validated experiment configurations to execute

        Returns
        -------
        list[Any]
            List of experiment results
        """
        results = []

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            task = progress.add_task("Running validated experiments", total=len(configurations))

            for config in configurations:
                try:
                    logger.info(f"🔬 Executing: {config.name}")

                    # Run experiment using validated runner
                    result = self.runner.run_from_config(config)
                    results.append(result)
                    self.completed_experiments += 1

                    logger.info(f"✅ Completed: {config.name}")

                except Exception as e:
                    logger.error(f"❌ Failed: {config.name} - {e}")
                    self.failed_experiments += 1

                progress.advance(task)

        return results

    @beartype
    def run_comprehensive_experiments(self) -> dict[str, Any]:
        """
        Run comprehensive validated paper experiments.

        Returns
        -------
        dict[str, Any]
            Complete experimental results summary
        """
        logger.info("🚀 Starting Comprehensive Validated Paper Experiments")
        logger.info(f"Phase 1 validation: {self.config.run_validation}")
        logger.info(f"Phase 2 LHS exploration: {self.config.run_lhs}")

        # Phase 1: Validation scenarios
        if self.config.run_validation:
            logger.info("📋 Phase 1: Running validation scenarios")

            # LK validation scenarios
            lk_configs = self.run_lk_validation_scenarios()
            lk_results = self.execute_experiments(lk_configs)
            self.all_results.extend(lk_results)

            # Cultural experiments (if not quick test)
            if not self.config.quick_test:
                cultural_configs = self.run_cultural_experiments()
                cultural_results = self.execute_experiments(cultural_configs)
                self.all_results.extend(cultural_results)

                # Combined experiments
                combined_configs = self.run_combined_experiments()
                combined_results = self.execute_experiments(combined_configs)
                self.all_results.extend(combined_results)

        # Phase 2: LHS exploration
        if self.config.run_lhs:
            logger.info("📊 Phase 2: Running LHS parameter exploration")
            lhs_configs = self.run_lhs_parameter_exploration()
            lhs_results = self.execute_experiments(lhs_configs)
            self.all_results.extend(lhs_results)

        # Generate summary
        summary = self._generate_summary()

        # Save results
        self._save_results(summary)

        return summary

    def _generate_summary(self) -> dict[str, Any]:
        """Generate comprehensive experiment summary."""
        end_time = time.time()
        total_duration = end_time - self.start_time

        # Get runner statistics
        runner_stats = self.runner.get_stats()

        summary = {
            "experiment_name": "validated_paper_experiments",
            "validation_system": "pydantic_v1.0",
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "total_duration_hours": total_duration / 3600,
            "total_experiments": len(self.all_results),
            "completed_experiments": runner_stats["experiments_run"],
            "failed_experiments": runner_stats["experiments_failed"],
            "success_rate": runner_stats["success_rate"],
            "configuration": {
                "quick_test": self.config.quick_test,
                "run_validation": self.config.run_validation,
                "run_lhs": self.config.run_lhs,
                "replications_per_condition": self.config.replications_per_condition,
                "n_generations": self.config.n_generations,
                "lhs_samples": self.config.lhs_samples,
            },
            "system_improvements": [
                "Type-safe Pydantic model validation",
                "Guaranteed population control (carrying_capacity validation)",
                "No dictionary manipulation errors",
                "Clear validation error messages",
                "Factory functions for common scenarios",
                "Automatic parameter normalization",
            ],
        }

        return summary

    def _save_results(self, summary: dict[str, Any]) -> None:
        """Save experimental results and summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main summary
        summary_file = self.output_dir / f"validated_paper_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"📊 Validated experiment summary saved: {summary_file}")

        # Save detailed results (if any)
        if self.all_results:
            results_file = self.output_dir / f"detailed_results_{timestamp}.json"
            serializable_results = []

            for result in self.all_results:
                try:
                    # Convert result to dictionary for JSON serialization
                    if hasattr(result, "model_dump"):
                        serializable_results.append(result.model_dump())
                    elif hasattr(result, "__dict__"):
                        serializable_results.append(result.__dict__)
                    else:
                        serializable_results.append(str(result))
                except Exception as e:
                    logger.warning(f"Could not serialize result: {e}")
                    serializable_results.append({"error": str(e)})

            with open(results_file, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"📋 Detailed results saved: {results_file}")


@beartype
def run_validated_paper_experiments(
    output_dir: str = "experiments/results/paper_data",
    quick_test: bool = False,
    run_validation: bool = True,
    run_lhs: bool = False,
    lhs_samples: int = 200,
    replications: int = 10,
    generations: int = 5000,
) -> dict[str, Any]:
    """
    Run validated paper experiments with Pydantic configuration system.

    Parameters
    ----------
    output_dir : str
        Directory to save experimental results
    quick_test : bool
        Run reduced scope experiments for testing
    run_validation : bool
        Run Phase 1 validation scenarios
    run_lhs : bool
        Run Phase 2 Latin Hypercube Sampling exploration
    lhs_samples : int
        Number of LHS parameter combinations
    replications : int
        Number of replications per parameter condition
    generations : int
        Number of generations per simulation

    Returns
    -------
    dict[str, Any]
        Experiment summary results
    """
    config = ValidatedPaperConfig(
        output_dir=output_dir,
        quick_test=quick_test,
        run_validation=run_validation,
        run_lhs=run_lhs,
        lhs_samples=lhs_samples,
        replications_per_condition=replications,
        n_generations=generations,
    )

    runner = ValidatedPaperRunner(config)
    return runner.run_comprehensive_experiments()


def main() -> None:
    """Main CLI entry point for validated paper experiments."""
    parser = argparse.ArgumentParser(
        description="Validated paper experiments using Pydantic configuration system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output experiments/results/paper_data
  %(prog)s --quick-test  # Fast validation with safe parameters
  %(prog)s --run-lhs --lhs-samples 100  # Parameter exploration
  %(prog)s --no-validation --run-lhs  # Skip validation, run LHS only

Features of the validated system:
  • Type-safe Pydantic models prevent parameter errors
  • Guaranteed population control (no more explosions!)
  • Clear validation error messages
  • Factory functions for common experiment scenarios
  • Automatic parameter normalization and validation
        """,
    )

    parser.add_argument(
        "--output", type=str, default="experiments/results/paper_data", help="Output directory for results"
    )
    parser.add_argument("--quick-test", action="store_true", help="Run reduced scope experiments for testing")
    parser.add_argument("--no-validation", action="store_true", help="Skip Phase 1 validation scenarios")
    parser.add_argument("--run-lhs", action="store_true", help="Run Phase 2 Latin Hypercube Sampling exploration")
    parser.add_argument("--lhs-samples", type=int, default=200, help="Number of LHS parameter combinations")
    parser.add_argument("--replications", type=int, default=10, help="Number of replications per condition")
    parser.add_argument("--generations", type=int, default=5000, help="Number of generations per simulation")

    args = parser.parse_args()

    try:
        # Show system information
        console.print("[bold green]🔬 LoveBug Validated Paper Experiments[/bold green]")
        console.print("Using type-safe Pydantic validation system")
        console.print("Population explosions are now prevented by design!\n")

        # Run experiments
        results = run_validated_paper_experiments(
            output_dir=args.output,
            quick_test=args.quick_test,
            run_validation=not args.no_validation,
            run_lhs=args.run_lhs,
            lhs_samples=args.lhs_samples,
            replications=args.replications,
            generations=args.generations,
        )

        # Display results
        console.print("[green]🎉 Validated experiments completed successfully![/green]")

        # Create results table
        table = Table(title="Validated Paper Experiments Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Experiments", str(results["total_experiments"]))
        table.add_row("Completed", str(results["completed_experiments"]))
        table.add_row("Failed", str(results["failed_experiments"]))
        table.add_row("Success Rate", f"{results['success_rate']:.1%}")
        table.add_row("Duration", f"{results['total_duration_hours']:.2f}h")
        table.add_row("Validation System", results["validation_system"])

        console.print(table)

        console.print(f"\n📁 Results saved to: [blue]{args.output}[/blue]")

    except KeyboardInterrupt:
        console.print("[yellow]⚠️  Experiments interrupted - partial results may be saved[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        logging.exception("Validated experiment runner failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
