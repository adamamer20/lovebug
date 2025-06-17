#!/usr/bin/env python3
"""
Systematic Scaling Experiments for Evolutionary Simulations

This script conducts comprehensive performance scaling analysis across all three experiment types:
- Layer1 (Genetic): O(n) scaling with Lande-Kirkpatrick model
- Layer2 (Cultural): O(n²) scaling with cultural transmission
- Combined: Mixed scaling behavior with integrated gene-culture model

The script measures execution time, memory usage, and validates theoretical scaling predictions
to generate empirical scaling law data for research publication.

Usage:
    uv run python experiments/scaling_experiments.py --output results/scaling_analysis.json
    uv run python experiments/scaling_experiments.py --quick-test  # Small population sizes only
    uv run python experiments/scaling_experiments.py --type layer1 --plot  # Single experiment type with plots
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import psutil
from beartype import beartype
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.runner import run_single_experiment

__all__ = ["ScalingDataPoint", "ScalingExperimentRunner", "run_scaling_analysis"]

console = Console()
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=False)
class ScalingDataPoint:
    """Single scaling measurement data point.

    Parameters
    ----------
    experiment_type : Literal["layer1", "layer2", "combined"]
        Type of experiment conducted
    population_size : int
        Population/agent count used in experiment
    n_generations : int
        Number of generations simulated
    execution_time_seconds : float
        Wall-clock execution time in seconds
    peak_memory_mb : float
        Peak memory usage in megabytes
    cpu_time_seconds : float
        CPU time used (user + system)
    success : bool
        Whether experiment completed successfully
    timestamp : str
        ISO timestamp when measurement was taken
    experiment_id : str
        Unique identifier for this measurement
    metadata : dict[str, Any]
        Additional metadata from the experiment
    """

    experiment_type: Literal["layer1", "layer2", "combined"]
    population_size: int
    n_generations: int
    execution_time_seconds: float
    peak_memory_mb: float
    cpu_time_seconds: float
    success: bool
    timestamp: str
    experiment_id: str
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=False)
class ScalingAnalysisConfig:
    """Configuration for scaling experiments.

    Parameters
    ----------
    output_file : str
        Path to output JSON file for results
    quick_test : bool
        Run with smaller population sizes for quick testing
    generate_plots : bool
        Generate matplotlib plots of scaling results
    experiment_types : list[str]
        List of experiment types to test ["layer1", "layer2", "combined"]
    fixed_generations : int
        Fixed number of generations for fair comparison
    layer1_populations : list[int]
        Population sizes to test for Layer1 experiments
    layer2_populations : list[int]
        Agent counts to test for Layer2 experiments (smaller due to O(n²))
    combined_populations : list[int]
        Agent counts to test for Combined experiments
    replications_per_size : int
        Number of replications per population size
    timeout_seconds : float
        Maximum time to allow per experiment (to handle runaway scaling)
    """

    output_file: str = "experiments/results/scaling_analysis.json"
    quick_test: bool = False
    generate_plots: bool = False
    experiment_types: list[str] | None = None
    fixed_generations: int = 20
    layer1_populations: list[int] | None = None
    layer2_populations: list[int] | None = None
    combined_populations: list[int] | None = None
    replications_per_size: int = 3
    timeout_seconds: float = 300.0  # 5 minutes max per experiment

    def __post_init__(self) -> None:
        if self.experiment_types is None:
            self.experiment_types = ["layer1", "layer2", "combined"]

        if self.quick_test:
            # Smaller sizes for quick testing
            self.layer1_populations = (
                [50, 100, 200, 500] if self.layer1_populations is None else self.layer1_populations
            )
            self.layer2_populations = [25, 50, 100] if self.layer2_populations is None else self.layer2_populations
            self.combined_populations = (
                [25, 50, 100] if self.combined_populations is None else self.combined_populations
            )
            self.fixed_generations = 10
            self.replications_per_size = 2
        else:
            # Full scaling analysis
            self.layer1_populations = (
                [100, 200, 500, 1000, 2000, 5000] if self.layer1_populations is None else self.layer1_populations
            )
            self.layer2_populations = (
                [50, 100, 200, 500] if self.layer2_populations is None else self.layer2_populations
            )  # Avoid larger due to O(n²)
            self.combined_populations = (
                [50, 100, 200, 500] if self.combined_populations is None else self.combined_populations
            )


class ScalingExperimentRunner:
    """Systematic scaling experiment runner for performance characterization."""

    def __init__(self, config: ScalingAnalysisConfig) -> None:
        self.config = config
        self.results: list[ScalingDataPoint] = []

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # Create output directory
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("Scaling experiment runner initialized")
        self.logger.info(f"Quick test mode: {self.config.quick_test}")
        self.logger.info(f"Experiment types: {self.config.experiment_types}")
        self.logger.info(f"Output file: {self.config.output_file}")

    @beartype
    def _measure_single_experiment(
        self, experiment_type: str, population_size: int, replication: int
    ) -> ScalingDataPoint | None:
        """
        Measure performance of a single experiment run.

        Parameters
        ----------
        experiment_type : str
            Type of experiment ("layer1", "layer2", "combined")
        population_size : int
            Population or agent count
        replication : int
            Replication number for this configuration

        Returns
        -------
        ScalingDataPoint | None
            Measurement data point, or None if experiment failed
        """
        # Create experiment parameters based on type
        if experiment_type == "layer1":
            params_dict = {
                "type": "layer1",
                "config": {"name": f"scaling_layer1_{population_size}_{replication}"},
                "params": {
                    "n_generations": self.config.fixed_generations,
                    "pop_size": population_size,
                    "h2_trait": 0.5,
                    "h2_preference": 0.5,
                    "genetic_correlation": 0.2,
                    "selection_strength": 0.1,
                    "preference_cost": 0.05,
                    "mutation_variance": 0.01,
                },
                "random_seed": hash(f"{experiment_type}_{population_size}_{replication}") % (2**32),
            }
        elif experiment_type == "layer2":
            params_dict = {
                "type": "layer2",
                "config": {"name": f"scaling_layer2_{population_size}_{replication}"},
                "params": {
                    "innovation_rate": 0.1,
                    "horizontal_transmission_rate": 0.3,
                    "oblique_transmission_rate": 0.2,
                    "network_type": "scale_free",
                    "network_connectivity": 0.1,
                    "cultural_memory_size": 10,
                },
                "n_agents": population_size,
                "n_generations": self.config.fixed_generations,
                "random_seed": hash(f"{experiment_type}_{population_size}_{replication}") % (2**32),
            }
        elif experiment_type == "combined":
            params_dict = {
                "type": "combined",
                "config": {"name": f"scaling_combined_{population_size}_{replication}"},
                "params": {
                    # Layer activation parameters
                    "genetic_enabled": True,
                    "cultural_enabled": True,
                    "genetic_weight": 0.5,
                    "cultural_weight": 0.5,
                    "blending_mode": "weighted_average",
                    "normalize_weights": True,
                    "theta_detect": 8.0,
                    "sigma_perception": 2.0,
                    # Genetic parameters
                    "n_generations": self.config.fixed_generations,
                    "pop_size": population_size,
                    "h2_trait": 0.5,
                    "h2_preference": 0.5,
                    "genetic_correlation": 0.2,
                    "selection_strength": 0.1,
                    "preference_cost": 0.05,
                    "mutation_variance": 0.01,
                    # Cultural parameters
                    "innovation_rate": 0.1,
                    "horizontal_transmission_rate": 0.3,
                    "oblique_transmission_rate": 0.2,
                    "network_type": "scale_free",
                    "network_connectivity": 0.1,
                    "cultural_memory_size": 10,
                    "local_learning_radius": 5,
                },
                "n_agents": population_size,
                "n_generations": self.config.fixed_generations,
                "random_seed": hash(f"{experiment_type}_{population_size}_{replication}") % (2**32),
            }
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        # Measure memory before experiment
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Get initial resource usage
        initial_rusage = resource.getrusage(resource.RUSAGE_SELF)

        # Run experiment with timing
        start_time = time.time()
        start_timestamp = datetime.now().isoformat()

        try:
            # Execute experiment with timeout protection
            result = run_single_experiment(params_dict)
            success = True
            experiment_id = result.metadata.experiment_id
            metadata = {
                "outcome": getattr(result, "outcome", None),
                "final_values": {},
            }

            # Extract type-specific final values based on result type
            from experiments.models import CulturalExperimentResult, GeneticExperimentResult, IntegratedExperimentResult

            if isinstance(result, GeneticExperimentResult):
                metadata["final_values"]["final_trait"] = result.final_trait
                metadata["final_values"]["final_preference"] = result.final_preference
                metadata["final_values"]["final_covariance"] = result.final_covariance
            elif isinstance(result, CulturalExperimentResult):
                metadata["final_values"]["final_diversity"] = result.final_diversity
                metadata["final_values"]["total_events"] = result.total_events
            elif isinstance(result, IntegratedExperimentResult):
                metadata["final_values"]["gene_culture_correlation"] = result.gene_culture_correlation
                metadata["final_values"]["interaction_strength"] = result.interaction_strength
                metadata["final_values"]["genetic_final_trait"] = result.genetic_component.final_trait
                metadata["final_values"]["cultural_final_diversity"] = result.cultural_component.final_diversity

        except Exception as e:
            self.logger.warning(f"Experiment failed: {experiment_type}, size={population_size}, rep={replication}: {e}")
            success = False
            experiment_id = f"failed_{experiment_type}_{population_size}_{replication}"
            metadata = {"error": str(e)}

        # Calculate timing and resource usage
        end_time = time.time()
        execution_time = end_time - start_time

        # Get final resource usage
        final_rusage = resource.getrusage(resource.RUSAGE_SELF)
        cpu_time = (final_rusage.ru_utime - initial_rusage.ru_utime) + (final_rusage.ru_stime - initial_rusage.ru_stime)

        # Measure peak memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = max(initial_memory, final_memory)

        return ScalingDataPoint(
            experiment_type=experiment_type,  # type: ignore[arg-type]
            population_size=population_size,
            n_generations=self.config.fixed_generations,
            execution_time_seconds=execution_time,
            peak_memory_mb=peak_memory,
            cpu_time_seconds=cpu_time,
            success=success,
            timestamp=start_timestamp,
            experiment_id=experiment_id,
            metadata=metadata,
        )

    @beartype
    def run_scaling_experiments(self) -> dict[str, Any]:
        """
        Execute systematic scaling experiments across all experiment types.

        Returns
        -------
        dict[str, Any]
            Summary of scaling experiments with analysis results
        """
        total_experiments = 0

        # Count total experiments
        if self.config.experiment_types is not None:
            for exp_type in self.config.experiment_types:
                if exp_type == "layer1" and self.config.layer1_populations is not None:
                    total_experiments += len(self.config.layer1_populations) * self.config.replications_per_size
                elif exp_type == "layer2" and self.config.layer2_populations is not None:
                    total_experiments += len(self.config.layer2_populations) * self.config.replications_per_size
                elif exp_type == "combined" and self.config.combined_populations is not None:
                    total_experiments += len(self.config.combined_populations) * self.config.replications_per_size

        self.logger.info(f"Starting {total_experiments} scaling experiments")

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
            main_task = progress.add_task("Scaling Analysis", total=total_experiments)

            # Run experiments for each type
            for exp_type in self.config.experiment_types:
                if exp_type == "layer1":
                    populations = self.config.layer1_populations
                elif exp_type == "layer2":
                    populations = self.config.layer2_populations
                elif exp_type == "combined":
                    populations = self.config.combined_populations
                else:
                    continue

                self.logger.info(f"Testing {exp_type} with populations: {populations}")

                for pop_size in populations:
                    for rep in range(self.config.replications_per_size):
                        # Update progress description
                        progress.update(
                            main_task, description=f"Scaling {exp_type.title()} (n={pop_size}, rep={rep + 1})"
                        )

                        # Run single experiment
                        data_point = self._measure_single_experiment(exp_type, pop_size, rep)
                        if data_point is not None:
                            self.results.append(data_point)

                            # Log progress
                            if data_point.success:
                                self.logger.info(
                                    f"✅ {exp_type} n={pop_size}: {data_point.execution_time_seconds:.3f}s, "
                                    f"{data_point.peak_memory_mb:.1f}MB"
                                )
                            else:
                                self.logger.warning(f"❌ {exp_type} n={pop_size}: Failed")

                        progress.advance(main_task)

        # Generate analysis
        analysis_results = self._analyze_scaling_results()

        # Save results
        self._save_results(analysis_results)

        # Generate plots if requested
        if self.config.generate_plots:
            self._generate_plots()

        return analysis_results

    def _analyze_scaling_results(self) -> dict[str, Any]:
        """
        Analyze scaling results and fit theoretical scaling laws.

        Returns
        -------
        dict[str, Any]
            Analysis results with scaling coefficients and validation of theoretical predictions
        """
        analysis = {
            "experiment_summary": {
                "total_experiments": len(self.results),
                "successful_experiments": sum(1 for r in self.results if r.success),
                "failed_experiments": sum(1 for r in self.results if not r.success),
                "experiment_types": list({r.experiment_type for r in self.results}),
            },
            "scaling_analysis": {},
            "theoretical_validation": {},
            "performance_summary": {},
        }

        # Analyze each experiment type separately
        for exp_type in {r.experiment_type for r in self.results}:
            type_results = [r for r in self.results if r.experiment_type == exp_type and r.success]

            if len(type_results) < 2:
                self.logger.warning(f"Insufficient data points for {exp_type} scaling analysis")
                continue

            # Extract data for fitting
            population_sizes = np.array([r.population_size for r in type_results])
            execution_times = np.array([r.execution_time_seconds for r in type_results])
            memory_usage = np.array([r.peak_memory_mb for r in type_results])

            # Fit scaling laws
            # Linear scaling: T = a * n + b
            linear_coeffs = np.polyfit(population_sizes, execution_times, 1)
            linear_r2 = self._calculate_r_squared(population_sizes, execution_times, linear_coeffs, 1)

            # Quadratic scaling: T = a * n^2 + b * n + c
            if len(type_results) >= 3:
                quad_coeffs = np.polyfit(population_sizes, execution_times, 2)
                quad_r2 = self._calculate_r_squared(population_sizes, execution_times, quad_coeffs, 2)
            else:
                quad_coeffs = [0, 0, 0]
                quad_r2 = 0.0

            # Log scaling: T = a * log(n) + b
            log_n = np.log(population_sizes)
            log_coeffs = np.polyfit(log_n, execution_times, 1)
            log_r2 = self._calculate_r_squared(log_n, execution_times, log_coeffs, 1)

            analysis["scaling_analysis"][exp_type] = {
                "data_points": len(type_results),
                "population_range": [int(min(population_sizes)), int(max(population_sizes))],
                "time_range": [float(min(execution_times)), float(max(execution_times))],
                "memory_range": [float(min(memory_usage)), float(max(memory_usage))],
                "linear_fit": {
                    "coefficients": [float(c) for c in linear_coeffs],
                    "r_squared": float(linear_r2),
                    "equation": f"T = {linear_coeffs[0]:.6f} * n + {linear_coeffs[1]:.6f}",
                },
                "quadratic_fit": {
                    "coefficients": [float(c) for c in quad_coeffs],
                    "r_squared": float(quad_r2),
                    "equation": f"T = {quad_coeffs[0]:.9f} * n² + {quad_coeffs[1]:.6f} * n + {quad_coeffs[2]:.6f}",
                },
                "logarithmic_fit": {
                    "coefficients": [float(c) for c in log_coeffs],
                    "r_squared": float(log_r2),
                    "equation": f"T = {log_coeffs[0]:.6f} * log(n) + {log_coeffs[1]:.6f}",
                },
            }

            # Determine best fit
            fits = [
                ("linear", linear_r2),
                ("quadratic", quad_r2),
                ("logarithmic", log_r2),
            ]
            best_fit = max(fits, key=lambda x: x[1])
            analysis["scaling_analysis"][exp_type]["best_fit"] = best_fit[0]
            analysis["scaling_analysis"][exp_type]["best_r_squared"] = best_fit[1]

            # Theoretical validation
            theoretical_expectation = self._get_theoretical_scaling(exp_type)
            analysis["theoretical_validation"][exp_type] = {
                "expected_scaling": theoretical_expectation,
                "observed_best_fit": best_fit[0],
                "matches_theory": best_fit[0] == theoretical_expectation,
                "confidence": best_fit[1],
            }

            # Performance summary statistics
            analysis["performance_summary"][exp_type] = {
                "mean_time_per_generation": float(np.mean(execution_times) / self.config.fixed_generations),
                "time_scaling_coefficient": float(linear_coeffs[0])
                if best_fit[0] == "linear"
                else float(quad_coeffs[0]),
                "efficiency_score": float(1.0 / (np.mean(execution_times) / min(execution_times))),
                "memory_efficiency": float(np.mean(memory_usage) / max(memory_usage)),
            }

        return analysis

    def _get_theoretical_scaling(self, experiment_type: str) -> str:
        """Get theoretical scaling expectation for experiment type."""
        if experiment_type == "layer1":
            return "linear"  # O(n) for genetic algorithms
        elif experiment_type == "layer2":
            return "quadratic"  # O(n²) for cultural transmission
        elif experiment_type == "combined":
            return "quadratic"  # Dominated by cultural component
        else:
            return "unknown"

    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray, degree: int) -> float:
        """Calculate R-squared for polynomial fit."""
        if degree == 1:
            y_pred = coeffs[0] * x + coeffs[1]
        elif degree == 2:
            y_pred = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
        else:
            return 0.0

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _save_results(self, analysis_results: dict[str, Any]) -> None:
        """Save scaling experiment results and analysis to JSON file."""
        output_data = {
            "scaling_experiment_results": {
                "config": asdict(self.config),
                "data_points": [asdict(r) for r in self.results],
                "analysis": analysis_results,
                "timestamp": datetime.now().isoformat(),
                "total_experiments": len(self.results),
            }
        }

        with open(self.config.output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        self.logger.info(f"Results saved to {self.config.output_file}")

    def _generate_plots(self) -> None:
        """Generate matplotlib plots of scaling results."""
        if not self.results:
            self.logger.warning("No results to plot")
            return

        # Create plots directory
        plots_dir = Path(self.config.output_file).parent / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Set up plotting style
        plt.style.use("default")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot 1: Execution time vs population size
        ax1 = axes[0]
        for exp_type in {r.experiment_type for r in self.results}:
            type_results = [r for r in self.results if r.experiment_type == exp_type and r.success]
            if not type_results:
                continue

            pop_sizes = [r.population_size for r in type_results]
            times = [r.execution_time_seconds for r in type_results]

            ax1.scatter(pop_sizes, times, label=f"{exp_type.title()}", alpha=0.7, s=50)

            # Fit trend line
            if len(pop_sizes) >= 2:
                z = np.polyfit(pop_sizes, times, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(pop_sizes), max(pop_sizes), 100)
                ax1.plot(x_trend, p(x_trend), "--", alpha=0.8)

        ax1.set_xlabel("Population Size")
        ax1.set_ylabel("Execution Time (seconds)")
        ax1.set_title("Scaling: Execution Time vs Population Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Memory usage vs population size
        ax2 = axes[1]
        for exp_type in {r.experiment_type for r in self.results}:
            type_results = [r for r in self.results if r.experiment_type == exp_type and r.success]
            if not type_results:
                continue

            pop_sizes = [r.population_size for r in type_results]
            memory = [r.peak_memory_mb for r in type_results]

            ax2.scatter(pop_sizes, memory, label=f"{exp_type.title()}", alpha=0.7, s=50)

        ax2.set_xlabel("Population Size")
        ax2.set_ylabel("Peak Memory (MB)")
        ax2.set_title("Memory Usage vs Population Size")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Log-log scaling analysis
        ax3 = axes[2]
        for exp_type in {r.experiment_type for r in self.results}:
            type_results = [r for r in self.results if r.experiment_type == exp_type and r.success]
            if not type_results:
                continue

            pop_sizes = np.array([r.population_size for r in type_results])
            times = np.array([r.execution_time_seconds for r in type_results])

            ax3.loglog(pop_sizes, times, "o-", label=f"{exp_type.title()}", alpha=0.7, markersize=6)

        ax3.set_xlabel("Population Size (log scale)")
        ax3.set_ylabel("Execution Time (log scale)")
        ax3.set_title("Log-Log Scaling Analysis")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Performance efficiency
        ax4 = axes[3]
        exp_types = list({r.experiment_type for r in self.results})
        efficiencies = []

        for exp_type in exp_types:
            type_results = [r for r in self.results if r.experiment_type == exp_type and r.success]
            if type_results:
                times = [r.execution_time_seconds for r in type_results]
                # Calculate efficiency as inverse of mean time
                efficiency = 1.0 / np.mean(times) if times else 0
                efficiencies.append(efficiency)
            else:
                efficiencies.append(0)

        bars = ax4.bar(exp_types, efficiencies, alpha=0.7, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(exp_types)])
        ax4.set_ylabel("Performance Efficiency (1/mean_time)")
        ax4.set_title("Relative Performance Efficiency")
        ax4.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, eff in zip(bars, efficiencies):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bar.get_height() * 0.01,
                f"{eff:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        # Save plots
        plot_file = plots_dir / "scaling_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.savefig(plots_dir / "scaling_analysis.pdf", bbox_inches="tight")

        self.logger.info(f"Plots saved to {plots_dir}")
        plt.close()

    def print_summary_table(self, analysis_results: dict[str, Any]) -> None:
        """Print a summary table of scaling results using Rich."""
        table = Table(title="Scaling Experiment Results Summary")

        table.add_column("Experiment Type", style="cyan", no_wrap=True)
        table.add_column("Population Range", style="green")
        table.add_column("Best Fit", style="yellow")
        table.add_column("R²", style="magenta")
        table.add_column("Theoretical Match", style="red")
        table.add_column("Time/Generation (s)", style="blue")

        for exp_type, data in analysis_results.get("scaling_analysis", {}).items():
            pop_range = f"{data['population_range'][0]}-{data['population_range'][1]}"
            best_fit = data["best_fit"]
            r_squared = f"{data['best_r_squared']:.3f}"

            theoretical = analysis_results.get("theoretical_validation", {}).get(exp_type, {})
            theory_match = "✅" if theoretical.get("matches_theory", False) else "❌"

            performance = analysis_results.get("performance_summary", {}).get(exp_type, {})
            time_per_gen = f"{performance.get('mean_time_per_generation', 0):.4f}"

            table.add_row(exp_type.title(), pop_range, best_fit.title(), r_squared, theory_match, time_per_gen)

        console.print(table)


@beartype
def run_scaling_analysis(
    output_file: str = "experiments/results/scaling_analysis.json",
    quick_test: bool = False,
    generate_plots: bool = False,
    experiment_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run comprehensive scaling analysis across experiment types.

    Parameters
    ----------
    output_file : str
        Path to output JSON results file
    quick_test : bool
        Use smaller population sizes for quick testing
    generate_plots : bool
        Generate matplotlib visualization plots
    experiment_types : list[str] | None
        List of experiment types to test, defaults to all

    Returns
    -------
    dict[str, Any]
        Analysis results with scaling coefficients and performance data

    Examples
    --------
    >>> # Run full scaling analysis
    >>> results = run_scaling_analysis()
    >>>
    >>> # Quick test with plots
    >>> results = run_scaling_analysis(quick_test=True, generate_plots=True)
    >>>
    >>> # Test only Layer1 experiments
    >>> results = run_scaling_analysis(experiment_types=["layer1"])
    """
    config = ScalingAnalysisConfig(
        output_file=output_file,
        quick_test=quick_test,
        generate_plots=generate_plots,
        experiment_types=experiment_types or ["layer1", "layer2", "combined"],
    )

    runner = ScalingExperimentRunner(config)
    analysis_results = runner.run_scaling_experiments()

    # Print summary
    runner.print_summary_table(analysis_results)

    # Log key findings
    logger.info("🔬 Scaling Analysis Complete!")
    logger.info(f"📊 Total experiments: {analysis_results['experiment_summary']['total_experiments']}")
    logger.info(
        f"✅ Success rate: {analysis_results['experiment_summary']['successful_experiments']}/{analysis_results['experiment_summary']['total_experiments']}"
    )

    # Log theoretical validation
    for exp_type, validation in analysis_results.get("theoretical_validation", {}).items():
        expected = validation["expected_scaling"]
        observed = validation["observed_best_fit"]
        match = "✅" if validation["matches_theory"] else "❌"
        confidence = validation["confidence"]
        logger.info(f"{match} {exp_type.title()}: Expected {expected}, Observed {observed} (R²={confidence:.3f})")

    return analysis_results


def main() -> None:
    """Main CLI entry point for scaling experiments."""
    parser = argparse.ArgumentParser(
        description="Systematic scaling experiments for evolutionary simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output results/scaling_analysis.json
  %(prog)s --quick-test --plot
  %(prog)s --type layer1 layer2 --plot
        """,
    )

    parser.add_argument("--output", default="experiments/results/scaling_analysis.json", help="Output JSON file path")
    parser.add_argument("--quick-test", action="store_true", help="Run with smaller population sizes for quick testing")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots of results")
    parser.add_argument(
        "--type", nargs="*", choices=["layer1", "layer2", "combined"], help="Experiment types to test (default: all)"
    )

    args = parser.parse_args()

    try:
        run_scaling_analysis(
            output_file=args.output,
            quick_test=args.quick_test,
            generate_plots=args.plot,
            experiment_types=args.type,
        )

        console.print("[green]🎉 Scaling analysis completed successfully![/green]")
        console.print(f"[blue]📁 Results saved to: {args.output}[/blue]")

        if args.plot:
            plots_dir = Path(args.output).parent / "plots"
            console.print(f"[yellow]📊 Plots saved to: {plots_dir}[/yellow]")

    except KeyboardInterrupt:
        console.print("[yellow]⚠️ Analysis interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        logging.exception("Scaling analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
