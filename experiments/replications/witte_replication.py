#!/usr/bin/env python3
"""
Witte Cultural Transmission Replication

This script replicates experiments showing that a socially learned preference
can establish itself and persist in a population through cultural transmission.

Reference: Witte, K., & Ryan, M. J. (2002). Mate choice copying in the sailfin
molly, Poecilia latipinna, in the wild. Animal Behaviour, 63, 943-949.

Key Mechanism: Cultural transmission can maintain novel preferences in a
population even when they are not genetically favored.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lovebug.config import (
    CulturalParams,
    GeneticParams,
    LayerBlendingParams,
    LoveBugConfig,
    PerceptualParams,
    SimulationParams,
)
from lovebug.unified_mesa_model import LoveModel

logger = logging.getLogger(__name__)


class WitteReplication:
    """Replicates Witte's cultural transmission persistence experiment."""

    def __init__(self, population_size: int = 100, n_generations: int = 200, seed: int = 42):
        self.population_size = population_size
        self.n_generations = n_generations
        self.seed = seed
        self.results: dict[str, Any] = {}

    def create_experimental_config(self) -> LoveBugConfig:
        """
        Create configuration for Witte replication.

        Pure cultural evolution to test persistence of socially learned traits.
        """
        return LoveBugConfig(
            name="witte_cultural_transmission",
            genetic=GeneticParams(
                h2_trait=0.0,  # No genetic influence on traits
                h2_preference=0.0,  # No genetic influence on preferences
                mutation_rate=0.0,  # No genetic mutation
                crossover_rate=0.0,  # No genetic recombination
                population_size=self.population_size,
                elitism=1,
                energy_decay=0.01,  # Slow aging to allow cultural transmission
                mutation_variance=0.0,
                max_age=self.n_generations,  # Live long enough for experiment
                carrying_capacity=self.population_size,
            ),
            cultural=CulturalParams(
                learning_rate=0.3,  # Moderate cultural learning
                innovation_rate=0.001,  # Very low innovation - test persistence
                network_type="small_world",  # Realistic social network
                network_connectivity=0.8,
                cultural_memory_size=5,
                memory_decay_rate=0.01,  # Slow memory decay
                horizontal_transmission_rate=0.3,  # Key parameter for transmission
                oblique_transmission_rate=0.1,  # Some cross-generational transmission
                local_learning_radius=10,
                memory_update_strength=0.5,
            ),
            blending=LayerBlendingParams(
                blend_mode="weighted",
                blend_weight=0.0,  # Pure cultural evolution
            ),
            perceptual=PerceptualParams(),
            simulation=SimulationParams(
                population_size=self.population_size,
                steps=self.n_generations,
                seed=self.seed,
            ),
        )

    def run_experiment(self) -> dict[str, Any]:
        """
        Run the Witte cultural transmission experiment.

        1. Initialize population with uniform preference (baseline)
        2. Introduce novel preference in small subset (innovation)
        3. Track persistence and spread of novel preference over time
        """
        logger.info("🐟 Starting Witte cultural transmission replication")

        config = self.create_experimental_config()
        model = LoveModel(config=config)

        # Initialize with uniform baseline preference
        baseline_preference = 128  # Middle value (bit pattern: 10000000)
        novel_preference = 200  # Novel preference (bit pattern: 11001000)

        logger.info(f"Baseline preference: {baseline_preference}")
        logger.info(f"Novel preference: {novel_preference}")

        # Set initial cultural preferences to baseline
        agents_data = model.agents._agents
        if "pref_culture" in agents_data.columns:
            model.agents._agents = agents_data.with_columns(pl.lit(baseline_preference).alias("pref_culture"))

        # Track preferences over time
        preference_history = []

        # Step 1: Run baseline for a few generations
        logger.info("Phase 1: Establishing baseline")
        for generation in range(5):
            model.step()

            # Record current preference distribution
            current_agents = model.agents._agents
            if "pref_culture" in current_agents.columns:
                prefs = current_agents["pref_culture"].to_list()
                preference_history.append(
                    {
                        "generation": generation,
                        "phase": "baseline",
                        "novel_frequency": sum(1 for p in prefs if abs(p - novel_preference) < 10) / len(prefs),
                        "mean_preference": np.mean(prefs),
                        "preference_variance": np.var(prefs),
                    }
                )

        # Step 2: Introduce novel preference in 5% of population
        logger.info("Phase 2: Introducing novel preference")
        current_agents = model.agents._agents
        n_innovators = max(1, int(0.05 * len(current_agents)))  # 5% of population

        # Randomly select innovators
        np.random.seed(self.seed + 100)
        innovator_indices = np.random.choice(len(current_agents), size=n_innovators, replace=False)

        # Update preferences for innovators
        if "pref_culture" in current_agents.columns:
            updated_prefs = current_agents["pref_culture"].to_list()
            for idx in innovator_indices:
                updated_prefs[idx] = novel_preference

            model.agents._agents = current_agents.with_columns(pl.Series("pref_culture", updated_prefs))

        logger.info(f"Introduced novel preference to {n_innovators} individuals")

        # Step 3: Track transmission and persistence
        logger.info("Phase 3: Tracking cultural transmission")
        for generation in range(5, self.n_generations):
            model.step()

            # Record current preference distribution
            current_agents = model.agents._agents
            if "pref_culture" in current_agents.columns:
                prefs = current_agents["pref_culture"].to_list()
                novel_frequency = sum(1 for p in prefs if abs(p - novel_preference) < 10) / len(prefs)

                preference_history.append(
                    {
                        "generation": generation,
                        "phase": "transmission",
                        "novel_frequency": novel_frequency,
                        "mean_preference": np.mean(prefs),
                        "preference_variance": np.var(prefs),
                    }
                )

                # Log progress at key points
                if generation % 50 == 0 or generation == self.n_generations - 1:
                    logger.info(f"Generation {generation}: Novel preference frequency = {novel_frequency:.3f}")

        # Analyze results
        final_frequency = preference_history[-1]["novel_frequency"]
        max_frequency = max(h["novel_frequency"] for h in preference_history if h["phase"] == "transmission")

        # Determine if preference persisted
        persistence_threshold = 0.01  # Must maintain at least 1% frequency
        final_persistent = final_frequency > persistence_threshold

        # Determine if preference spread
        spread_threshold = 0.1  # Must reach at least 10% at some point
        max_spread = max_frequency > spread_threshold

        # Compile results
        results = {
            "experiment": "witte_cultural_transmission",
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "baseline_preference": baseline_preference,
            "novel_preference": novel_preference,
            "n_innovators": n_innovators,
            "initial_frequency": n_innovators / self.population_size,
            "final_frequency": final_frequency,
            "max_frequency": max_frequency,
            "persistent": final_persistent,
            "spread": max_spread,
            "success": final_persistent and max_spread,
            "preference_history": preference_history,
        }

        logger.info(f"Final novel preference frequency: {final_frequency:.3f}")
        logger.info(f"Maximum frequency reached: {max_frequency:.3f}")
        logger.info(f"Preference persistent: {final_persistent}")
        logger.info(f"Preference spread: {max_spread}")
        logger.info(f"Experiment success: {results['success']}")

        self.results = results
        return results

    def get_summary(self) -> str:
        """Generate human-readable summary of results."""
        if not self.results:
            return "No results available - run experiment first"

        summary = f"""
Witte Cultural Transmission Replication Results
==============================================

Population size: {self.results["population_size"]}
Generations: {self.results["n_generations"]}
Innovators: {self.results["n_innovators"]} ({self.results["initial_frequency"]:.1%})

Novel preference frequency:
  Initial: {self.results["initial_frequency"]:.3f}
  Maximum: {self.results["max_frequency"]:.3f}
  Final: {self.results["final_frequency"]:.3f}

Outcomes:
  Preference spread: {self.results["spread"]}
  Preference persistent: {self.results["persistent"]}
  Experiment successful: {self.results["success"]}

Interpretation:
{"✅ SUCCESS: Novel preference established and persisted through cultural transmission" if self.results["success"] else "❌ FAILURE: Novel preference failed to establish or persist"}
{"✅ SPREAD: Novel preference reached significant frequency" if self.results["spread"] else "❌ NO SPREAD: Novel preference remained rare"}
{"✅ PERSISTENT: Novel preference maintained in final generation" if self.results["persistent"] else "❌ LOST: Novel preference disappeared"}
        """
        return summary.strip()

    def plot_transmission_dynamics(self) -> None:
        """Plot preference dynamics over time (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            if not self.results or not self.results["preference_history"]:
                print("No data to plot")
                return

            history = self.results["preference_history"]
            generations = [h["generation"] for h in history]
            frequencies = [h["novel_frequency"] for h in history]

            plt.figure(figsize=(10, 6))
            plt.plot(generations, frequencies, "b-", linewidth=2, label="Novel preference frequency")
            plt.axhline(y=0.05, color="r", linestyle="--", alpha=0.7, label="5% threshold")
            plt.axhline(y=0.1, color="orange", linestyle="--", alpha=0.7, label="10% spread threshold")
            plt.axvline(x=5, color="gray", linestyle=":", alpha=0.7, label="Introduction point")

            plt.xlabel("Generation")
            plt.ylabel("Novel preference frequency")
            plt.title("Witte Cultural Transmission Dynamics")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib not available - cannot plot dynamics")


def main():
    """Run Witte replication experiment."""
    logging.basicConfig(level=logging.INFO)

    replication = WitteReplication(population_size=100, n_generations=200, seed=42)
    results = replication.run_experiment()

    print(replication.get_summary())

    # Optionally plot results
    try:
        replication.plot_transmission_dynamics()
    except Exception as e:
        print(f"Could not plot results: {e}")

    return results


if __name__ == "__main__":
    main()
