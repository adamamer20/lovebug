#!/usr/bin/env python3
"""
Witte Cultural Transmission Replication (Adapted for Refactored Model)

This script replicates the core mechanism: a socially learned preference
can establish itself and persist in a population through cultural transmission.

Note: Adapted for refactored model using the cultural learning system.

Reference: Witte, K., & Ryan, M. J. (2002). Mate choice copying in the sailfin
molly, Poecilia latipinna, in the wild. Animal Behaviour, 63, 943-949.
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

from lovebug import LoveModelRefactored
from lovebug.config import (
    CulturalParams,
    GeneticParams,
    LayerConfig,
    LoveBugConfig,
    SimulationParams,
)

logger = logging.getLogger(__name__)


class WitteReplication:
    """
    Replicates Witte's cultural transmission persistence experiment.

    Adapted for refactored model with built-in cultural learning system.
    Tests persistence of novel preferences through social transmission.
    """

    def __init__(self, population_size: int = 100, n_generations: int = 200, seed: int = 42):
        self.population_size = population_size
        self.n_generations = n_generations
        self.seed = seed
        self.results: dict[str, Any] = {}

    def create_experimental_config(self) -> LoveBugConfig:
        """
        Create configuration for Witte replication.

        High cultural transmission to test persistence of socially learned traits.
        """
        return LoveBugConfig(
            name="witte_cultural_transmission",
            genetic=GeneticParams(
                h2_trait=0.1,  # Low heritability - focus on cultural transmission
                h2_preference=0.1,
                mutation_rate=0.005,  # Small but non-zero to prevent genetic drift issues
                crossover_rate=0.3,  # Modest recombination to maintain diversity
                elitism=1,
                energy_decay=0.008,  # Reduced to prevent population crash
                mutation_variance=0.01,
                max_age=200,  # Longer lifespan for cultural transmission
                carrying_capacity=self.population_size * 3,  # K = 300 for stability
                energy_replenishment_rate=0.0027,  # r = d * N₀/K = 0.008 * 100/300 = 0.0027
                parental_investment_rate=0.4,  # Reduced investment for more reproduction
                energy_min_mating=0.5,  # Lower mating threshold
                juvenile_cost=0.3,  # Lower juvenile cost for population growth
                display_cost_scalar=0.1,  # Reduced display costs
                search_cost=0.005,  # Lower search costs
                base_energy=15.0,  # Higher base energy for sustainability
            ),
            cultural=CulturalParams(
                innovation_rate=0.01,  # Higher innovation to maintain cultural variants
                memory_span=15,  # Longer memory for better persistence
                network_type="small_world",  # Realistic social network
                network_connectivity=0.9,  # Higher connectivity for better transmission
                cultural_memory_size=15,  # Larger memory buffer
                memory_decay_rate=0.001,  # Much slower decay for persistence
                horizontal_transmission_rate=0.6,  # Higher transmission rate
                oblique_transmission_rate=0.3,  # More intergenerational transmission
                local_learning_radius=10,  # Larger learning radius
                memory_update_strength=1.0,
                learning_strategy="success-biased",  # More effective for spread
            ),
            layer=LayerConfig(
                genetic_enabled=True,
                cultural_enabled=True,
                blending_mode="weighted",
                genetic_weight=0.0,  # Pure cultural evolution
                cultural_weight=1.0,
                sigma_perception=0.0,
                theta_detect=0.0,
                sigmoid_steepness=1.5,
            ),
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
        model = LoveModelRefactored(config=config)

        # Initialize with uniform baseline preference
        baseline_preference = 32768  # Mid-range 16-bit value (max 65535)
        novel_preference = 52000  # Novel preference value (high-end 16-bit)

        logger.info(f"Baseline preference: {baseline_preference}")
        logger.info(f"Novel preference: {novel_preference}")

        # Set initial cultural preferences to baseline
        current_df = model.get_agent_dataframe()
        baseline_prefs = np.full(len(current_df), baseline_preference, dtype=np.uint16)

        updated_df = current_df.with_columns(pl.Series("cultural_preference", baseline_prefs, dtype=pl.UInt16))
        model.agents._agentsets[0].agents = updated_df

        # Track preferences over time
        preference_history = []

        # Step 1: Run baseline for a few generations
        logger.info("Phase 1: Establishing baseline")
        for generation in range(5):
            model.step()

            # Record current preference distribution
            current_agents = model.get_agent_dataframe()
            if "cultural_preference" in current_agents.columns:
                prefs = current_agents["cultural_preference"].to_numpy()
                novel_frequency = np.sum(np.abs(prefs - novel_preference) < 1000) / len(prefs)
                preference_history.append(
                    {
                        "generation": generation,
                        "phase": "baseline",
                        "novel_frequency": novel_frequency,
                        "mean_preference": float(np.mean(prefs)),
                        "preference_variance": float(np.var(prefs)),
                        "population_size": len(prefs),
                    }
                )

        # Step 2: Introduce novel preference in 5% of population
        logger.info("Phase 2: Introducing novel preference")
        current_agents = model.get_agent_dataframe()
        n_innovators = max(1, int(0.05 * len(current_agents)))  # 5% of population

        # Randomly select innovators
        np.random.seed(self.seed + 100)
        innovator_indices = np.random.choice(len(current_agents), size=n_innovators, replace=False)

        # Update preferences for innovators
        if "cultural_preference" in current_agents.columns:
            updated_prefs = current_agents["cultural_preference"].to_numpy().copy()
            updated_prefs[innovator_indices] = novel_preference

            updated_df = current_agents.with_columns(pl.Series("cultural_preference", updated_prefs, dtype=pl.UInt16))
            model.agents._agentsets[0].agents = updated_df

        logger.info(f"Introduced novel preference to {n_innovators} individuals")

        # Step 3: Track transmission and persistence
        logger.info("Phase 3: Tracking cultural transmission")
        for generation in range(5, self.n_generations):
            model.step()

            # Record current preference distribution
            current_agents = model.get_agent_dataframe()
            if "cultural_preference" in current_agents.columns:
                prefs = current_agents["cultural_preference"].to_numpy()
                novel_frequency = np.sum(np.abs(prefs - novel_preference) < 1000) / len(prefs)

                preference_history.append(
                    {
                        "generation": generation,
                        "phase": "transmission",
                        "novel_frequency": novel_frequency,
                        "mean_preference": float(np.mean(prefs)),
                        "preference_variance": float(np.var(prefs)),
                        "population_size": len(prefs),
                    }
                )

                # Log progress at key points
                if generation % 50 == 0 or generation == self.n_generations - 1:
                    logger.info(
                        f"Generation {generation}: Novel preference frequency = {novel_frequency:.3f}, Population = {len(prefs)}"
                    )

        # Analyze results
        if preference_history:
            final_frequency = preference_history[-1]["novel_frequency"]
            transmission_history = [h for h in preference_history if h["phase"] == "transmission"]
            max_frequency = max(h["novel_frequency"] for h in transmission_history) if transmission_history else 0

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
        else:
            results = {
                "experiment": "witte_cultural_transmission",
                "error": "No preference data collected",
                "success": False,
            }

        logger.info(f"Final novel preference frequency: {results.get('final_frequency', 0):.3f}")
        logger.info(f"Maximum frequency reached: {results.get('max_frequency', 0):.3f}")
        logger.info(f"Preference persistent: {results.get('persistent', False)}")
        logger.info(f"Preference spread: {results.get('spread', False)}")
        logger.info(f"Experiment success: {results.get('success', False)}")

        self.results = results
        return results

    def get_summary(self) -> str:
        """Generate human-readable summary of results."""
        if not self.results:
            return "No results available - run experiment first"

        if "error" in self.results:
            return f"Experiment failed: {self.results['error']}"

        summary = f"""
Witte Cultural Transmission Replication Results (Adapted)
=========================================================

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

            if not self.results or not self.results.get("preference_history"):
                print("No data to plot")
                return

            history = self.results["preference_history"]
            generations = [h["generation"] for h in history]
            frequencies = [h["novel_frequency"] for h in history]
            populations = [h["population_size"] for h in history]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Plot frequency dynamics
            ax1.plot(generations, frequencies, "b-", linewidth=2, label="Novel preference frequency")
            ax1.axhline(y=0.05, color="r", linestyle="--", alpha=0.7, label="5% threshold")
            ax1.axhline(y=0.1, color="orange", linestyle="--", alpha=0.7, label="10% spread threshold")
            ax1.axvline(x=5, color="gray", linestyle=":", alpha=0.7, label="Introduction point")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Novel preference frequency")
            ax1.set_title("Witte Cultural Transmission Dynamics")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot population dynamics
            ax2.plot(generations, populations, "purple", linewidth=2, label="Population size")
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Population size")
            ax2.set_title("Population Dynamics")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

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
