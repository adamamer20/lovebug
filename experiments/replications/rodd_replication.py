#!/usr/bin/env python3
"""
Rodd Sensory Bias Replication (Adapted for Refactored Model)

This script replicates the core mechanism: a pre-existing sensory bias
(preference that did not co-evolve with the trait) can drive evolution
of a corresponding trait through sexual selection.

Note: Adapted for refactored model using the unlinked gene architecture.

Reference: Rodd, F. H., Hughes, K. A., Grether, G. F., & Baril, C. T. (2002).
A possible non-sexual origin of mate preference: are male guppies mimicking
fruit? Proceedings of the Royal Society B, 269, 475-481.
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


class RoddReplication:
    """
    Replicates Rodd's sensory bias evolution experiment.

    Adapted for refactored model with unlinked genes.
    Tests evolution of display traits toward biased preferences.
    """

    def __init__(self, population_size: int = 1000, n_generations: int = 500, seed: int = 42):
        self.population_size = population_size
        self.n_generations = n_generations
        self.seed = seed
        self.results: dict[str, Any] = {}

    def create_experimental_config(self) -> LoveBugConfig:
        """
        Create configuration for Rodd replication.

        Pure genetic evolution to test whether display traits evolve
        to match pre-existing preference bias.
        """
        return LoveBugConfig(
            name="rodd_sensory_bias",
            genetic=GeneticParams(
                mutation_rate=0.01,  # Moderate mutation for trait evolution
                crossover_rate=0.7,  # Allow recombination of unlinked genes
                max_age=50,  # Reasonable lifespan
                energy_decay=0.01,  # Rodd: High-flow Trinidad pools
                energy_replenishment_rate=0.0067,  # Rule: energy_decay * N₀/K = 0.01 * 2000/3000
                carrying_capacity=self.population_size + 1000,  # K = 3000
            ),
            cultural=CulturalParams(
                learning_rate=0.0,  # Pure genetic evolution - no cultural learning
                horizontal_transmission_rate=0.0,
                oblique_transmission_rate=0.0,
                innovation_rate=0.0,
                network_type="random",
                network_connectivity=0.0,
                cultural_memory_size=1,
                memory_decay_rate=0.0,
                local_learning_radius=1,
                memory_update_strength=0.0,
                learning_strategy="conformist",
            ),
            layer=LayerConfig(
                genetic_enabled=True,
                cultural_enabled=False,  # Genetic-only evolution
            ),
            simulation=SimulationParams(
                population_size=self.population_size,
                steps=self.n_generations,
                seed=self.seed,
            ),
        )

    def initialize_biased_population(self, model: LoveModelRefactored, bias_preference: int = 50000) -> None:
        """
        Initialize population with biased preferences but random display traits.

        This simulates a pre-existing sensory bias in the population.
        All agents have the biased preference while display traits are random.
        """
        current_df = model.get_agent_dataframe()

        # Set all agents to have the biased preference (16-bit value)
        biased_preferences = np.full(len(current_df), bias_preference, dtype=np.uint16)

        # Update the agents with biased preferences but keep random display traits
        updated_df = current_df.with_columns(pl.Series("gene_preference", biased_preferences, dtype=pl.UInt16))

        # Apply the update
        model.agents._agentsets[0].agents = updated_df

        logger.info(f"Initialized population with biased preference: {bias_preference}")

    def run_experiment(self) -> dict[str, Any]:
        """
        Run the Rodd sensory bias experiment.

        1. Initialize with biased preferences but random display traits
        2. Track evolution of display traits toward the biased preference
        3. Monitor trait-preference matching over time
        """
        logger.info("🐠 Starting Rodd sensory bias replication")

        config = self.create_experimental_config()
        model = LoveModelRefactored(config=config)

        # Set the bias preference value
        bias_preference = 50000  # Mid-range 16-bit value representing the bias

        # Initialize with biased preferences
        self.initialize_biased_population(model, bias_preference)

        # Track evolution over time
        evolution_history = []

        logger.info("Tracking trait evolution under biased preference")

        # Record initial state before any evolution
        agents_df = model.get_agent_dataframe()
        if len(agents_df) > 0:
            display_traits = agents_df["gene_display"].to_numpy()
            preferences = agents_df["gene_preference"].to_numpy()

            evolution_history.append(
                {
                    "generation": -1,  # Initial state before evolution
                    "mean_display_trait": np.mean(display_traits),
                    "var_display_trait": np.var(display_traits),
                    "mean_preference": np.mean(preferences),
                    "trait_pref_correlation": 0.0,
                    "distance_from_bias": abs(np.mean(display_traits) - bias_preference),
                    "population_size": len(agents_df),
                }
            )

        for generation in range(self.n_generations):
            model.step()

            # Analyze current population every 25 generations
            if generation % 25 == 0 or generation == self.n_generations - 1:
                agents_df = model.get_agent_dataframe()

                if len(agents_df) > 0:
                    # Extract display traits and preferences
                    display_traits = agents_df["gene_display"].to_numpy()
                    preferences = agents_df["gene_preference"].to_numpy()

                    # Calculate statistics
                    mean_display_trait = np.mean(display_traits)
                    var_display_trait = np.var(display_traits)
                    mean_preference = np.mean(preferences)

                    # Calculate trait-preference correlation (will be NaN with fixed preferences)
                    # Since all preferences are biased to the same value, correlation is undefined
                    # The key metric is convergence of traits toward the bias, not correlation
                    trait_pref_correlation = 0.0  # Always 0 with fixed preferences

                    # Distance from bias preference
                    distance_from_bias = abs(mean_display_trait - bias_preference)

                    evolution_history.append(
                        {
                            "generation": generation,
                            "mean_display_trait": mean_display_trait,
                            "var_display_trait": var_display_trait,
                            "mean_preference": mean_preference,
                            "trait_pref_correlation": trait_pref_correlation,
                            "distance_from_bias": distance_from_bias,
                            "population_size": len(agents_df),
                        }
                    )

                    if generation % 100 == 0:
                        logger.info(
                            f"Gen {generation}: Display trait = {mean_display_trait:.1f}, "
                            f"Distance from bias = {distance_from_bias:.1f}, "
                            f"Population = {len(agents_df)}"
                        )

        # Analyze results
        if evolution_history:
            # Find initial measurement (generation -1 or 0)
            initial_record = next((h for h in evolution_history if h["generation"] in [-1, 0]), evolution_history[0])
            final_record = evolution_history[-1]

            initial_distance = initial_record["distance_from_bias"]
            final_distance = final_record["distance_from_bias"]
            initial_trait = initial_record["mean_display_trait"]
            final_trait = final_record["mean_display_trait"]

            # Check if trait evolved toward bias
            trait_convergence = initial_distance - final_distance
            convergence_percentage = (trait_convergence / initial_distance) * 100 if initial_distance > 0 else 0
            convergence_significant = convergence_percentage > 25  # 25% improvement

            # Check direction of evolution
            evolution_direction = "toward" if final_distance < initial_distance else "away from"

            results = {
                "experiment": "rodd_sensory_bias",
                "population_size": self.population_size,
                "n_generations": self.n_generations,
                "bias_preference": bias_preference,
                "initial_trait": float(initial_trait),
                "final_trait": float(final_trait),
                "initial_distance": float(initial_distance),
                "final_distance": float(final_distance),
                "trait_convergence": float(trait_convergence),
                "convergence_percentage": float(convergence_percentage),
                "convergence_significant": convergence_significant,
                "evolution_direction": evolution_direction,
                "success": convergence_significant and evolution_direction == "toward",
                "evolution_history": evolution_history,
            }
        else:
            results = {
                "experiment": "rodd_sensory_bias",
                "error": "No evolution data collected",
                "success": False,
            }

        logger.info(f"Trait evolution: {results.get('initial_trait', 0):.1f} → {results.get('final_trait', 0):.1f}")
        logger.info(
            f"Distance change: {results.get('trait_convergence', 0):.1f} ({results.get('convergence_percentage', 0):.1f}%)"
        )
        logger.info(f"Evolution direction: {results.get('evolution_direction', 'unknown')}")
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
Rodd Sensory Bias Replication Results (Adapted)
===============================================

Population size: {self.results["population_size"]}
Generations: {self.results["n_generations"]}
Biased preference value: {self.results["bias_preference"]}

Trait Evolution:
  Initial display trait: {self.results["initial_trait"]:.1f}
  Final display trait: {self.results["final_trait"]:.1f}
  Initial distance from bias: {self.results["initial_distance"]:.1f}
  Final distance from bias: {self.results["final_distance"]:.1f}

Convergence Analysis:
  Trait convergence: {self.results["trait_convergence"]:.1f}
  Convergence percentage: {self.results["convergence_percentage"]:.1f}%
  Evolution direction: {self.results["evolution_direction"]} bias
  Convergence significant: {self.results["convergence_significant"]}

Overall Success: {self.results["success"]}

Interpretation:
{"✅ SUCCESS: Display traits evolved toward biased preference" if self.results["success"] else "❌ FAILURE: Display traits did not evolve toward bias"}
{"✅ CONVERGENCE: Traits moved significantly toward bias" if self.results["convergence_significant"] else "❌ NO CONVERGENCE: Trait evolution was insufficient"}
{"📈 DIRECTION: Evolution proceeded toward bias as expected" if self.results["evolution_direction"] == "toward" else "📉 DIRECTION: Evolution moved away from bias"}
        """
        return summary.strip()

    def plot_evolution_dynamics(self) -> None:
        """Plot trait evolution over time (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            if not self.results or not self.results.get("evolution_history"):
                print("No data to plot")
                return

            history = self.results["evolution_history"]
            generations = [h["generation"] for h in history]
            traits = [h["mean_display_trait"] for h in history]
            distances = [h["distance_from_bias"] for h in history]
            populations = [h["population_size"] for h in history]

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

            # Plot trait evolution
            ax1.plot(generations, traits, "b-", linewidth=2, label="Mean display trait")
            ax1.axhline(
                y=self.results["bias_preference"],
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Biased preference ({self.results['bias_preference']})",
            )
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Mean display trait value")
            ax1.set_title("Evolution of Display Traits Under Biased Preference")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot distance from bias
            ax2.plot(generations, distances, "g-", linewidth=2, label="Distance from bias")
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Distance from biased preference")
            ax2.set_title("Convergence to Biased Preference")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot population size
            ax3.plot(generations, populations, "purple", linewidth=2, label="Population size")
            ax3.set_xlabel("Generation")
            ax3.set_ylabel("Population size")
            ax3.set_title("Population Dynamics")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib not available - cannot plot dynamics")


def main():
    """Run Rodd replication experiment."""
    logging.basicConfig(level=logging.INFO)

    replication = RoddReplication(population_size=1000, n_generations=500, seed=42)
    results = replication.run_experiment()

    print(replication.get_summary())

    # Optionally plot results
    try:
        replication.plot_evolution_dynamics()
    except Exception as e:
        print(f"Could not plot results: {e}")

    return results


if __name__ == "__main__":
    main()
