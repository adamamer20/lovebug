#!/usr/bin/env python3
"""
Rodd Sensory Bias Replication

This script replicates experiments showing that a pre-existing sensory bias
(a preference that did not co-evolve with the trait) can drive the evolution
of a corresponding male trait.

Reference: Rodd, F. H., Hughes, K. A., Grether, G. F., & Baril, C. T. (2002).
A possible non-sexual origin of mate preference: are male guppies mimicking
fruit? Proceedings of the Royal Society B, 269, 475-481.

Key Mechanism: Pre-existing biased preferences can drive evolution of matching
traits even without initial genetic correlation.
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


class RoddReplication:
    """Replicates Rodd's sensory bias evolution experiment."""

    def __init__(self, population_size: int = 1000, n_generations: int = 500, seed: int = 42):
        self.population_size = population_size
        self.n_generations = n_generations
        self.seed = seed
        self.results: dict[str, Any] = {}

    def create_experimental_config(self) -> LoveBugConfig:
        """
        Create configuration for Rodd replication.

        Pure genetic evolution with biased initial preferences to test
        whether traits evolve to match pre-existing preference bias.
        """
        return LoveBugConfig(
            name="rodd_sensory_bias",
            genetic=GeneticParams(
                h2_trait=0.8,  # High heritability for trait evolution
                h2_preference=0.9,  # High heritability to maintain preference bias
                mutation_rate=0.01,  # Moderate mutation for trait evolution
                crossover_rate=0.7,  # Allow recombination
                population_size=self.population_size,
                elitism=5,  # Some elitism to maintain good genotypes
                energy_decay=0.02,  # Moderate aging
                mutation_variance=0.02,  # Moderate mutation effects
                max_age=50,  # Reasonable lifespan
                carrying_capacity=self.population_size,
            ),
            cultural=CulturalParams(
                learning_rate=0.0,  # No cultural learning
                innovation_rate=0.0,
                network_type="scale_free",
                network_connectivity=0.0,
                cultural_memory_size=1,
                memory_decay_rate=1.0,
                horizontal_transmission_rate=0.0,
                oblique_transmission_rate=0.0,
                local_learning_radius=1,
                memory_update_strength=0.0,
            ),
            blending=LayerBlendingParams(
                blend_mode="weighted",
                blend_weight=1.0,  # Pure genetic evolution
            ),
            perceptual=PerceptualParams(),
            simulation=SimulationParams(
                population_size=self.population_size,
                steps=self.n_generations,
                seed=self.seed,
            ),
        )

    def initialize_biased_population(self, model: LoveModel, bias_preference: int = 240) -> None:
        """
        Initialize population with biased preferences but random traits.

        This simulates a pre-existing sensory bias in the population.
        Females have a biased preference (e.g., for "orange" = high bit values)
        while male traits are initially random.
        """
        agents_data = model.agents._agents

        # Set all females to have the biased preference
        # Preference is stored in bits 16-23 of the genome
        biased_preference_bits = (bias_preference & 0xFF) << 16

        # Keep other genome parts but set preference bits for females
        updated_genomes = []
        for row in agents_data.iter_rows(named=True):
            genome = row["genome"]
            if row["sex"] == "female":
                # Clear preference bits and set to biased value
                genome = (genome & ~0x00FF0000) | biased_preference_bits
            updated_genomes.append(genome)

        # Update the agents data
        model.agents._agents = agents_data.with_columns(pl.Series("genome", updated_genomes, dtype=pl.UInt32))

        logger.info(f"Initialized population with biased female preference: {bias_preference}")

    def run_experiment(self) -> dict[str, Any]:
        """
        Run the Rodd sensory bias experiment.

        1. Initialize with biased female preferences but random male traits
        2. Track evolution of male traits toward the biased preference
        3. Monitor genetic correlation development over time
        """
        logger.info("🐠 Starting Rodd sensory bias replication")

        config = self.create_experimental_config()
        model = LoveModel(config=config)

        # Set the bias preference value (high values = "orange-like")
        bias_preference = 240  # High bit pattern representing "orange"

        # Initialize with biased preferences
        self.initialize_biased_population(model, bias_preference)

        # Track evolution over time
        evolution_history = []

        logger.info("Tracking trait evolution under biased preference")

        for generation in range(self.n_generations):
            model.step()

            # Analyze current population every 25 generations
            if generation % 25 == 0 or generation == self.n_generations - 1:
                agents_data = model.agents._agents

                # Separate males and females
                males = agents_data.filter(pl.col("sex") == "male")
                females = agents_data.filter(pl.col("sex") == "female")

                if len(males) > 0 and len(females) > 0:
                    # Extract traits and preferences from genomes
                    male_traits = [(genome & 0xFFFF) for genome in males["genome"]]
                    female_prefs = [((genome >> 16) & 0xFF) for genome in females["genome"]]

                    # Calculate statistics
                    mean_male_trait = np.mean(male_traits)
                    var_male_trait = np.var(male_traits)
                    mean_female_pref = np.mean(female_prefs)

                    # Calculate trait-preference correlation within individuals
                    # (This should start at 0 and potentially increase due to assortative mating)
                    all_traits = [(genome & 0xFFFF) for genome in agents_data["genome"]]
                    all_prefs = [((genome >> 16) & 0xFF) for genome in agents_data["genome"]]

                    if len(all_traits) > 1:
                        trait_pref_correlation = np.corrcoef(all_traits, all_prefs)[0, 1]
                        if np.isnan(trait_pref_correlation):
                            trait_pref_correlation = 0.0
                    else:
                        trait_pref_correlation = 0.0

                    # Distance from bias preference
                    distance_from_bias = abs(mean_male_trait - bias_preference)

                    evolution_history.append(
                        {
                            "generation": generation,
                            "mean_male_trait": mean_male_trait,
                            "var_male_trait": var_male_trait,
                            "mean_female_pref": mean_female_pref,
                            "trait_pref_correlation": trait_pref_correlation,
                            "distance_from_bias": distance_from_bias,
                        }
                    )

                    if generation % 100 == 0:
                        logger.info(
                            f"Gen {generation}: Male trait = {mean_male_trait:.1f}, "
                            f"Distance from bias = {distance_from_bias:.1f}, "
                            f"Correlation = {trait_pref_correlation:.3f}"
                        )

        # Analyze results
        if evolution_history:
            initial_distance = evolution_history[0]["distance_from_bias"]
            final_distance = evolution_history[-1]["distance_from_bias"]
            initial_correlation = evolution_history[0]["trait_pref_correlation"]
            final_correlation = evolution_history[-1]["trait_pref_correlation"]

            # Check if trait evolved toward bias
            trait_convergence = initial_distance - final_distance
            convergence_significant = trait_convergence > (initial_distance * 0.25)  # 25% improvement

            # Check if genetic correlation developed
            correlation_development = final_correlation - initial_correlation
            correlation_significant = correlation_development > 0.2  # Substantial correlation

            results = {
                "experiment": "rodd_sensory_bias",
                "population_size": self.population_size,
                "n_generations": self.n_generations,
                "bias_preference": bias_preference,
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "trait_convergence": trait_convergence,
                "convergence_significant": convergence_significant,
                "initial_correlation": initial_correlation,
                "final_correlation": final_correlation,
                "correlation_development": correlation_development,
                "correlation_significant": correlation_significant,
                "success": convergence_significant and correlation_significant,
                "evolution_history": evolution_history,
            }
        else:
            results = {
                "experiment": "rodd_sensory_bias",
                "error": "No evolution data collected",
                "success": False,
            }

        logger.info(f"Trait convergence: {results.get('trait_convergence', 0):.1f}")
        logger.info(f"Convergence significant: {results.get('convergence_significant', False)}")
        logger.info(f"Correlation development: {results.get('correlation_development', 0):.3f}")
        logger.info(f"Correlation significant: {results.get('correlation_significant', False)}")
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
Rodd Sensory Bias Replication Results
====================================

Population size: {self.results["population_size"]}
Generations: {self.results["n_generations"]}
Biased preference value: {self.results["bias_preference"]}

Trait Evolution:
  Initial distance from bias: {self.results["initial_distance"]:.1f}
  Final distance from bias: {self.results["final_distance"]:.1f}
  Trait convergence: {self.results["trait_convergence"]:.1f}
  Convergence significant: {self.results["convergence_significant"]}

Genetic Correlation:
  Initial correlation: {self.results["initial_correlation"]:.3f}
  Final correlation: {self.results["final_correlation"]:.3f}
  Correlation development: {self.results["correlation_development"]:.3f}
  Correlation significant: {self.results["correlation_significant"]}

Overall Success: {self.results["success"]}

Interpretation:
{"✅ SUCCESS: Male traits evolved toward biased female preference" if self.results["convergence_significant"] else "❌ NO CONVERGENCE: Male traits did not evolve toward bias"}
{"✅ CORRELATION: Genetic correlation developed between trait and preference" if self.results["correlation_significant"] else "❌ NO CORRELATION: No genetic correlation developed"}
{"✅ FULL SUCCESS: Both trait evolution and correlation development occurred" if self.results["success"] else "⚠️  PARTIAL/NO SUCCESS: Sensory bias mechanism not fully demonstrated"}
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
            traits = [h["mean_male_trait"] for h in history]
            correlations = [h["trait_pref_correlation"] for h in history]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Plot trait evolution
            ax1.plot(generations, traits, "b-", linewidth=2, label="Mean male trait")
            ax1.axhline(
                y=self.results["bias_preference"],
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Biased preference ({self.results['bias_preference']})",
            )
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Mean male trait value")
            ax1.set_title("Evolution of Male Traits Under Biased Preference")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot correlation development
            ax2.plot(generations, correlations, "g-", linewidth=2, label="Trait-preference correlation")
            ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.7)
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Genetic correlation")
            ax2.set_title("Development of Trait-Preference Genetic Correlation")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

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
