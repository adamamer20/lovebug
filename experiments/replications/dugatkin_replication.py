#!/usr/bin/env python3
"""
Dugatkin Mate-Choice Copying Replication (Adapted for Refactored Model)

This script replicates the core mechanism of Dugatkin's experiment:
social learning can override genetic preferences when observed choices
have high salience/prestige.

Note: Adapted for the refactored model which doesn't have explicit sex
differentiation. Instead, we simulate preference copying between agents.

Reference: Dugatkin, L. A. (1992). Sexual selection and imitation: females copy
the mate choice of others. American Naturalist, 139, 1384-1389.
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
    LayerConfig,
    LoveBugConfig,
    SimulationParams,
)
from lovebug.model import LoveModelRefactored

logger = logging.getLogger(__name__)


class DugatkinReplication:
    """
    Replicates the core mechanism of Dugatkin's mate-choice copying experiment.

    Adapted for refactored model without explicit sex differentiation.
    Tests whether agents can copy preferences from high-prestige individuals.
    """

    def __init__(self, population_size: int = 20, seed: int = 42):
        self.population_size = population_size
        self.seed = seed
        self.results: dict[str, Any] = {}

    def create_experimental_config(self) -> LoveBugConfig:
        """
        Create configuration for Dugatkin replication.

        Small population with high cultural transmission to model
        the salience of observed preference copying.
        """
        return LoveBugConfig(
            name="dugatkin_preference_copying",
            genetic=GeneticParams(
                h2_trait=0.8,
                h2_preference=0.8,
                mutation_rate=0.001,  # Very low to maintain initial preferences
                crossover_rate=0.0,  # No recombination to keep preferences stable
                elitism=1,
                energy_decay=0.010,  # Reduced from 0.015 for stability
                mutation_variance=0.01,
                max_age=1000,  # Effectively immortal for short experiment
                carrying_capacity=60,  # Increased from 40 for more stable environment
                energy_replenishment_rate=0.0033,  # r = d * N₀/K = 0.010 * 20/60 = 0.0033
                parental_investment_rate=0.6,
                energy_min_mating=1.0,
                juvenile_cost=0.5,
                display_cost_scalar=0.2,
                search_cost=0.01,
                base_energy=10.0,
            ),
            cultural=CulturalParams(
                innovation_rate=0.0,  # No innovation during observation phase
                memory_span=5,
                network_type="random",  # Fully connected-like for small population
                network_connectivity=1.0,
                cultural_memory_size=5,
                memory_decay_rate=0.01,
                horizontal_transmission_rate=0.9,  # Very high - models high salience
                oblique_transmission_rate=0.1,
                local_learning_radius=5,
                memory_update_strength=1.0,
                learning_strategy="success-biased",  # Copy successful individuals
            ),
            layer=LayerConfig(
                genetic_enabled=True,
                cultural_enabled=True,
                blending_mode="weighted",
                genetic_weight=0.2,  # Cultural learning can override genetics
                cultural_weight=0.8,
                sigma_perception=0.0,
                theta_detect=0.0,
            ),
            simulation=SimulationParams(
                population_size=self.population_size,
                steps=20,  # Short experiment
                seed=self.seed,
            ),
        )

    def run_experiment(self) -> dict[str, Any]:
        """
        Run the adapted Dugatkin experiment.

        Phase A: Establish initial preferences
        Phase B: Simulate high-prestige choice (by boosting mating success)
        Phase C: Test if preferences copied toward prestige choice
        """
        logger.info("🐠 Starting Dugatkin preference copying replication")

        config = self.create_experimental_config()
        model = LoveModelRefactored(config=config)

        # Phase A: Establish baseline and identify preference diversity
        logger.info("Phase A: Establishing baseline preferences")

        # Run a few steps to establish baseline
        for _ in range(3):
            model.step()

        # Get initial agent state
        initial_df = model.get_agent_dataframe()

        if len(initial_df) == 0:
            raise ValueError("No agents found in population")

        # Record initial genetic and cultural preferences
        initial_genetic_prefs = initial_df["gene_preference"].to_numpy()
        initial_cultural_prefs = initial_df["cultural_preference"].to_numpy()

        # Identify an agent with a distinct preference (the "observer")
        observer_idx = 0
        observer_genetic_pref = initial_genetic_prefs[observer_idx]
        observer_cultural_pref = initial_cultural_prefs[observer_idx]

        # Find the most different preference in population (the "model's choice")
        preference_distances = np.abs(initial_genetic_prefs - observer_genetic_pref)
        most_different_idx = np.argmax(preference_distances)
        target_preference = initial_genetic_prefs[most_different_idx]

        logger.info(f"Observer preference: {observer_genetic_pref}")
        logger.info(f"Target (different) preference: {target_preference}")

        # Phase B: Create prestige signal - boost mating success of target preference holders
        logger.info("Phase B: Creating prestige signal for target preference")

        # Identify agents with preferences similar to target
        similarity_threshold = 1000  # Relatively broad similarity
        target_similar_mask = np.abs(initial_genetic_prefs - target_preference) < similarity_threshold

        if not target_similar_mask.any():
            # If no similar agents, pick the closest one
            closest_idx = np.argmin(np.abs(initial_genetic_prefs - target_preference))
            target_similar_mask = np.zeros(len(initial_genetic_prefs), dtype=bool)
            target_similar_mask[closest_idx] = True

        # Artificially boost mating success for target preference holders
        current_df = model.get_agent_dataframe()
        boosted_success = current_df["mating_success"].to_numpy().copy()
        boosted_success[target_similar_mask] = 10  # High mating success signal

        # Update the model (this is a simplification - normally would be through mating)
        # For the refactored model, we need to update the agents dataframe
        updated_df = current_df.with_columns(pl.Series("mating_success", boosted_success, dtype=pl.UInt16))

        # Apply the update (this is a hack for the experiment)
        model.agents._agentsets[0].agents = updated_df

        logger.info(f"Boosted mating success for {target_similar_mask.sum()} agents")

        # Phase C: Allow cultural transmission to occur
        logger.info("Phase C: Allowing preference copying")

        # Run several steps to allow cultural learning
        for _ in range(5):
            model.step()

        # Phase D: Measure preference change
        logger.info("Phase D: Measuring preference change")

        final_df = model.get_agent_dataframe()
        final_cultural_prefs = final_df["cultural_preference"].to_numpy()

        # Check if observer copied toward target preference
        observer_final_cultural = final_cultural_prefs[observer_idx]
        initial_distance = abs(observer_cultural_pref - target_preference)
        final_distance = abs(observer_final_cultural - target_preference)
        preference_shift = initial_distance - final_distance

        # Calculate population-level preference shift toward target
        initial_pop_distance = np.mean(np.abs(initial_cultural_prefs - target_preference))
        final_pop_distance = np.mean(np.abs(final_cultural_prefs - target_preference))
        population_shift = initial_pop_distance - final_pop_distance

        # Compile results
        results = {
            "experiment": "dugatkin_preference_copying",
            "population_size": self.population_size,
            "observer_initial_genetic": int(observer_genetic_pref),
            "observer_initial_cultural": int(observer_cultural_pref),
            "observer_final_cultural": int(observer_final_cultural),
            "target_preference": int(target_preference),
            "observer_preference_shift": float(preference_shift),
            "population_preference_shift": float(population_shift),
            "observer_copied": preference_shift > 0,
            "population_copied": population_shift > 0,
            "success": preference_shift > 0 and population_shift > 0,
        }

        logger.info(f"Observer preference shift: {preference_shift}")
        logger.info(f"Population preference shift: {population_shift}")
        logger.info(f"Observer copied: {results['observer_copied']}")
        logger.info(f"Population copied: {results['population_copied']}")
        logger.info(f"Experiment success: {results['success']}")

        self.results = results
        return results

    def get_summary(self) -> str:
        """Generate human-readable summary of results."""
        if not self.results:
            return "No results available - run experiment first"

        summary = f"""
Dugatkin Preference Copying Replication Results (Adapted)
========================================================

Population size: {self.results["population_size"]}
Observer initial preference: {self.results["observer_initial_cultural"]}
Target preference: {self.results["target_preference"]}
Observer final preference: {self.results["observer_final_cultural"]}

Preference shifts:
  Observer: {self.results["observer_preference_shift"]:.1f}
  Population: {self.results["population_preference_shift"]:.1f}

Outcomes:
  Observer copied: {self.results["observer_copied"]}
  Population copied: {self.results["population_copied"]}
  Experiment successful: {self.results["success"]}

Interpretation:
{"✅ SUCCESS: Agents copied preferences toward high-prestige choice" if self.results["success"] else "❌ FAILURE: No preference copying occurred"}
{"✅ INDIVIDUAL: Observer agent shifted preference" if self.results["observer_copied"] else "❌ NO INDIVIDUAL EFFECT"}
{"✅ POPULATION: Overall population shifted preferences" if self.results["population_copied"] else "❌ NO POPULATION EFFECT"}
        """
        return summary.strip()


def main():
    """Run Dugatkin replication experiment."""
    logging.basicConfig(level=logging.INFO)

    replication = DugatkinReplication(population_size=20, seed=42)
    results = replication.run_experiment()

    print(replication.get_summary())

    return results


if __name__ == "__main__":
    main()
