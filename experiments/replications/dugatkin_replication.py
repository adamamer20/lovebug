#!/usr/bin/env python3
"""
Dugatkin Mate-Choice Copying Replication

This script replicates the famous Dugatkin experiment showing that an observer
female's preference can be reversed by watching a "model" female choose a
previously non-preferred male.

Reference: Dugatkin, L. A. (1992). Sexual selection and imitation: females copy
the mate choice of others. American Naturalist, 139, 1384-1389.

Key Mechanism: Social learning can override genetic preferences when the
observed choice has high salience/prestige.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

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


class DugatkinReplication:
    """Replicates Dugatkin's mate-choice copying experiment."""

    def __init__(self, population_size: int = 20, seed: int = 42):
        self.population_size = population_size
        self.seed = seed
        self.results: dict[str, Any] = {}

    def create_experimental_config(self) -> LoveBugConfig:
        """
        Create configuration for Dugatkin replication.

        Small population mimicking lab tank conditions, with high cultural
        transmission to model the salience of observed mate choice.
        """
        return LoveBugConfig(
            name="dugatkin_mate_choice_copying",
            genetic=GeneticParams(
                h2_trait=0.8,  # Strong genetic preferences initially
                h2_preference=0.8,
                mutation_rate=0.001,  # Very low to maintain initial preferences
                crossover_rate=0.0,  # No recombination to keep preferences stable
                population_size=self.population_size,
                elitism=1,
                energy_decay=0.0,  # No aging effects
                mutation_variance=0.001,
                max_age=1000,  # Effectively immortal for short experiment
                carrying_capacity=self.population_size,
            ),
            cultural=CulturalParams(
                learning_rate=0.9,  # Very high - models high salience of observation
                innovation_rate=0.0,  # No innovation during observation phase
                network_type="complete",  # Everyone can observe everyone
                network_connectivity=1.0,
                cultural_memory_size=1,  # Only remember most recent observation
                memory_decay_rate=0.0,  # Perfect memory during experiment
                horizontal_transmission_rate=0.9,  # High transmission of observed choice
                oblique_transmission_rate=0.0,  # No oblique transmission
                local_learning_radius=self.population_size,  # Can learn from anyone
                memory_update_strength=1.0,  # Perfect memory updates
            ),
            blending=LayerBlendingParams(
                blend_mode="weighted",
                blend_weight=0.2,  # Cultural learning can override genetics
            ),
            perceptual=PerceptualParams(),
            simulation=SimulationParams(
                population_size=self.population_size,
                steps=20,  # Short experiment
                seed=self.seed,
            ),
        )

    def run_experiment(self) -> dict[str, Any]:
        """
        Run the three-phase Dugatkin experiment.

        Phase A (Pre-test): Identify observer's initial preference
        Phase B (Observation): Force high-prestige choice of non-preferred male
        Phase C (Post-test): Test if observer's preference has changed
        """
        logger.info("🐠 Starting Dugatkin mate-choice copying replication")

        config = self.create_experimental_config()
        model = LoveModel(config=config)

        # Phase A: Pre-test - identify initial preferences
        logger.info("Phase A: Pre-test - identifying initial preferences")

        # Run a few steps to establish baseline
        for _ in range(3):
            model.step()

        # Record initial state
        agents_df = model.agents._agents

        # Identify observer female (first female) and her genetic preference
        females = agents_df.filter(pl.col("sex") == "female")
        if len(females) == 0:
            raise ValueError("No females found in population")

        observer_id = females[0, "unique_id"]
        observer_genome = females[0, "genome"]

        # Extract genetic preference from genome (bits 16-23)
        genetic_preference = (observer_genome >> 16) & 0xFF

        # Identify males and find the one closest/farthest from preference
        males = agents_df.filter(pl.col("sex") == "male")
        if len(males) < 2:
            raise ValueError("Need at least 2 males for preference reversal test")

        # Calculate display trait values for males (bits 0-15)
        male_traits = [(male_id, (genome & 0xFFFF)) for male_id, genome in zip(males["unique_id"], males["genome"])]

        # Find preferred and non-preferred males based on genetic preference
        preference_distances = [(male_id, abs(trait - genetic_preference)) for male_id, trait in male_traits]
        preference_distances.sort(key=lambda x: x[1])

        preferred_male_id = preference_distances[0][0]  # Closest to preference
        non_preferred_male_id = preference_distances[-1][0]  # Farthest from preference

        logger.info(f"Observer {observer_id} genetically prefers male {preferred_male_id}")
        logger.info(f"Non-preferred male: {non_preferred_male_id}")

        # Phase B: Observation - force model female to choose non-preferred male
        logger.info("Phase B: Observation - model female chooses non-preferred male")

        # Find a model female (not the observer)
        model_females = females.filter(pl.col("unique_id") != observer_id)
        if len(model_females) == 0:
            raise ValueError("Need at least 2 females for model/observer setup")

        model_female_id = model_females[0, "unique_id"]

        # Temporarily modify cultural layer to force high-prestige choice
        # This simulates the experimental manipulation where the model female
        # is observed choosing the non-preferred male

        # Store original cultural preferences
        original_cultural_prefs = {}
        agents_data = model.agents._agents
        if "pref_culture" in agents_data.columns:
            for row in agents_data.iter_rows(named=True):
                original_cultural_prefs[row["unique_id"]] = row.get("pref_culture", 0)

        # Set model female's cultural preference to the non-preferred male's trait
        non_preferred_trait = next(trait for male_id, trait in male_traits if male_id == non_preferred_male_id)

        # Update model female's cultural preference to demonstrate choice
        # This simulates the experimental manipulation
        if hasattr(model, "cultural_layer") and model.cultural_layer is not None:
            # Manually set cultural preference for model female
            agents_data = model.agents._agents
            if "pref_culture" in agents_data.columns:
                model.agents._agents = agents_data.with_columns(
                    pl.when(pl.col("unique_id") == model_female_id)
                    .then(pl.lit(non_preferred_trait))
                    .otherwise(pl.col("pref_culture"))
                    .alias("pref_culture")
                )
                logger.info(f"Model female {model_female_id} now culturally prefers trait {non_preferred_trait}")

        # Run observation phase
        for _ in range(3):
            model.step()

        # Phase C: Post-test - check if observer's preference changed
        logger.info("Phase C: Post-test - checking preference change")

        # Get updated agent state
        final_agents_df = model.agents._agents
        observer_final = final_agents_df.filter(pl.col("unique_id") == observer_id)

        if len(observer_final) == 0:
            raise ValueError("Observer female not found in final state")

        # Check cultural preference of observer
        final_cultural_pref = None
        if "pref_culture" in final_agents_df.columns:
            observer_row = final_agents_df.filter(pl.col("unique_id") == observer_id)
            if len(observer_row) > 0:
                final_cultural_pref = observer_row[0, "pref_culture"]

        # Calculate preference shift
        preference_shift = 0
        if final_cultural_pref is not None:
            original_distance_to_nonpreferred = abs(genetic_preference - non_preferred_trait)
            final_distance_to_nonpreferred = abs(final_cultural_pref - non_preferred_trait)
            preference_shift = original_distance_to_nonpreferred - final_distance_to_nonpreferred

        # Compile results
        results = {
            "experiment": "dugatkin_mate_choice_copying",
            "population_size": self.population_size,
            "observer_id": observer_id,
            "model_female_id": model_female_id,
            "preferred_male_id": preferred_male_id,
            "non_preferred_male_id": non_preferred_male_id,
            "initial_genetic_preference": genetic_preference,
            "non_preferred_trait": non_preferred_trait,
            "final_cultural_preference": final_cultural_pref,
            "preference_shift": preference_shift,
            "preference_reversed": preference_shift > (genetic_preference * 0.5),  # Significant shift
            "success": preference_shift > 0,  # Any movement toward non-preferred
        }

        logger.info(f"Preference shift: {preference_shift}")
        logger.info(f"Preference reversed: {results['preference_reversed']}")
        logger.info(f"Experiment success: {results['success']}")

        self.results = results
        return results

    def get_summary(self) -> str:
        """Generate human-readable summary of results."""
        if not self.results:
            return "No results available - run experiment first"

        summary = f"""
Dugatkin Mate-Choice Copying Replication Results
===============================================

Population size: {self.results["population_size"]}
Observer female: {self.results["observer_id"]}
Model female: {self.results["model_female_id"]}

Initial genetic preference: {self.results["initial_genetic_preference"]}
Non-preferred male trait: {self.results["non_preferred_trait"]}
Final cultural preference: {self.results["final_cultural_preference"]}

Preference shift: {self.results["preference_shift"]}
Preference reversed: {self.results["preference_reversed"]}
Experiment successful: {self.results["success"]}

Interpretation:
{"✅ SUCCESS: Observer female shifted preference toward non-preferred male after observation" if self.results["success"] else "❌ FAILURE: Observer female did not shift preference"}
{"✅ STRONG EFFECT: Preference was significantly reversed" if self.results["preference_reversed"] else "⚠️  WEAK EFFECT: Preference shift was modest"}
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
