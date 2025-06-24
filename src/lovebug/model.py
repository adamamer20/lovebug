"""
Refactored Unified Mesa-Frames Model implementing the biological refactoring plan.

This module provides a scientifically robust, biologically defensible model
implementing unlinked multi-gene architecture, endogenous population regulation,
and configurable social learning strategies.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from beartype import beartype
from mesa_frames import AgentSetPolars, ModelDF

from lovebug.config import LoveBugConfig
from lovebug.layer2.network import NetworkTopology, SocialNetwork

__all__ = ["LoveAgentsRefactored", "LoveModelRefactored", "ParameterHelper"]

logger = logging.getLogger(__name__)


class ParameterHelper:
    """
    Helper class for calculating biologically meaningful parameter ratios.

    This addresses the "too many knobs" problem by providing functions to compute
    raw parameter values from biologically interpretable ratios.
    """

    @staticmethod
    def calculate_energy_balance_ratio(energy_replenishment_rate: float, energy_decay: float) -> float:
        """Calculate daily net energy balance ratio (raw, without density effects)."""
        return (energy_replenishment_rate - energy_decay) / energy_decay

    @staticmethod
    def calculate_effective_energy_balance_ratio(
        energy_replenishment_rate: float, energy_decay: float, current_pop: int, carrying_capacity: int
    ) -> float:
        """Calculate effective energy balance ratio accounting for current density."""
        density_factor = max(0.1, 1.0 - (current_pop / carrying_capacity) ** 2)
        effective_replenishment = energy_replenishment_rate * density_factor
        return (effective_replenishment - energy_decay) / energy_decay

    @staticmethod
    def calculate_female_choosiness_ratio(mean_gene_threshold: float) -> float:
        """Calculate female choosiness as fraction of max threshold (16)."""
        return mean_gene_threshold / 16.0

    @staticmethod
    def calculate_display_cost_ratio(display_cost_scalar: float, mean_display_bits: float) -> float:
        """Calculate viability cost of display as fraction of foraging efficiency."""
        return display_cost_scalar * (mean_display_bits / 16.0)

    @staticmethod
    def calculate_juvenile_bottleneck_ratio(
        juvenile_cost: float, parental_investment_rate: float, base_energy: float
    ) -> float:
        """Calculate juvenile bottleneck intensity."""
        return juvenile_cost / (parental_investment_rate * base_energy)

    @staticmethod
    def tune_energy_balance(target_ratio: float = 0.05, energy_decay: float = 0.01) -> float:
        """
        Calculate energy_replenishment_rate for a target net energy balance.

        Target ratio of ~0.05-0.1 keeps populations near carrying capacity.
        """
        return energy_decay * (1 + target_ratio)

    @staticmethod
    def tune_female_choosiness(target_ratio: float = 0.8) -> float:
        """
        Calculate gene_threshold for target female choosiness.

        Target ratio ≥ 0.8 creates strong sensory bias.
        """
        return target_ratio * 16.0

    @staticmethod
    def tune_display_cost(target_ratio: float = 0.1) -> float:
        """
        Calculate display_cost_scalar for target viability cost.

        Target ratio ≤ 0.1 for bias tests, > 0.2 for handicap tests.
        """
        # Assumes mean display will be ~8 bits (half of 16)
        mean_expected_bits = 8.0
        return target_ratio / (mean_expected_bits / 16.0)

    @staticmethod
    def check_parameter_health(config, actual_thresholds=None) -> dict[str, float]:
        """
        Check key parameter ratios for biological plausibility.

        Args:
            config: Model configuration
            actual_thresholds: Optional array of actual population thresholds for accurate choosiness calculation

        Returns a dict of key ratios that should be monitored.
        """
        energy_balance = ParameterHelper.calculate_energy_balance_ratio(
            config.genetic.energy_replenishment_rate, config.genetic.energy_decay
        )

        # Use actual population mean if provided, otherwise estimate from 4-bit range
        if actual_thresholds is not None:
            mean_threshold = float(np.mean(actual_thresholds))
        else:
            # Default estimate for 4-bit threshold: uniform 0-15 → mean ≈ 7.5
            mean_threshold = 7.5

        choosiness = ParameterHelper.calculate_female_choosiness_ratio(mean_threshold)

        # Estimate display cost using scalar and assuming 8 average bits
        display_cost = ParameterHelper.calculate_display_cost_ratio(config.genetic.display_cost_scalar, 8.0)

        juvenile_bottleneck = ParameterHelper.calculate_juvenile_bottleneck_ratio(
            config.genetic.juvenile_cost, config.genetic.parental_investment_rate, config.genetic.base_energy
        )

        return {
            "energy_balance_ratio": energy_balance,
            "female_choosiness_ratio": choosiness,
            "display_cost_ratio": display_cost,
            "juvenile_bottleneck_ratio": juvenile_bottleneck,
        }


def hamming_similarity_16bit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate Hamming similarity between two 16-bit arrays using optimized NumPy operations.

    PERFORMANCE OPTIMIZATION: Uses vectorized bit counting instead of Python loops.
    Returns similarity score as 16 - Hamming distance.
    """
    diff = a ^ b
    # Vectorized bit counting using Brian Kernighan's algorithm optimized for NumPy
    hamming_distance = np.zeros(len(diff), dtype=np.uint8)
    for i in range(16):  # 16 bits maximum
        hamming_distance += (diff >> i) & 1
    return 16 - hamming_distance


def hamming_similarity_16bit_polars(a: pl.Series, b: pl.Series) -> pl.Series:
    """
    Calculate Hamming similarity using fully native Polars operations.

    ULTIMATE PERFORMANCE: Uses Rust-optimized bitwise_count_ones() with zero conversions.
    Returns similarity score as 16 - Hamming distance.
    """
    # Ensure series are of the correct type
    a = a.cast(pl.UInt16)
    b = b.cast(pl.UInt16)

    # XOR to find differing bits, then count them using Rust-optimized function
    hamming_distance = (a ^ b).bitwise_count_ones()
    return 16 - hamming_distance


class LoveAgentsRefactored(AgentSetPolars):
    """
    Refactored agent set implementing unlinked multi-gene architecture.

    Key changes from original:
    1. Replaced single 'genome' with separate gene columns
    2. Implemented endogenous population regulation via energy system
    3. Added configurable social learning strategies
    4. Standardized 16-bit trait-preference comparison
    """

    def __init__(self, n: int, model: LoveModelRefactored):
        super().__init__(model)
        self.config = model.config

        # Initialize unlinked genes
        gene_display = np.random.randint(0, 2**16, size=n, dtype=np.uint16)
        gene_preference = np.random.randint(0, 2**16, size=n, dtype=np.uint16)
        gene_threshold = np.random.randint(0, 2**4, size=n, dtype=np.uint8)  # 4-bit value (0-15)
        gene_foraging_efficiency = np.random.randint(0, 2**8, size=n, dtype=np.uint8)

        # Initialize sex differentiation (0 = female, 1 = male)
        sex = np.random.randint(0, 2, size=n, dtype=np.uint8)

        # Initialize with stable age distribution
        max_age = self.config.genetic.max_age
        ages = np.random.randint(0, max_age, size=n, dtype=np.uint16)

        # Initialize energy based on age and foraging efficiency
        base_energy = self.config.genetic.base_energy
        age_penalty = ages * self.config.genetic.energy_decay
        efficiency_bonus = (gene_foraging_efficiency / 255.0) * 5.0
        initial_energy = base_energy - age_penalty + efficiency_bonus
        initial_energy = np.maximum(initial_energy, 0.1)  # Minimum survival energy

        # Create base dataframe with genetic traits
        df_data = {
            "sex": pl.Series(sex, dtype=pl.UInt8),
            "gene_display": pl.Series(gene_display, dtype=pl.UInt16),
            "gene_preference": pl.Series(gene_preference, dtype=pl.UInt16),
            "gene_threshold": pl.Series(gene_threshold, dtype=pl.UInt8),
            "gene_foraging_efficiency": pl.Series(gene_foraging_efficiency, dtype=pl.UInt8),
            "energy": pl.Series(initial_energy, dtype=pl.Float32),
            "age": pl.Series(ages, dtype=pl.UInt16),
            "mating_success": pl.Series([0] * n, dtype=pl.UInt16),
        }

        # Add cultural traits if cultural layer is enabled
        if self.config.layer.cultural_enabled:
            df_data["cultural_preference"] = pl.Series(
                np.random.randint(0, 2**16, size=n, dtype=np.uint16), dtype=pl.UInt16
            )
            df_data["cultural_innovation_count"] = pl.Series([0] * n, dtype=pl.UInt16)
            df_data["social_network_neighbors"] = pl.Series([[] for _ in range(n)], dtype=pl.List(pl.UInt32))

        # Add effective preference for combined models
        if self.config.layer.is_combined():
            df_data["effective_preference"] = pl.Series(gene_preference, dtype=pl.UInt16)

        self += pl.DataFrame(df_data)
        self._mask = pl.repeat(True, len(self.agents), dtype=pl.Boolean, eager=True)

        # Initialize social network if cultural layer enabled
        if self.config.layer.cultural_enabled:
            self._initialize_social_network()

        logger.debug(f"Initialized {n} refactored agents with unlinked genes")
        # Log dtypes for debugging Int32/UInt32 mismatch
        logger.info(
            f"social_network_neighbors dtype: {self.agents['social_network_neighbors'].dtype if 'social_network_neighbors' in self.agents.columns else 'N/A'}"
        )

    def _initialize_social_network(self) -> None:
        """Initialize social network for cultural transmission."""
        if not self.config.layer.cultural_enabled:
            return

        topology = NetworkTopology(
            network_type=self.config.cultural.network_type,
            connectivity=self.config.cultural.network_connectivity,
        )
        self.social_network = SocialNetwork(len(self), topology)

        # Populate neighbor data in DataFrame
        neighbors_data = []
        for i in range(len(self)):
            neighbors = self.social_network.get_neighbors(i, max_neighbors=10)
            neighbors_data.append(neighbors)

        self["social_network_neighbors"] = pl.Series(neighbors_data, dtype=pl.List(pl.UInt32))
        # Log neighbor dtype for debugging
        logger.info(
            f"Assigned neighbors dtype: {type(neighbors_data[0][0]) if neighbors_data and neighbors_data[0] else 'Empty'}"
        )

    def step(self) -> None:
        """
        Execute one timestep with new simulation flow:
        1. Ecology & Natural Selection Phase
        2. Cultural Evolution Phase
        3. Mating & Reproduction Phase
        """
        n_start_step = len(self)
        if n_start_step == 0:
            return

        # 1. Ecology & Natural Selection Phase
        self._ecology_phase()

        # 2. Cultural Evolution Phase
        if self.config.layer.cultural_enabled:
            self._cultural_evolution_phase()

        # 3. Mating & Reproduction Phase
        self._mating_reproduction_phase()

    def _ecology_phase(self) -> None:
        """
        Ecology & Natural Selection Phase:
        A. Energy Acquisition (density-dependent)
        B. Metabolism (energy decay)
        C. Survival Check (energy and age)
        """
        # A. Energy Acquisition with improved stability
        carrying_capacity = self.config.genetic.carrying_capacity
        energy_replenishment_rate = self.config.genetic.energy_replenishment_rate
        current_population = len(self.agents)

        if current_population > 0:
            # Enhanced density-dependent energy with better stability
            # Use smoother density dependence to avoid sudden crashes
            density_factor = max(0.1, 1.0 - (current_population / carrying_capacity) ** 2)
            base_energy_per_agent = energy_replenishment_rate * density_factor

            # Individual energy gain scaled by foraging efficiency
            foraging_efficiency = self.agents["gene_foraging_efficiency"].cast(pl.Float32) / 255.0

            # Display-survival trade-off: higher display traits reduce foraging efficiency
            display_cost_scalar = self.config.genetic.display_cost_scalar
            display_bits = self.agents["gene_display"]
            # Count number of set bits as proxy for display elaboration
            display_cost = display_bits.bitwise_count_ones().cast(pl.Float32) / 16.0 * display_cost_scalar

            # Effective foraging efficiency reduced by display cost
            effective_foraging = (foraging_efficiency - display_cost).clip(0.1, 1.0)  # Keep minimum 0.1

            # More stable energy gain with guaranteed minimum
            energy_gain = (base_energy_per_agent * (0.5 + effective_foraging)).clip(0.01, None)
        else:
            energy_gain = pl.lit(0.0, dtype=pl.Float32)

        # B. Metabolism
        energy_decay = self.config.genetic.energy_decay

        # C. Update energy and filter survivors
        max_age = self.config.genetic.max_age

        survivors_df = self.agents.with_columns(
            [(pl.col("age") + 1).alias("age"), (pl.col("energy") + energy_gain - energy_decay).alias("energy")]
        ).filter((pl.col("age") < max_age) & (pl.col("energy") > 0))

        # Update agent set
        self.agents = survivors_df

    def _cultural_evolution_phase(self) -> None:
        """
        Cultural Evolution Phase implementing configurable learning strategies.
        """
        n = len(self.agents)
        if n == 0:
            return

        learning_strategy = self.config.cultural.learning_strategy

        # Get learning parameters
        horizontal_rate = self.config.cultural.horizontal_transmission_rate
        innovation_rate = self.config.cultural.innovation_rate

        # Innovation process
        if innovation_rate > 0:
            innovators = np.random.random(n) < innovation_rate
            if innovators.any():
                cultural_prefs = self.agents["cultural_preference"].to_numpy().copy()
                n_innovate = int(np.sum(innovators))
                cultural_prefs[innovators] = np.random.randint(0, 2**16, n_innovate, dtype=np.uint16)
                self["cultural_preference"] = pl.Series(cultural_prefs, dtype=pl.UInt16)

        # Social learning based on strategy
        if horizontal_rate > 0:
            learners = np.random.random(n) < horizontal_rate
            if learners.any():
                self._apply_learning_strategy(learning_strategy, learners)

    def _apply_learning_strategy(self, strategy: str, learners: np.ndarray) -> None:
        """
        Apply the specified learning strategy using fully vectorized Polars operations.

        PERFORMANCE OPTIMIZATION: Eliminates Python loops for O(n log n) performance.
        """
        if not learners.any():
            return

        n = len(self.agents)
        learner_indices = np.where(learners)[0]

        if len(learner_indices) == 0:
            return

        # Get neighbor data using fully vectorized social network operations
        if hasattr(self, "social_network") and self.social_network:
            # ULTIMATE PERFORMANCE: Use vectorized network operation
            learner_indices_series = pl.Series(learner_indices, dtype=pl.Int32)
            learner_neighbor_df = self.social_network.get_neighbors_vectorized(learner_indices_series).rename(
                {"agent_id": "learner_id", "neighbor_id": "neighbor_id"}
            )

            # Add self-connections for learners with no neighbors as fallback
            learners_with_neighbors = learner_neighbor_df["learner_id"].unique()
            isolated_learners = pl.Series(learner_indices, dtype=pl.Int32).filter(
                ~pl.Series(learner_indices, dtype=pl.Int32).is_in(learners_with_neighbors)
            )

            if len(isolated_learners) > 0:
                self_connections = pl.DataFrame(
                    {"learner_id": isolated_learners.cast(pl.UInt32), "neighbor_id": isolated_learners.cast(pl.UInt32)}
                )
                # Ensure both DataFrames have Int64 columns for concatenation
                learner_neighbor_df = learner_neighbor_df.with_columns(
                    [pl.col("learner_id").cast(pl.Int64), pl.col("neighbor_id").cast(pl.Int64)]
                )
                self_connections = self_connections.with_columns(
                    [pl.col("learner_id").cast(pl.Int64), pl.col("neighbor_id").cast(pl.Int64)]
                )
                learner_neighbor_df = pl.concat([learner_neighbor_df, self_connections], how="diagonal")
        else:
            # Random neighborhood fallback using vectorized operations
            n_neighbors_per_learner = min(10, n)
            all_neighbors = []
            for learner_idx in learner_indices:
                neighbors = np.random.choice(n, size=n_neighbors_per_learner, replace=False).tolist()
                all_neighbors.extend([(learner_idx, neighbor) for neighbor in neighbors])

            learner_neighbor_df = pl.DataFrame(
                {"learner_id": [pair[0] for pair in all_neighbors], "neighbor_id": [pair[1] for pair in all_neighbors]}
            )

        if len(learner_neighbor_df) == 0:
            return

        # Join with agent data to get neighbor attributes
        agents_df = self.agents.with_row_index("agent_idx")

        neighbor_data = learner_neighbor_df.join(
            agents_df.select(["agent_idx", "cultural_preference", "mating_success", "energy", "age"]),
            left_on="neighbor_id",
            right_on="agent_idx",
            how="left",
        )

        # Apply strategy-specific vectorized learning
        if strategy == "conformist":
            # Find most common preference for each learner
            new_prefs = neighbor_data.group_by("learner_id").agg(
                [pl.col("cultural_preference").mode().first().alias("new_preference")]
            )

        elif strategy == "success-biased":
            # Copy preference of most successful neighbor
            new_prefs = (
                neighbor_data.sort(["learner_id", "mating_success"], descending=[False, True])
                .group_by("learner_id")
                .agg([pl.col("cultural_preference").first().alias("new_preference")])
            )

        elif strategy == "condition-dependent":
            # Copy preference of neighbor with highest energy
            new_prefs = (
                neighbor_data.sort(["learner_id", "energy"], descending=[False, True])
                .group_by("learner_id")
                .agg([pl.col("cultural_preference").first().alias("new_preference")])
            )

        elif strategy == "age-biased":
            # Copy preference of oldest neighbor
            new_prefs = (
                neighbor_data.sort(["learner_id", "age"], descending=[False, True])
                .group_by("learner_id")
                .agg([pl.col("cultural_preference").first().alias("new_preference")])
            )
        else:
            return

        # Apply learned preferences using vectorized operations
        if len(new_prefs) > 0:
            # Create update mapping
            learner_ids = new_prefs["learner_id"].to_numpy()
            new_preferences = new_prefs["new_preference"].to_numpy()

            # Update cultural preferences efficiently
            current_prefs = self.agents["cultural_preference"].to_numpy().copy()
            current_prefs[learner_ids] = new_preferences

            self["cultural_preference"] = pl.Series(current_prefs, dtype=pl.UInt16)

    def _mating_reproduction_phase(self) -> None:
        """
        Mating & Reproduction Phase:
        A. Preference Blending
        B. Courtship
        C. Mating
        D. Reproduction & Inheritance
        """
        n = len(self.agents)
        if n < 2:
            return

        # A. Preference Blending
        self._update_effective_preferences()

        # B. Courtship
        offspring_df = self._courtship_and_reproduction()

        # Deduct energy cost of reproduction from BOTH parents
        if (
            offspring_df is not None
            and "parent_a_idx" in offspring_df.columns
            and "parent_b_idx" in offspring_df.columns
        ):
            # Aggregate costs for parent A
            parent_a_costs = offspring_df.group_by("parent_a_idx").agg(
                (pl.col("parent_a_investment").sum()).alias("total_cost")
            )

            # Aggregate costs for parent B
            parent_b_costs = offspring_df.group_by("parent_b_idx").agg(
                (pl.col("parent_b_investment").sum()).alias("total_cost")
            )

            # Combine all parent costs
            all_parent_costs = (
                pl.concat(
                    [
                        parent_a_costs.rename({"parent_a_idx": "parent_id"}),
                        parent_b_costs.rename({"parent_b_idx": "parent_id"}),
                    ]
                )
                .group_by("parent_id")
                .agg(pl.col("total_cost").sum())
            )

            # Update energy for all parents in a single operation
            updated_agents_df = (
                self.agents.join(all_parent_costs, left_on="unique_id", right_on="parent_id", how="left")
                .with_columns((pl.col("energy") - pl.col("total_cost").fill_null(0.0)).alias("energy"))
                .drop("total_cost")
            )

            # Update energy column using the vectorized result
            self["energy"] = updated_agents_df["energy"]

        # C. Add offspring to population using += operator
        if offspring_df is not None and len(offspring_df) > 0:
            # Remove parent tracking columns before adding offspring
            clean_offspring_df = offspring_df.drop(
                ["parent_a_idx", "parent_a_investment", "parent_b_idx", "parent_b_investment"]
            )
            # Use += operator to properly add offspring to AgentSetPolars
            self += clean_offspring_df

        # Update mating success counters for accepted matings before reset
        if offspring_df is not None and len(offspring_df) > 0:
            # Mark females who successfully mated
            parent_females = offspring_df.filter(pl.col("parent_a_idx").is_not_null())["parent_a_idx"].unique()
            parent_indices = (
                self.agents.with_row_index("idx").filter(pl.col("unique_id").is_in(parent_females))["idx"].to_numpy()
            )

            if len(parent_indices) > 0:
                current_success = self.agents["mating_success"].to_numpy().copy()
                current_success[parent_indices] += 1
                self["mating_success"] = pl.Series(current_success, dtype=pl.UInt16)

        # Reset mating success counters for next timestep
        n = len(self.agents)
        if n > 0:
            self["mating_success"] = pl.Series([0] * n, dtype=pl.UInt16)

    def _update_effective_preferences(self) -> None:
        """Update effective preferences by blending genetic and cultural layers."""
        if not self.config.layer.is_combined():
            return

        n = len(self.agents)
        if n == 0:
            return

        genetic_prefs = self.agents["gene_preference"].to_numpy()
        cultural_prefs = self.agents["cultural_preference"].to_numpy()

        # Simple weighted blending
        genetic_weight = self.config.layer.genetic_weight
        cultural_weight = self.config.layer.cultural_weight

        # Blend at the bit level
        effective_prefs = (
            genetic_prefs.astype(np.uint32) * genetic_weight + cultural_prefs.astype(np.uint32) * cultural_weight
        ).astype(np.uint16)

        self["effective_preference"] = pl.Series(effective_prefs, dtype=pl.UInt16)

    def _courtship_and_reproduction(self) -> pl.DataFrame | None:
        """
        Optimized courtship with standardized 16-bit trait-preference comparison.
        """
        n = len(self.agents)
        if n < 2:
            return None

        # Get sex data
        sex_data = self.agents["sex"].to_numpy()

        # Separate males and females
        male_indices = np.where(sex_data == 1)[0]
        female_indices = np.where(sex_data == 0)[0]

        n_males = len(male_indices)
        n_females = len(female_indices)

        # Ensure we have both sexes
        if n_males == 0 or n_females == 0:
            return None

        # Create cross-sex pairings: each male paired with a female
        # Use the smaller of the two populations to determine number of pairs
        n_pairs = min(n_males, n_females)

        # Randomly permute and pair
        shuffled_males = np.random.permutation(male_indices)[:n_pairs]
        shuffled_females = np.random.permutation(female_indices)[:n_pairs]

        # Create partner mapping for the full population
        partners = np.full(n, -1, dtype=np.int32)  # -1 indicates no partner

        # Assign partners (males get female partners, females get male partners)
        partners[shuffled_males] = shuffled_females
        partners[shuffled_females] = shuffled_males

        # Filter to only those with valid partners
        valid_pairs = partners >= 0
        if not valid_pairs.any():
            return None

        # Update arrays to only include valid pairs
        valid_indices = np.where(valid_pairs)[0]
        valid_partners = partners[valid_indices]
        n_valid = len(valid_indices)

        # Get genetic traits for valid pairs only
        gene_display_self = self.agents["gene_display"].to_numpy()[valid_indices]
        gene_display_partner = self.agents["gene_display"].to_numpy()[valid_partners]
        sex_self = self.agents["sex"].to_numpy()[valid_indices]

        # Get effective preferences for valid pairs only
        if self.config.layer.is_combined():
            effective_pref_self = self.agents["effective_preference"].to_numpy()[valid_indices]
            effective_pref_partner = self.agents["effective_preference"].to_numpy()[valid_partners]
        elif self.config.layer.is_cultural_only():
            effective_pref_self = self.agents["cultural_preference"].to_numpy()[valid_indices]
            effective_pref_partner = self.agents["cultural_preference"].to_numpy()[valid_partners]
        else:
            effective_pref_self = self.agents["gene_preference"].to_numpy()[valid_indices]
            effective_pref_partner = self.agents["gene_preference"].to_numpy()[valid_partners]

        # Standardized 16-bit Hamming similarity calculation using Polars-native operations
        sim_self = hamming_similarity_16bit_polars(
            pl.Series(gene_display_partner, dtype=pl.UInt16), pl.Series(effective_pref_self, dtype=pl.UInt16)
        ).to_numpy()
        sim_partner = hamming_similarity_16bit_polars(
            pl.Series(gene_display_self, dtype=pl.UInt16), pl.Series(effective_pref_partner, dtype=pl.UInt16)
        ).to_numpy()

        # Apply perceptual constraints
        sigma_perception = self.config.layer.sigma_perception
        theta_detect = self.config.layer.theta_detect

        if sigma_perception > 0:
            noise_self = np.random.normal(0, sigma_perception, size=n_valid)
            noise_partner = np.random.normal(0, sigma_perception, size=n_valid)
            sim_self = sim_self + noise_self
            sim_partner = sim_partner + noise_partner

        # Detection threshold
        sim_self = np.where(sim_self >= theta_detect, sim_self, 0.0)
        sim_partner = np.where(sim_partner >= theta_detect, sim_partner, 0.0)

        # Compute probabilistic acceptance per individual (not per pair)
        # This ensures each female evaluates males based on her own threshold

        # Probabilistic female choice using sigmoid function for stronger selection gradients
        # P(accept) = sigmoid(k * (similarity - threshold))
        k = self.config.layer.sigmoid_steepness

        # Get full population data for individual acceptance computation
        full_sex = self.agents["sex"].to_numpy()
        full_thresholds = self.agents["gene_threshold"].to_numpy()

        # Build similarity arrays for each individual's perspective
        # Each individual needs to know how similar their partner appears to them
        full_similarities = np.zeros(len(self.agents))
        full_similarities[valid_indices] = sim_self  # How partner appears to self
        full_similarities[valid_partners] = sim_partner  # How self appears to partner

        # Compute acceptance probability for each individual
        is_female = full_sex == 0
        logit_acceptance = k * (full_similarities - full_thresholds)
        female_prob = 1.0 / (1.0 + np.exp(-logit_acceptance))

        # Set acceptance probability: males always accept (1.0), females use sigmoid
        accept_prob = np.where(is_female, female_prob, 1.0)

        # Draw independent acceptance decisions for each individual
        individual_accepts = np.random.random(len(self.agents)) < accept_prob

        # Extract acceptance decisions for the paired individuals
        accept_self = individual_accepts[valid_indices]
        accept_partner = individual_accepts[valid_partners]

        # Apply mating cap: each female can only mate once per timestep
        current_mating_success = self.agents["mating_success"].to_numpy()[valid_indices]
        already_mated = current_mating_success > 0
        can_mate = ~already_mated

        # Only females that haven't mated yet can participate
        female_can_mate = np.where(sex_self == 0, can_mate, True)  # Females limited, males unrestricted

        # Apply search/assessment energy cost to females before acceptance
        search_cost = self.config.genetic.search_cost
        female_indices = valid_indices[sex_self == 0]  # Get female indices

        if len(female_indices) > 0:
            # Deduct search cost from females who participated in courtship
            current_energy = self.agents["energy"].to_numpy().copy()
            current_energy[female_indices] -= search_cost
            self["energy"] = pl.Series(current_energy, dtype=pl.Float32)

        # Mutual acceptance required (but now asymmetric) with mating constraints
        accepted = accept_self & accept_partner & female_can_mate

        if not accepted.any():
            return None

        # Calculate and log selection differential for males (avoiding double-counting)
        # This measures the actual selection gradient that drives evolution
        # Only count each pair once by keeping pairs where valid_indices < valid_partners
        unique_pair_mask = valid_indices < valid_partners

        male_mask = sex_self == 1
        unique_males_mask = male_mask & unique_pair_mask

        if unique_males_mask.any():
            # All males who participated in unique pairs
            all_unique_males = gene_display_self[unique_males_mask]
            # Males who were accepted in unique pairs
            accepted_unique_males = gene_display_self[unique_males_mask & accepted]

            if len(accepted_unique_males) > 0 and len(all_unique_males) > 0:
                mean_all_males = float(np.mean(all_unique_males))
                mean_accepted_males = float(np.mean(accepted_unique_males))
                selection_differential = mean_accepted_males - mean_all_males

                # Store selection differential in model for logging
                if not hasattr(self.model, "selection_differentials"):
                    self.model.selection_differentials = []
                self.model.selection_differentials.append(selection_differential)

        # Create offspring through genetic recombination
        return self._create_offspring(valid_indices, valid_partners, accepted)

    def _create_offspring(self, valid_indices: np.ndarray, partners: np.ndarray, accepted: np.ndarray) -> pl.DataFrame:
        """Create offspring through genetic recombination of unlinked genes."""
        accepted_mask = np.where(accepted)[0]
        idx = valid_indices[accepted_mask]
        partner_idx = partners[accepted_mask]

        # Minimum energy gate: both parents must have sufficient energy to reproduce
        E_MIN = self.config.genetic.energy_min_mating
        parent_energy_a = self.agents["energy"].to_numpy()[idx]
        parent_energy_b = self.agents["energy"].to_numpy()[partner_idx]
        viable = (parent_energy_a > E_MIN) & (parent_energy_b > E_MIN)

        # Filter to only viable parents
        idx = idx[viable]
        partner_idx = partner_idx[viable]
        n_offspring = len(idx)

        if n_offspring == 0:
            return pl.DataFrame()

        # Get parent genes
        parent_a_display = self.agents["gene_display"].to_numpy()[idx]
        parent_a_preference = self.agents["gene_preference"].to_numpy()[idx]
        parent_a_threshold = self.agents["gene_threshold"].to_numpy()[idx]
        parent_a_foraging = self.agents["gene_foraging_efficiency"].to_numpy()[idx]

        parent_b_display = self.agents["gene_display"].to_numpy()[partner_idx]
        parent_b_preference = self.agents["gene_preference"].to_numpy()[partner_idx]
        parent_b_threshold = self.agents["gene_threshold"].to_numpy()[partner_idx]
        parent_b_foraging = self.agents["gene_foraging_efficiency"].to_numpy()[partner_idx]

        # Genetic recombination for unlinked genes
        offspring_display = np.where(np.random.random(n_offspring) < 0.5, parent_a_display, parent_b_display)
        offspring_preference = np.where(np.random.random(n_offspring) < 0.5, parent_a_preference, parent_b_preference)
        offspring_threshold = np.where(np.random.random(n_offspring) < 0.5, parent_a_threshold, parent_b_threshold)
        offspring_foraging = np.where(np.random.random(n_offspring) < 0.5, parent_a_foraging, parent_b_foraging)

        # Apply mutations using fully vectorized operations
        mutation_rate = self.config.genetic.mutation_rate
        if mutation_rate > 0:
            # PERFORMANCE OPTIMIZATION: Vectorized mutation replaces Python loops
            # Generate mutation masks for all genes simultaneously

            # Display gene mutations (16-bit)
            mutate_display = np.random.random(n_offspring) < mutation_rate
            if mutate_display.any():
                bit_positions = np.random.randint(0, 16, size=n_offspring)
                mutation_masks = np.uint16(1) << bit_positions
                offspring_display = np.where(mutate_display, offspring_display ^ mutation_masks, offspring_display)

            # Preference gene mutations (16-bit)
            mutate_preference = np.random.random(n_offspring) < mutation_rate
            if mutate_preference.any():
                bit_positions = np.random.randint(0, 16, size=n_offspring)
                mutation_masks = np.uint16(1) << bit_positions
                offspring_preference = np.where(
                    mutate_preference, offspring_preference ^ mutation_masks, offspring_preference
                )

            # Threshold gene mutations (4-bit to match similarity scale)
            mutate_threshold = np.random.random(n_offspring) < mutation_rate
            if mutate_threshold.any():
                bit_positions = np.random.randint(0, 4, size=n_offspring)
                mutation_masks = np.uint8(1) << bit_positions
                offspring_threshold = np.where(
                    mutate_threshold, offspring_threshold ^ mutation_masks, offspring_threshold
                )

            # Foraging efficiency gene mutations (8-bit)
            mutate_foraging = np.random.random(n_offspring) < mutation_rate
            if mutate_foraging.any():
                bit_positions = np.random.randint(0, 8, size=n_offspring)
                mutation_masks = np.uint8(1) << bit_positions
                offspring_foraging = np.where(mutate_foraging, offspring_foraging ^ mutation_masks, offspring_foraging)

        # Calculate offspring energy from parental investment
        parent_energy_a = self.agents["energy"].to_numpy()[idx]
        parent_energy_b = self.agents["energy"].to_numpy()[partner_idx]
        parental_contribution_rate = self.config.genetic.parental_investment_rate

        parent_a_investment = parent_energy_a * parental_contribution_rate
        parent_b_investment = parent_energy_b * parental_contribution_rate
        offspring_energy = parent_a_investment + parent_b_investment

        # Juvenile start-up debit: subtract fixed cost for yolk/early mortality
        juvenile_cost = self.config.genetic.juvenile_cost
        offspring_energy = offspring_energy - juvenile_cost

        # Filter out offspring with insufficient energy to survive
        energy_viable = offspring_energy > 0

        # Apply density-dependent juvenile survival
        current_population = len(self.agents)
        carrying_capacity = self.config.genetic.carrying_capacity
        density_survival_rate = max(0.0, 1.0 - (current_population / carrying_capacity))

        # Random survival based on density
        density_survivors = np.random.random(n_offspring) < density_survival_rate

        # Combine energy and density constraints
        viable_offspring = energy_viable & density_survivors
        if not viable_offspring.any():
            return pl.DataFrame()

        # Apply viability filter to all offspring attributes
        idx = idx[viable_offspring]
        partner_idx = partner_idx[viable_offspring]
        offspring_display = offspring_display[viable_offspring]
        offspring_preference = offspring_preference[viable_offspring]
        offspring_threshold = offspring_threshold[viable_offspring]
        offspring_foraging = offspring_foraging[viable_offspring]
        parent_energy_a = parent_energy_a[viable_offspring]
        parent_energy_b = parent_energy_b[viable_offspring]
        parent_a_investment = parent_a_investment[viable_offspring]
        parent_b_investment = parent_b_investment[viable_offspring]
        offspring_energy = offspring_energy[viable_offspring]
        n_offspring = len(idx)

        # Create offspring DataFrame
        offspring_data = {
            # Track BOTH parent indices and investments for energy deduction
            "parent_a_idx": pl.Series(self.agents["unique_id"].to_numpy()[idx], dtype=pl.UInt64),
            "parent_a_investment": pl.Series(parent_a_investment, dtype=pl.Float32),
            "parent_b_idx": pl.Series(self.agents["unique_id"].to_numpy()[partner_idx], dtype=pl.UInt64),
            "parent_b_investment": pl.Series(parent_b_investment, dtype=pl.Float32),
            # --- Offspring genes and initial state ---
            "gene_display": pl.Series(offspring_display, dtype=pl.UInt16),
            "gene_preference": pl.Series(offspring_preference, dtype=pl.UInt16),
            "gene_threshold": pl.Series(offspring_threshold, dtype=pl.UInt8),
            "gene_foraging_efficiency": pl.Series(offspring_foraging, dtype=pl.UInt8),
            "energy": pl.Series(offspring_energy, dtype=pl.Float32),
            "age": pl.Series([0] * n_offspring, dtype=pl.UInt16),
            "mating_success": pl.Series([0] * n_offspring, dtype=pl.UInt16),
        }

        # Initialize offspring sex (50/50 split)
        offspring_sex = np.random.randint(0, 2, size=n_offspring, dtype=np.uint8)
        offspring_data["sex"] = pl.Series(offspring_sex, dtype=pl.UInt8)

        # Add cultural traits for offspring if cultural layer enabled
        if self.config.layer.cultural_enabled:
            # Cultural inheritance with innovation
            parent_cultural_a = self.agents["cultural_preference"].to_numpy()[idx]
            parent_cultural_b = self.agents["cultural_preference"].to_numpy()[partner_idx]

            offspring_cultural = np.where(np.random.random(n_offspring) < 0.5, parent_cultural_a, parent_cultural_b)

            # Cultural innovation
            innovation_rate = self.config.cultural.innovation_rate
            innovate = np.random.random(n_offspring) < innovation_rate
            if innovate.any():
                n_innovate = int(np.sum(innovate))
                offspring_cultural[innovate] = np.random.randint(0, 2**16, n_innovate, dtype=np.uint16)

            offspring_data["cultural_preference"] = pl.Series(offspring_cultural, dtype=pl.UInt16)
            offspring_data["cultural_innovation_count"] = pl.Series([0] * n_offspring, dtype=pl.UInt16)
            offspring_data["social_network_neighbors"] = pl.Series(
                [[] for _ in range(n_offspring)], dtype=pl.List(pl.UInt32)
            )

        # Add effective preference for combined models
        if self.config.layer.is_combined():
            offspring_data["effective_preference"] = pl.Series(offspring_preference, dtype=pl.UInt16)

        return pl.DataFrame(offspring_data)


class LoveModelRefactored(ModelDF):
    """
    Refactored LoveBug model implementing the biological refactoring plan.

    Key improvements:
    1. Unlinked multi-gene architecture
    2. Endogenous population regulation
    3. Configurable social learning strategies
    4. Standardized trait-preference comparison
    """

    @beartype
    def __init__(self, config: LoveBugConfig) -> None:
        super().__init__()
        self.config = config

        # Initialize agents with refactored architecture
        self.agents += LoveAgentsRefactored(self.config.simulation.population_size, self)

        # Track metrics
        self.history: list[dict[str, Any]] = []
        self.step_count = 0
        self.selection_differentials: list[float] = []

        # Check parameter health at initialization
        self._check_parameter_health()

        logger.info(f"LoveModelRefactored initialized: n_agents={self.config.simulation.population_size}")

    def step(self) -> None:
        """Execute one model timestep and collect metrics."""
        self.agents.do("step")

        # Collect metrics
        metrics = self._collect_metrics()
        self.history.append(metrics)
        self.step_count += 1

        if self.step_count % 50 == 0:
            logger.debug(f"Step {self.step_count}: {len(self.agents)} agents")

    def _collect_metrics(self) -> dict[str, Any]:
        """Collect comprehensive metrics for current step."""
        if len(self.agents) == 0:
            return {"step": self.step_count, "population_size": 0}

        df = self.agents._agentsets[0].agents

        # Basic metrics
        metrics: dict[str, Any] = {
            "step": self.step_count,
            "population_size": len(df),
            "mean_age": float(df["age"].mean()) if df["age"].mean() is not None else 0.0,
            "mean_energy": float(df["energy"].mean()) if df["energy"].mean() is not None else 0.0,
        }

        # Add selection differential if available
        if hasattr(self, "selection_differentials") and self.selection_differentials:
            metrics["selection_differential"] = self.selection_differentials[-1]  # Most recent
            metrics["mean_selection_differential"] = float(np.mean(self.selection_differentials))
        else:
            metrics["selection_differential"] = 0.0
            metrics["mean_selection_differential"] = 0.0

        # Genetic metrics
        if self.config.layer.genetic_enabled:
            display_mean = df["gene_display"].cast(pl.Float32).mean()
            preference_mean = df["gene_preference"].cast(pl.Float32).mean()
            threshold_mean = df["gene_threshold"].cast(pl.Float32).mean()
            foraging_mean = df["gene_foraging_efficiency"].cast(pl.Float32).mean()

            metrics.update(
                {
                    "mean_gene_display": float(display_mean) if display_mean is not None else 0.0,
                    "mean_gene_preference": float(preference_mean) if preference_mean is not None else 0.0,
                    "mean_gene_threshold": float(threshold_mean) if threshold_mean is not None else 0.0,
                    "mean_gene_foraging": float(foraging_mean) if foraging_mean is not None else 0.0,
                    "var_gene_display": float(df["gene_display"].cast(pl.Float32).var())
                    if df["gene_display"].cast(pl.Float32).var() is not None
                    else 0.0,
                    "var_gene_preference": float(df["gene_preference"].cast(pl.Float32).var())
                    if df["gene_preference"].cast(pl.Float32).var() is not None
                    else 0.0,
                }
            )

        # Cultural metrics
        if self.config.layer.cultural_enabled and "cultural_preference" in df.columns:
            cultural_mean = df["cultural_preference"].cast(pl.Float32).mean()
            cultural_var = df["cultural_preference"].cast(pl.Float32).var()

            metrics.update(
                {
                    "mean_cultural_preference": float(cultural_mean) if cultural_mean is not None else 0.0,
                    "var_cultural_preference": float(cultural_var) if cultural_var is not None else 0.0,
                }
            )

        return metrics

    @beartype
    def run(self, n_steps: int = 100) -> dict[str, Any]:
        """Run simulation for specified number of steps."""
        logger.info(f"Starting refactored simulation: {n_steps} steps")

        for _ in range(n_steps):
            self.step()

        results = {
            "layer_config": self.config.layer.model_dump(),
            "n_steps": n_steps,
            "final_population": len(self.agents),
            "trajectory": self.history,
            "final_metrics": self.history[-1] if self.history else {},
        }

        logger.info(f"Refactored simulation completed: final population {len(self.agents)}")
        return results

    def _check_parameter_health(self) -> None:
        """Check parameter ratios and warn about potential issues."""
        # Get actual population thresholds for accurate choosiness calculation
        actual_thresholds = None
        current_pop = len(self.agents)
        if current_pop > 0:
            actual_thresholds = self.get_agent_dataframe()["gene_threshold"].to_numpy()

        ratios = ParameterHelper.check_parameter_health(self.config, actual_thresholds)

        # Also calculate effective energy balance at current density
        if current_pop > 0:
            effective_energy_ratio = ParameterHelper.calculate_effective_energy_balance_ratio(
                self.config.genetic.energy_replenishment_rate,
                self.config.genetic.energy_decay,
                current_pop,
                self.config.genetic.carrying_capacity,
            )
            ratios["effective_energy_balance_ratio"] = effective_energy_ratio

        warnings = []

        # Check energy balance (use effective ratio if available, otherwise raw ratio)
        energy_ratio_key = (
            "effective_energy_balance_ratio" if "effective_energy_balance_ratio" in ratios else "energy_balance_ratio"
        )
        energy_ratio = ratios[energy_ratio_key]

        if energy_ratio < 0.01:
            warnings.append(f"Energy balance ratio {energy_ratio:.3f} may cause population crashes")
        elif energy_ratio > 0.2:
            warnings.append(f"Energy balance ratio {energy_ratio:.3f} may cause explosive growth")

        # Report both raw and effective ratios if different
        if "effective_energy_balance_ratio" in ratios:
            raw_ratio = ratios["energy_balance_ratio"]
            effective_ratio = ratios["effective_energy_balance_ratio"]
            if abs(raw_ratio - effective_ratio) > 0.05:
                logger.info(f"Energy balance: raw={raw_ratio:.3f}, effective@N={current_pop}={effective_ratio:.3f}")

        # Check female choosiness
        if ratios["female_choosiness_ratio"] < 0.5:
            warnings.append(
                f"Female choosiness ratio {ratios['female_choosiness_ratio']:.3f} may result in weak sexual selection"
            )

        # Check display costs
        if ratios["display_cost_ratio"] > 0.3:
            warnings.append(f"Display cost ratio {ratios['display_cost_ratio']:.3f} may overwhelm sexual selection")

        # Check juvenile bottleneck
        if ratios["juvenile_bottleneck_ratio"] > 0.8:
            warnings.append(
                f"Juvenile bottleneck ratio {ratios['juvenile_bottleneck_ratio']:.3f} may prevent reproduction"
            )

        if warnings:
            logger.warning("Parameter health check found potential issues:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("Parameter health check passed - ratios look reasonable")

    @beartype
    def get_agent_dataframe(self) -> pl.DataFrame:
        """Get the current agent dataframe."""
        return self.agents._agentsets[0].agents.clone()
