"""
Vectorized learning algorithms for cultural transmission.

Contains the core vectorized implementations of all cultural learning mechanisms:
eligibility computation, oblique transmission, horizontal transmission,
innovation, and memory decay. All operations use Polars DataFrames and NumPy
arrays for maximum performance.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from beartype import beartype

from ..config import Layer2Config

__all__ = [
    "LearningEligibilityComputer",
    "ObliqueTransmissionEngine",
    "HorizontalTransmissionEngine",
    "CulturalInnovationEngine",
    "MemoryDecayEngine",
]

logger = logging.getLogger(__name__)


class LearningEligibilityComputer:
    """
    Vectorized computation of learning eligibility using boolean masks.

    Replaces sequential per-agent loops with bulk DataFrame operations to
    determine which agents can participate in cultural learning.
    """

    def __init__(self, config: Layer2Config) -> None:
        self.config = config

    @beartype
    def compute_eligibility(self, df: pl.DataFrame, generation: int) -> pl.Series:
        """
        Vectorized computation of which agents can learn this generation.

        Parameters
        ----------
        df : pl.DataFrame
            Agent DataFrame with required columns
        generation : int
            Current generation number

        Returns
        -------
        pl.Series
            Boolean Series indicating learning eligibility

        Examples
        --------
        >>> computer = LearningEligibilityComputer(Layer2Config())
        >>> eligibility = computer.compute_eligibility(agents_df, generation=10)
        """
        # Define minimum requirements for learning eligibility
        min_age = 1  # Agents must be at least 1 generation old
        min_energy = 1.0  # Minimum energy required for learning
        learning_cooldown = 1  # Minimum generations between learning events

        eligibility_conditions = df.select(
            [
                # Age-based eligibility
                (pl.col("age") >= min_age).alias("age_eligible"),
                # Energy-based eligibility
                (pl.col("energy") >= min_energy).alias("energy_eligible"),
                # Cooldown eligibility (time since last learning)
                ((generation - pl.col("last_learning_event")) >= learning_cooldown).alias("cooldown_eligible"),
                # Network eligibility (has neighbors to learn from)
                (pl.col("network_degree") > 0).alias("network_eligible"),
            ]
        )

        # Combine all eligibility conditions
        eligible = eligibility_conditions.select(
            (
                pl.col("age_eligible")
                & pl.col("energy_eligible")
                & pl.col("cooldown_eligible")
                & pl.col("network_eligible")
            ).alias("eligible")
        ).get_column("eligible")

        logger.debug(f"Generation {generation}: {eligible.sum()} agents eligible for learning")
        return eligible


class ObliqueTransmissionEngine:
    """
    Vectorized oblique transmission (parent-offspring cultural learning).

    Replaces per-agent parent selection with bulk operations using prestige-based
    selection from older generations.
    """

    def __init__(self, config: Layer2Config) -> None:
        self.config = config

    @beartype
    def execute_transmission(self, df: pl.DataFrame, learner_mask: pl.Series) -> pl.DataFrame:
        """
        Execute vectorized oblique transmission.

        Parameters
        ----------
        df : pl.DataFrame
            Full agent DataFrame
        learner_mask : pl.Series
            Boolean mask of agents eligible for oblique learning

        Returns
        -------
        pl.DataFrame
            DataFrame with learner_id, new_preference, teacher_id columns
        """
        if not learner_mask.any():
            return pl.DataFrame({"learner_id": [], "new_preference": [], "teacher_id": [], "learning_type": []})

        learners = df.filter(learner_mask)
        n_learners = len(learners)

        # Find potential teachers (older agents)
        age_threshold = df.select(pl.col("age").quantile(0.75)).item()
        potential_teachers = df.filter(pl.col("age") >= age_threshold)

        if len(potential_teachers) == 0:
            # No older agents available - return empty result
            return pl.DataFrame({"learner_id": [], "new_preference": [], "teacher_id": [], "learning_type": []})

        # Vectorized prestige-based teacher selection
        teacher_weights = self._compute_teacher_weights(potential_teachers)

        # Random teacher assignment for each learner
        teacher_indices = np.random.choice(len(potential_teachers), size=n_learners, p=teacher_weights, replace=True)

        selected_teachers = potential_teachers[teacher_indices]

        # Create learning events DataFrame
        learning_events = pl.DataFrame(
            {
                "learner_id": learners.get_column("agent_id"),
                "new_preference": selected_teachers.get_column("pref_culture"),
                "teacher_id": selected_teachers.get_column("agent_id"),
                "learning_type": ["oblique"] * n_learners,
            }
        )

        logger.debug(f"Oblique transmission: {n_learners} learning events")
        return learning_events

    def _compute_teacher_weights(self, teachers: pl.DataFrame) -> np.ndarray:
        """Compute prestige-based selection weights for teachers."""
        # Use prestige score with fallback to mating success
        if "prestige_score" in teachers.columns:
            weights = teachers.get_column("prestige_score").to_numpy()
        else:
            weights = teachers.get_column("mating_success").to_numpy().astype(float)

        # Add small base weight to avoid zero probabilities
        weights = weights + 0.1

        # Normalize to probabilities
        weights = weights / np.sum(weights)
        return weights


class HorizontalTransmissionEngine:
    """
    Vectorized horizontal transmission (peer-to-peer learning).

    Uses network neighbor operations with conformist and prestige-based
    learning modes implemented through DataFrame joins and aggregations.
    """

    def __init__(self, config: Layer2Config) -> None:
        self.config = config

    @beartype
    def execute_transmission(self, df: pl.DataFrame, learner_mask: pl.Series) -> pl.DataFrame:
        """
        Execute vectorized horizontal transmission.

        Parameters
        ----------
        df : pl.DataFrame
            Full agent DataFrame
        learner_mask : pl.Series
            Boolean mask of agents eligible for horizontal learning

        Returns
        -------
        pl.DataFrame
            DataFrame with learner_id, new_preference, teacher_id columns
        """
        if not learner_mask.any():
            return pl.DataFrame({"learner_id": [], "new_preference": [], "teacher_id": [], "learning_type": []})

        learners = df.filter(learner_mask)

        # Explode neighbor lists to create learner-neighbor pairs
        learner_neighbor_pairs = (
            learners.select([pl.col("agent_id").alias("learner_id"), pl.col("neighbors")])
            .explode("neighbors")
            .rename({"neighbors": "neighbor_id"})
        )

        if len(learner_neighbor_pairs) == 0:
            return pl.DataFrame({"learner_id": [], "new_preference": [], "teacher_id": [], "learning_type": []})

        # Ensure neighbor_id has correct data type (not null)
        if learner_neighbor_pairs.select(pl.col("neighbor_id").null_count()).item() == len(learner_neighbor_pairs):
            return pl.DataFrame({"learner_id": [], "new_preference": [], "teacher_id": [], "learning_type": []})

        # Join with neighbor attributes
        enriched_pairs = learner_neighbor_pairs.join(
            df.select(
                [
                    pl.col("agent_id").alias("neighbor_id"),
                    pl.col("pref_culture").alias("neighbor_culture"),
                    pl.col("prestige_score").alias("neighbor_prestige"),
                ]
            ),
            on="neighbor_id",
            how="inner",
        )

        if len(enriched_pairs) == 0:
            return pl.DataFrame({"learner_id": [], "new_preference": [], "teacher_id": [], "learning_type": []})

        # Choose learning strategy (conformist vs prestige-based)
        conformist_probability = 0.7  # 70% conformist learning
        learning_modes = self._compute_learning_choices(enriched_pairs, conformist_probability)

        return learning_modes

    def _compute_learning_choices(self, enriched_pairs: pl.DataFrame, conformist_prob: float) -> pl.DataFrame:
        """Compute learning choices using conformist or prestige-based strategies."""
        # Group by learner and compute learning options
        learning_options = enriched_pairs.group_by("learner_id").agg(
            [
                # Conformist choice: most common neighbor preference
                pl.col("neighbor_culture").mode().first().alias("conformist_choice"),
                # Prestige choice: preference of highest prestige neighbor
                pl.col("neighbor_culture")
                .filter(pl.col("neighbor_prestige") == pl.col("neighbor_prestige").max())
                .first()
                .alias("prestige_choice"),
                # Get corresponding neighbor IDs for tracking
                pl.col("neighbor_id")
                .filter(pl.col("neighbor_culture") == pl.col("neighbor_culture").mode().first())
                .first()
                .alias("conformist_teacher"),
                pl.col("neighbor_id")
                .filter(pl.col("neighbor_prestige") == pl.col("neighbor_prestige").max())
                .first()
                .alias("prestige_teacher"),
            ]
        )

        # Vectorized strategy selection
        n_learners = len(learning_options)
        use_conformist = np.random.random(n_learners) < conformist_prob

        # Select final choices based on strategy
        final_choices = learning_options.with_columns(
            [
                pl.when(pl.Series(use_conformist))
                .then(pl.col("conformist_choice"))
                .otherwise(pl.col("prestige_choice"))
                .alias("new_preference"),
                pl.when(pl.Series(use_conformist))
                .then(pl.col("conformist_teacher"))
                .otherwise(pl.col("prestige_teacher"))
                .alias("teacher_id"),
                pl.lit("horizontal").alias("learning_type"),
            ]
        ).select(["learner_id", "new_preference", "teacher_id", "learning_type"])

        logger.debug(f"Horizontal transmission: {n_learners} learning events")
        return final_choices


class CulturalInnovationEngine:
    """
    Vectorized cultural innovation through random mutation.

    Implements bulk random preference generation using different mutation
    strategies (bit-flip, random, neighbor-blend).
    """

    def __init__(self, config: Layer2Config) -> None:
        self.config = config

    @beartype
    def execute_innovation(self, df: pl.DataFrame, innovation_rate: float) -> pl.DataFrame:
        """
        Execute vectorized cultural innovation.

        Parameters
        ----------
        df : pl.DataFrame
            Full agent DataFrame
        innovation_rate : float
            Rate of innovation (0-1)

        Returns
        -------
        pl.DataFrame
            DataFrame with agent_id, new_preference, learning_type columns
        """
        n_agents = len(df)

        # Vectorized innovation selection
        innovators_mask = np.random.random(n_agents) < innovation_rate
        n_innovators = int(np.sum(innovators_mask))

        if n_innovators == 0:
            return pl.DataFrame({"learner_id": [], "new_preference": [], "teacher_id": [], "learning_type": []})

        # Get current preferences of innovators
        innovators = df.filter(pl.Series(innovators_mask))
        current_prefs = innovators.get_column("pref_culture").to_numpy()

        # Generate new preferences using different strategies
        new_preferences = self._generate_innovations(current_prefs, n_innovators)

        innovation_events = pl.DataFrame(
            {
                "learner_id": innovators.get_column("agent_id"),
                "new_preference": new_preferences,
                "teacher_id": [None] * n_innovators,  # No teacher for innovation
                "learning_type": ["innovation"] * n_innovators,
            }
        )

        logger.debug(f"Cultural innovation: {n_innovators} innovation events")
        return innovation_events

    def _generate_innovations(self, current_prefs: np.ndarray, n_innovators: int) -> np.ndarray:
        """Generate new cultural preferences through different innovation mechanisms."""
        # Choose innovation type for each innovator
        innovation_types = np.random.choice(
            ["bit_flip", "random", "neighbor_blend"],
            size=n_innovators,
            p=[0.5, 0.3, 0.2],  # Bit-flip most common, random moderate, blend rare
        )

        new_preferences = np.zeros_like(current_prefs)

        # Bit-flip mutations (flip random bit)
        bit_flip_mask = innovation_types == "bit_flip"
        if np.any(bit_flip_mask):
            flip_bits = np.random.randint(0, 8, size=np.sum(bit_flip_mask))
            new_preferences[bit_flip_mask] = current_prefs[bit_flip_mask] ^ (1 << flip_bits)

        # Random new preferences
        random_mask = innovation_types == "random"
        if np.any(random_mask):
            new_preferences[random_mask] = np.random.randint(0, 256, size=np.sum(random_mask))

        # Neighbor-influenced innovations (simplified for now)
        blend_mask = innovation_types == "neighbor_blend"
        if np.any(blend_mask):
            # For now, just do random variation around current preference
            variations = np.random.randint(-50, 51, size=np.sum(blend_mask))
            new_preferences[blend_mask] = np.clip(current_prefs[blend_mask].astype(int) + variations, 0, 255)

        return new_preferences.astype(np.uint8)


class MemoryDecayEngine:
    """
    Vectorized memory decay using column shifting operations.

    Implements efficient memory decay by shifting memory columns and applying
    decay rates in bulk operations.
    """

    def __init__(self, config: Layer2Config) -> None:
        self.config = config

    @beartype
    def apply_memory_decay(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply vectorized memory decay to all agents.

        Parameters
        ----------
        df : pl.DataFrame
            Agent DataFrame with memory columns

        Returns
        -------
        pl.DataFrame
            Updated DataFrame with decayed memories
        """
        memory_size = self.config.cultural_memory_size
        decay_rate = self.config.memory_decay_rate

        if memory_size <= 0:
            return df

        # Build memory update operations
        memory_updates = {}

        # First shift older memories (shift right before adding new)
        for i in range(memory_size - 1, 0, -1):  # Work backwards to avoid overwriting
            prev_memory_col = f"cultural_memory_{i - 1}"
            current_memory_col = f"cultural_memory_{i}"
            prev_weight_col = f"memory_weights_{i - 1}"
            current_weight_col = f"memory_weights_{i}"

            if prev_memory_col in df.columns:
                memory_updates[current_memory_col] = pl.col(prev_memory_col)
            if prev_weight_col in df.columns:
                memory_updates[current_weight_col] = pl.col(prev_weight_col) * (1.0 - decay_rate)

        # Then add current preference to position 0 (most recent)
        memory_updates["cultural_memory_0"] = pl.col("pref_culture").cast(pl.Float32)
        memory_updates["memory_weights_0"] = pl.lit(self.config.memory_update_strength)

        # Decay remaining memory weights that weren't shifted
        for i in range(memory_size):
            weight_col = f"memory_weights_{i}"
            if weight_col in df.columns and weight_col not in memory_updates:
                memory_updates[weight_col] = pl.col(weight_col) * (1.0 - decay_rate)

        # Apply all memory updates in single operation
        if memory_updates:
            updated_df = df.with_columns([expr.alias(col_name) for col_name, expr in memory_updates.items()])
        else:
            updated_df = df

        # Remove very weak memories (optional optimization)
        threshold = 0.01
        for i in range(memory_size):
            weight_col = f"memory_weights_{i}"
            memory_col = f"cultural_memory_{i}"

            if weight_col in updated_df.columns:
                updated_df = updated_df.with_columns(
                    [
                        pl.when(pl.col(weight_col) < threshold)
                        .then(pl.lit(0.0))
                        .otherwise(pl.col(memory_col))
                        .alias(memory_col),
                        pl.when(pl.col(weight_col) < threshold)
                        .then(pl.lit(0.0))
                        .otherwise(pl.col(weight_col))
                        .alias(weight_col),
                    ]
                )

        logger.debug(f"Applied memory decay to {len(df)} agents")
        return updated_df

    @beartype
    def compute_effective_preference(self, df: pl.DataFrame) -> pl.Series:
        """
        Compute effective cultural preference incorporating memory.

        Parameters
        ----------
        df : pl.DataFrame
            Agent DataFrame with memory columns

        Returns
        -------
        pl.Series
            Series of effective cultural preferences
        """
        memory_size = self.config.cultural_memory_size

        if memory_size <= 0 or "cultural_memory_0" not in df.columns:
            return df.get_column("pref_culture")

        # Compute weighted average of memories
        memory_expr = pl.col("pref_culture").cast(pl.Float32)  # Current preference weight = 1.0
        weight_expr = pl.lit(1.0)

        for i in range(memory_size):
            memory_col = f"cultural_memory_{i}"
            weight_col = f"memory_weights_{i}"

            if memory_col in df.columns and weight_col in df.columns:
                memory_expr = memory_expr + (pl.col(memory_col) * pl.col(weight_col))
                weight_expr = weight_expr + pl.col(weight_col)

        # Compute weighted average and clamp to valid range [0, 255]
        effective_pref = (memory_expr / weight_expr).round().cast(pl.UInt8)

        return df.select([effective_pref.alias("effective_pref")]).get_column("effective_pref")
