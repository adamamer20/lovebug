"""
Mesa-frames native vectorized cultural transmission layer.

Provides the main integration class that replaces the sequential
CulturalTransmissionManager with a fully vectorized implementation using
Polars DataFrames and coordinated learning algorithms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from beartype import beartype

from .config import Layer2Config
from .learning_algorithms import (
    CulturalInnovationEngine,
    HorizontalTransmissionEngine,
    LearningEligibilityComputer,
    MemoryDecayEngine,
    ObliqueTransmissionEngine,
)
from .network import NetworkTopology, VectorizedSocialNetwork

if TYPE_CHECKING:
    from lovebug.unified_mesa_model import UnifiedLoveBugs

__all__ = ["VectorizedCulturalLayer"]

logger = logging.getLogger(__name__)


class VectorizedCulturalLayer:
    """
    Mesa-frames native vectorized cultural transmission system.

    Coordinates all cultural learning mechanisms using fully vectorized
    operations on Polars DataFrames. Replaces the sequential per-agent
    loops with bulk operations for O(n log n) performance.

    Parameters
    ----------
    agent_set : UnifiedLoveBugs
        Mesa-frames agent set to operate on
    config : Layer2Config
        Configuration for cultural transmission parameters

    Examples
    --------
    >>> from lovebug.layer2.config import Layer2Config
    >>> config = Layer2Config(innovation_rate=0.1)
    >>> cultural_layer = VectorizedCulturalLayer(agents, config)
    >>> cultural_layer.step()
    """

    def __init__(self, agent_set: UnifiedLoveBugs, config: Layer2Config) -> None:
        self.agents = agent_set
        self.config = config
        self.generation = 0

        # Initialize network topology
        topology = NetworkTopology(
            network_type=config.network_type,
            connectivity=config.network_connectivity,
            rewiring_prob=0.1,  # Could be added to config
            degree_preference=1.0,
        )

        # Initialize vectorized components
        self.network = VectorizedSocialNetwork(len(agent_set), topology)
        self.eligibility_computer = LearningEligibilityComputer(config)
        self.oblique_engine = ObliqueTransmissionEngine(config)
        self.horizontal_engine = HorizontalTransmissionEngine(config)
        self.innovation_engine = CulturalInnovationEngine(config)
        self.memory_engine = MemoryDecayEngine(config)

        # Event tracking
        self.learning_events: list[dict[str, Any]] = []
        self.generation_stats: dict[str, Any] = {}

        # Initialize agent DataFrame with required columns
        self._ensure_cultural_columns()
        self._initialize_network_data()

        logger.info(f"Initialized vectorized cultural layer with {len(agent_set)} agents")

    def _ensure_cultural_columns(self) -> None:
        """Ensure agent DataFrame has all required cultural columns."""
        df = self.agents.agents
        required_columns = {
            "pref_culture": pl.UInt8,
            "cultural_innovation_count": pl.UInt16,
            "prestige_score": pl.Float32,
            "last_learning_event": pl.UInt32,
            "learning_eligibility": pl.Boolean,
        }

        # Only add effective_preference if not already present
        if "effective_preference" not in df.columns:
            required_columns["effective_preference"] = pl.UInt8

        # Add missing columns with default values
        new_columns = []
        for col_name, col_type in required_columns.items():
            if col_name not in df.columns:
                if col_type == pl.UInt8:
                    default_val = 128  # Middle of 0-255 range
                elif col_type == pl.UInt16:
                    default_val = 0
                elif col_type == pl.UInt32:
                    default_val = 0
                elif col_type == pl.Float32:
                    default_val = 0.0
                elif col_type == pl.Boolean:
                    default_val = True
                else:
                    default_val = 0

                new_columns.append(pl.lit(default_val).cast(col_type).alias(col_name))

        if new_columns:
            self.agents.agents = df.with_columns(new_columns)
            logger.debug(f"Added {len(new_columns)} missing cultural columns")

        # Add cultural memory columns if configured
        if self.config.cultural_memory_size > 0:
            self._add_memory_columns()

    def _add_memory_columns(self) -> None:
        """Add cultural memory columns to agent DataFrame."""
        df = self.agents.agents
        memory_columns = {}

        for i in range(self.config.cultural_memory_size):
            memory_col = f"cultural_memory_{i}"
            weight_col = f"memory_weights_{i}"

            if memory_col not in df.columns:
                memory_columns[memory_col] = pl.lit(0.0).cast(pl.Float32)
            if weight_col not in df.columns:
                memory_columns[weight_col] = pl.lit(0.0).cast(pl.Float32)

        if memory_columns:
            # Use a more explicit column creation method
            df_with_memory = df
            for col_name, col_expr in memory_columns.items():
                df_with_memory = df_with_memory.with_columns([col_expr.alias(col_name)])

            self.agents.agents = df_with_memory
            logger.debug(f"Added {len(memory_columns)} memory columns")

    def _initialize_network_data(self) -> None:
        """Initialize network data in agent DataFrame."""
        df = self.agents.agents

        # Add agent_id column if it doesn't exist (using row index)
        if "agent_id" not in df.columns:
            df = df.with_row_count("agent_id")

        # Ensure agent_id is UInt32 to match network schema
        df = df.with_columns([pl.col("agent_id").cast(pl.UInt32)])

        # Get network data and merge with agent DataFrame
        network_df = self.network.adjacency_df

        # Join network data with agent DataFrame
        try:
            updated_df = df.join(
                network_df.select(["agent_id", "neighbors", "degree"]), on="agent_id", how="left"
            ).with_columns(
                [
                    # Fill missing network data for new agents
                    pl.col("neighbors").fill_null([]),
                    pl.col("degree").fill_null(0).cast(pl.UInt32).alias("network_degree"),
                ]
            )

            self.agents.agents = updated_df
            logger.debug("Initialized network data in agent DataFrame")
        except Exception as e:
            logger.warning(f"Could not initialize network data: {e}")
            # Add basic network columns with default values
            self.agents.agents = df.with_columns(
                [
                    pl.lit([]).cast(pl.List(pl.UInt32)).alias("neighbors"),
                    pl.lit(0).cast(pl.UInt32).alias("network_degree"),
                ]
            )

    @beartype
    def step(self) -> None:
        """
        Execute one generation of vectorized cultural transmission.

        This replaces the sequential per-agent processing with coordinated
        vectorized operations across all cultural learning mechanisms.
        """
        df = self.agents.agents
        n_agents = len(df)

        if n_agents == 0:
            return

        # Update network size if population changed
        if n_agents != self.network.n_agents:
            self.network.update_network_size(n_agents)
            self._initialize_network_data()
            df = self.agents.agents  # Refresh after network update

        # 1. Compute learning eligibility (vectorized)
        eligibility_mask = self.eligibility_computer.compute_eligibility(df, self.generation)
        n_eligible = int(eligibility_mask.sum())

        # Track generation statistics
        self.generation_stats = {
            "generation": self.generation,
            "total_agents": n_agents,
            "eligible_agents": n_eligible,
            "learning_events": 0,
            "innovation_events": 0,
            "oblique_events": 0,
            "horizontal_events": 0,
        }

        if n_eligible == 0:
            logger.debug(f"Generation {self.generation}: No agents eligible for learning")
            self.generation += 1
            return

        # 2. Execute learning mechanisms in parallel
        all_learning_events = []

        # Oblique transmission (parent-offspring learning)
        if self.config.oblique_transmission_rate > 0:
            oblique_random = pl.Series(np.random.random(n_agents) < self.config.oblique_transmission_rate)
            oblique_learners = eligibility_mask & oblique_random
            if oblique_learners.any():
                oblique_events = self.oblique_engine.execute_transmission(df, oblique_learners)
                if len(oblique_events) > 0:
                    all_learning_events.append(oblique_events)
                    self.generation_stats["oblique_events"] = len(oblique_events)

        # Horizontal transmission (peer-to-peer learning)
        if self.config.horizontal_transmission_rate > 0:
            horizontal_random = pl.Series(np.random.random(n_agents) < self.config.horizontal_transmission_rate)
            horizontal_learners = eligibility_mask & horizontal_random
            if horizontal_learners.any():
                horizontal_events = self.horizontal_engine.execute_transmission(df, horizontal_learners)
                if len(horizontal_events) > 0:
                    all_learning_events.append(horizontal_events)
                    self.generation_stats["horizontal_events"] = len(horizontal_events)

        # Cultural innovation
        if self.config.innovation_rate > 0:
            innovation_events = self.innovation_engine.execute_innovation(df, self.config.innovation_rate)
            if len(innovation_events) > 0:
                all_learning_events.append(innovation_events)
                self.generation_stats["innovation_events"] = len(innovation_events)

        # 3. Apply all learning updates in single DataFrame operation
        if all_learning_events:
            self._apply_learning_updates(all_learning_events)
            self.generation_stats["learning_events"] = sum(len(events) for events in all_learning_events)

        # 4. Update prestige scores (vectorized)
        self._update_prestige_scores()

        # 5. Apply memory decay (vectorized)
        if self.config.cultural_memory_size > 0:
            self.agents.agents = self.memory_engine.apply_memory_decay(self.agents.agents)

        # 6. Update derived metrics
        self._update_learning_metadata()

        self.generation += 1

        logger.debug(
            f"Generation {self.generation}: "
            f"{self.generation_stats['learning_events']} learning events, "
            f"{self.generation_stats['innovation_events']} innovations"
        )

    def _apply_learning_updates(self, learning_events: list[pl.DataFrame]) -> None:
        """Apply all learning events to agent cultural preferences."""
        # Combine all learning events
        all_events = pl.concat(learning_events)

        if len(all_events) == 0:
            return

        # Group by learner to handle multiple events per agent (take last)
        unique_updates = all_events.group_by("learner_id").agg(
            [
                pl.col("new_preference").last().alias("new_preference"),
                pl.col("learning_type").last().alias("learning_type"),
            ]
        )

        # Apply updates to agent DataFrame using join and conditional update
        df = self.agents.agents
        updated_df = (
            df.join(
                unique_updates.select(
                    [pl.col("learner_id").alias("agent_id"), pl.col("new_preference"), pl.col("learning_type")]
                ),
                on="agent_id",
                how="left",
            )
            .with_columns(
                [
                    # Update cultural preference where learning occurred
                    pl.when(pl.col("new_preference").is_not_null())
                    .then(pl.col("new_preference"))
                    .otherwise(pl.col("pref_culture"))
                    .alias("pref_culture"),
                    # Update last learning event generation
                    pl.when(pl.col("new_preference").is_not_null())
                    .then(pl.lit(self.generation))
                    .otherwise(pl.col("last_learning_event"))
                    .alias("last_learning_event"),
                    # Update innovation count for innovation events
                    pl.when(pl.col("learning_type") == "innovation")
                    .then(pl.col("cultural_innovation_count") + 1)
                    .otherwise(pl.col("cultural_innovation_count"))
                    .alias("cultural_innovation_count"),
                ]
            )
            .drop(["new_preference", "learning_type"])
        )

        self.agents.agents = updated_df

        # Store events for analysis
        self.learning_events.extend(all_events.to_dicts())

    def _update_prestige_scores(self) -> None:
        """Update prestige scores using vectorized operations."""
        df = self.agents.agents

        # Compute prestige based on multiple factors
        prestige_expr = pl.lit(0.0)

        # Mating success component
        if "mating_success" in df.columns:
            max_mating = df.select(pl.col("mating_success").max()).item() or 1.0
            prestige_expr = prestige_expr + (pl.col("mating_success").cast(pl.Float32) / max_mating) * 0.5

        # Age/survival component
        if "age" in df.columns:
            max_age = df.select(pl.col("age").max()).item() or 1.0
            prestige_expr = prestige_expr + (pl.col("age").cast(pl.Float32) / max_age) * 0.3

        # Energy component
        if "energy" in df.columns:
            max_energy = df.select(pl.col("energy").max()).item() or 1.0
            prestige_expr = prestige_expr + (pl.col("energy").cast(pl.Float32) / max_energy) * 0.2

        # Update prestige scores
        self.agents.agents = df.with_columns([prestige_expr.alias("prestige_score")])

    def _update_learning_metadata(self) -> None:
        """Update learning eligibility and effective preferences."""
        df = self.agents.agents

        # Update learning eligibility for next generation
        eligibility = self.eligibility_computer.compute_eligibility(df, self.generation + 1)

        # Compute effective preferences (incorporating memory if enabled)
        if self.config.cultural_memory_size > 0:
            effective_pref = self.memory_engine.compute_effective_preference(df)
        else:
            effective_pref = df.get_column("pref_culture")

        # Apply updates
        self.agents.agents = df.with_columns(
            [eligibility.alias("learning_eligibility"), effective_pref.alias("effective_preference")]
        )

    @beartype
    def get_generation_statistics(self) -> dict[str, Any]:
        """
        Get statistics for the current generation.

        Returns
        -------
        dict[str, Any]
            Dictionary containing generation statistics
        """
        return self.generation_stats.copy()

    @beartype
    def get_learning_events_dataframe(self) -> pl.DataFrame:
        """
        Get all learning events as a Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame containing all learning events
        """
        if not self.learning_events:
            return pl.DataFrame(
                {
                    "learner_id": [],
                    "teacher_id": [],
                    "learning_type": [],
                    "old_preference": [],
                    "new_preference": [],
                    "generation": [],
                }
            )

        return pl.DataFrame(self.learning_events)

    @beartype
    def get_network_statistics(self) -> dict[str, Any]:
        """
        Get social network statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing network metrics
        """
        return self.network.get_network_statistics()

    @beartype
    def compute_cultural_diversity(self) -> float:
        """
        Compute population-level cultural diversity.

        Returns
        -------
        float
            Shannon diversity index of cultural preferences
        """
        df = self.agents.agents
        if len(df) == 0:
            return 0.0

        # Get cultural preference distribution
        pref_counts = df.group_by("pref_culture").len()
        if len(pref_counts) <= 1:
            return 0.0

        # Compute Shannon diversity
        counts = pref_counts.get_column("len").to_numpy()
        total = np.sum(counts)
        proportions = counts / total

        # Shannon entropy: -sum(p * log(p))
        diversity = -np.sum(proportions * np.log(proportions + 1e-10))
        return float(diversity)

    @beartype
    def reset(self) -> None:
        """Reset all cultural learning state."""
        self.generation = 0
        self.learning_events.clear()
        self.generation_stats.clear()

        # Reset agent cultural state to initial values
        df = self.agents.agents
        reset_columns = {
            "pref_culture": 128,  # Middle value
            "cultural_innovation_count": 0,
            "prestige_score": 0.0,
            "last_learning_event": 0,
            "learning_eligibility": True,
        }

        # Reset memory columns if they exist
        if self.config.cultural_memory_size > 0:
            for i in range(self.config.cultural_memory_size):
                reset_columns[f"cultural_memory_{i}"] = 0.0
                reset_columns[f"memory_weights_{i}"] = 0.0

        reset_updates = {
            col: pl.lit(val).cast(df.schema[col]) if col in df.columns else pl.lit(val)
            for col, val in reset_columns.items()
        }

        self.agents.agents = df.with_columns(reset_updates)

        logger.info("Reset vectorized cultural layer state")

    @property
    def current_generation(self) -> int:
        """Get the current generation number."""
        return self.generation

    @property
    def total_learning_events(self) -> int:
        """Get total number of learning events across all generations."""
        return len(self.learning_events)
