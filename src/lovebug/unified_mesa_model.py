"""
Enhanced Unified Mesa-Frames Model for Layer Activation.

This module provides a mesa-frames based unified model that can run genetic-only,
cultural-only, or combined evolution based on LayerActivationConfig settings.
Built on vectorized polars architecture with advanced cultural features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
from beartype import beartype
from mesa_frames import AgentSetPolars, ModelDF

from lovebug.lande_kirkpatrick import LandeKirkpatrickParams
from lovebug.layer2.config import Layer2Config
from lovebug.layer2.cultural_layer import CulturalLayer
from lovebug.layer2.network import NetworkTopology, SocialNetwork
from lovebug.layer_activation import LayerActivationConfig

__all__ = ["LoveAgents", "LoveModel"]

logger = logging.getLogger(__name__)

# ── Bit masks & shifts (same as original model) ────────────────────────────
DISPLAY_MASK = np.uint32(0x0000_FFFF)
PREF_MASK = np.uint32(0x00FF_0000)
BEHAV_MASK = np.uint32(0xFF00_0000)

DISPLAY_SHIFT = 0
PREF_SHIFT = 16
BEHAV_SHIFT = 24

# Default evolution parameters
DEFAULT_MUTATION_RATE = 1e-4
DEFAULT_ENERGY_DECAY = 0.2
DEFAULT_MAX_AGE = 100


def _bit_count_u32(values: np.ndarray) -> np.ndarray:
    """Return the number of set bits for each 32-bit integer."""
    return pl.Series(values.astype(np.uint32), dtype=pl.UInt32).bitwise_count_ones().to_numpy()


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return similarity (16 – Hamming distance) for lower 16 bits."""
    diff = (
        (pl.Series(a.astype(np.uint32), dtype=pl.UInt32) ^ pl.Series(b.astype(np.uint32), dtype=pl.UInt32))
        & DISPLAY_MASK
    ).bitwise_count_ones()
    return (16 - diff).to_numpy()


class LoveAgents(AgentSetPolars):
    """
    Enhanced unified agent set with advanced cultural features.

    All agents live inside a single Polars DataFrame with vectorised operations.
    Supports genetic-only, cultural-only, or combined evolution modes with
    advanced cultural learning mechanisms.
    """

    def __init__(self, n: int, model: LoveModel):
        super().__init__(model)
        self.layer_config = model.layer_config

        # Initialize genomes
        genomes = np.random.randint(0, 2**32, size=n, dtype=np.uint32)

        # Create base dataframe with genetic traits
        df_data = {
            "genome": genomes,
            "energy": pl.Series([10.0] * n, dtype=pl.Float32),
            "age": pl.Series([0] * n, dtype=pl.UInt16),
            "mating_success": pl.Series([0] * n, dtype=pl.UInt16),  # Track in DataFrame
        }

        # Add cultural traits if cultural layer is enabled
        if self.layer_config.cultural_enabled:
            df_data["pref_culture"] = ((genomes & PREF_MASK) >> PREF_SHIFT).astype(np.uint8)
            df_data["cultural_innovation_count"] = pl.Series([0] * n, dtype=pl.UInt16)
            df_data["prestige_score"] = pl.Series([0.0] * n, dtype=pl.Float32)

            # Add cultural memory columns
            if model.cultural_params and model.cultural_params.cultural_memory_size > 0:
                memory_size = model.cultural_params.cultural_memory_size
                for i in range(memory_size):
                    df_data[f"cultural_memory_{i}"] = pl.Series([0.0] * n, dtype=pl.Float32)

            # Add social network data
            df_data["social_network_neighbors"] = pl.Series([[] for _ in range(n)], dtype=pl.List(pl.Int32))

        # Add fields for tracking layer effects
        if self.layer_config.is_combined():
            df_data["effective_preference"] = ((genomes & PREF_MASK) >> PREF_SHIFT).astype(np.uint8)

        self += pl.DataFrame(df_data)
        self._mask = pl.repeat(True, len(self.agents), dtype=pl.Boolean, eager=True)

        # Initialize social network if cultural layer enabled
        if self.layer_config.cultural_enabled and model.cultural_params:
            self._initialize_social_network()

        # Initialize vectorized cultural layer if enabled and configured
        self.vectorized_cultural_layer = None
        if (
            self.layer_config.cultural_enabled
            and model.cultural_params
            and getattr(model, "use_vectorized_cultural_layer", True)
        ):
            try:
                self.vectorized_cultural_layer = CulturalLayer(self, model.cultural_params)
                logger.info("Initialized vectorized cultural layer")
            except Exception as e:
                logger.warning(f"Failed to initialize vectorized cultural layer: {e}")
                logger.info("Falling back to sequential cultural learning")

        logger.debug(f"Initialized {n} agents with layer config: {self.layer_config}")

    def _initialize_social_network(self) -> None:
        """Initialize social network and populate neighbor data."""
        if not self.model.cultural_params:
            return

        topology = NetworkTopology(
            network_type=self.model.cultural_params.network_type,
            connectivity=self.model.cultural_params.network_connectivity,
        )
        self.social_network = SocialNetwork(len(self), topology)

        # Populate neighbor data in DataFrame using vectorized operations
        neighbors_data = []
        for i in range(len(self)):
            neighbors = self.social_network.get_neighbors(i, max_neighbors=10)
            neighbors_data.append(neighbors)

        # Update the social network neighbors column
        self["social_network_neighbors"] = pl.Series(neighbors_data, dtype=pl.List(pl.Int32))

    # ── Properties for lazy genome slices ──────────────────────────────────
    @property
    def display(self):
        return (pl.col("genome") & pl.lit(DISPLAY_MASK, dtype=pl.UInt32)).alias("display")

    @property
    def preference(self):
        return ((pl.col("genome") & pl.lit(PREF_MASK, dtype=pl.UInt32)) // (1 << PREF_SHIFT)).alias("preference")

    @property
    def threshold(self):
        return ((pl.col("genome") & pl.lit(BEHAV_MASK, dtype=pl.UInt32)) // (1 << BEHAV_SHIFT)).alias("threshold")

    # ── Main timestep ──────────────────────────────────────────────────────
    def step(self) -> None:
        """Execute one timestep with active layers."""
        # Update effective preferences based on layer activation
        if self.layer_config.is_combined():
            self._update_effective_preferences()

        # Run main evolutionary processes
        self.courtship()

        if self.layer_config.cultural_enabled:
            if self.vectorized_cultural_layer:
                # Use vectorized cultural learning
                self.vectorized_cultural_layer.step()
                # Extract statistics for model tracking
                stats = self.vectorized_cultural_layer.get_generation_statistics()
                self.model._cultural_learning_events = stats.get("learning_events", 0)
                self.model._cultural_innovation_events = stats.get("innovation_events", 0)
            else:
                # Fallback to sequential cultural learning
                self._vectorized_cultural_learning()

        self.metabolism()
        self.age_and_die()

    def _update_effective_preferences(self) -> None:
        """Update effective preferences by blending genetic and cultural layers."""
        if not self.layer_config.is_combined():
            return

        n = len(self)
        if n == 0:
            return

        genetic_prefs = ((self.agents["genome"].to_numpy() & PREF_MASK) >> PREF_SHIFT).astype(np.uint8)
        cultural_prefs = self.agents["pref_culture"].to_numpy().astype(np.uint8)

        if self.layer_config.blending_mode == "weighted_average":
            # Weighted average blending
            genetic_weight = self.layer_config.get_effective_genetic_weight()
            cultural_weight = self.layer_config.get_effective_cultural_weight()

            # Blend at bit level for more realistic interaction
            effective_prefs = (genetic_weight * genetic_prefs + cultural_weight * cultural_prefs).astype(np.uint8)

        elif self.layer_config.blending_mode == "probabilistic":
            # Probabilistic switching
            genetic_weight = self.layer_config.get_effective_genetic_weight()
            choose_genetic = np.random.random(n) < genetic_weight
            effective_prefs = genetic_prefs.copy()
            effective_prefs[~choose_genetic] = cultural_prefs[~choose_genetic]

        elif self.layer_config.blending_mode == "competitive":
            # Competitive: stronger preference wins
            genetic_strength = np.abs(genetic_prefs.astype(np.float32) - 128)
            cultural_strength = np.abs(cultural_prefs.astype(np.float32) - 128)
            use_genetic = genetic_strength >= cultural_strength
            effective_prefs = genetic_prefs.copy()
            effective_prefs[~use_genetic] = cultural_prefs[~use_genetic]

        else:
            raise ValueError(f"Unknown blending mode: {self.layer_config.blending_mode}")

        self["effective_preference"] = pl.Series(effective_prefs, dtype=pl.UInt8)

    def _apply_perceptual_constraints(self, similarity_scores: np.ndarray) -> np.ndarray:
        """
        Apply perceptual constraints (θ_detect, σ_perception) from paper theory.

        Implements perceptual noise and detection thresholds that agents
        experience when evaluating potential mates' display traits.

        Parameters
        ----------
        similarity_scores : np.ndarray
            Raw similarity scores between displays and preferences

        Returns
        -------
        np.ndarray
            Perceived similarity scores after applying constraints
        """
        # Apply perceptual noise using σ_perception parameter
        noise = np.random.normal(0, self.layer_config.sigma_perception, size=similarity_scores.shape)
        noisy_scores = similarity_scores.astype(np.float32) + noise

        # Apply detection threshold - signals below θ_detect are undetectable
        perceived_scores = np.where(
            noisy_scores >= self.layer_config.theta_detect,
            noisy_scores,
            0.0,  # Undetectable signals score as 0
        )

        # Clamp to valid similarity range [0, 16]
        return np.clip(perceived_scores, 0.0, 16.0)

    def get_effective_pref(self) -> np.ndarray:
        """Get effective preference based on layer activation."""
        if self.layer_config.is_genetic_only():
            # Genetic only
            return ((self.agents["genome"].to_numpy() & PREF_MASK) >> PREF_SHIFT).astype(np.uint8)
        elif self.layer_config.is_cultural_only():
            # Cultural only
            return self.agents["pref_culture"].to_numpy().astype(np.uint8)
        else:
            # Combined - use pre-computed effective preference
            return self.agents["effective_preference"].to_numpy().astype(np.uint8)

    # ── Vectorised sub‑routines ────────────────────────────────────────────
    def courtship(self) -> None:
        """Randomly pair agents, create offspring when mutual acceptance."""
        n = len(self)
        if n < 2:
            return

        partners = np.random.permutation(n)
        genomes_self = self.agents["genome"].to_numpy().astype(np.uint32)
        genomes_partner = genomes_self[partners]

        # Reset mating success for this generation
        self["mating_success"] = pl.Series([0] * n, dtype=pl.UInt16)

        # Slice genomes into fields
        disp_self = genomes_self & DISPLAY_MASK
        pref_self = self.get_effective_pref()
        thr_self = (genomes_self & BEHAV_MASK) >> BEHAV_SHIFT

        disp_partner = genomes_partner & DISPLAY_MASK
        pref_partner = pref_self[partners]
        thr_partner = (genomes_partner & BEHAV_MASK) >> BEHAV_SHIFT

        # Similarity scores (0‑16) with perceptual constraints from paper theory
        sim_self = hamming_similarity(disp_partner, pref_self)
        sim_partner = hamming_similarity(disp_self, pref_partner)

        # Apply perceptual noise and detection thresholds (θ_detect, σ_perception)
        sim_self_perceived = self._apply_perceptual_constraints(sim_self)
        sim_partner_perceived = self._apply_perceptual_constraints(sim_partner)

        accepted = (sim_self_perceived >= thr_self) & (sim_partner_perceived >= thr_partner)
        if not accepted.any():
            return

        idx = np.where(accepted)[0]
        partner_idx = partners[idx]
        parents_a = genomes_self[idx]
        parents_b = genomes_partner[idx]

        # Update mating success in DataFrame
        mating_success = self.agents["mating_success"].to_numpy().copy()
        np.add.at(mating_success, idx, 1)
        np.add.at(mating_success, partner_idx, 1)
        self["mating_success"] = pl.Series(mating_success, dtype=pl.UInt16)

        # Apply genetic parameters if available
        mutation_rate = DEFAULT_MUTATION_RATE
        if self.model.genetic_params:
            mutation_rate = self.model.genetic_params.mutation_variance

        # Uniform crossover via random mask
        mask = np.random.randint(0, 2**32, size=len(idx))
        offspring_genomes = (parents_a & mask) | (parents_b & ~mask)

        # Bit‑flip mutation
        if mutation_rate > 0:
            flips = np.random.binomial(1, mutation_rate, size=(len(idx), 32)).astype(bool)
            rows, bits = np.where(flips)
            if rows.size:
                offspring_genomes[rows] ^= np.left_shift(np.uint32(1), bits.astype(np.uint32))

        # Prepare offspring data
        offspring_data = {
            "genome": offspring_genomes,
            "energy": pl.Series([10.0] * len(idx), dtype=pl.Float32),
            "age": pl.Series([0] * len(idx), dtype=pl.UInt16),
            "mating_success": pl.Series([0] * len(idx), dtype=pl.UInt16),
        }

        # Add cultural traits for offspring if cultural layer enabled
        if self.layer_config.cultural_enabled:
            # Inherit cultural preferences with some innovation
            parent_cultures_a = self.agents["pref_culture"].to_numpy()[idx]
            parent_cultures_b = self.agents["pref_culture"].to_numpy()[partner_idx]

            innovation_rate = 0.05
            if self.model.cultural_params:
                innovation_rate = self.model.cultural_params.innovation_rate

            # Cultural inheritance (choose randomly from parents)
            inherit_from_a = np.random.random(len(idx)) < 0.5
            offspring_culture = parent_cultures_a.copy()
            offspring_culture[~inherit_from_a] = parent_cultures_b[~inherit_from_a]

            # Apply innovation
            innovate = np.random.random(len(idx)) < innovation_rate
            if innovate.any():
                n_innovate = int(np.sum(innovate))
                offspring_culture[innovate] = np.random.randint(0, 256, n_innovate, dtype=np.uint8)

            offspring_data["pref_culture"] = offspring_culture.astype(np.uint8)
            offspring_data["cultural_innovation_count"] = pl.Series([0] * len(idx), dtype=pl.UInt16)
            offspring_data["prestige_score"] = pl.Series([0.0] * len(idx), dtype=pl.Float32)
            offspring_data["social_network_neighbors"] = pl.Series(
                [[] for _ in range(len(idx))], dtype=pl.List(pl.Int32)
            )

            # Add cultural memory columns for offspring
            if self.model.cultural_params and self.model.cultural_params.cultural_memory_size > 0:
                memory_size = self.model.cultural_params.cultural_memory_size
                for i in range(memory_size):
                    offspring_data[f"cultural_memory_{i}"] = pl.Series([0.0] * len(idx), dtype=pl.Float32)

        # Add effective preference for combined models
        if self.layer_config.is_combined():
            genetic_pref = ((offspring_genomes & PREF_MASK) >> PREF_SHIFT).astype(np.uint8)
            offspring_data["effective_preference"] = genetic_pref

        # Append offspring
        self += pl.DataFrame(offspring_data)

    def _vectorized_cultural_learning(self) -> None:
        """Enhanced vectorized cultural learning with advanced features."""
        if not self.layer_config.cultural_enabled:
            return

        n = len(self)
        if n == 0:
            return

        # Get learning parameters
        p_learn = 0.1
        innovation_rate = 0.05
        oblique_rate = 0.0

        if self.model.cultural_params:
            p_learn = self.model.cultural_params.horizontal_transmission_rate
            innovation_rate = self.model.cultural_params.innovation_rate
            oblique_rate = self.model.cultural_params.oblique_transmission_rate

        # Track events for this generation
        cultural_events = []
        innovation_events = []

        # 1. Update prestige scores using vectorized operations
        self._update_prestige_scores()

        # 2. Innovation process (vectorized)
        if innovation_rate > 0:
            innovators = np.random.random(n) < innovation_rate
            if innovators.any():
                culture_array = self.agents["pref_culture"].to_numpy().copy()
                n_innovate = int(np.sum(innovators))
                culture_array[innovators] = np.random.randint(0, 256, n_innovate, dtype=np.uint8)
                self["pref_culture"] = pl.Series(culture_array, dtype=pl.UInt8)
                innovation_events.extend([i for i in range(n) if innovators[i]])

        # 3. Horizontal transmission (prestige-based, vectorized)
        if p_learn > 0:
            learners = np.random.random(n) < p_learn
            if learners.any():
                self._vectorized_prestige_learning(learners)
                cultural_events.extend([i for i in range(n) if learners[i]])

        # 4. Oblique transmission (vectorized)
        if oblique_rate > 0:
            oblique_learners = np.random.random(n) < oblique_rate
            if oblique_learners.any():
                self._vectorized_oblique_transmission(oblique_learners)

        # 5. Update cultural memory (vectorized)
        if self.model.cultural_params and self.model.cultural_params.cultural_memory_size > 0:
            self._update_cultural_memory_vectorized()

        # Store event counts for metrics
        self.model._cultural_learning_events = len(cultural_events)
        self.model._cultural_innovation_events = len(innovation_events)

    def _update_prestige_scores(self) -> None:
        """Update prestige scores using vectorized operations."""
        n = len(self)
        if n == 0:
            return

        # Get data from DataFrame - all arrays are same size
        mating_scores = self.agents["mating_success"].to_numpy().astype(np.float32)
        ages = self.agents["age"].to_numpy().astype(np.float32)
        energies = self.agents["energy"].to_numpy().astype(np.float32)

        # Verify all arrays have the same size
        assert len(mating_scores) == len(ages) == len(energies) == n, (
            f"Array size mismatch: mating={len(mating_scores)}, ages={len(ages)}, energies={len(energies)}, n={n}"
        )

        # Normalize scores
        max_mating = max(1.0, float(np.max(mating_scores)))
        max_age = max(1.0, float(np.max(ages)))
        max_energy = max(1.0, float(np.max(energies)))

        prestige_scores = 0.5 * (mating_scores / max_mating) + 0.3 * (ages / max_age) + 0.2 * (energies / max_energy)

        self["prestige_score"] = pl.Series(prestige_scores, dtype=pl.Float32)

    def _vectorized_prestige_learning(self, learners: np.ndarray) -> None:
        """
        Vectorized prestige-based learning with local radius constraint.

        Uses local_learning_radius from Layer2Config to respect spatial/social
        constraints on cultural transmission as described in paper theory.
        """
        if not learners.any():
            return

        n = len(self)
        prestige_scores = self.agents["prestige_score"].to_numpy()
        culture_array = self.agents["pref_culture"].to_numpy().copy()
        learner_indices = np.where(learners)[0]

        # Get local learning radius from config
        local_radius = 5  # Default fallback
        if self.model.cultural_params:
            local_radius = self.model.cultural_params.local_learning_radius

        # Apply learning with local radius constraint
        for learner_idx in learner_indices:
            # Get neighbors within local learning radius
            if hasattr(self, "social_network") and self.social_network:
                potential_teachers = self.social_network.get_local_neighbors(int(learner_idx), radius=local_radius)
            else:
                # Fallback to random subset if no social network
                potential_teachers = np.random.choice(n, size=min(local_radius * 2, n), replace=False).tolist()

            if not potential_teachers:
                continue

            # Filter by prestige within local neighborhood
            teacher_prestige = prestige_scores[potential_teachers]

            if np.sum(teacher_prestige) > 0:
                # Prestige-weighted selection within local neighborhood
                teacher_probs = teacher_prestige / np.sum(teacher_prestige)
                selected_teacher = np.random.choice(potential_teachers, p=teacher_probs)
            else:
                # Random selection if no prestige differences
                selected_teacher = np.random.choice(potential_teachers)

            # Apply cultural transmission
            culture_array[learner_idx] = culture_array[selected_teacher]

        self["pref_culture"] = pl.Series(culture_array, dtype=pl.UInt8)

    def _vectorized_oblique_transmission(self, learners: np.ndarray) -> None:
        """Vectorized oblique transmission from older agents."""
        if not learners.any():
            return

        ages = self.agents["age"].to_numpy()
        culture_array = self.agents["pref_culture"].to_numpy().copy()

        # Find older agents
        age_threshold = np.percentile(ages, 75)  # Top 25% by age
        older_agents = ages >= age_threshold

        if not older_agents.any():
            return

        older_indices = np.where(older_agents)[0]
        n_learners = int(np.sum(learners))

        # Random selection from older agents
        teacher_indices = np.random.choice(older_indices, size=n_learners, replace=True)
        learner_indices = np.where(learners)[0]

        culture_array[learner_indices] = culture_array[teacher_indices]
        self["pref_culture"] = pl.Series(culture_array, dtype=pl.UInt8)

    def _update_cultural_memory_vectorized(self) -> None:
        """Update cultural memory using vectorized operations."""
        if not self.model.cultural_params:
            return

        memory_size = self.model.cultural_params.cultural_memory_size
        decay_rate = self.model.cultural_params.memory_decay_rate

        # Shift memories (move older memories down)
        for i in range(memory_size - 1, 0, -1):
            old_col = f"cultural_memory_{i - 1}"
            new_col = f"cultural_memory_{i}"
            if old_col in self.agents.columns:
                decayed_values = self.agents[old_col] * (1.0 - decay_rate)
                self[new_col] = decayed_values

        # Add current cultural preference to memory_0
        current_prefs = self.agents["pref_culture"].cast(pl.Float32)
        self["cultural_memory_0"] = current_prefs

    def social_learning(self) -> None:
        """Legacy method - replaced by _vectorized_cultural_learning."""
        self._vectorized_cultural_learning()

    def metabolism(self) -> None:
        """Apply energy decay and remove depleted agents."""
        energy_decay = DEFAULT_ENERGY_DECAY
        if self.model.genetic_params:
            # Could add energy parameters to genetic_params if needed
            pass

        self["energy"] -= energy_decay
        self.select(self.energy > 0)

    def age_and_die(self) -> None:
        """Age agents and remove those exceeding max age."""
        max_age = DEFAULT_MAX_AGE
        if self.model.genetic_params:
            # Could add age parameters to genetic_params if needed
            pass

        self["age"] += 1
        self.select(self.age < max_age)


class LoveModel(ModelDF):
    """
    Enhanced Unified Mesa-Frames model supporting advanced layer activation.

    This model can run genetic-only, cultural-only, or combined evolution
    based on LayerActivationConfig settings while maintaining the vectorized
    efficiency and adding advanced cultural features.

    Parameters
    ----------
    layer_config : LayerActivationConfig
        Configuration specifying which layers are active and their weights
    genetic_params : LandeKirkpatrickParams | None, optional
        Parameters for genetic evolution
    cultural_params : Layer2Config | None, optional
        Parameters for cultural evolution
    n_agents : int, default=1000
        Number of agents in the population
    use_vectorized_cultural_layer : bool, default=True
        Whether to use the vectorized cultural layer implementation

    Examples
    --------
    >>> # Pure genetic evolution
    >>> genetic_config = LayerActivationConfig.genetic_only()
    >>> genetic_params = LandeKirkpatrickParams()
    >>> model = LoveModel(genetic_config, genetic_params, None)
    >>> model.run(100)

    >>> # Combined evolution
    >>> combined_config = LayerActivationConfig.balanced_combined(0.6)
    >>> cultural_params = Layer2Config()
    >>> model = LoveModel(combined_config, genetic_params, cultural_params)
    >>> model.run(100)
    """

    @beartype
    def __init__(
        self,
        layer_config: LayerActivationConfig,
        genetic_params: LandeKirkpatrickParams | None = None,
        cultural_params: Layer2Config | None = None,
        n_agents: int = 1000,
        use_vectorized_cultural_layer: bool = True,
    ) -> None:
        super().__init__()

        self.layer_config = layer_config
        self.genetic_params = genetic_params
        self.cultural_params = cultural_params
        self.use_vectorized_cultural_layer = use_vectorized_cultural_layer

        # Event tracking
        self._cultural_learning_events = 0
        self._cultural_innovation_events = 0

        # Validate required parameters
        if layer_config.genetic_enabled and genetic_params is None:
            logger.warning("genetic_params not provided for genetic layer - using defaults")
        if layer_config.cultural_enabled and cultural_params is None:
            logger.warning("cultural_params not provided for cultural layer - using defaults")

        # Initialize agents
        self.agents += LoveAgents(n_agents, self)

        # Track metrics
        self.history: list[dict[str, Any]] = []
        self.step_count = 0

        logger.info(
            f"Enhanced LoveModel initialized: genetic={layer_config.genetic_enabled}, "
            f"cultural={layer_config.cultural_enabled}, n_agents={n_agents}"
        )

    def step(self) -> None:
        """Execute one model timestep and collect metrics."""
        # Reset event counters
        self._cultural_learning_events = 0
        self._cultural_innovation_events = 0

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
        metrics = {
            "step": self.step_count,
            "population_size": len(df),
        }

        # Safe metric collection with proper null handling
        age_mean = df["age"].mean()
        energy_mean = df["energy"].mean()

        metrics.update(
            {
                "mean_age": float(age_mean) if age_mean is not None else 0.0,
                "mean_energy": float(energy_mean) if energy_mean is not None else 0.0,
            }
        )

        # Genetic metrics
        if self.layer_config.genetic_enabled:
            gene_pref = ((df["genome"] & PREF_MASK) // (1 << PREF_SHIFT)).cast(pl.UInt32)
            gene_pref_mean = gene_pref.mean()
            gene_pref_var = gene_pref.var()

            metrics.update(
                {
                    "mean_genetic_preference": float(gene_pref_mean) if gene_pref_mean is not None else 0.0,
                    "var_genetic_preference": float(gene_pref_var) if gene_pref_var is not None else 0.0,
                }
            )

        # Cultural metrics
        if self.layer_config.cultural_enabled and "pref_culture" in df.columns:
            culture_pref = df["pref_culture"].cast(pl.UInt32)
            culture_pref_mean = culture_pref.mean()
            culture_pref_var = culture_pref.var()

            metrics.update(
                {
                    "mean_cultural_preference": float(culture_pref_mean) if culture_pref_mean is not None else 0.0,
                    "var_cultural_preference": float(culture_pref_var) if culture_pref_var is not None else 0.0,
                    "cultural_learning_events": self._cultural_learning_events,
                    "cultural_innovation_events": self._cultural_innovation_events,
                }
            )

            # Prestige metrics
            if "prestige_score" in df.columns:
                prestige = df["prestige_score"]
                prestige_mean = prestige.mean()
                prestige_var = prestige.var()

                metrics.update(
                    {
                        "mean_prestige_score": float(prestige_mean) if prestige_mean is not None else 0.0,
                        "var_prestige_score": float(prestige_var) if prestige_var is not None else 0.0,
                    }
                )

            # Gene-culture distance
            if self.layer_config.genetic_enabled:
                gene = ((df["genome"] & PREF_MASK) // (1 << PREF_SHIFT)).cast(pl.UInt32)
                culture = df["pref_culture"].cast(pl.UInt32)
                dist = ((gene ^ culture) & 0xFF).bitwise_count_ones().mean()
                metrics["gene_culture_distance"] = float(dist) if dist is not None else 0.0

        # Combined model metrics
        if self.layer_config.is_combined() and "effective_preference" in df.columns:
            eff_pref = df["effective_preference"].cast(pl.UInt32)
            eff_pref_mean = eff_pref.mean()
            eff_pref_var = eff_pref.var()

            metrics.update(
                {
                    "mean_effective_preference": float(eff_pref_mean) if eff_pref_mean is not None else 0.0,
                    "var_effective_preference": float(eff_pref_var) if eff_pref_var is not None else 0.0,
                }
            )

        return metrics

    @beartype
    def run(self, n_steps: int = 100) -> dict[str, Any]:
        """
        Run simulation for specified number of steps.

        Parameters
        ----------
        n_steps : int, default=100
            Number of steps to run

        Returns
        -------
        dict[str, Any]
            Complete simulation results including trajectory data
        """
        logger.info(f"Starting enhanced unified simulation: {n_steps} steps")

        for _ in range(n_steps):
            self.step()

        # Compile results
        results = {
            "layer_config": self.layer_config.to_dict(),
            "n_steps": n_steps,
            "final_population": len(self.agents),
            "trajectory": self.history,
            "final_metrics": self.history[-1] if self.history else {},
        }

        # Add layer-specific summaries
        if self.layer_config.genetic_enabled:
            results["genetic_summary"] = self._summarize_genetic_evolution()

        if self.layer_config.cultural_enabled:
            results["cultural_summary"] = self._summarize_cultural_evolution()

        if self.layer_config.is_combined():
            results["interaction_summary"] = self._summarize_layer_interactions()

        logger.info(f"Enhanced unified simulation completed: final population {len(self.agents)}")
        return results

    def _summarize_genetic_evolution(self) -> dict[str, Any]:
        """Summarize genetic evolution outcomes."""
        if not self.history:
            return {}

        initial = self.history[0]
        final = self.history[-1]

        return {
            "trait_evolution": final.get("mean_genetic_trait", 0.0) - initial.get("mean_genetic_trait", 0.0),
            "preference_evolution": final.get("mean_genetic_preference", 0.0)
            - initial.get("mean_genetic_preference", 0.0),
            "final_genetic_variance": final.get("var_genetic_preference", 0.0),
            "generations_to_fixation": self._calculate_fixation_time("mean_genetic_preference"),
        }

    def _summarize_cultural_evolution(self) -> dict[str, Any]:
        """Summarize cultural evolution outcomes."""
        if not self.history:
            return {}

        initial = self.history[0]
        final = self.history[-1]

        total_cultural_events = sum(h.get("cultural_learning_events", 0) for h in self.history)
        total_innovation_events = sum(h.get("cultural_innovation_events", 0) for h in self.history)

        return {
            "cultural_preference_evolution": final.get("mean_cultural_preference", 0.0)
            - initial.get("mean_cultural_preference", 0.0),
            "final_cultural_variance": final.get("var_cultural_preference", 0.0),
            "total_cultural_events": total_cultural_events,
            "total_innovation_events": total_innovation_events,
            "cultural_learning_rate": total_cultural_events / max(1, len(self.history) * len(self.agents)),
        }

    def _summarize_layer_interactions(self) -> dict[str, Any]:
        """Summarize interactions between genetic and cultural layers."""
        if not self.history:
            return {}

        correlations = [h.get("gene_culture_correlation", 0.0) for h in self.history]
        distances = [h.get("gene_culture_distance", 0.0) for h in self.history]

        return {
            "final_gene_culture_correlation": correlations[-1] if correlations else 0.0,
            "max_correlation": max(correlations) if correlations else 0.0,
            "mean_gene_culture_distance": np.mean(distances) if distances else 0.0,
            "final_gene_culture_distance": distances[-1] if distances else 0.0,
            "interaction_strength": self.layer_config.genetic_weight * self.layer_config.cultural_weight,
        }

    def _calculate_fixation_time(self, trait_name: str) -> int | None:
        """Calculate generations to fixation for a trait."""
        if len(self.history) < 2:
            return None

        # Simple heuristic: when variance drops below threshold
        variance_key = trait_name.replace("mean_", "var_")

        for i, metrics in enumerate(self.history):
            if metrics.get(variance_key, 1.0) < 0.01:
                return i

        return None

    @beartype
    def get_agent_dataframe(self) -> pl.DataFrame:
        """
        Get the current agent dataframe.

        Returns
        -------
        pl.DataFrame
            Current agent population as polars DataFrame
        """
        return self.agents._agentsets[0].agents.clone()
