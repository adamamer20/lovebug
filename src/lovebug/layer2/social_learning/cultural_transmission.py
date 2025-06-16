"""
Cultural transmission manager for Layer 2 social learning mechanisms.

Coordinates oblique transmission, horizontal learning, innovation, and prestige-based
learning with social network effects and cultural memory systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl
from beartype import beartype

from ..config import Layer2Config
from .social_networks import SocialNetwork

__all__ = ["CulturalTransmissionManager", "LearningEvent", "LearningType"]

logger = logging.getLogger(__name__)


class LearningType(Enum):
    """Types of cultural learning events."""

    OBLIQUE = "oblique"  # Parent-offspring transmission
    HORIZONTAL = "horizontal"  # Peer-to-peer learning
    INNOVATION = "innovation"  # Cultural mutation/creativity
    PRESTIGE = "prestige"  # Success-based model selection


@dataclass(slots=True, frozen=False)
class LearningEvent:
    """
    Record of a cultural learning event.

    Parameters
    ----------
    learner_id : int
        ID of the learning agent
    teacher_id : int | None
        ID of the teaching agent (None for innovation)
    learning_type : LearningType
        Type of learning mechanism
    old_preference : int
        Previous cultural preference
    new_preference : int
        New cultural preference after learning
    success_metric : float
        Success metric of teacher (for prestige learning)
    generation : int
        Generation when learning occurred
    """

    learner_id: int
    teacher_id: int | None
    learning_type: LearningType
    old_preference: int
    new_preference: int
    success_metric: float = 0.0
    generation: int = 0


@runtime_checkable
class AgentDataProtocol(Protocol):
    """Protocol for agent data access."""

    def get_cultural_preferences(self) -> np.ndarray:
        """Get array of cultural preferences."""
        ...

    def get_genetic_preferences(self) -> np.ndarray:
        """Get array of genetic preferences."""
        ...

    def get_mating_success(self) -> np.ndarray:
        """Get array of mating success scores."""
        ...

    def get_ages(self) -> np.ndarray:
        """Get array of agent ages."""
        ...

    def get_agent_ids(self) -> np.ndarray:
        """Get array of agent IDs."""
        ...

    def update_cultural_preference(self, agent_id: int, new_preference: int) -> None:
        """Update an agent's cultural preference."""
        ...


class CulturalTransmissionManager:
    """
    Manages all cultural transmission mechanisms for Layer 2.

    Coordinates different types of cultural learning including oblique transmission,
    horizontal learning, innovation, and prestige-based learning within social
    network structures.

    Parameters
    ----------
    config : Layer2Config
        Configuration for cultural transmission parameters
    social_network : SocialNetwork
        Social network for managing agent connections

    Examples
    --------
    >>> config = Layer2Config(oblique_transmission_rate=0.3)
    >>> network = SocialNetwork(1000, NetworkTopology("small_world"))
    >>> manager = CulturalTransmissionManager(config, network)
    >>> events = manager.process_cultural_learning(agent_data, generation=10)
    """

    def __init__(self, config: Layer2Config, social_network: SocialNetwork) -> None:
        self.config = config
        self.social_network = social_network
        self.learning_events: list[LearningEvent] = []
        self.generation_count = 0

        # Cultural memory system
        self._cultural_memory: dict[int, list[int]] = {}
        self._memory_strengths: dict[int, list[float]] = {}

    @beartype
    def process_cultural_learning(self, agent_data: AgentDataProtocol, generation: int) -> list[LearningEvent]:
        """
        Process all cultural learning for one generation.

        Parameters
        ----------
        agent_data : AgentDataProtocol
            Interface to agent data
        generation : int
            Current generation number

        Returns
        -------
        list[LearningEvent]
            List of cultural learning events that occurred
        """
        self.generation_count = generation
        generation_events: list[LearningEvent] = []

        agent_ids = agent_data.get_agent_ids()
        cultural_prefs = agent_data.get_cultural_preferences()

        # Update network size if population changed
        if len(agent_ids) != self.social_network.n_agents:
            self.social_network.update_network_size(len(agent_ids))

        # Process each type of cultural learning
        for agent_id in agent_ids:
            if agent_id >= len(cultural_prefs):
                continue

            current_pref = cultural_prefs[agent_id]

            # Try different learning mechanisms
            event = self._attempt_cultural_learning(agent_id, current_pref, agent_data)

            if event is not None:
                event.generation = generation
                generation_events.append(event)
                self.learning_events.append(event)

                # Update agent's cultural preference
                agent_data.update_cultural_preference(event.learner_id, event.new_preference)

                # Update cultural memory
                self._update_cultural_memory(event.learner_id, event.new_preference)

        # Decay cultural memories
        self._decay_cultural_memories()

        logger.info(f"Generation {generation}: {len(generation_events)} cultural learning events")
        return generation_events

    def _attempt_cultural_learning(
        self, agent_id: int, current_pref: int, agent_data: AgentDataProtocol
    ) -> LearningEvent | None:
        """Attempt cultural learning for a single agent."""

        # Determine which learning mechanism to use
        learning_prob = np.random.random()
        cumulative_prob = 0.0

        # Oblique transmission (parent-offspring)
        cumulative_prob += self.config.oblique_transmission_rate
        if learning_prob < cumulative_prob:
            return self._oblique_transmission(agent_id, current_pref, agent_data)

        # Horizontal transmission (peer learning)
        cumulative_prob += self.config.horizontal_transmission_rate
        if learning_prob < cumulative_prob:
            return self._horizontal_transmission(agent_id, current_pref, agent_data)

        # Innovation (cultural mutation)
        cumulative_prob += self.config.innovation_rate
        if learning_prob < cumulative_prob:
            return self._cultural_innovation(agent_id, current_pref)

        # No learning occurred
        return None

    def _oblique_transmission(
        self, agent_id: int, current_pref: int, agent_data: AgentDataProtocol
    ) -> LearningEvent | None:
        """
        Oblique transmission: learning from older generation agents.

        Agents learn cultural preferences from older, successful agents
        representing parent-generation cultural transmission.
        """
        ages = agent_data.get_ages()
        mating_success = agent_data.get_mating_success()
        cultural_prefs = agent_data.get_cultural_preferences()
        agent_ids = agent_data.get_agent_ids()

        # Find older agents (potential parents)
        agent_age = ages[agent_id] if agent_id < len(ages) else 0
        older_agents = [
            aid
            for aid in agent_ids
            if aid < len(ages) and ages[aid] > agent_age + 5  # At least 5 timesteps older
        ]

        if not older_agents:
            return None

        # Select based on success (prestige-based selection within older generation)
        older_success = [mating_success[aid] for aid in older_agents if aid < len(mating_success)]
        if not older_success or all(s == 0 for s in older_success):
            # Random selection if no success differences
            teacher_id = np.random.choice(older_agents)
            teacher_success = 0.0
        else:
            # Probability proportional to success
            success_probs = np.array(older_success, dtype=float)
            success_probs += 0.1  # Small base probability
            success_probs /= success_probs.sum()

            teacher_idx = np.random.choice(len(older_agents), p=success_probs)
            teacher_id = older_agents[teacher_idx]
            teacher_success = older_success[teacher_idx]

        if teacher_id >= len(cultural_prefs):
            return None

        new_preference = cultural_prefs[teacher_id]

        return LearningEvent(
            learner_id=agent_id,
            teacher_id=teacher_id,
            learning_type=LearningType.OBLIQUE,
            old_preference=current_pref,
            new_preference=new_preference,
            success_metric=teacher_success,
        )

    def _horizontal_transmission(
        self, agent_id: int, current_pref: int, agent_data: AgentDataProtocol
    ) -> LearningEvent | None:
        """
        Horizontal transmission: peer-to-peer learning within generation.

        Agents learn from their social network neighbors using conformist
        or prestige-based selection.
        """
        # Get neighbors from social network
        neighbors = self.social_network.get_neighbors(agent_id, max_neighbors=5)

        if not neighbors:
            return None

        cultural_prefs = agent_data.get_cultural_preferences()
        mating_success = agent_data.get_mating_success()

        # Filter valid neighbors
        valid_neighbors = [n for n in neighbors if n < len(cultural_prefs)]
        if not valid_neighbors:
            return None

        # Choose learning strategy
        if np.random.random() < 0.7:  # 70% conformist, 30% prestige-based
            # Conformist learning: adopt majority preference
            neighbor_prefs = [cultural_prefs[n] for n in valid_neighbors]

            # Find most common preference
            unique_prefs, counts = np.unique(neighbor_prefs, return_counts=True)
            majority_pref = unique_prefs[np.argmax(counts)]

            # Select a random agent with the majority preference as teacher
            majority_agents = [n for n in valid_neighbors if cultural_prefs[n] == majority_pref]
            teacher_id = np.random.choice(majority_agents)
            teacher_success = mating_success[teacher_id] if teacher_id < len(mating_success) else 0.0

        else:
            # Prestige-based learning: learn from most successful neighbor
            neighbor_success = [mating_success[n] if n < len(mating_success) else 0.0 for n in valid_neighbors]

            if all(s == 0 for s in neighbor_success):
                # Random selection if no success differences
                teacher_id = np.random.choice(valid_neighbors)
                teacher_success = 0.0
            else:
                best_idx = np.argmax(neighbor_success)
                teacher_id = valid_neighbors[best_idx]
                teacher_success = neighbor_success[best_idx]

        new_preference = cultural_prefs[teacher_id]

        return LearningEvent(
            learner_id=agent_id,
            teacher_id=teacher_id,
            learning_type=LearningType.HORIZONTAL,
            old_preference=current_pref,
            new_preference=new_preference,
            success_metric=teacher_success,
        )

    def _cultural_innovation(self, agent_id: int, current_pref: int) -> LearningEvent | None:
        """
        Cultural innovation: random mutation of cultural preferences.

        Represents creativity, experimentation, or random errors in
        cultural transmission.
        """
        # Generate new preference through mutation
        if np.random.random() < 0.5:
            # Bit-flip mutation (similar to genetic mutation)
            new_pref = current_pref ^ (1 << np.random.randint(0, 8))  # Flip random bit
        else:
            # Random new preference
            new_pref = np.random.randint(0, 256)  # 8-bit preference space

        return LearningEvent(
            learner_id=agent_id,
            teacher_id=None,  # No teacher for innovation
            learning_type=LearningType.INNOVATION,
            old_preference=current_pref,
            new_preference=new_pref,
            success_metric=0.0,
        )

    def _update_cultural_memory(self, agent_id: int, new_preference: int) -> None:
        """Update an agent's cultural memory with new experience."""
        if agent_id not in self._cultural_memory:
            self._cultural_memory[agent_id] = []
            self._memory_strengths[agent_id] = []

        memory = self._cultural_memory[agent_id]
        strengths = self._memory_strengths[agent_id]

        # Add new preference to memory
        memory.append(new_preference)
        strengths.append(self.config.memory_update_strength)

        # Limit memory size
        if len(memory) > self.config.cultural_memory_size:
            memory.pop(0)
            strengths.pop(0)

    def _decay_cultural_memories(self) -> None:
        """Apply decay to all cultural memories."""
        for agent_id in self._memory_strengths:
            strengths = self._memory_strengths[agent_id]
            # Apply exponential decay
            for i in range(len(strengths)):
                strengths[i] *= 1.0 - self.config.memory_decay_rate

            # Remove very weak memories
            threshold = 0.01
            indices_to_keep = [i for i, s in enumerate(strengths) if s > threshold]

            if len(indices_to_keep) < len(strengths):
                self._cultural_memory[agent_id] = [self._cultural_memory[agent_id][i] for i in indices_to_keep]
                self._memory_strengths[agent_id] = [strengths[i] for i in indices_to_keep]

    @beartype
    def get_effective_cultural_preference(self, agent_id: int, current_pref: int) -> int:
        """
        Get effective cultural preference combining current preference with memory.

        Parameters
        ----------
        agent_id : int
            ID of the agent
        current_pref : int
            Current cultural preference

        Returns
        -------
        int
            Effective cultural preference weighted by memory
        """
        if agent_id not in self._cultural_memory or not self._cultural_memory[agent_id]:
            return current_pref

        memory = self._cultural_memory[agent_id]
        strengths = self._memory_strengths[agent_id]

        # Weight current preference and memories
        total_weight = 1.0 + sum(strengths)  # Current pref has weight 1.0

        # Calculate weighted average (treating preferences as continuous for averaging)
        weighted_sum = float(current_pref) + sum(pref * strength for pref, strength in zip(memory, strengths))

        effective_pref = int(round(weighted_sum / total_weight))
        return max(0, min(255, effective_pref))  # Clamp to valid range

    def get_learning_statistics(self) -> dict[str, Any]:
        """
        Get statistics about cultural learning events.

        Returns
        -------
        dict[str, Any]
            Dictionary of learning statistics
        """
        if not self.learning_events:
            return {}

        stats = {}

        # Count events by type
        type_counts = {}
        for event in self.learning_events:
            event_type = event.learning_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

        total_events = len(self.learning_events)
        stats["total_events"] = total_events
        stats["events_by_type"] = type_counts

        # Calculate proportions
        for event_type, count in type_counts.items():
            stats[f"{event_type}_proportion"] = count / total_events

        # Success metrics for prestige-based learning
        prestige_events = [e for e in self.learning_events if e.learning_type == LearningType.PRESTIGE]
        if prestige_events:
            success_values = [e.success_metric for e in prestige_events]
            stats["prestige_success_mean"] = np.mean(success_values)
            stats["prestige_success_std"] = np.std(success_values)

        # Memory statistics
        stats["agents_with_memory"] = len(self._cultural_memory)
        if self._cultural_memory:
            memory_sizes = [len(memory) for memory in self._cultural_memory.values()]
            stats["mean_memory_size"] = np.mean(memory_sizes)
            stats["max_memory_size"] = np.max(memory_sizes)

        return stats

    def get_events_dataframe(self) -> pl.DataFrame:
        """
        Get learning events as a Polars DataFrame for analysis.

        Returns
        -------
        pl.DataFrame
            DataFrame containing all learning events
        """
        if not self.learning_events:
            return pl.DataFrame()

        data = {
            "learner_id": [e.learner_id for e in self.learning_events],
            "teacher_id": [e.teacher_id for e in self.learning_events],
            "learning_type": [e.learning_type.value for e in self.learning_events],
            "old_preference": [e.old_preference for e in self.learning_events],
            "new_preference": [e.new_preference for e in self.learning_events],
            "success_metric": [e.success_metric for e in self.learning_events],
            "generation": [e.generation for e in self.learning_events],
        }

        return pl.DataFrame(data)

    def reset(self) -> None:
        """Reset all cultural learning data."""
        self.learning_events.clear()
        self._cultural_memory.clear()
        self._memory_strengths.clear()
        self.generation_count = 0

        logger.info("Cultural transmission manager reset")
