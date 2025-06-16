#!/usr/bin/env python3
"""
Clean Data Models for Experiment Results

This module defines type-safe data structures for evolutionary simulation experiments,
eliminating null pollution through proper separation of concerns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

__all__ = [
    "ExperimentMetadata",
    "CommonParameters",
    "GeneticExperimentResult",
    "CulturalExperimentResult",
    "IntegratedExperimentResult",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ExperimentMetadata:
    """Common metadata for all experiment types.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for this experiment run
    name : str
        Human-readable experiment name
    experiment_type : Literal["genetic", "cultural", "integrated"]
        Type of experiment conducted
    start_time : datetime
        When the experiment started
    duration_seconds : float
        Total execution time in seconds
    success : bool
        Whether the experiment completed successfully
    process_id : int
        OS process ID that ran this experiment

    Examples
    --------
    >>> from datetime import datetime
    >>> metadata = ExperimentMetadata(
    ...     experiment_id="exp_001",
    ...     name="genetic_test",
    ...     experiment_type="genetic",
    ...     start_time=datetime.now(),
    ...     duration_seconds=1.5,
    ...     success=True,
    ...     process_id=12345
    ... )
    >>> metadata.experiment_type
    'genetic'
    """

    experiment_id: str
    name: str
    experiment_type: Literal["genetic", "cultural", "integrated"]
    start_time: datetime
    duration_seconds: float
    success: bool
    process_id: int


@dataclass(slots=True, frozen=True)
class CommonParameters:
    """Parameters shared across different experiment types.

    These parameters have equivalent meaning across genetic and cultural evolution
    experiments, enabling direct cross-experiment comparisons.

    Parameters
    ----------
    n_generations : int
        Number of generations to simulate
    population_size : int
        Size of population (pop_size for genetic, n_agents for cultural)
    random_seed : int | None, optional
        Random seed for reproducible experiments, by default None

    Examples
    --------
    >>> params = CommonParameters(
    ...     n_generations=1000,
    ...     population_size=2000,
    ...     random_seed=42
    ... )
    >>> params.population_size
    2000
    """

    n_generations: int
    population_size: int
    random_seed: int | None = None


@dataclass(slots=True, frozen=True)
class GeneticExperimentResult:
    """Results from Layer1 (Lande-Kirkpatrick) genetic evolution experiments.

    Captures all results and parameters from genetic sexual selection simulations
    without any null fields from other experiment types.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Common experiment metadata
    common_params : CommonParameters
        Parameters shared with other experiment types
    final_trait : float
        Final mean display trait value
    final_preference : float
        Final mean female preference value
    final_covariance : float
        Final genetic covariance between trait and preference
    outcome : Literal["runaway", "stasis", "equilibrium"]
        Classified evolutionary outcome
    generations_completed : int
        Number of generations actually simulated
    h2_trait : float
        Heritability of display trait (0-1)
    h2_preference : float
        Heritability of preference (0-1)
    genetic_correlation : float
        Initial genetic correlation between trait and preference
    selection_strength : float
        Strength of natural selection against extreme traits
    preference_cost : float
        Cost of having strong preferences (0-1)
    mutation_variance : float
        Variance of mutational effects per generation

    Examples
    --------
    >>> metadata = ExperimentMetadata("exp_001", "test", "genetic",
    ...                               datetime.now(), 1.0, True, 123)
    >>> params = CommonParameters(1000, 2000, 42)
    >>> result = GeneticExperimentResult(
    ...     metadata=metadata,
    ...     common_params=params,
    ...     final_trait=0.5,
    ...     final_preference=0.3,
    ...     final_covariance=0.2,
    ...     outcome="equilibrium",
    ...     generations_completed=1000,
    ...     h2_trait=0.5,
    ...     h2_preference=0.5,
    ...     genetic_correlation=0.2,
    ...     selection_strength=0.1,
    ...     preference_cost=0.05,
    ...     mutation_variance=0.01
    ... )
    >>> result.outcome
    'equilibrium'
    """

    metadata: ExperimentMetadata
    common_params: CommonParameters

    # Core Results
    final_trait: float
    final_preference: float
    final_covariance: float
    outcome: Literal["runaway", "stasis", "equilibrium"]
    generations_completed: int

    # Genetic-Specific Parameters
    h2_trait: float
    h2_preference: float
    genetic_correlation: float
    selection_strength: float
    preference_cost: float
    mutation_variance: float


@dataclass(slots=True, frozen=True)
class CulturalExperimentResult:
    """Results from Layer2 cultural transmission experiments.

    Captures all results and parameters from cultural evolution simulations
    without any null fields from other experiment types.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Common experiment metadata
    common_params : CommonParameters
        Parameters shared with other experiment types
    final_diversity : float
        Final cultural diversity (0-1)
    diversity_trend : float
        Linear trend in diversity over time
    total_events : int
        Total number of cultural transmission events
    cultural_outcome : Literal["high_diversity", "moderate_diversity", "low_diversity"]
        Classified cultural evolution outcome
    generations_completed : int
        Number of generations actually simulated
    innovation_rate : float
        Rate of cultural innovation
    horizontal_transmission_rate : float
        Rate of horizontal cultural transmission
    oblique_transmission_rate : float
        Rate of oblique cultural transmission
    network_type : str
        Social network topology ("random", "grid", "scale_free", "small_world")
    network_connectivity : float
        Network connectivity parameter
    cultural_memory_size : int
        Size of cultural memory buffer

    Examples
    --------
    >>> metadata = ExperimentMetadata("exp_002", "test", "cultural",
    ...                               datetime.now(), 2.0, True, 124)
    >>> params = CommonParameters(1000, 2000, 42)
    >>> result = CulturalExperimentResult(
    ...     metadata=metadata,
    ...     common_params=params,
    ...     final_diversity=0.7,
    ...     diversity_trend=-0.002,
    ...     total_events=50000,
    ...     cultural_outcome="high_diversity",
    ...     generations_completed=1000,
    ...     innovation_rate=0.1,
    ...     horizontal_transmission_rate=0.3,
    ...     oblique_transmission_rate=0.2,
    ...     network_type="scale_free",
    ...     network_connectivity=0.1,
    ...     cultural_memory_size=10
    ... )
    >>> result.cultural_outcome
    'high_diversity'
    """

    metadata: ExperimentMetadata
    common_params: CommonParameters

    # Core Results
    final_diversity: float
    diversity_trend: float
    total_events: int
    cultural_outcome: Literal["high_diversity", "moderate_diversity", "low_diversity"]
    generations_completed: int

    # Cultural-Specific Parameters
    innovation_rate: float
    horizontal_transmission_rate: float
    oblique_transmission_rate: float
    network_type: str
    network_connectivity: float
    cultural_memory_size: int


@dataclass(slots=True, frozen=True)
class IntegratedExperimentResult:
    """Results from future integrated genetic+cultural experiments.

    This structure is reserved for future experiments that truly integrate
    both genetic and cultural evolution in a single simulation.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Common experiment metadata
    common_params : CommonParameters
        Parameters shared with other experiment types
    genetic_component : GeneticExperimentResult
        Genetic evolution results from the integrated simulation
    cultural_component : CulturalExperimentResult
        Cultural evolution results from the integrated simulation
    gene_culture_correlation : float
        Correlation between genetic and cultural traits
    interaction_strength : float
        Strength of gene-culture interaction effects
    emergent_properties : dict[str, float]
        Additional emergent properties from integration

    Examples
    --------
    >>> # This is a placeholder for future integrated experiments
    >>> # Implementation will depend on the integrated model design
    """

    metadata: ExperimentMetadata
    common_params: CommonParameters

    # Integration Results
    genetic_component: GeneticExperimentResult
    cultural_component: CulturalExperimentResult
    gene_culture_correlation: float
    interaction_strength: float
    emergent_properties: dict[str, float]


if __name__ == "__main__":
    # Simple validation that models can be instantiated
    from datetime import datetime

    metadata = ExperimentMetadata(
        experiment_id="test_001",
        name="validation_test",
        experiment_type="genetic",
        start_time=datetime.now(),
        duration_seconds=0.1,
        success=True,
        process_id=12345,
    )

    params = CommonParameters(n_generations=100, population_size=1000, random_seed=42)

    genetic_result = GeneticExperimentResult(
        metadata=metadata,
        common_params=params,
        final_trait=0.1,
        final_preference=0.05,
        final_covariance=0.02,
        outcome="equilibrium",
        generations_completed=100,
        h2_trait=0.5,
        h2_preference=0.5,
        genetic_correlation=0.2,
        selection_strength=0.1,
        preference_cost=0.05,
        mutation_variance=0.01,
    )

    logger.info(f"✅ Models validated: {genetic_result.outcome}")
    print("✅ Clean data models created successfully!")
    print(f"   Genetic experiment: {genetic_result.metadata.name}")
    print(f"   Population size: {genetic_result.common_params.population_size}")
    print(f"   Outcome: {genetic_result.outcome}")
