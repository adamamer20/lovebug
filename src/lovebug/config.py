"""Unified configuration models for LoveBug simulation.

This module defines all Pydantic models for simulation configuration,
consolidating genetic, cultural, layer, and simulation parameters into
a single source of truth.
"""

from typing import Literal

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, model_validator

__all__ = [
    "GeneticParams",
    "CulturalParams",
    "SimulationParams",
    "LayerConfig",
    "LoveBugConfig",
]


class GeneticParams(BaseModel):
    """Genetic algorithm parameters."""

    h2_trait: float = Field(0.5, ge=0.0, le=1.0, description="Heritability of display trait (0-1).")
    h2_preference: float = Field(0.5, ge=0.0, le=1.0, description="Heritability of preference trait (0-1).")
    mutation_rate: float = Field(0.01, ge=0.0, le=1.0, description="Mutation rate (0-1).")
    crossover_rate: float = Field(0.7, ge=0.0, le=1.0, description="Crossover rate (0-1).")
    elitism: NonNegativeInt = Field(1, description="Number of elite individuals to preserve.")
    energy_decay: float = Field(0.01, ge=0.0, description="Per-generation decay rate of energy (>= 0).")
    mutation_variance: float = Field(0.01, ge=0.0, description="Variance of mutation effect (>= 0).")
    max_age: PositiveInt = Field(100, description="Maximum age for individuals.")
    carrying_capacity: PositiveInt = Field(1000, description="Maximum population size (carrying capacity).")
    energy_replenishment_rate: float = Field(
        0.01, ge=0.0, description="Per-generation energy replenishment rate (>= 0)."
    )
    parental_investment_rate: float = Field(
        0.6, ge=0.0, le=1.0, description="Fraction of a parent's energy invested in each offspring (0-1)."
    )
    energy_min_mating: float = Field(1.0, ge=0.0, description="Minimum energy required for mating (>= 0).")
    juvenile_cost: float = Field(0.5, ge=0.0, description="Energy cost for juveniles (>= 0).")
    display_cost_scalar: float = Field(
        0.2, ge=0.0, description="Cost scalar for display traits on foraging efficiency (>= 0)."
    )
    search_cost: float = Field(0.01, ge=0.0, description="Energy cost for courtship assessment (>= 0).")
    base_energy: float = Field(10.0, ge=0.0, description="Base energy level for new agents (>= 0).")


class CulturalParams(BaseModel):
    """Cultural learning parameters."""

    innovation_rate: float = Field(0.01, ge=0.0, le=1.0, description="Innovation rate (0-1).")
    memory_span: PositiveInt = Field(5, description="Number of generations to remember.")
    network_type: str = Field("small_world", description="Network topology for cultural transmission.")
    network_connectivity: float = Field(1.0, ge=0.0, le=1.0, description="Network connectivity (0-1).")
    cultural_memory_size: PositiveInt = Field(5, description="Number of memory slots for cultural memory.")
    memory_decay_rate: float = Field(0.01, ge=0.0, le=1.0, description="Decay rate for cultural memory (0-1).")
    horizontal_transmission_rate: float = Field(
        0.1, ge=0.0, le=1.0, description="Horizontal cultural transmission rate (0-1)."
    )
    oblique_transmission_rate: float = Field(
        0.1, ge=0.0, le=1.0, description="Oblique cultural transmission rate (0-1)."
    )
    local_learning_radius: PositiveInt = Field(5, description="Radius for local cultural learning.")
    memory_update_strength: float = Field(1.0, ge=0.0, le=1.0, description="Strength of memory updates (0-1).")
    learning_strategy: Literal["conformist", "success-biased", "condition-dependent", "age-biased"] = Field(
        "conformist", description="Social learning strategy for the entire population."
    )


class SimulationParams(BaseModel):
    """Single simulation run parameters."""

    population_size: PositiveInt
    steps: PositiveInt = Field(1000, description="Number of steps per simulation run.")
    seed: NonNegativeInt = Field(42, description="Random seed for reproducibility.")


class LayerConfig(BaseModel):
    """Layer activation and blending configuration for unified model."""

    genetic_enabled: bool = Field(True, description="Enable genetic evolution layer.")
    cultural_enabled: bool = Field(False, description="Enable cultural evolution layer.")
    blending_mode: Literal["additive", "multiplicative", "weighted"] = Field(
        "weighted", description="Method for blending genetic and cultural contributions."
    )
    genetic_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for genetic contribution (0-1).")
    cultural_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for cultural contribution (0-1).")
    sigma_perception: float = Field(0.0, ge=0.0, description="Perceptual noise level (>= 0).")
    theta_detect: float = Field(0.0, ge=0.0, description="Detection threshold (>= 0).")

    @model_validator(mode="after")
    def validate_weights(self) -> "LayerConfig":
        if self.blending_mode == "weighted":
            total_weight = self.genetic_weight + self.cultural_weight
            if abs(total_weight - 1.0) > 1e-6:
                raise ValueError(
                    f"Genetic and cultural weights must sum to 1.0 for weighted blending, got {total_weight}"
                )
        return self

    def is_combined(self) -> bool:
        return self.genetic_enabled and self.cultural_enabled

    def is_genetic_only(self) -> bool:
        return self.genetic_enabled and not self.cultural_enabled

    def is_cultural_only(self) -> bool:
        return self.cultural_enabled and not self.genetic_enabled


class LoveBugConfig(BaseModel):
    """Unified configuration for LoveBug simulation."""

    name: str = Field(..., description="Unique name for this configuration (used in logs and error messages).")
    genetic: GeneticParams = Field(default_factory=lambda: GeneticParams())
    cultural: CulturalParams = Field(default_factory=lambda: CulturalParams())
    simulation: SimulationParams = Field(..., description="Simulation parameters including population size.")
    layer: LayerConfig = Field(default_factory=lambda: LayerConfig())

    @property
    def h2_trait(self) -> float:
        """Heritability of display trait (0-1)."""
        return self.genetic.h2_trait

    @property
    def h2_preference(self) -> float:
        """Heritability of preference trait (0-1)."""
        return self.genetic.h2_preference

    @property
    def population_size(self) -> int:
        """Population size for the simulation."""
        return self.simulation.population_size

    @property
    def random_seed(self) -> int:
        """Random seed for the simulation."""
        return self.simulation.seed

    @property
    def steps(self) -> int:
        """Number of steps for the simulation."""
        return self.simulation.steps
