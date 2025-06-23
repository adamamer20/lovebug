"""Unified configuration models for LoveBug simulation.

This module defines all Pydantic models for simulation configuration,
consolidating genetic, cultural, blending, perceptual, and simulation
parameters into a single source of truth.
"""

from typing import Literal

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

__all__ = [
    "GeneticParams",
    "CulturalParams",
    "LayerBlendingParams",
    "PerceptualParams",
    "SimulationParams",
    "LayerConfig",
    "LoveBugConfig",
]


class GeneticParams(BaseModel):
    """Genetic algorithm parameters."""

    h2_trait: float = Field(0.5, ge=0.0, le=1.0, description="Heritability of display trait (0-1).")
    h2_preference: float = Field(0.5, ge=0.0, le=1.0, description="Heritability of preference trait (0-1).")
    mutation_rate: float = Field(0.01, ge=0.0, le=1.0)
    crossover_rate: float = Field(0.7, ge=0.0, le=1.0)
    population_size: PositiveInt = 100
    elitism: NonNegativeInt = 1
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

    @model_validator(mode="after")
    def check_rates(self) -> "GeneticParams":
        if not (0.0 <= self.h2_trait <= 1.0):
            raise ValueError("h2_trait must be in [0, 1]")
        if not (0.0 <= self.h2_preference <= 1.0):
            raise ValueError("h2_preference must be in [0, 1]")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1]")
        if self.energy_decay < 0.0:
            raise ValueError("energy_decay must be >= 0")
        if self.mutation_variance < 0.0:
            raise ValueError("mutation_variance must be >= 0")

        # Population Viability Check
        # Ensure that the energy replenishment can support the population.
        # We assume an average foraging_multiplier of 1.0 for this check.
        avg_energy_gain = (self.carrying_capacity * self.energy_replenishment_rate) / self.population_size
        net_energy_balance = avg_energy_gain - self.energy_decay

        # The net balance should be positive to be sustainable.
        # A small positive margin is recommended to account for stochasticity.
        viability_margin = 0.001
        if net_energy_balance < viability_margin:
            raise ValueError(
                f"Population not viable. Net energy balance is negative ({net_energy_balance:.4f}). "
                f"Energy Gain ({avg_energy_gain:.4f}) < Energy Decay ({self.energy_decay:.4f}). "
                "Consider increasing 'energy_replenishment_rate' or 'carrying_capacity', "
                "or decreasing 'energy_decay' or 'population_size'."
            )

        return self


class CulturalParams(BaseModel):
    """Cultural learning parameters."""

    learning_rate: float = Field(0.05, ge=0.0, le=1.0)
    innovation_rate: float = Field(0.01, ge=0.0, le=1.0)
    memory_span: PositiveInt = 5
    network_type: str = Field("small_world", description="Network topology for cultural transmission.")
    network_connectivity: float = Field(1.0, ge=0.0, le=1.0, description="Network connectivity [0, 1].")
    cultural_memory_size: PositiveInt = Field(5, description="Number of memory slots for cultural memory.")
    memory_decay_rate: float = Field(0.01, ge=0.0, le=1.0, description="Decay rate for cultural memory [0, 1].")
    horizontal_transmission_rate: float = Field(
        0.1, ge=0.0, le=1.0, description="Horizontal cultural transmission rate [0, 1]."
    )
    oblique_transmission_rate: float = Field(
        0.1, ge=0.0, le=1.0, description="Oblique cultural transmission rate [0, 1]."
    )
    local_learning_radius: PositiveInt = Field(5, description="Radius for local cultural learning.")
    memory_update_strength: float = Field(1.0, ge=0.0, le=1.0, description="Strength of memory updates [0, 1].")
    learning_strategy: Literal["conformist", "success-biased", "condition-dependent", "age-biased"] = Field(
        "conformist", description="Social learning strategy for the entire population."
    )

    @model_validator(mode="after")
    def check_rates(self) -> "CulturalParams":
        if not (0.0 <= self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in [0, 1]")
        if not (0.0 <= self.innovation_rate <= 1.0):
            raise ValueError("innovation_rate must be in [0, 1]")
        return self


class LayerBlendingParams(BaseModel):
    """Parameters for blending genetic and cultural layers."""

    blend_mode: Literal["additive", "multiplicative", "weighted"] = "weighted"
    blend_weight: float = Field(0.5, ge=0.0, le=1.0)

    def get_effective_genetic_weight(self) -> float:
        if self.blend_mode == "weighted":
            return self.blend_weight
        return 0.5

    def get_effective_cultural_weight(self) -> float:
        if self.blend_mode == "weighted":
            return 1.0 - self.blend_weight
        return 0.5

    @model_validator(mode="after")
    def check_weight(self) -> "LayerBlendingParams":
        if self.blend_mode == "weighted" and not (0.0 <= self.blend_weight <= 1.0):
            raise ValueError("blend_weight must be in [0, 1] for 'weighted' mode")
        return self


class PerceptualParams(BaseModel):
    """Perceptual and environmental parameters."""

    noise_level: NonNegativeFloat = 0.0
    perception_radius: PositiveFloat = 1.0


class SimulationParams(BaseModel):
    """Global simulation parameters."""

    population_size: PositiveInt
    steps: PositiveInt = 1000
    seed: NonNegativeInt = 42
    n_generations: PositiveInt = 100
    replications: PositiveInt = 1


class LayerConfig(BaseModel):
    """Layer activation and blending configuration for unified model."""

    genetic_enabled: bool = True
    cultural_enabled: bool = False
    blending_mode: str = "weighted_average"
    genetic_weight: float = 0.5
    cultural_weight: float = 0.5
    sigma_perception: float = 0.0
    theta_detect: float = 0.0

    def is_combined(self) -> bool:
        return self.genetic_enabled and self.cultural_enabled

    def is_genetic_only(self) -> bool:
        return self.genetic_enabled and not self.cultural_enabled

    def is_cultural_only(self) -> bool:
        return self.cultural_enabled and not self.genetic_enabled


class LoveBugConfig(BaseModel):
    """Unified configuration for LoveBug simulation."""

    name: str = Field(..., description="Unique name for this configuration (used in logs and error messages).")
    genetic: GeneticParams = Field(default_factory=GeneticParams)
    cultural: CulturalParams = Field(default_factory=CulturalParams)
    blending: LayerBlendingParams = Field(default_factory=LayerBlendingParams)
    perceptual: PerceptualParams = Field(default_factory=PerceptualParams)
    simulation: SimulationParams = Field(default_factory=SimulationParams)
    layer: LayerConfig = Field(default_factory=LayerConfig)

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

    @property
    def n_generations(self) -> int:
        """Number of generations (steps) to run in each experiment."""
        return self.simulation.n_generations

    @property
    def replications(self) -> int:
        """Number of replications per experiment condition."""
        return self.simulation.replications
