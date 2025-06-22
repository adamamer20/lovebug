#!/usr/bin/env python3
"""
Pydantic models for validated experiment configuration.

This module replaces fragile dictionary-based parameter passing with
type-safe Pydantic models that provide automatic validation, serialization,
and clear interfaces for experiment configuration.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

__all__ = [
    "BaseExperimentConfig",
    "GeneticExperimentConfig",
    "CulturalExperimentConfig",
    "CombinedExperimentConfig",
    "ExperimentTask",
]


class BaseExperimentConfig(BaseModel):
    """Base configuration for all experiment types with common parameters."""

    model_config = {"extra": "forbid", "validate_assignment": True}

    # Experiment identification
    name: str = Field(..., description="Human-readable experiment name")
    experiment_type: Literal["genetic", "cultural", "combined"] = Field(..., description="Type of experiment")

    # Common simulation parameters
    n_generations: int = Field(default=1000, ge=1, le=10000, description="Number of simulation generations")
    population_size: int = Field(default=1000, ge=10, le=100000, description="Initial population size")
    random_seed: int | None = Field(default=None, description="Random seed for reproducibility")

    # Population regulation parameters (critical for preventing explosions)
    carrying_capacity: int = Field(default=2000, ge=50, description="Maximum population before culling")
    energy_decay: float = Field(default=0.2, ge=0.0, le=1.0, description="Energy decay rate per timestep")
    max_age: int = Field(default=100, ge=1, le=1000, description="Maximum agent age before death")

    @field_validator("carrying_capacity")
    @classmethod
    def validate_carrying_capacity(cls, v: int, info) -> int:
        """Ensure carrying capacity is reasonable relative to population size."""
        if hasattr(info, "data") and "population_size" in info.data:
            pop_size = info.data["population_size"]
            if v < pop_size * 0.5:
                raise ValueError(f"carrying_capacity ({v}) should be at least 50% of population_size ({pop_size})")
        return v


class GeneticExperimentConfig(BaseExperimentConfig):
    """Validated configuration for genetic evolution experiments (Lande-Kirkpatrick model)."""

    experiment_type: Literal["genetic"] = Field(default="genetic", frozen=True, exclude=True)

    # Lande-Kirkpatrick parameters with validation
    h2_trait: float = Field(default=0.5, ge=0.0, le=1.0, description="Heritability of male display trait")
    h2_preference: float = Field(default=0.5, ge=0.0, le=1.0, description="Heritability of female preference")
    selection_strength: float = Field(default=0.1, ge=0.0, le=1.0, description="Strength of natural selection")
    genetic_correlation: float = Field(
        default=0.2, ge=-1.0, le=1.0, description="Genetic correlation between trait and preference"
    )
    mutation_variance: float = Field(default=0.01, gt=0.0, le=0.1, description="Variance of mutational effects")
    preference_cost: float = Field(default=0.05, ge=0.0, le=1.0, description="Cost of having strong preferences")

    @computed_field
    @property
    def scenario_type(self) -> str:
        """Automatically classify the experiment scenario based on parameters."""
        if abs(self.genetic_correlation) < 0.01 and self.preference_cost == 0.0:
            return "stasis"
        elif self.genetic_correlation > 0.2 and self.preference_cost == 0.0:
            return "runaway"
        elif self.genetic_correlation > 0.2 and self.preference_cost > 0.1:
            return "costly_choice"
        else:
            return "equilibrium"


class CulturalExperimentConfig(BaseExperimentConfig):
    """Validated configuration for cultural evolution experiments."""

    experiment_type: Literal["cultural"] = Field(default="cultural", frozen=True, exclude=True)

    # Cultural transmission parameters
    innovation_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Rate of cultural innovation")
    horizontal_transmission_rate: float = Field(default=0.3, ge=0.0, le=1.0, description="Horizontal transmission rate")
    oblique_transmission_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Oblique transmission rate")

    # Network topology parameters
    network_type: Literal["scale_free", "small_world", "random", "lattice"] = Field(
        default="scale_free", description="Social network topology type"
    )
    network_connectivity: float = Field(default=0.1, ge=0.01, le=1.0, description="Network connectivity parameter")

    # Cultural memory and learning
    cultural_memory_size: int = Field(default=10, ge=1, le=100, description="Size of cultural memory buffer")
    local_learning_radius: int = Field(default=5, ge=1, le=20, description="Radius for local learning interactions")

    @field_validator("horizontal_transmission_rate", "oblique_transmission_rate", "innovation_rate")
    @classmethod
    def validate_transmission_rates(cls, v: float) -> float:
        """Ensure transmission rates are reasonable."""
        if v > 0.8:
            raise ValueError(f"Transmission rate {v} is unrealistically high (>0.8)")
        return v


class CombinedExperimentConfig(BaseExperimentConfig):
    """Validated configuration for combined genetic+cultural evolution experiments."""

    experiment_type: Literal["combined"] = Field(default="combined", frozen=True, exclude=True)

    # Layer activation parameters
    genetic_enabled: bool = Field(default=True, description="Enable genetic evolution layer")
    cultural_enabled: bool = Field(default=True, description="Enable cultural evolution layer")
    genetic_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for genetic layer influence")
    cultural_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for cultural layer influence")
    blending_mode: Literal["weighted_average", "probabilistic", "competitive"] = Field(
        default="weighted_average", description="Method for blending genetic and cultural influences"
    )
    normalize_weights: bool = Field(default=True, description="Automatically normalize layer weights to sum to 1.0")

    # Perceptual constraint parameters
    theta_detect: float = Field(
        default=8.0, ge=1.0, le=16.0, description="Detection threshold for perceptual constraints"
    )
    sigma_perception: float = Field(default=2.0, ge=0.1, le=5.0, description="Perceptual noise standard deviation")

    # Genetic parameters (embedded)
    h2_trait: float = Field(default=0.5, ge=0.0, le=1.0, description="Heritability of male display trait")
    h2_preference: float = Field(default=0.5, ge=0.0, le=1.0, description="Heritability of female preference")
    selection_strength: float = Field(default=0.1, ge=0.0, le=1.0, description="Strength of natural selection")
    genetic_correlation: float = Field(
        default=0.2, ge=-1.0, le=1.0, description="Genetic correlation between trait and preference"
    )
    mutation_variance: float = Field(default=0.01, gt=0.0, le=0.1, description="Variance of mutational effects")
    preference_cost: float = Field(default=0.05, ge=0.0, le=1.0, description="Cost of having strong preferences")

    # Cultural parameters (embedded)
    innovation_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Rate of cultural innovation")
    horizontal_transmission_rate: float = Field(default=0.3, ge=0.0, le=1.0, description="Horizontal transmission rate")
    oblique_transmission_rate: float = Field(default=0.2, ge=0.0, le=1.0, description="Oblique transmission rate")
    network_type: Literal["scale_free", "small_world", "random", "lattice"] = Field(
        default="scale_free", description="Social network topology type"
    )
    network_connectivity: float = Field(default=0.1, ge=0.01, le=1.0, description="Network connectivity parameter")
    cultural_memory_size: int = Field(default=10, ge=1, le=100, description="Size of cultural memory buffer")
    local_learning_radius: int = Field(default=5, ge=1, le=20, description="Radius for local learning interactions")

    @model_validator(mode="after")
    def validate_layer_configuration(self) -> CombinedExperimentConfig:
        """Validate that at least one layer is enabled and weights are reasonable."""
        if not self.genetic_enabled and not self.cultural_enabled:
            raise ValueError("At least one layer (genetic or cultural) must be enabled")

        if self.normalize_weights:
            total_weight = self.genetic_weight + self.cultural_weight
            if total_weight <= 0:
                raise ValueError("Total layer weights must be positive when normalize_weights=True")

        return self

    @computed_field
    @property
    def normalized_genetic_weight(self) -> float:
        """Get normalized genetic weight."""
        if not self.normalize_weights:
            return self.genetic_weight
        total = self.genetic_weight + self.cultural_weight
        return self.genetic_weight / total if total > 0 else 0.0

    @computed_field
    @property
    def normalized_cultural_weight(self) -> float:
        """Get normalized cultural weight."""
        if not self.normalize_weights:
            return self.cultural_weight
        total = self.genetic_weight + self.cultural_weight
        return self.cultural_weight / total if total > 0 else 0.0


class ExperimentTask(BaseModel):
    """Type-safe wrapper for experiment tasks with validated configuration."""

    model_config = {"extra": "forbid"}

    config: GeneticExperimentConfig | CulturalExperimentConfig | CombinedExperimentConfig = Field(
        ..., description="Validated experiment configuration"
    )
    replication_id: int = Field(default=0, ge=0, description="Replication number for this configuration")

    @computed_field
    @property
    def experiment_type(self) -> str:
        """Get experiment type from config."""
        return self.config.experiment_type

    @computed_field
    @property
    def full_name(self) -> str:
        """Get full experiment name including replication."""
        return f"{self.config.name}_rep_{self.replication_id}"

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to legacy dictionary format for backward compatibility."""
        data = self.config.model_dump()
        return {
            "type": self.experiment_type,
            "config": {"name": self.full_name},
            "params": data,
            "replication": self.replication_id,
        }


# Factory functions for common experiment configurations


def create_lk_stasis_config(
    name: str = "lk_stasis",
    population_size: int = 1000,
    n_generations: int = 1000,
    carrying_capacity: int | None = None,
) -> GeneticExperimentConfig:
    """Create a Lande-Kirkpatrick stasis scenario configuration."""
    if carrying_capacity is None:
        carrying_capacity = population_size * 2

    return GeneticExperimentConfig(
        name=name,
        population_size=population_size,
        n_generations=n_generations,
        carrying_capacity=carrying_capacity,
        h2_trait=0.3,
        h2_preference=0.2,
        genetic_correlation=0.0,
        selection_strength=0.3,
        preference_cost=0.0,
        mutation_variance=0.01,
    )


def create_lk_runaway_config(
    name: str = "lk_runaway",
    population_size: int = 1000,
    n_generations: int = 1000,
    carrying_capacity: int | None = None,
) -> GeneticExperimentConfig:
    """Create a Lande-Kirkpatrick runaway scenario configuration."""
    if carrying_capacity is None:
        carrying_capacity = population_size * 2

    return GeneticExperimentConfig(
        name=name,
        population_size=population_size,
        n_generations=n_generations,
        carrying_capacity=carrying_capacity,
        h2_trait=0.6,
        h2_preference=0.7,
        genetic_correlation=0.3,
        selection_strength=0.05,
        preference_cost=0.0,
        mutation_variance=0.01,
    )


def create_lk_costly_choice_config(
    name: str = "lk_costly_choice",
    population_size: int = 1000,
    n_generations: int = 1000,
    carrying_capacity: int | None = None,
) -> GeneticExperimentConfig:
    """Create a Lande-Kirkpatrick costly choice scenario configuration."""
    if carrying_capacity is None:
        carrying_capacity = population_size * 2

    return GeneticExperimentConfig(
        name=name,
        population_size=population_size,
        n_generations=n_generations,
        carrying_capacity=carrying_capacity,
        h2_trait=0.6,
        h2_preference=0.7,
        genetic_correlation=0.3,
        selection_strength=0.05,
        preference_cost=0.15,
        mutation_variance=0.01,
    )
