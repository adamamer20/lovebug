#!/usr/bin/env python3
"""
Model factory with validated configuration conversion.

This module provides functions to convert validated Pydantic configurations
into the objects needed for model initialization, replacing fragile dictionary
manipulation with type-safe conversions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import config models for runtime use
from experiments.config_models import (
    CombinedExperimentConfig,
    CulturalExperimentConfig,
    GeneticExperimentConfig,
)
from lovebug.layer2.config import Layer2Config
from lovebug.layer_activation import LayerActivationConfig
from lovebug.parameters import LandeKirkpatrickParams
from lovebug.unified_mesa_model import LoveModel

if TYPE_CHECKING:
    pass

__all__ = [
    "create_lande_kirkpatrick_params",
    "create_layer2_config",
    "create_layer_activation_config",
    "create_love_model_from_config",
]


def create_lande_kirkpatrick_params(
    config: GeneticExperimentConfig | CombinedExperimentConfig,
) -> LandeKirkpatrickParams:
    """
    Create validated LandeKirkpatrickParams from Pydantic config.

    Parameters
    ----------
    config : GeneticExperimentConfig | CombinedExperimentConfig
        Validated experiment configuration containing genetic parameters

    Returns
    -------
    LandeKirkpatrickParams
        Properly initialized genetic parameters object

    Raises
    ------
    ValueError
        If required genetic parameters are missing or invalid
    """
    return LandeKirkpatrickParams(
        n_generations=config.n_generations,
        pop_size=config.population_size,
        h2_trait=config.h2_trait,
        h2_preference=config.h2_preference,
        selection_strength=config.selection_strength,
        genetic_correlation=config.genetic_correlation,
        mutation_variance=config.mutation_variance,
        preference_cost=config.preference_cost,
        energy_decay=config.energy_decay,
        max_age=config.max_age,
        carrying_capacity=config.carrying_capacity,
    )


def create_layer2_config(config: CulturalExperimentConfig | CombinedExperimentConfig) -> Layer2Config:
    """
    Create validated Layer2Config from Pydantic config.

    Parameters
    ----------
    config : CulturalExperimentConfig | CombinedExperimentConfig
        Validated experiment configuration containing cultural parameters

    Returns
    -------
    Layer2Config
        Properly initialized cultural parameters object

    Raises
    ------
    ValueError
        If required cultural parameters are missing or invalid
    """
    return Layer2Config(
        innovation_rate=config.innovation_rate,
        horizontal_transmission_rate=config.horizontal_transmission_rate,
        oblique_transmission_rate=config.oblique_transmission_rate,
        network_type=config.network_type,
        network_connectivity=config.network_connectivity,
        cultural_memory_size=config.cultural_memory_size,
        local_learning_radius=config.local_learning_radius,
    )


def create_layer_activation_config(config: CombinedExperimentConfig) -> LayerActivationConfig:
    """
    Create validated LayerActivationConfig from Pydantic config.

    Parameters
    ----------
    config : CombinedExperimentConfig
        Validated combined experiment configuration

    Returns
    -------
    LayerActivationConfig
        Properly initialized layer activation configuration

    Raises
    ------
    ValueError
        If layer activation parameters are invalid
    """
    return LayerActivationConfig(
        genetic_enabled=config.genetic_enabled,
        cultural_enabled=config.cultural_enabled,
        genetic_weight=config.normalized_genetic_weight,
        cultural_weight=config.normalized_cultural_weight,
        blending_mode=config.blending_mode,
        normalize_weights=False,  # Already normalized by Pydantic model
        theta_detect=config.theta_detect,
        sigma_perception=config.sigma_perception,
    )


def create_love_model_from_genetic_config(config: GeneticExperimentConfig) -> LoveModel:
    """
    Create a LoveModel for genetic-only experiments from validated config.

    Parameters
    ----------
    config : GeneticExperimentConfig
        Validated genetic experiment configuration

    Returns
    -------
    LoveModel
        Properly initialized model for genetic evolution
    """
    # Create layer activation for genetic-only mode
    layer_config = LayerActivationConfig(
        genetic_enabled=True,
        cultural_enabled=False,
        genetic_weight=1.0,
        cultural_weight=0.0,
    )

    # Create genetic parameters
    genetic_params = create_lande_kirkpatrick_params(config)

    return LoveModel(
        layer_config=layer_config,
        genetic_params=genetic_params,
        cultural_params=None,
        n_agents=config.population_size,
    )


def create_love_model_from_cultural_config(config: CulturalExperimentConfig) -> LoveModel:
    """
    Create a LoveModel for cultural-only experiments from validated config.

    Parameters
    ----------
    config : CulturalExperimentConfig
        Validated cultural experiment configuration

    Returns
    -------
    LoveModel
        Properly initialized model for cultural evolution
    """
    # Create layer activation for cultural-only mode
    layer_config = LayerActivationConfig(
        genetic_enabled=False,
        cultural_enabled=True,
        genetic_weight=0.0,
        cultural_weight=1.0,
    )

    # Create cultural parameters
    cultural_params = create_layer2_config(config)

    return LoveModel(
        layer_config=layer_config,
        genetic_params=None,
        cultural_params=cultural_params,
        n_agents=config.population_size,
    )


def create_love_model_from_combined_config(config: CombinedExperimentConfig) -> LoveModel:
    """
    Create a LoveModel for combined experiments from validated config.

    Parameters
    ----------
    config : CombinedExperimentConfig
        Validated combined experiment configuration

    Returns
    -------
    LoveModel
        Properly initialized model for combined evolution
    """
    # Create layer activation configuration
    layer_config = create_layer_activation_config(config)

    # Create genetic parameters if genetic layer is enabled
    genetic_params = None
    if config.genetic_enabled:
        genetic_params = create_lande_kirkpatrick_params(config)

    # Create cultural parameters if cultural layer is enabled
    cultural_params = None
    if config.cultural_enabled:
        cultural_params = create_layer2_config(config)

    return LoveModel(
        layer_config=layer_config,
        genetic_params=genetic_params,
        cultural_params=cultural_params,
        n_agents=config.population_size,
    )


def create_love_model_from_config(
    config: GeneticExperimentConfig | CulturalExperimentConfig | CombinedExperimentConfig,
) -> LoveModel:
    """
    Create a LoveModel from any validated experiment configuration.

    This is the main factory function that dispatches to the appropriate
    specialized creation function based on the configuration type.

    Parameters
    ----------
    config : GeneticExperimentConfig | CulturalExperimentConfig | CombinedExperimentConfig
        Validated experiment configuration of any type

    Returns
    -------
    LoveModel
        Properly initialized model for the specified experiment type

    Raises
    ------
    ValueError
        If configuration type is not recognized or parameters are invalid
    """
    if isinstance(config, GeneticExperimentConfig):
        return create_love_model_from_genetic_config(config)
    elif isinstance(config, CulturalExperimentConfig):
        return create_love_model_from_cultural_config(config)
    elif isinstance(config, CombinedExperimentConfig):
        return create_love_model_from_combined_config(config)
    else:
        raise ValueError(f"Unknown experiment configuration type: {type(config)}")


# Convenience functions for common validation scenarios


def validate_and_create_genetic_model(
    name: str,
    population_size: int,
    n_generations: int,
    carrying_capacity: int | None = None,
    **genetic_params,
) -> tuple[GeneticExperimentConfig, LoveModel]:
    """
    Validate parameters and create a genetic model in one step.

    Parameters
    ----------
    name : str
        Experiment name
    population_size : int
        Initial population size
    n_generations : int
        Number of generations
    carrying_capacity : int | None
        Maximum population size (defaults to 2x population_size)
    **genetic_params
        Additional genetic parameters (h2_trait, h2_preference, etc.)

    Returns
    -------
    tuple[GeneticExperimentConfig, LoveModel]
        Validated configuration and initialized model

    Raises
    ------
    ValidationError
        If any parameters are invalid
    """
    from experiments.config_models import GeneticExperimentConfig

    if carrying_capacity is None:
        carrying_capacity = population_size * 2

    config = GeneticExperimentConfig(
        name=name,
        population_size=population_size,
        n_generations=n_generations,
        carrying_capacity=carrying_capacity,
        **genetic_params,
    )

    model = create_love_model_from_genetic_config(config)
    return config, model


def validate_experiment_parameters(
    experiment_type: str,
    parameters: dict,
) -> GeneticExperimentConfig | CulturalExperimentConfig | CombinedExperimentConfig:
    """
    Validate raw experiment parameters using appropriate Pydantic model.

    This function serves as a bridge between the old dictionary-based
    parameter passing and the new validated configuration system.

    Parameters
    ----------
    experiment_type : str
        Type of experiment ("genetic", "cultural", or "combined")
    parameters : dict
        Raw experiment parameters dictionary

    Returns
    -------
    GeneticExperimentConfig | CulturalExperimentConfig | CombinedExperimentConfig
        Validated configuration object

    Raises
    ------
    ValidationError
        If parameters are invalid or missing required fields
    ValueError
        If experiment_type is not recognized
    """
    from experiments.config_models import (
        CombinedExperimentConfig,
        CulturalExperimentConfig,
        GeneticExperimentConfig,
    )

    if experiment_type == "genetic":
        return GeneticExperimentConfig(**parameters)
    elif experiment_type == "cultural":
        return CulturalExperimentConfig(**parameters)
    elif experiment_type == "combined":
        return CombinedExperimentConfig(**parameters)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
