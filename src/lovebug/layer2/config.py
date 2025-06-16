"""
Configuration system for Layer 2 research extension.

Provides structured configuration management for social learning mechanisms,
experimental parameters, and monitoring options.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from beartype import beartype

__all__ = ["Layer2Config"]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=False)
class Layer2Config:
    """
    Configuration for Layer 2 social learning mechanisms.

    Parameters
    ----------
    oblique_transmission_rate : float
        Rate of parent-offspring cultural transmission (0-1)
    horizontal_transmission_rate : float
        Rate of peer-to-peer cultural learning (0-1)
    innovation_rate : float
        Rate of cultural innovation/mutation (0-1)
    network_type : str
        Type of social network ('random', 'small_world', 'scale_free')
    network_connectivity : float
        Network connectivity parameter (0-1)
    local_learning_radius : int
        Radius for local learning interactions
    cultural_memory_size : int
        Number of cultural experiences to remember
    memory_decay_rate : float
        Rate of cultural memory decay per generation
    memory_update_strength : float
        Strength of new cultural experiences
    prestige_mechanisms : list[str]
        List of prestige indicators ('mating_success', 'survival_time')
    prestige_decay_rate : float
        Rate of prestige decay over time
    log_cultural_events : bool
        Whether to log individual cultural learning events
    log_every_n_generations : int
        Frequency of detailed logging
    save_detailed_data : bool
        Whether to save detailed cultural data

    Examples
    --------
    >>> config = Layer2Config(
    ...     oblique_transmission_rate=0.4,
    ...     innovation_rate=0.08,
    ...     log_cultural_events=True
    ... )
    >>> config.save_to_file("experiment_config.yaml")
    """

    # Cultural transmission parameters
    oblique_transmission_rate: float = 0.3
    horizontal_transmission_rate: float = 0.2
    innovation_rate: float = 0.05

    # Social network parameters
    network_type: str = "small_world"
    network_connectivity: float = 0.1
    local_learning_radius: int = 5

    # Memory parameters
    cultural_memory_size: int = 10
    memory_decay_rate: float = 0.05
    memory_update_strength: float = 1.0

    # Prestige learning parameters
    prestige_mechanisms: list[str] = field(default_factory=lambda: ["mating_success", "survival_time"])
    prestige_decay_rate: float = 0.1

    # Monitoring parameters
    log_cultural_events: bool = True
    log_every_n_generations: int = 10
    save_detailed_data: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate that all parameters are within acceptable ranges."""
        # Rate parameters should be in [0, 1]
        rate_params = [
            ("oblique_transmission_rate", self.oblique_transmission_rate),
            ("horizontal_transmission_rate", self.horizontal_transmission_rate),
            ("innovation_rate", self.innovation_rate),
            ("network_connectivity", self.network_connectivity),
            ("memory_decay_rate", self.memory_decay_rate),
            ("prestige_decay_rate", self.prestige_decay_rate),
        ]

        for param_name, value in rate_params:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{param_name} must be between 0 and 1, got {value}")

        # Integer parameters should be positive
        int_params = [
            ("local_learning_radius", self.local_learning_radius),
            ("cultural_memory_size", self.cultural_memory_size),
            ("log_every_n_generations", self.log_every_n_generations),
        ]

        for param_name, value in int_params:
            if value <= 0:
                raise ValueError(f"{param_name} must be positive, got {value}")

        # Network type validation
        valid_networks = {"random", "small_world", "scale_free", "grid"}
        if self.network_type not in valid_networks:
            raise ValueError(f"network_type must be one of {valid_networks}, got {self.network_type}")

        # Prestige mechanisms validation
        valid_mechanisms = {"mating_success", "survival_time", "energy_level"}
        invalid_mechanisms = set(self.prestige_mechanisms) - valid_mechanisms
        if invalid_mechanisms:
            raise ValueError(f"Invalid prestige mechanisms: {invalid_mechanisms}")

    @beartype
    def save_to_file(self, filepath: str | Path) -> None:
        """
        Save configuration to YAML file.

        Parameters
        ----------
        filepath : str | Path
            Path to save configuration file

        Raises
        ------
        OSError
            If file cannot be written
        """
        filepath = Path(filepath)
        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and save
            config_dict = {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

            with filepath.open("w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)

            logger.info(f"Configuration saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save configuration to {filepath}: {e}")
            raise

    @classmethod
    @beartype
    def load_from_file(cls, filepath: str | Path) -> Layer2Config:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        filepath : str | Path
            Path to configuration file

        Returns
        -------
        Layer2Config
            Loaded configuration instance

        Raises
        ------
        FileNotFoundError
            If configuration file doesn't exist
        ValueError
            If configuration is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            with filepath.open("r") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise ValueError("Configuration file must contain a dictionary")

            logger.info(f"Configuration loaded from {filepath}")
            return cls(**data)

        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            raise

    @beartype
    def update(self, **kwargs: Any) -> Layer2Config:
        """
        Create a new configuration with updated parameters.

        Parameters
        ----------
        **kwargs
            Parameters to update

        Returns
        -------
        Layer2Config
            New configuration instance with updated parameters

        Examples
        --------
        >>> config = Layer2Config()
        >>> new_config = config.update(innovation_rate=0.1, log_cultural_events=False)
        """
        # Get current values
        current_values = {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

        # Update with new values
        current_values.update(kwargs)

        return Layer2Config(**current_values)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

    def __repr__(self) -> str:
        """Rich representation of configuration."""
        params = []
        for field_obj in self.__dataclass_fields__.values():
            value = getattr(self, field_obj.name)
            if isinstance(value, (int, float)):
                params.append(f"{field_obj.name}={value}")
            elif isinstance(value, str):
                params.append(f"{field_obj.name}='{value}'")
            else:
                params.append(f"{field_obj.name}={value}")

        return f"Layer2Config({', '.join(params)})"
