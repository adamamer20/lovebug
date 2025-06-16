"""
Layer activation configuration for selective evolutionary layer control.

This module provides the configuration system for enabling/disabling and weighting
genetic and cultural evolutionary layers in unified simulations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from beartype import beartype

__all__ = ["LayerActivationConfig"]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=False)
class LayerActivationConfig:
    """
    Configuration for selective layer activation in evolutionary simulations.

    This class controls which evolutionary layers (genetic, cultural) are active
    during simulation runs and how their effects are blended together.

    Parameters
    ----------
    genetic_enabled : bool, default=True
        Whether genetic evolution layer is active
    cultural_enabled : bool, default=True
        Whether cultural evolution layer is active
    genetic_weight : float, default=0.5
        Relative weight of genetic layer influence (0.0-1.0)
    cultural_weight : float, default=0.5
        Relative weight of cultural layer influence (0.0-1.0)
    blending_mode : str, default="weighted_average"
        Strategy for blending genetic and cultural effects
        Options: "weighted_average", "probabilistic", "competitive"
    normalize_weights : bool, default=True
        Whether to auto-normalize weights to sum to 1.0

    Raises
    ------
    ValueError
        If weights are outside [0.0, 1.0] range or no layers are enabled

    Examples
    --------
    >>> # Pure genetic evolution
    >>> config = LayerActivationConfig(
    ...     genetic_enabled=True,
    ...     cultural_enabled=False
    ... )
    >>> config.genetic_weight
    1.0

    >>> # Balanced combined evolution
    >>> config = LayerActivationConfig(
    ...     genetic_enabled=True,
    ...     cultural_enabled=True,
    ...     genetic_weight=0.6,
    ...     cultural_weight=0.4
    ... )
    >>> config.genetic_weight
    0.6

    >>> # Auto-normalized weights
    >>> config = LayerActivationConfig(
    ...     genetic_weight=0.8,
    ...     cultural_weight=0.6,
    ...     normalize_weights=True
    ... )
    >>> abs(config.genetic_weight + config.cultural_weight - 1.0) < 1e-10
    True
    """

    # Layer activation flags
    genetic_enabled: bool = True
    cultural_enabled: bool = True

    # Layer influence weights (0.0 = no influence, 1.0 = full influence)
    genetic_weight: float = 0.5
    cultural_weight: float = 0.5

    # Blending strategy
    blending_mode: Literal["weighted_average", "probabilistic", "competitive"] = "weighted_average"

    # Validation parameters
    normalize_weights: bool = True

    def __post_init__(self) -> None:
        """Validate configuration and normalize weights if requested."""
        self._validate_parameters()
        if self.normalize_weights:
            self._normalize_weights()

        logger.debug(
            f"LayerActivationConfig initialized: genetic={self.genetic_enabled}, "
            f"cultural={self.cultural_enabled}, weights=({self.genetic_weight:.3f}, "
            f"{self.cultural_weight:.3f}), mode={self.blending_mode}"
        )

    def _validate_parameters(self) -> None:
        """
        Validate parameter ranges and logical consistency.

        Raises
        ------
        ValueError
            If parameters are invalid or inconsistent
        """
        # Validate weight ranges
        if not (0.0 <= self.genetic_weight <= 1.0):
            raise ValueError(f"genetic_weight must be in [0.0, 1.0], got {self.genetic_weight}")
        if not (0.0 <= self.cultural_weight <= 1.0):
            raise ValueError(f"cultural_weight must be in [0.0, 1.0], got {self.cultural_weight}")

        # Check logical consistency - disable weights if layers are disabled
        if not self.genetic_enabled and self.genetic_weight > 0:
            logger.warning("Setting genetic_weight=0.0 because genetic_enabled=False")
            self.genetic_weight = 0.0
        if not self.cultural_enabled and self.cultural_weight > 0:
            logger.warning("Setting cultural_weight=0.0 because cultural_enabled=False")
            self.cultural_weight = 0.0

        # Ensure at least one layer is enabled
        if not self.genetic_enabled and not self.cultural_enabled:
            raise ValueError("At least one layer must be enabled")

        # Validate blending mode
        valid_modes = {"weighted_average", "probabilistic", "competitive"}
        if self.blending_mode not in valid_modes:
            raise ValueError(f"blending_mode must be one of {valid_modes}, got {self.blending_mode}")

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0 if both are positive."""
        total_weight = self.genetic_weight + self.cultural_weight
        if total_weight > 0:
            self.genetic_weight /= total_weight
            self.cultural_weight /= total_weight
            logger.debug(
                f"Normalized weights to sum=1.0: genetic={self.genetic_weight:.3f}, cultural={self.cultural_weight:.3f}"
            )

    @beartype
    def is_genetic_only(self) -> bool:
        """
        Check if configuration represents genetic-only evolution.

        Returns
        -------
        bool
            True if only genetic layer is enabled
        """
        return self.genetic_enabled and not self.cultural_enabled

    @beartype
    def is_cultural_only(self) -> bool:
        """
        Check if configuration represents cultural-only evolution.

        Returns
        -------
        bool
            True if only cultural layer is enabled
        """
        return self.cultural_enabled and not self.genetic_enabled

    @beartype
    def is_combined(self) -> bool:
        """
        Check if configuration represents combined evolution.

        Returns
        -------
        bool
            True if both layers are enabled
        """
        return self.genetic_enabled and self.cultural_enabled

    @beartype
    def get_effective_genetic_weight(self) -> float:
        """
        Get effective genetic weight (0.0 if layer disabled).

        Returns
        -------
        float
            Effective genetic weight considering layer activation
        """
        return self.genetic_weight if self.genetic_enabled else 0.0

    @beartype
    def get_effective_cultural_weight(self) -> float:
        """
        Get effective cultural weight (0.0 if layer disabled).

        Returns
        -------
        float
            Effective cultural weight considering layer activation
        """
        return self.cultural_weight if self.cultural_enabled else 0.0

    @beartype
    def to_dict(self) -> dict[str, bool | float | str]:
        """
        Convert configuration to dictionary representation.

        Returns
        -------
        dict[str, bool | float | str]
            Dictionary representation of configuration
        """
        return {
            "genetic_enabled": self.genetic_enabled,
            "cultural_enabled": self.cultural_enabled,
            "genetic_weight": self.genetic_weight,
            "cultural_weight": self.cultural_weight,
            "blending_mode": self.blending_mode,
            "normalize_weights": self.normalize_weights,
        }

    @classmethod
    @beartype
    def from_dict(cls, config_dict: dict[str, bool | float | str]) -> LayerActivationConfig:
        """
        Create configuration from dictionary representation.

        Parameters
        ----------
        config_dict : dict[str, bool | float | str]
            Dictionary containing configuration parameters

        Returns
        -------
        LayerActivationConfig
            Configuration instance

        Raises
        ------
        ValueError
            If dictionary contains invalid parameters
        """
        # Type-safe extraction of parameters with validation
        blending_mode_value = str(config_dict.get("blending_mode", "weighted_average"))
        valid_modes = {"weighted_average", "probabilistic", "competitive"}
        if blending_mode_value not in valid_modes:
            raise ValueError(f"Invalid blending_mode: {blending_mode_value}. Must be one of {valid_modes}")

        return cls(
            genetic_enabled=bool(config_dict.get("genetic_enabled", True)),
            cultural_enabled=bool(config_dict.get("cultural_enabled", True)),
            genetic_weight=float(config_dict.get("genetic_weight", 0.5)),
            cultural_weight=float(config_dict.get("cultural_weight", 0.5)),
            blending_mode=blending_mode_value,  # type: ignore[arg-type]
            normalize_weights=bool(config_dict.get("normalize_weights", True)),
        )

    @classmethod
    @beartype
    def genetic_only(cls) -> LayerActivationConfig:
        """
        Create configuration for genetic-only evolution.

        Returns
        -------
        LayerActivationConfig
            Configuration with only genetic layer enabled
        """
        return cls(
            genetic_enabled=True,
            cultural_enabled=False,
            genetic_weight=1.0,
            cultural_weight=0.0,
            normalize_weights=False,
        )

    @classmethod
    @beartype
    def cultural_only(cls) -> LayerActivationConfig:
        """
        Create configuration for cultural-only evolution.

        Returns
        -------
        LayerActivationConfig
            Configuration with only cultural layer enabled
        """
        return cls(
            genetic_enabled=False,
            cultural_enabled=True,
            genetic_weight=0.0,
            cultural_weight=1.0,
            normalize_weights=False,
        )

    @classmethod
    @beartype
    def balanced_combined(cls, genetic_weight: float = 0.5) -> LayerActivationConfig:
        """
        Create configuration for balanced combined evolution.

        Parameters
        ----------
        genetic_weight : float, default=0.5
            Weight for genetic layer (cultural weight = 1 - genetic_weight)

        Returns
        -------
        LayerActivationConfig
            Configuration with both layers enabled and balanced weights
        """
        return cls(
            genetic_enabled=True,
            cultural_enabled=True,
            genetic_weight=genetic_weight,
            cultural_weight=1.0 - genetic_weight,
            normalize_weights=False,
        )

    def __repr__(self) -> str:
        """Rich representation of configuration."""
        layer_desc = []
        if self.genetic_enabled:
            layer_desc.append(f"genetic={self.genetic_weight:.3f}")
        if self.cultural_enabled:
            layer_desc.append(f"cultural={self.cultural_weight:.3f}")

        return f"LayerActivationConfig({', '.join(layer_desc)}, mode={self.blending_mode})"
