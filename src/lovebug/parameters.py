"""
Parameter classes for evolutionary models.

This module contains parameter dataclasses for different evolutionary models
used throughout the LoveBug simulation framework.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LandeKirkpatrickParams"]


@dataclass(slots=True, frozen=False)
class LandeKirkpatrickParams:
    """Parameters for the Lande-Kirkpatrick model.

    Parameters
    ----------
    n_generations : int
        Number of generations to simulate
    pop_size : int
        Population size (assumed constant)
    h2_trait : float
        Heritability of male display trait (0-1)
    h2_preference : float
        Heritability of female preference (0-1)
    selection_strength : float
        Strength of natural selection against extreme traits
    genetic_correlation : float
        Initial genetic correlation between trait and preference
    mutation_variance : float
        Variance of mutational effects per generation
    preference_cost : float
        Cost of having strong preferences (0-1)
    """

    n_generations: int = 500
    pop_size: int = 1000
    h2_trait: float = 0.5
    h2_preference: float = 0.3
    selection_strength: float = 0.1
    genetic_correlation: float = 0.1
    mutation_variance: float = 0.01
    preference_cost: float = 0.05
