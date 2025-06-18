"""
Lande-Kirkpatrick Model Implementation for Sexual Selection

This module implements the classic Lande-Kirkpatrick model for understanding
trait-preference coevolution in sexual selection, with full type annotations
and comprehensive documentation.

The model describes how male display traits and female mating preferences
can coevolve through genetic correlations, demonstrating conditions for
"runaway" sexual selection. The implementation assumes mutation-selection
balance maintains genetic variance, with simplified linkage dynamics
governing covariance evolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from beartype import beartype

__all__ = ["LandeKirkpatrickParams", "simulate_lande_kirkpatrick", "plot_trajectory"]

# Configure logging
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=False)
class LandeKirkpatrickParams:
    """Parameters for the Lande-Kirkpatrick model.

    This model assumes mutation-selection balance where genetic variance
    is maintained by mutation input (mutation_variance) counterbalanced
    by selection-driven variance loss (variance_decay_rate). The simplified
    linkage dynamics approach models covariance buildup proportional to
    correlated selection pressures.

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
    variance_decay_rate : float
        Rate of variance loss per generation due to selection (0-1)
    stochastic_noise_interval : int
        Interval (generations) for environmental noise injection
    stochastic_noise_sigma : float
        Standard deviation of environmental noise
    covariance_buildup_rate : float
        Rate constant for genetic covariance accumulation
    """

    n_generations: int = 500
    pop_size: int = 1000
    h2_trait: float = 0.5
    h2_preference: float = 0.3
    selection_strength: float = 0.1
    genetic_correlation: float = 0.1
    mutation_variance: float = 0.01
    preference_cost: float = 0.05
    variance_decay_rate: float = 0.01
    stochastic_noise_interval: int = 10
    stochastic_noise_sigma: float = 0.001
    covariance_buildup_rate: float = 0.1

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if not (0 <= self.h2_trait <= 1):
            raise ValueError(f"h2_trait must be between 0 and 1, got {self.h2_trait}")
        if not (0 <= self.h2_preference <= 1):
            raise ValueError(f"h2_preference must be between 0 and 1, got {self.h2_preference}")
        if self.pop_size <= 0:
            raise ValueError(f"pop_size must be positive, got {self.pop_size}")
        if self.n_generations <= 0:
            raise ValueError(f"n_generations must be positive, got {self.n_generations}")
        if not (0 <= self.variance_decay_rate <= 1):
            raise ValueError(f"variance_decay_rate must be in [0,1], got {self.variance_decay_rate}")
        if self.stochastic_noise_interval <= 0:
            raise ValueError(f"stochastic_noise_interval must be positive, got {self.stochastic_noise_interval}")
        if self.stochastic_noise_sigma < 0:
            raise ValueError(f"stochastic_noise_sigma must be non-negative, got {self.stochastic_noise_sigma}")
        if self.covariance_buildup_rate < 0:
            raise ValueError(f"covariance_buildup_rate must be non-negative, got {self.covariance_buildup_rate}")


@beartype
def simulate_lande_kirkpatrick(params: LandeKirkpatrickParams) -> pl.DataFrame:
    """
    Simulate the Lande-Kirkpatrick model of trait-preference coevolution.

    This function implements the core evolutionary dynamics of the Lande-Kirkpatrick
    model, tracking the evolution of mean trait values, preferences, variances,
    and genetic covariances over multiple generations.

    The model includes:
    - Mutation-selection balance: Genetic variance maintained by balance between
      mutational input and selection-driven variance loss
    - Simplified linkage dynamics: Covariance buildup proportional to correlated
      selection on traits and preferences
    - Stochastic noise injection: Environmental fluctuations added periodically
      to model realistic population dynamics

    Parameters
    ----------
    params : LandeKirkpatrickParams
        Model parameters specifying simulation conditions

    Returns
    -------
    pl.DataFrame
        Simulation results with columns:
        - generation: Generation number (0 to n_generations-1)
        - mean_trait: Mean male display trait value
        - mean_preference: Mean female preference strength
        - trait_variance: Variance in display traits
        - preference_variance: Variance in preferences
        - genetic_covariance: Genetic covariance between trait and preference
        - selection_differential_trait: Selection differential on traits
        - selection_differential_preference: Selection differential on preferences
        - effective_selection_trait: Effective selection accounting for heritability
        - effective_selection_preference: Effective selection accounting for heritability
        - population_size: Population size (constant in this model)

    Raises
    ------
    ValueError
        If parameters are outside valid ranges (e.g., heritabilities not in [0,1])

    Examples
    --------
    >>> params = LandeKirkpatrickParams(n_generations=100, genetic_correlation=0.2)
    >>> result = simulate_lande_kirkpatrick(params)
    >>> len(result)
    100
    >>> "mean_trait" in result.columns
    True
    """
    logger.info(f"Starting Lande-Kirkpatrick simulation: {params.n_generations} generations")

    # Initialize tracking arrays
    results: list[dict[str, Any]] = []

    # Initial conditions
    mean_trait = 0.0
    mean_preference = 0.0
    trait_variance = 1.0
    preference_variance = 1.0
    genetic_covariance = params.genetic_correlation * np.sqrt(trait_variance * preference_variance)

    # Simulation loop
    for generation in range(params.n_generations):
        # Natural selection against extreme traits (stabilizing selection)
        selection_trait = -params.selection_strength * mean_trait

        # Sexual selection: preferences drive trait evolution
        # Strength depends on preference mean and genetic covariance
        sexual_selection_trait = (genetic_covariance / trait_variance) * mean_preference

        # Selection on preferences
        # Direct cost of being choosy
        direct_selection_preference = -params.preference_cost * mean_preference
        # Indirect selection through genetic correlation with traits
        indirect_selection_preference = (genetic_covariance / preference_variance) * (
            selection_trait + sexual_selection_trait
        )

        # Total selection pressures
        total_selection_trait = selection_trait + sexual_selection_trait
        total_selection_preference = direct_selection_preference + indirect_selection_preference

        # Store current generation data
        results.append(
            {
                "generation": generation,
                "mean_trait": mean_trait,
                "mean_preference": mean_preference,
                "trait_variance": trait_variance,
                "preference_variance": preference_variance,
                "genetic_covariance": genetic_covariance,
                "selection_differential_trait": total_selection_trait,
                "selection_differential_preference": total_selection_preference,
                "effective_selection_trait": total_selection_trait * params.h2_trait,
                "effective_selection_preference": total_selection_preference * params.h2_preference,
                "population_size": params.pop_size,
            }
        )

        # Evolution: response to selection
        delta_trait = params.h2_trait * total_selection_trait
        delta_preference = params.h2_preference * total_selection_preference

        # Update population means
        mean_trait += delta_trait
        mean_preference += delta_preference

        # Update genetic variances using mutation-selection balance
        # Variance increases by mutation, decreases by selection
        trait_variance = max(
            0.1, trait_variance + params.mutation_variance - params.variance_decay_rate * trait_variance
        )
        preference_variance = max(
            0.1, preference_variance + params.mutation_variance - params.variance_decay_rate * preference_variance
        )

        # Update genetic covariance using simplified linkage dynamics
        # Covariance builds up through correlated selection pressures
        covariance_change = (
            params.h2_trait
            * params.h2_preference
            * total_selection_trait
            * total_selection_preference
            * params.covariance_buildup_rate
        )
        genetic_covariance += covariance_change

        # Add stochastic environmental noise at regular intervals
        if generation % params.stochastic_noise_interval == 0 and generation > 0:
            mean_trait += np.random.normal(0, params.stochastic_noise_sigma)
            mean_preference += np.random.normal(0, params.stochastic_noise_sigma)
            genetic_covariance += np.random.normal(0, params.stochastic_noise_sigma)

    logger.info("Simulation completed successfully")
    return pl.DataFrame(results)


@beartype
def interpret_results(
    final_trait: float, final_preference: float, final_covariance: float, params: LandeKirkpatrickParams
) -> str:
    """
    Generate biological interpretation of simulation results.

    Parameters
    ----------
    final_trait : float
        Final mean trait value
    final_preference : float
        Final mean preference value
    final_covariance : float
        Final genetic covariance
    params : LandeKirkpatrickParams
        Model parameters used in simulation

    Returns
    -------
    str
        Human-readable interpretation of evolutionary dynamics

    Examples
    --------
    >>> params = LandeKirkpatrickParams()
    >>> interp = interpret_results(0.5, 0.3, 0.2, params)
    >>> "evolution" in interp.lower()
    True
    """
    interpretations = []

    # Analyze trait evolution
    if abs(final_trait) > 1.0:
        interpretations.append(
            f"Strong trait evolution (|T| = {abs(final_trait):.2f}) indicates intense sexual selection"
        )
    elif abs(final_trait) < 0.1:
        interpretations.append("Minimal trait evolution suggests balanced selection pressures")
    else:
        interpretations.append(f"Moderate trait evolution (T = {final_trait:.2f})")

    # Analyze preference evolution
    if abs(final_preference) > 0.5:
        interpretations.append(
            f"Strong preference evolution (|P| = {abs(final_preference):.2f}) shows choosiness benefits outweigh costs"
        )
    elif final_preference < -0.1:
        interpretations.append("Preference reduction indicates high costs of choosiness")
    else:
        interpretations.append(f"Moderate preference evolution (P = {final_preference:.2f})")

    # Analyze covariance dynamics
    if final_covariance > 0.3:
        interpretations.append("Strong positive genetic correlation enables runaway sexual selection")
    elif final_covariance < -0.1:
        interpretations.append("Negative genetic correlation constrains trait-preference coevolution")
    elif abs(final_covariance) < 0.05:
        interpretations.append("Minimal genetic correlation limits coevolutionary dynamics")
    else:
        interpretations.append(f"Moderate genetic correlation (r = {final_covariance:.3f})")

    # Parameter-specific insights
    if params.preference_cost > 0.1:
        interpretations.append(f"High preference costs ({params.preference_cost:.2f}) limit choosiness evolution")

    if params.selection_strength > 0.2:
        interpretations.append(f"Strong natural selection ({params.selection_strength:.2f}) opposes trait exaggeration")

    return " • ".join(interpretations)


@beartype
def plot_trajectory(data: pl.DataFrame, title: str = "Lande-Kirkpatrick Model") -> Any:
    """
    Create a trajectory plot of trait-preference coevolution.

    Parameters
    ----------
    data : pl.DataFrame
        Simulation results from simulate_lande_kirkpatrick
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Figure object with trajectory plot

    Examples
    --------
    >>> params = LandeKirkpatrickParams(n_generations=50)
    >>> data = simulate_lande_kirkpatrick(params)
    >>> fig = plot_trajectory(data)
    >>> fig.get_size_inches()[0] > 0  # Check that figure was created
    True
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting") from e

    # Convert to pandas for matplotlib compatibility
    df = data.to_pandas()

    # Create subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Trait and preference evolution
    ax1.plot(df["generation"], df["mean_trait"], "b-", linewidth=2.5, label="Male Display Trait", alpha=0.8)
    ax1.plot(df["generation"], df["mean_preference"], "r-", linewidth=2.5, label="Female Preference", alpha=0.8)

    # Add confidence bands using variance
    trait_std = np.sqrt(df["trait_variance"])
    pref_std = np.sqrt(df["preference_variance"])

    ax1.fill_between(
        df["generation"],
        df["mean_trait"] - 1.96 * trait_std,
        df["mean_trait"] + 1.96 * trait_std,
        alpha=0.2,
        color="blue",
    )
    ax1.fill_between(
        df["generation"],
        df["mean_preference"] - 1.96 * pref_std,
        df["mean_preference"] + 1.96 * pref_std,
        alpha=0.2,
        color="red",
    )

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean Phenotype Value")
    ax1.set_title(f"{title}: Trait-Preference Coevolution", fontsize=14, fontweight="bold")
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Genetic covariance evolution
    ax2.plot(df["generation"], df["genetic_covariance"], "g-", linewidth=2.5, label="Genetic Covariance")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="Neutral Line")
    ax2.axhline(y=0.1, color="orange", linestyle=":", alpha=0.7, label="Runaway Threshold")

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Genetic Covariance (r_TP)")
    ax2.set_title("Genetic Correlation Dynamics", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
