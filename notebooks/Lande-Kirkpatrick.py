"""
Lande-Kirkpatrick Model Implementation and Visualization

A comprehensive notebook implementing the Lande-Kirkpatrick model for
trait-preference coevolution in sexual selection, with publication-quality
visualizations using the LoveBug visualization framework.

The Lande-Kirkpatrick model describes the coevolution of male display traits
and female mating preferences through genetic correlations, demonstrating
conditions for "runaway" sexual selection.

Author: Adam Amer
Date: 2025-06-15
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import marimo
import numpy as np
import polars as pl
from beartype import beartype

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__generated_with = "0.13.15"
app = marimo.App(width="medium")


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


@beartype
def simulate_lande_kirkpatrick(params: LandeKirkpatrickParams) -> pl.DataFrame:
    """
    Simulate the Lande-Kirkpatrick model of trait-preference coevolution.

    Parameters
    ----------
    params : LandeKirkpatrickParams
        Model parameters

    Returns
    -------
    pl.DataFrame
        Simulation results with columns:
        - generation: Generation number
        - mean_trait: Mean male display trait value
        - mean_preference: Mean female preference strength
        - trait_variance: Variance in display traits
        - preference_variance: Variance in preferences
        - genetic_covariance: Genetic covariance between trait and preference
        - selection_differential_trait: Selection differential on traits
        - selection_differential_preference: Selection differential on preferences
        - effective_selection_trait: Effective selection including preference costs
        - effective_selection_preference: Effective selection including costs
        - population_size: Population size (constant in this model)

    Raises
    ------
    ValueError
        If parameters are outside valid ranges
    """
    # Validate parameters
    if not (0 <= params.h2_trait <= 1):
        raise ValueError(f"h2_trait must be between 0 and 1, got {params.h2_trait}")
    if not (0 <= params.h2_preference <= 1):
        raise ValueError(f"h2_preference must be between 0 and 1, got {params.h2_preference}")
    if params.pop_size <= 0:
        raise ValueError(f"pop_size must be positive, got {params.pop_size}")

    logger.info(f"Starting Lande-Kirkpatrick simulation with {params.n_generations} generations")

    # Initialize arrays to store results
    results = []

    # Initial conditions
    mean_trait = 0.0
    mean_preference = 0.0
    trait_variance = 1.0
    preference_variance = 1.0
    genetic_covariance = params.genetic_correlation * np.sqrt(trait_variance * preference_variance)

    # Simulation loop
    for generation in range(params.n_generations):
        # Natural selection against extreme traits
        selection_trait = -params.selection_strength * mean_trait

        # Sexual selection: preferences drive trait evolution
        # Strength depends on preference variance and genetic covariance
        sexual_selection_trait = (genetic_covariance / trait_variance) * mean_preference

        # Selection on preferences: cost of being choosy + indirect selection
        direct_selection_preference = -params.preference_cost * mean_preference
        indirect_selection_preference = (genetic_covariance / preference_variance) * (
            selection_trait + sexual_selection_trait
        )

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

        # Evolution: change in means due to selection and heritability
        delta_trait = params.h2_trait * total_selection_trait
        delta_preference = params.h2_preference * total_selection_preference

        # Update means
        mean_trait += delta_trait
        mean_preference += delta_preference

        # Update genetic variances and covariance
        # Simplified: assume variances remain roughly constant with mutation
        trait_variance = max(0.1, trait_variance + params.mutation_variance - 0.01 * trait_variance)
        preference_variance = max(0.1, preference_variance + params.mutation_variance - 0.01 * preference_variance)

        # Genetic covariance evolves due to selection and linkage
        # This is a simplified version of the full quantitative genetics
        covariance_change = (
            params.h2_trait * params.h2_preference * total_selection_trait * total_selection_preference * 0.1
        )
        genetic_covariance += covariance_change

        # Add some noise to prevent unrealistic precision
        if generation % 10 == 0:
            mean_trait += np.random.normal(0, 0.001)
            mean_preference += np.random.normal(0, 0.001)

    logger.info("Simulation completed successfully")
    return pl.DataFrame(results)


@app.cell
def import_libraries():
    """Import required libraries and set up environment."""
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns

    # Set up plotting style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    mo.md("""
    # 🧬 Lande-Kirkpatrick Model: Trait-Preference Coevolution

    This notebook implements and visualizes the classic **Lande-Kirkpatrick model**
    of sexual selection, which describes how male display traits and female mating
    preferences can coevolve through genetic correlations.

    ## Key Concepts

    - **Genetic Correlation**: The fundamental driver of coevolution
    - **Runaway Selection**: When preferences and traits escalate together
    - **Selection Balance**: Natural vs. sexual selection trade-offs
    - **Heritability Effects**: How genetic transmission shapes evolution
    """)
    return mo, np, pl, Path, plt, sns


@app.cell
def setup_parameters(mo):
    """Interactive parameter setup for the model."""

    # Create interactive sliders for key parameters
    n_generations = mo.ui.slider(100, 1000, value=500, step=50, label="Generations")
    pop_size = mo.ui.slider(500, 5000, value=1000, step=250, label="Population Size")
    h2_trait = mo.ui.slider(0.1, 0.9, value=0.5, step=0.1, label="Trait Heritability")
    h2_preference = mo.ui.slider(0.1, 0.9, value=0.3, step=0.1, label="Preference Heritability")
    selection_strength = mo.ui.slider(0.01, 0.5, value=0.1, step=0.01, label="Natural Selection Strength")
    genetic_correlation = mo.ui.slider(-0.5, 0.5, value=0.1, step=0.05, label="Initial Genetic Correlation")
    preference_cost = mo.ui.slider(0.0, 0.2, value=0.05, step=0.01, label="Preference Cost")

    mo.md(f"""
    ## 🎛️ Model Parameters

    Adjust the parameters below to explore different evolutionary scenarios:

    **Population Parameters:**
    - {n_generations}
    - {pop_size}

    **Genetic Parameters:**
    - {h2_trait}
    - {h2_preference}
    - {genetic_correlation}

    **Selection Parameters:**
    - {selection_strength}
    - {preference_cost}

    **Current Configuration:**
    - Generations: {n_generations.value}
    - Population: {pop_size.value}
    - Trait h²: {h2_trait.value}
    - Preference h²: {h2_preference.value}
    - Selection: {selection_strength.value}
    - r(T,P): {genetic_correlation.value}
    - Cost: {preference_cost.value}
    """)

    return (n_generations, pop_size, h2_trait, h2_preference, selection_strength, genetic_correlation, preference_cost)


@app.cell
def run_simulation(
    n_generations,
    pop_size,
    h2_trait,
    h2_preference,
    selection_strength,
    genetic_correlation,
    preference_cost,
    mo,
    np,
    pl,
):
    """Run the Lande-Kirkpatrick simulation with current parameters."""

    # Create parameter object
    params = LandeKirkpatrickParams(
        n_generations=n_generations.value,
        pop_size=pop_size.value,
        h2_trait=h2_trait.value,
        h2_preference=h2_preference.value,
        selection_strength=selection_strength.value,
        genetic_correlation=genetic_correlation.value,
        mutation_variance=0.01,
        preference_cost=preference_cost.value,
    )

    # Run simulation
    try:
        simulation_data = simulate_lande_kirkpatrick(params)

        # Calculate additional metrics for analysis
        final_trait = simulation_data.select(pl.col("mean_trait").last()).item()
        final_preference = simulation_data.select(pl.col("mean_preference").last()).item()
        final_covariance = simulation_data.select(pl.col("genetic_covariance").last()).item()

        # Determine evolutionary outcome
        if abs(final_covariance) > 0.5:
            outcome = "🚀 **Runaway Evolution**" if final_covariance > 0 else "🔄 **Negative Coevolution**"
        elif abs(final_trait) < 0.1 and abs(final_preference) < 0.1:
            outcome = "⚖️ **Stable Equilibrium**"
        else:
            outcome = "📈 **Directional Evolution**"

        mo.md(f"""
        ## 📊 Simulation Results

        **Evolutionary Outcome:** {outcome}

        **Final Values:**
        - Mean Trait: {final_trait:.3f}
        - Mean Preference: {final_preference:.3f}
        - Genetic Covariance: {final_covariance:.3f}

        **Interpretation:**
        {_interpret_results(final_trait, final_preference, final_covariance, params)}
        """)

    except Exception as e:
        mo.md(f"""
        ## ❌ Simulation Error

        Failed to run simulation: {str(e)}

        Please check parameter values and try again.
        """)
        simulation_data = None

    return simulation_data, params


@beartype
def _interpret_results(trait: float, preference: float, covariance: float, params: LandeKirkpatrickParams) -> str:
    """Generate interpretation of simulation results."""

    interpretation = []

    # Trait evolution
    if abs(trait) > 1.0:
        interpretation.append(
            f"• **Strong trait evolution** (|T| = {abs(trait):.2f}) indicates intense sexual selection"
        )
    elif abs(trait) < 0.1:
        interpretation.append("• **Minimal trait evolution** suggests balanced selection pressures")

    # Preference evolution
    if abs(preference) > 0.5:
        interpretation.append(
            f"• **Strong preference evolution** (|P| = {abs(preference):.2f}) shows choosiness pays off"
        )
    elif preference < -0.1:
        interpretation.append("• **Preference reduction** indicates high costs of choosiness")

    # Covariance dynamics
    if covariance > 0.3:
        interpretation.append("• **Positive genetic correlation** enables runaway sexual selection")
    elif covariance < -0.1:
        interpretation.append("• **Negative genetic correlation** constrains trait-preference coevolution")

    # Parameter effects
    if params.preference_cost > 0.1:
        interpretation.append(f"• **High preference costs** ({params.preference_cost:.2f}) limit choosiness evolution")

    if params.selection_strength > 0.2:
        interpretation.append(
            f"• **Strong natural selection** ({params.selection_strength:.2f}) opposes trait exaggeration"
        )

    return "\n".join(interpretation) if interpretation else "• Standard evolutionary dynamics observed"


@app.cell
def create_trajectory_plot(simulation_data, mo, plt, sns):
    """Create the main trajectory plot showing trait-preference coevolution."""

    if simulation_data is None:
        return mo.md("No simulation data available for plotting.")

    # Convert to pandas for matplotlib compatibility
    df = simulation_data.to_pandas()

    # Create the trajectory plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Trait and Preference Evolution
    ax1.plot(df["generation"], df["mean_trait"], "b-", linewidth=2.5, label="Male Display Trait", alpha=0.8)
    ax1.plot(df["generation"], df["mean_preference"], "r-", linewidth=2.5, label="Female Preference", alpha=0.8)

    # Add confidence bands (using variance as approximate confidence)
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
    ax1.set_title("Lande-Kirkpatrick Model: Trait-Preference Coevolution", fontsize=14, fontweight="bold")
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Genetic Covariance Evolution
    ax2.plot(df["generation"], df["genetic_covariance"], "g-", linewidth=2.5, label="Genetic Covariance")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="Neutral Line")
    ax2.axhline(y=0.1, color="orange", linestyle=":", alpha=0.7, label="Runaway Threshold")

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Genetic Covariance (rTP)")
    ax2.set_title("Genetic Correlation Dynamics", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


@app.cell
def create_phase_portrait(simulation_data, mo, plt, sns):
    """Create a phase portrait showing the trait-preference evolutionary trajectory."""

    if simulation_data is None:
        return mo.md("No simulation data available for phase portrait.")

    df = simulation_data.to_pandas()

    # Create phase portrait
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectory with color gradient representing time
    points = ax.scatter(
        df["mean_trait"],
        df["mean_preference"],
        c=df["generation"],
        cmap="plasma",
        s=30,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
    )

    # Add arrow to show direction
    for i in range(0, len(df) - 1, len(df) // 20):  # Every 5% of the trajectory
        ax.annotate(
            "",
            xy=(df.iloc[i + 1]["mean_trait"], df.iloc[i + 1]["mean_preference"]),
            xytext=(df.iloc[i]["mean_trait"], df.iloc[i]["mean_preference"]),
            arrowprops={"arrowstyle": "->", "color": "black", "alpha": 0.6, "lw": 1},
        )

    # Mark start and end points
    ax.plot(
        df.iloc[0]["mean_trait"],
        df.iloc[0]["mean_preference"],
        "go",
        markersize=10,
        label="Start",
        markeredgecolor="white",
        markeredgewidth=2,
    )
    ax.plot(
        df.iloc[-1]["mean_trait"],
        df.iloc[-1]["mean_preference"],
        "ro",
        markersize=10,
        label="End",
        markeredgecolor="white",
        markeredgewidth=2,
    )

    # Add reference lines
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label("Generation", rotation=270, labelpad=20)

    ax.set_xlabel("Male Display Trait")
    ax.set_ylabel("Female Preference Strength")
    ax.set_title("Phase Portrait: Evolutionary Trajectory in Trait-Preference Space", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


@app.cell
def create_selection_analysis(simulation_data, mo, plt, sns):
    """Analyze and visualize selection pressures throughout evolution."""

    if simulation_data is None:
        return mo.md("No simulation data available for selection analysis.")

    df = simulation_data.to_pandas()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Selection Differentials
    ax1.plot(df["generation"], df["selection_differential_trait"], "b-", label="Trait Selection", alpha=0.8)
    ax1.plot(df["generation"], df["selection_differential_preference"], "r-", label="Preference Selection", alpha=0.8)
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Selection Differential")
    ax1.set_title("Selection Pressures Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Effective Selection (accounting for heritability)
    ax2.plot(df["generation"], df["effective_selection_trait"], "b-", label="Effective Trait Selection", alpha=0.8)
    ax2.plot(
        df["generation"], df["effective_selection_preference"], "r-", label="Effective Preference Selection", alpha=0.8
    )
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Effective Selection")
    ax2.set_title("Heritable Selection Effects")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Variance Evolution
    ax3.plot(df["generation"], df["trait_variance"], "b-", label="Trait Variance", alpha=0.8)
    ax3.plot(df["generation"], df["preference_variance"], "r-", label="Preference Variance", alpha=0.8)
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Phenotypic Variance")
    ax3.set_title("Variance Dynamics")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Selection Strength Distribution
    selection_magnitudes = np.sqrt(
        df["selection_differential_trait"] ** 2 + df["selection_differential_preference"] ** 2
    )
    ax4.hist(selection_magnitudes, bins=30, alpha=0.7, color="green", edgecolor="black")
    ax4.axvline(
        x=selection_magnitudes.mean(), color="red", linestyle="--", label=f"Mean = {selection_magnitudes.mean():.3f}"
    )
    ax4.set_xlabel("Selection Magnitude")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of Selection Intensities")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Selection Analysis: Lande-Kirkpatrick Model", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return fig


@app.cell
def evolutionary_outcomes_comparison(mo, np, pl, plt, sns):
    """Compare evolutionary outcomes under different parameter regimes."""

    # Define parameter scenarios to compare
    scenarios = {
        "Weak Selection": LandeKirkpatrickParams(selection_strength=0.01, preference_cost=0.01),
        "Strong Selection": LandeKirkpatrickParams(selection_strength=0.3, preference_cost=0.01),
        "High Preference Cost": LandeKirkpatrickParams(selection_strength=0.1, preference_cost=0.15),
        "Low Heritability": LandeKirkpatrickParams(h2_trait=0.2, h2_preference=0.2),
        "Strong Correlation": LandeKirkpatrickParams(genetic_correlation=0.3),
        "Negative Correlation": LandeKirkpatrickParams(genetic_correlation=-0.2),
    }

    # Run simulations for each scenario
    results = {}
    for name, params in scenarios.items():
        params.n_generations = 300  # Shorter runs for comparison
        try:
            data = simulate_lande_kirkpatrick(params)
            results[name] = data
        except Exception as e:
            logger.error(f"Failed to simulate {name}: {e}")
            continue

    if not results:
        return mo.md("Failed to run comparison simulations.")

    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    colors = sns.color_palette("husl", len(results))

    # Plot 1: Trait evolution comparison
    for i, (name, data) in enumerate(results.items()):
        df = data.to_pandas()
        ax1.plot(df["generation"], df["mean_trait"], color=colors[i], label=name, linewidth=2, alpha=0.8)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean Trait Value")
    ax1.set_title("Trait Evolution: Parameter Comparison")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Preference evolution comparison
    for i, (name, data) in enumerate(results.items()):
        df = data.to_pandas()
        ax2.plot(df["generation"], df["mean_preference"], color=colors[i], label=name, linewidth=2, alpha=0.8)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Preference Value")
    ax2.set_title("Preference Evolution: Parameter Comparison")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Genetic covariance comparison
    for i, (name, data) in enumerate(results.items()):
        df = data.to_pandas()
        ax3.plot(df["generation"], df["genetic_covariance"], color=colors[i], label=name, linewidth=2, alpha=0.8)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Genetic Covariance")
    ax3.set_title("Genetic Correlation: Parameter Comparison")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final outcomes summary
    final_traits = []
    final_preferences = []
    scenario_names = []

    for name, data in results.items():
        final_traits.append(data.select(pl.col("mean_trait").last()).item())
        final_preferences.append(data.select(pl.col("mean_preference").last()).item())
        scenario_names.append(name)

    ax4.scatter(
        final_traits,
        final_preferences,
        c=range(len(final_traits)),
        cmap="viridis",
        s=100,
        alpha=0.8,
        edgecolors="white",
        linewidth=2,
    )

    for i, name in enumerate(scenario_names):
        ax4.annotate(
            name, (final_traits[i], final_preferences[i]), xytext=(5, 5), textcoords="offset points", fontsize=9
        )

    ax4.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax4.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax4.set_xlabel("Final Trait Value")
    ax4.set_ylabel("Final Preference Value")
    ax4.set_title("Evolutionary Endpoints")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Parameter Sensitivity Analysis: Lande-Kirkpatrick Model", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return fig, results


@app.cell
def model_summary_and_interpretation(mo):
    """Provide model summary and biological interpretation."""

    mo.md("""
    ## 🧬 Model Summary & Biological Interpretation

    ### The Lande-Kirkpatrick Model

    The **Lande-Kirkpatrick model** is a foundational theoretical framework in sexual selection
    that describes how male ornamental traits and female mating preferences can coevolve through
    genetic correlations. This model helps explain the evolution of elaborate traits that seem
    costly or maladaptive from a survival perspective.

    ### Key Mechanisms

    1. **Genetic Correlation (r_TP)**: The crucial link between trait and preference genes
       - Positive correlation enables runaway sexual selection
       - Negative correlation constrains coevolution
       - Correlation strength determines evolutionary dynamics

    2. **Selection Balance**: Trade-off between different selective forces
       - **Sexual selection**: Preferences favor extreme traits
       - **Natural selection**: Survival costs oppose trait exaggeration
       - **Preference costs**: Choosiness itself can be costly

    3. **Heritability Effects**: Genetic transmission shapes evolutionary response
       - Higher heritability → stronger evolutionary response
       - Different heritabilities for traits vs preferences create asymmetries

    ### Evolutionary Outcomes

    - **🚀 Runaway Evolution**: Traits and preferences escalate together indefinitely
    - **⚖️ Stable Equilibrium**: Forces balance, evolution stops
    - **📈 Directional Evolution**: Steady change toward new equilibrium
    - **🔄 Oscillatory Dynamics**: Cyclical changes in trait values

    ### Real-World Examples

    - **Peacock tails** and peahen preferences for eye-spots
    - **Bird song complexity** and female choosiness
    - **Fish coloration** and mate recognition systems
    - **Human cultural evolution** of aesthetic preferences

    ### Model Limitations

    - Assumes infinite population (no genetic drift)
    - Simplified genetics (additive effects only)
    - Constant environmental conditions
    - No mutation-selection balance
    - Single trait-preference pair

    ### Extensions & Future Directions

    - **Multi-trait models**: Multiple correlated traits and preferences
    - **Finite population effects**: Genetic drift and stochastic dynamics
    - **Environmental variation**: Fluctuating selection pressures
    - **Gene flow**: Migration between populations
    - **Cultural transmission**: Learning-based preference evolution
    """)


@app.cell
def export_simulation_data(simulation_data, params, mo, Path):
    """Export simulation results for further analysis."""

    if simulation_data is None:
        return mo.md("No simulation data to export.")

    # Create output directory
    output_dir = Path("simulation_output")
    output_dir.mkdir(exist_ok=True)

    # Generate filename with parameter summary
    filename = f"lande_kirkpatrick_h2t{params.h2_trait}_h2p{params.h2_preference}_sel{params.selection_strength}_corr{params.genetic_correlation:.2f}.parquet"
    output_path = output_dir / filename

    # Export data
    try:
        simulation_data.write_parquet(output_path)

        # Also export parameters as JSON
        import json

        param_dict = {
            "n_generations": params.n_generations,
            "pop_size": params.pop_size,
            "h2_trait": params.h2_trait,
            "h2_preference": params.h2_preference,
            "selection_strength": params.selection_strength,
            "genetic_correlation": params.genetic_correlation,
            "mutation_variance": params.mutation_variance,
            "preference_cost": params.preference_cost,
        }

        param_path = output_dir / f"parameters_{filename.replace('.parquet', '.json')}"
        with open(param_path, "w") as f:
            json.dump(param_dict, f, indent=2)

        mo.md(f"""
        ## 💾 Data Export Complete

        **Simulation data saved to:** `{output_path}`
        **Parameters saved to:** `{param_path}`

        **File size:** {output_path.stat().st_size / 1024:.1f} KB
        **Records:** {len(simulation_data):,}

        This data can be loaded into the LoveBug visualization framework for
        publication-quality plots and further analysis.
        """)

    except Exception as e:
        mo.md(f"""
        ## ❌ Export Failed

        Could not export simulation data: {str(e)}
        """)

    return output_path


if __name__ == "__main__":
    app.run()
