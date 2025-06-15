#!/usr/bin/env python3
"""
Lande-Kirkpatrick Model Demonstration

This script demonstrates the Lande-Kirkpatrick model of trait-preference
coevolution with various parameter scenarios and publication-quality
visualizations.

Usage:
    uv run python examples/lande_kirkpatrick_demo.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from lovebug.lande_kirkpatrick import (
    LandeKirkpatrickParams,
    interpret_results,
    plot_trajectory,
    simulate_lande_kirkpatrick,
)

# Configure logging and plotting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up publication-quality plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def run_scenario_analysis() -> dict[str, pl.DataFrame]:
    """
    Run multiple parameter scenarios to demonstrate model dynamics.

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary mapping scenario names to simulation results
    """
    scenarios = {
        "Baseline": LandeKirkpatrickParams(
            n_generations=200, genetic_correlation=0.1, selection_strength=0.1, preference_cost=0.05
        ),
        "Strong Correlation": LandeKirkpatrickParams(
            n_generations=200, genetic_correlation=0.3, selection_strength=0.1, preference_cost=0.05
        ),
        "High Selection": LandeKirkpatrickParams(
            n_generations=200, genetic_correlation=0.1, selection_strength=0.3, preference_cost=0.05
        ),
        "High Preference Cost": LandeKirkpatrickParams(
            n_generations=200, genetic_correlation=0.1, selection_strength=0.1, preference_cost=0.15
        ),
        "Low Heritability": LandeKirkpatrickParams(
            n_generations=200,
            genetic_correlation=0.1,
            selection_strength=0.1,
            preference_cost=0.05,
            h2_trait=0.2,
            h2_preference=0.2,
        ),
        "Negative Correlation": LandeKirkpatrickParams(
            n_generations=200, genetic_correlation=-0.2, selection_strength=0.1, preference_cost=0.05
        ),
    }

    results = {}
    for name, params in scenarios.items():
        logger.info(f"Running scenario: {name}")
        try:
            data = simulate_lande_kirkpatrick(params)
            results[name] = data

            # Print final outcomes
            final_trait = data.select(pl.col("mean_trait").last()).item()
            final_preference = data.select(pl.col("mean_preference").last()).item()
            final_covariance = data.select(pl.col("genetic_covariance").last()).item()

            print(f"\n{name}:")
            print(f"  Final trait: {final_trait:.4f}")
            print(f"  Final preference: {final_preference:.4f}")
            print(f"  Final covariance: {final_covariance:.4f}")
            print(f"  Interpretation: {interpret_results(final_trait, final_preference, final_covariance, params)}")

        except Exception as e:
            logger.error(f"Failed to simulate {name}: {e}")
            continue

    return results


def create_comparison_plots(results: dict[str, pl.DataFrame]) -> plt.Figure:
    """
    Create comprehensive comparison plots across scenarios.

    Parameters
    ----------
    results : dict[str, pl.DataFrame]
        Simulation results for each scenario

    Returns
    -------
    plt.Figure
        Figure with comparison plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    colors = sns.color_palette("husl", len(results))

    # Plot 1: Trait evolution comparison
    for i, (name, data) in enumerate(results.items()):
        df = data.to_pandas()
        ax1.plot(df["generation"], df["mean_trait"], color=colors[i], label=name, linewidth=2, alpha=0.8)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean Trait Value")
    ax1.set_title("Trait Evolution: Parameter Comparison", fontsize=14, fontweight="bold")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Preference evolution comparison
    for i, (name, data) in enumerate(results.items()):
        df = data.to_pandas()
        ax2.plot(df["generation"], df["mean_preference"], color=colors[i], label=name, linewidth=2, alpha=0.8)

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean Preference Value")
    ax2.set_title("Preference Evolution: Parameter Comparison", fontsize=14, fontweight="bold")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Genetic covariance comparison
    for i, (name, data) in enumerate(results.items()):
        df = data.to_pandas()
        ax3.plot(df["generation"], df["genetic_covariance"], color=colors[i], label=name, linewidth=2, alpha=0.8)

    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.axhline(y=0.1, color="orange", linestyle=":", alpha=0.7, label="Runaway Threshold")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Genetic Covariance")
    ax3.set_title("Genetic Correlation: Parameter Comparison", fontsize=14, fontweight="bold")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final outcomes scatter
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
        s=150,
        alpha=0.8,
        edgecolors="white",
        linewidth=2,
    )

    # Add annotations
    for i, name in enumerate(scenario_names):
        ax4.annotate(
            name,
            (final_traits[i], final_preferences[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": colors[i], "alpha": 0.3},
        )

    ax4.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax4.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax4.set_xlabel("Final Trait Value")
    ax4.set_ylabel("Final Preference Value")
    ax4.set_title("Evolutionary Endpoints", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Lande-Kirkpatrick Model: Parameter Sensitivity Analysis", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()

    return fig


def create_phase_portrait(data: pl.DataFrame, title: str = "Evolutionary Trajectory") -> plt.Figure:
    """
    Create a phase portrait showing trait-preference evolutionary trajectory.

    Parameters
    ----------
    data : pl.DataFrame
        Simulation results
    title : str
        Plot title

    Returns
    -------
    plt.Figure
        Phase portrait figure
    """
    df = data.to_pandas()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectory with color gradient representing time
    points = ax.scatter(
        df["mean_trait"],
        df["mean_preference"],
        c=df["generation"],
        cmap="plasma",
        s=50,
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
    )

    # Add directional arrows
    n_arrows = 15
    arrow_indices = np.linspace(0, len(df) - 2, n_arrows, dtype=int)

    for i in arrow_indices:
        dx = df.iloc[i + 1]["mean_trait"] - df.iloc[i]["mean_trait"]
        dy = df.iloc[i + 1]["mean_preference"] - df.iloc[i]["mean_preference"]

        if abs(dx) > 1e-6 or abs(dy) > 1e-6:  # Only draw if there's movement
            ax.annotate(
                "",
                xy=(df.iloc[i + 1]["mean_trait"], df.iloc[i + 1]["mean_preference"]),
                xytext=(df.iloc[i]["mean_trait"], df.iloc[i]["mean_preference"]),
                arrowprops={"arrowstyle": "->", "color": "black", "alpha": 0.6, "lw": 1.5},
            )

    # Mark start and end points
    ax.plot(
        df.iloc[0]["mean_trait"],
        df.iloc[0]["mean_preference"],
        "go",
        markersize=12,
        label="Start",
        markeredgecolor="white",
        markeredgewidth=2,
    )
    ax.plot(
        df.iloc[-1]["mean_trait"],
        df.iloc[-1]["mean_preference"],
        "ro",
        markersize=12,
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
    ax.set_title(f"{title}: Phase Portrait", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_sample_data(results: dict[str, pl.DataFrame], output_dir: Path) -> None:
    """
    Save sample simulation data for use with visualization framework.

    Parameters
    ----------
    results : dict[str, pl.DataFrame]
        Simulation results
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in results.items():
        # Clean filename
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        filename = f"lande_kirkpatrick_{clean_name}.parquet"
        filepath = output_dir / filename

        # Add scenario identifier
        data_with_scenario = data.with_columns(
            [pl.lit(name).alias("scenario"), pl.lit("lande_kirkpatrick").alias("model_type")]
        )

        data_with_scenario.write_parquet(filepath)
        logger.info(f"Saved {name} data to {filepath}")


def main() -> None:
    """Run the complete Lande-Kirkpatrick model demonstration."""
    print("🧬 Lande-Kirkpatrick Model Demonstration")
    print("=" * 50)

    # Run scenario analysis
    print("\n📊 Running parameter sensitivity analysis...")
    results = run_scenario_analysis()

    if not results:
        print("❌ No scenarios completed successfully!")
        return

    # Create output directory
    output_dir = Path("output/lande_kirkpatrick")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\n📈 Creating visualizations...")

    # Individual trajectory plots for interesting scenarios
    interesting_scenarios = ["Baseline", "Strong Correlation", "Negative Correlation"]
    for scenario_name in interesting_scenarios:
        if scenario_name in results:
            fig = plot_trajectory(results[scenario_name], f"{scenario_name} Scenario")
            fig.savefig(
                output_dir / f"trajectory_{scenario_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)

    # Comparison plots
    comparison_fig = create_comparison_plots(results)
    comparison_fig.savefig(output_dir / "parameter_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(comparison_fig)

    # Phase portraits for select scenarios
    for scenario_name in interesting_scenarios:
        if scenario_name in results:
            phase_fig = create_phase_portrait(results[scenario_name], scenario_name)
            phase_fig.savefig(
                output_dir / f"phase_portrait_{scenario_name.lower().replace(' ', '_')}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(phase_fig)

    # Save data for further analysis
    print("\n💾 Saving simulation data...")
    save_sample_data(results, output_dir / "data")

    print(f"\n✅ Demo completed! Results saved to {output_dir}")
    print(f"📁 Generated {len(list(output_dir.glob('*.png')))} plots and {len(results)} datasets")

    # Summary statistics
    print("\n📈 Summary Statistics:")
    for name, data in results.items():
        final_trait = data.select(pl.col("mean_trait").last()).item()
        final_preference = data.select(pl.col("mean_preference").last()).item()
        final_covariance = data.select(pl.col("genetic_covariance").last()).item()

        outcome = "Runaway" if abs(final_covariance) > 0.3 else "Equilibrium"
        print(f"  {name}: {outcome} (r={final_covariance:.3f}, T={final_trait:.3f}, P={final_preference:.3f})")


if __name__ == "__main__":
    main()
