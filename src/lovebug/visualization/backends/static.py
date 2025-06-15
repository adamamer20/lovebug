"""
Static Backend for Publication-Quality Visualizations

This module implements the matplotlib-based static backend for creating
high-quality, publication-ready visualizations of LoveBug simulation results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

from ..charts.base import BaseChart
from .base import BackendNotAvailableError, BaseBackend, RenderingError


class StaticBackend(BaseBackend):
    """
    Matplotlib-based backend for static, publication-quality visualizations.

    Provides high-DPI output suitable for academic papers and presentations,
    with professional styling and customizable academic color schemes.
    """

    def _setup_backend(self) -> None:
        """Setup matplotlib backend and default styling."""
        if not MATPLOTLIB_AVAILABLE:
            raise BackendNotAvailableError(
                "Matplotlib and seaborn are required for StaticBackend. Install with: pip install matplotlib seaborn"
            )

        # Set default matplotlib parameters for publication quality
        plt.rcParams.update(
            {
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.titlesize": 16,
                "lines.linewidth": 2,
                "lines.markersize": 8,
                "axes.linewidth": 1,
                "grid.linewidth": 0.5,
                "font.family": "serif",
                "text.usetex": False,  # Can be enabled if LaTeX is available
            }
        )

        # Set seaborn style for better defaults
        if sns is not None:
            sns.set_style("whitegrid")
            sns.set_palette("colorblind")

        # Default configuration
        self.config.update(
            {
                "style": "academic",
                "color_palette": "academic",
                "export_formats": ["png", "pdf", "svg"],
                "transparent_background": False,
            }
        )

    def render_chart(self, chart: BaseChart) -> Figure:
        """
        Render a chart using matplotlib.

        Args:
            chart: Chart instance to render

        Returns:
            Matplotlib Figure object
        """
        try:
            # Get chart data and configuration
            render_data = chart.get_data_for_rendering()
            chart_type = chart.get_chart_type()

            # Create figure and axes
            figsize = render_data.get("figure_size", (10, 6))
            fig, ax = plt.subplots(figsize=figsize, dpi=render_data.get("dpi", 300))

            # Apply styling
            self._apply_academic_styling(fig, ax)

            # Render based on chart type
            if chart_type == "trajectory":
                self._render_trajectory_chart(ax, render_data)
            else:
                raise RenderingError(f"Chart type '{chart_type}' not supported by StaticBackend")

            # Apply final formatting
            self._apply_final_formatting(fig, ax, render_data)

            return fig

        except Exception as e:
            raise RenderingError(f"Failed to render chart: {e}") from e

    def _apply_academic_styling(self, fig: Figure, ax: Axes) -> None:
        """Apply academic publication styling."""
        # Set background colors
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Configure grid
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        # Configure spines
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1)

        # Ensure tight layout
        fig.tight_layout()

    def _render_trajectory_chart(self, ax: Axes, render_data: dict[str, Any]) -> None:
        """Render trajectory-specific chart elements."""
        trajectory_type = render_data["trajectory_type"]

        if trajectory_type == "trait_preference":
            self._render_trait_preference_trajectory(ax, render_data)
        elif trajectory_type == "covariance":
            self._render_covariance_trajectory(ax, render_data)
        elif trajectory_type == "cultural_genetic":
            self._render_cultural_genetic_trajectory(ax, render_data)
        elif trajectory_type == "population_dynamics":
            self._render_population_trajectory(ax, render_data)
        else:
            raise RenderingError(f"Trajectory type '{trajectory_type}' not implemented")

    def _render_trait_preference_trajectory(self, ax: Axes, render_data: dict[str, Any]) -> None:
        """Render trait and preference evolution trajectories."""
        data = render_data["data"]
        colors = render_data["color_palette"][:2]  # Use first two colors

        # Convert Polars DataFrame to numpy for matplotlib
        steps = data["step"].to_numpy()
        display_values = data["mean_display"].to_numpy()
        preference_values = data["mean_preference"].to_numpy()

        # Plot main lines
        ax.plot(steps, display_values, color=colors[0], label="Display Traits", linewidth=2)
        ax.plot(steps, preference_values, color=colors[1], label="Mate Preferences", linewidth=2)

        # Add confidence bands if available and requested
        if render_data.get("show_confidence_bands", False):
            if "display_lower" in data.columns and "display_upper" in data.columns:
                display_lower = data["display_lower"].to_numpy()
                display_upper = data["display_upper"].to_numpy()
                ax.fill_between(steps, display_lower, display_upper, color=colors[0], alpha=0.2, label="Display 95% CI")

            if "preference_lower" in data.columns and "preference_upper" in data.columns:
                preference_lower = data["preference_lower"].to_numpy()
                preference_upper = data["preference_upper"].to_numpy()
                ax.fill_between(
                    steps, preference_lower, preference_upper, color=colors[1], alpha=0.2, label="Preference 95% CI"
                )

        ax.legend(loc="best")

    def _render_covariance_trajectory(self, ax: Axes, render_data: dict[str, Any]) -> None:
        """Render genetic covariance evolution trajectory."""
        data = render_data["data"]
        colors = render_data["color_palette"]

        steps = data["step"].to_numpy()
        covariance = data["trait_preference_covariance"].to_numpy()

        # Plot main covariance line
        ax.plot(steps, covariance, color=colors[0], label="Trait-Preference Covariance", linewidth=2)

        # Add reference lines
        if "runaway_threshold" in data.columns:
            threshold = data["runaway_threshold"].to_numpy()[0]
            ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.7, label="Runaway Threshold")

        if "neutral_line" in data.columns:
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5, label="Neutral (r=0)")

        ax.legend(loc="best")

    def _render_cultural_genetic_trajectory(self, ax: Axes, render_data: dict[str, Any]) -> None:
        """Render cultural vs genetic preference trajectories."""
        data = render_data["data"]
        colors = render_data["color_palette"][:2]

        steps = data["step"].to_numpy()
        genetic_prefs = data["genetic_preference"].to_numpy()
        cultural_prefs = data["cultural_preference"].to_numpy()

        ax.plot(steps, genetic_prefs, color=colors[0], label="Genetic Preferences", linewidth=2)
        ax.plot(steps, cultural_prefs, color=colors[1], label="Cultural Preferences", linewidth=2)

        ax.legend(loc="best")

    def _render_population_trajectory(self, ax: Axes, render_data: dict[str, Any]) -> None:
        """Render population dynamics trajectory."""
        data = render_data["data"]
        colors = render_data["color_palette"]

        steps = data["step"].to_numpy()
        population = data["population_size"].to_numpy()

        ax.plot(steps, population, color=colors[0], label="Population Size", linewidth=2)

        # Add secondary axes for age and energy if available
        if "mean_age" in data.columns and "mean_energy" in data.columns:
            ax2 = ax.twinx()

            mean_age = data["mean_age"].to_numpy()
            mean_energy = data["mean_energy"].to_numpy()

            ax2.plot(steps, mean_age, color=colors[1], linestyle="--", alpha=0.7, label="Mean Age")
            ax2.plot(steps, mean_energy, color=colors[2], linestyle=":", alpha=0.7, label="Mean Energy")

            ax2.set_ylabel("Age / Energy", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            ax.legend(loc="best")

    def _apply_final_formatting(self, fig: Figure, ax: Axes, render_data: dict[str, Any]) -> None:
        """Apply final formatting to the chart."""
        # Set title
        title = render_data.get("title", "")
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Set axis labels
        axes_labels = render_data.get("axes_labels", {})
        ax.set_xlabel(axes_labels.get("x", "X Axis"), fontsize=12)
        ax.set_ylabel(axes_labels.get("y", "Y Axis"), fontsize=12)

        # Adjust layout
        fig.tight_layout()

    def supports_interactivity(self) -> bool:
        """Static backend does not support interactivity."""
        return False

    def supports_animation(self) -> bool:
        """Static backend does not support animations."""
        return False

    def get_supported_formats(self) -> list[str]:
        """Get supported output formats."""
        return ["png", "pdf", "svg", "eps", "jpg", "tiff"]

    def save_output(self, rendered_chart: Figure, filepath: str | Path, format: str | None = None, **kwargs) -> None:
        """
        Save rendered chart to file.

        Args:
            rendered_chart: Matplotlib Figure object
            filepath: Output file path
            format: Output format (if None, infer from filepath)
            **kwargs: Additional save parameters
        """
        filepath_obj = Path(filepath)

        # Infer format from extension if not provided
        if format is None:
            format = filepath_obj.suffix.lstrip(".").lower()

        # Validate format
        if format not in self.get_supported_formats():
            raise ValueError(f"Unsupported format: {format}. Supported formats: {self.get_supported_formats()}")

        # Ensure output directory exists
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        # Set default save parameters
        save_kwargs = {
            "dpi": kwargs.get("dpi", 300),
            "bbox_inches": kwargs.get("bbox_inches", "tight"),
            "facecolor": kwargs.get("facecolor", "white"),
            "edgecolor": kwargs.get("edgecolor", "none"),
            "transparent": kwargs.get("transparent", False),
        }

        # Save the figure
        try:
            rendered_chart.savefig(str(filepath_obj), format=format, **save_kwargs)
            print(f"Saved chart to {filepath_obj}")
        except Exception as e:
            raise RenderingError(f"Failed to save chart: {e}") from e
        finally:
            # Clean up the figure to free memory
            plt.close(rendered_chart)


# Convenience function for quick static chart creation
def create_static_chart(chart: BaseChart, save_path: str | None = None, **save_kwargs) -> Figure:
    """
    Convenience function for creating static charts.

    Args:
        chart: Chart instance to render
        save_path: Optional path to save the chart
        **save_kwargs: Additional save parameters

    Returns:
        Matplotlib Figure object
    """
    backend = StaticBackend()
    fig = backend.render_chart(chart)

    if save_path:
        backend.save_output(fig, save_path, **save_kwargs)

    return fig
