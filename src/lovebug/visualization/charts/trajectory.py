"""
Trajectory Chart for Sexual Selection Dynamics

This module implements trajectory visualizations for evolutionary dynamics
over time, including trait-preference coevolution, genetic covariance evolution,
and cultural vs genetic preference dynamics.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from .base import BaseChart, ChartError, DataProcessingError


class TrajectoryChart(BaseChart):
    """
    Trajectory chart for visualizing evolutionary dynamics over time.

    Specialized for sexual selection research, this chart can display:
    - Trait and preference evolution trajectories
    - Genetic covariance dynamics
    - Cultural vs genetic preference evolution
    - Population size effects on evolutionary dynamics
    - Multiple mechanism comparisons

    Example:
        chart = TrajectoryChart(data_loader, {
            'title': 'Trait-Preference Coevolution',
            'trajectory_type': 'trait_preference',
            'show_confidence_bands': True,
            'mechanisms': ['genetic_only', 'combined']
        })
    """

    # Supported trajectory types
    TRAJECTORY_TYPES = {
        "trait_preference": "Trait and preference evolution",
        "covariance": "Genetic covariance evolution",
        "cultural_genetic": "Cultural vs genetic preferences",
        "population_dynamics": "Population dynamics over time",
        "mechanism_comparison": "Multiple mechanism comparison",
    }

    def _validate_config(self) -> None:
        """Validate trajectory-specific configuration."""
        trajectory_type = self.config.get("trajectory_type", "trait_preference")

        if trajectory_type not in self.TRAJECTORY_TYPES:
            raise ValueError(
                f"Invalid trajectory_type: {trajectory_type}. Must be one of: {list(self.TRAJECTORY_TYPES.keys())}"
            )

        # Validate mechanism comparison config
        if trajectory_type == "mechanism_comparison":
            mechanisms = self.config.get("mechanisms", [])
            if not mechanisms:
                raise ValueError("mechanism_comparison requires 'mechanisms' list in config")

    def _preprocess_data(self) -> None:
        """Preprocess data for trajectory visualization."""
        try:
            trajectory_type = self.config.get("trajectory_type", "trait_preference")

            # Get base data with time filtering
            base_data = self.filter_data_by_time(self.data_loader.data)

            if trajectory_type == "trait_preference":
                self._processed_data = self._prepare_trait_preference_data(base_data)
            elif trajectory_type == "covariance":
                self._processed_data = self._prepare_covariance_data(base_data)
            elif trajectory_type == "cultural_genetic":
                self._processed_data = self._prepare_cultural_genetic_data(base_data)
            elif trajectory_type == "population_dynamics":
                self._processed_data = self._prepare_population_data(base_data)
            elif trajectory_type == "mechanism_comparison":
                self._processed_data = self._prepare_mechanism_comparison_data()

        except Exception as e:
            raise DataProcessingError(f"Failed to preprocess trajectory data: {e}") from e

    def _prepare_trait_preference_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Prepare data for trait and preference evolution visualization."""
        # Select relevant columns and ensure they exist
        required_cols = ["step", "mean_display", "mean_preference"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataProcessingError(f"Missing required columns: {missing_cols}")

        # Calculate additional metrics if confidence bands are requested
        if self.should_show_confidence_bands():
            # For now, use variance to estimate confidence bands
            # In a real implementation, this would come from multiple runs
            result = data.select(
                [
                    pl.col("step"),
                    pl.col("mean_display"),
                    pl.col("mean_preference"),
                    pl.col("display_variance").sqrt().alias("display_std"),
                    pl.col("preference_variance").sqrt().alias("preference_std"),
                ]
            )

            # Calculate confidence bands (assuming normal distribution)
            confidence_multiplier = 1.96  # 95% confidence
            result = result.with_columns(
                [
                    (pl.col("mean_display") - confidence_multiplier * pl.col("display_std")).alias("display_lower"),
                    (pl.col("mean_display") + confidence_multiplier * pl.col("display_std")).alias("display_upper"),
                    (pl.col("mean_preference") - confidence_multiplier * pl.col("preference_std")).alias(
                        "preference_lower"
                    ),
                    (pl.col("mean_preference") + confidence_multiplier * pl.col("preference_std")).alias(
                        "preference_upper"
                    ),
                ]
            )
        else:
            result = data.select([pl.col("step"), pl.col("mean_display"), pl.col("mean_preference")])

        return result.sort("step")

    def _prepare_covariance_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Prepare data for genetic covariance evolution visualization."""
        required_cols = ["step", "trait_preference_covariance"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataProcessingError(f"Missing required columns: {missing_cols}")

        result = data.select([pl.col("step"), pl.col("trait_preference_covariance")])

        # Add runaway threshold line (this is a theoretical value)
        # In real implementation, this would be calculated based on model parameters
        runaway_threshold = 0.1  # Placeholder value
        result = result.with_columns(
            [pl.lit(runaway_threshold).alias("runaway_threshold"), pl.lit(0.0).alias("neutral_line")]
        )

        return result.sort("step")

    def _prepare_cultural_genetic_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Prepare data for cultural vs genetic preference comparison."""
        required_cols = ["step", "mean_preference", "cultural_genetic_distance"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataProcessingError(f"Missing required columns: {missing_cols}")

        # Calculate approximate cultural preference from genetic + distance
        result = data.select(
            [pl.col("step"), pl.col("mean_preference").alias("genetic_preference"), pl.col("cultural_genetic_distance")]
        )

        # This is a simplified calculation - in practice, you'd track cultural preferences directly
        result = result.with_columns(
            [(pl.col("genetic_preference") + pl.col("cultural_genetic_distance")).alias("cultural_preference")]
        )

        return result.sort("step")

    def _prepare_population_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Prepare data for population dynamics visualization."""
        required_cols = ["step", "population_size"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataProcessingError(f"Missing required columns: {missing_cols}")

        result = data.select(
            [
                pl.col("step"),
                pl.col("population_size"),
                pl.col("mean_age").fill_null(0),
                pl.col("mean_energy").fill_null(0),
            ]
        )

        return result.sort("step")

    def _prepare_mechanism_comparison_data(self) -> pl.DataFrame:
        """Prepare data for mechanism comparison (placeholder implementation)."""
        # This would typically load data from multiple simulation runs
        # For now, return the base data as a placeholder
        base_data = self.filter_data_by_time(self.data_loader.data)

        # Add mechanism identifier (placeholder)
        result = base_data.with_columns([pl.lit("combined").alias("mechanism")])

        return result.sort("step")

    def get_data_for_rendering(self) -> dict[str, Any]:
        """Get processed data ready for backend rendering."""
        if self._processed_data is None:
            raise ChartError("Data not preprocessed. Call _preprocess_data() first.")

        trajectory_type = self.config.get("trajectory_type", "trait_preference")

        render_data = {
            "data": self._processed_data,
            "trajectory_type": trajectory_type,
            "title": self.get_title(),
            "axes_labels": self.get_axes_labels(),
            "show_confidence_bands": self.should_show_confidence_bands(),
            "color_palette": self.get_color_palette(),
            "figure_size": self.get_figure_size(),
            "dpi": self.get_dpi(),
        }

        # Add trajectory-specific rendering hints
        if trajectory_type == "trait_preference":
            render_data.update(
                {
                    "primary_lines": ["mean_display", "mean_preference"],
                    "line_labels": ["Display Traits", "Mate Preferences"],
                    "confidence_bands": ["display_lower", "display_upper", "preference_lower", "preference_upper"]
                    if self.should_show_confidence_bands()
                    else [],
                }
            )
        elif trajectory_type == "covariance":
            render_data.update(
                {
                    "primary_lines": ["trait_preference_covariance"],
                    "line_labels": ["Trait-Preference Covariance"],
                    "reference_lines": ["runaway_threshold", "neutral_line"],
                    "reference_labels": ["Runaway Threshold", "Neutral (r=0)"],
                }
            )
        elif trajectory_type == "cultural_genetic":
            render_data.update(
                {
                    "primary_lines": ["genetic_preference", "cultural_preference"],
                    "line_labels": ["Genetic Preferences", "Cultural Preferences"],
                }
            )
        elif trajectory_type == "population_dynamics":
            render_data.update(
                {
                    "primary_lines": ["population_size"],
                    "line_labels": ["Population Size"],
                    "secondary_lines": ["mean_age", "mean_energy"],
                    "secondary_labels": ["Mean Age", "Mean Energy"],
                }
            )
        elif trajectory_type == "mechanism_comparison":
            mechanisms = self.config.get("mechanisms", [])
            render_data.update(
                {
                    "mechanisms": mechanisms,
                    "comparison_metric": self.config.get("comparison_metric", "trait_preference_covariance"),
                }
            )

        return render_data

    def get_axes_labels(self) -> dict[str, str]:
        """Get appropriate axis labels for the trajectory type."""
        trajectory_type = self.config.get("trajectory_type", "trait_preference")

        base_labels = super().get_axes_labels()

        # Set x-axis to generations for all trajectory types
        base_labels["x"] = self.config.get("x_label", "Generation")

        # Set y-axis based on trajectory type
        if trajectory_type == "trait_preference":
            base_labels["y"] = self.config.get("y_label", "Mean Value")
        elif trajectory_type == "covariance":
            base_labels["y"] = self.config.get("y_label", "Genetic Covariance")
        elif trajectory_type == "cultural_genetic":
            base_labels["y"] = self.config.get("y_label", "Preference Value")
        elif trajectory_type == "population_dynamics":
            base_labels["y"] = self.config.get("y_label", "Population Size")
        elif trajectory_type == "mechanism_comparison":
            metric = self.config.get("comparison_metric", "trait_preference_covariance")
            base_labels["y"] = self.config.get("y_label", metric.replace("_", " ").title())

        return base_labels

    def get_title(self) -> str:
        """Get appropriate title for the trajectory type."""
        if self.config.get("title"):
            return self.config.get("title")

        trajectory_type = self.config.get("trajectory_type", "trait_preference")
        return self.TRAJECTORY_TYPES.get(trajectory_type, "Trajectory Chart")

    @classmethod
    def get_supported_trajectory_types(cls) -> dict[str, str]:
        """Get all supported trajectory types and their descriptions."""
        return cls.TRAJECTORY_TYPES.copy()

    def get_trajectory_description(self) -> str:
        """Get description of the current trajectory type."""
        trajectory_type = self.config.get("trajectory_type", "trait_preference")
        return self.TRAJECTORY_TYPES.get(trajectory_type, "Unknown trajectory type")
