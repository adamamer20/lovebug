"""
Base Chart Class for LoveBug Visualizations

This module defines the abstract base class for all chart types,
providing a consistent interface for data processing and chart
configuration across different visualization types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl

from ..data import DataLoader


class BaseChart(ABC):
    """
    Abstract base class for all chart types.

    Provides common functionality for data loading, preprocessing,
    and configuration management. Subclasses implement specific
    chart logic for different visualization types.
    """

    def __init__(self, data_loader: DataLoader, config: dict[str, Any]):
        """
        Initialize chart with data and configuration.

        Args:
            data_loader: DataLoader instance for accessing simulation data
            config: Configuration dictionary for chart customization
        """
        self.data_loader = data_loader
        self.config = ChartConfig(**config)
        self._processed_data: pl.DataFrame | None = None

        # Validate configuration
        self._validate_config()

        # Preprocess data if needed
        if self.config.get("preprocess_data", True):
            self._preprocess_data()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate chart-specific configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def _preprocess_data(self) -> None:
        """
        Preprocess data for visualization.

        This method should prepare the data needed for rendering,
        performing any necessary transformations, aggregations,
        or statistical calculations.
        """
        pass

    @abstractmethod
    def get_data_for_rendering(self) -> dict[str, Any]:
        """
        Get processed data ready for backend rendering.

        Returns:
            Dictionary containing all data needed for rendering
        """
        pass

    def get_chart_type(self) -> str:
        """
        Get the chart type identifier.

        Returns:
            String identifier for this chart type
        """
        return self.__class__.__name__.replace("Chart", "").lower()

    def get_title(self) -> str:
        """
        Get the chart title.

        Returns:
            Chart title string
        """
        return self.config.get("title", f"{self.get_chart_type().title()} Chart")

    def get_axes_labels(self) -> dict[str, str]:
        """
        Get axis labels for the chart.

        Returns:
            Dictionary with 'x' and 'y' axis labels
        """
        return {"x": self.config.get("x_label", "X Axis"), "y": self.config.get("y_label", "Y Axis")}

    def get_time_range(self) -> tuple[int, int | None]:
        """
        Get the time range for the chart.

        Returns:
            Tuple of (start_step, end_step)
        """
        start_step = self.config.get("start_step", 0)
        end_step = self.config.get("end_step", None)
        return start_step, end_step

    def filter_data_by_time(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Filter data by the configured time range.

        Args:
            data: Input DataFrame with 'step' column

        Returns:
            Filtered DataFrame
        """
        start_step, end_step = self.get_time_range()

        if end_step is None:
            return data.filter(pl.col("step") >= start_step)
        else:
            return data.filter((pl.col("step") >= start_step) & (pl.col("step") <= end_step))

    def compute_confidence_bands(
        self, data: pl.DataFrame, value_col: str, group_col: str = "step", confidence_level: float = 0.95
    ) -> pl.DataFrame:
        """
        Compute confidence bands for time series data.

        Args:
            data: Input DataFrame
            value_col: Column name for values
            group_col: Column name for grouping (default: 'step')
            confidence_level: Confidence level (default: 0.95)

        Returns:
            DataFrame with mean, lower, and upper confidence bounds
        """
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        return (
            data.group_by(group_col)
            .agg(
                [
                    pl.col(value_col).mean().alias(f"{value_col}_mean"),
                    pl.col(value_col).quantile(lower_percentile / 100).alias(f"{value_col}_lower"),
                    pl.col(value_col).quantile(upper_percentile / 100).alias(f"{value_col}_upper"),
                    pl.col(value_col).std().alias(f"{value_col}_std"),
                    pl.col(value_col).count().alias(f"{value_col}_count"),
                ]
            )
            .sort(group_col)
        )

    def get_color_palette(self) -> list[str]:
        """
        Get the color palette for the chart.

        Returns:
            List of color strings
        """
        palette_name = self.config.get("color_palette", "viridis")

        # Default color palettes
        palettes = {
            "viridis": [
                "#440154",
                "#481567",
                "#482677",
                "#453781",
                "#404788",
                "#39568c",
                "#33638d",
                "#2d708e",
                "#287d8e",
                "#238a8d",
            ],
            "academic": [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ],
            "sexual_selection": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"],
        }

        return palettes.get(palette_name, palettes["academic"])

    def should_show_confidence_bands(self) -> bool:
        """
        Check if confidence bands should be displayed.

        Returns:
            True if confidence bands should be shown
        """
        return self.config.get("show_confidence_bands", True)

    def get_figure_size(self) -> tuple[float, float]:
        """
        Get the figure size.

        Returns:
            Tuple of (width, height) in inches
        """
        return self.config.get("figsize", (10, 6))

    def get_dpi(self) -> int:
        """
        Get the DPI for high-quality output.

        Returns:
            DPI value
        """
        return self.config.get("dpi", 300)


class ChartConfig:
    """
    Configuration object for chart customization.

    Provides typed access to chart configuration parameters
    with validation and default values.
    """

    def __init__(self, **kwargs):
        """Initialize with configuration parameters."""
        # Chart appearance
        self.title: str = kwargs.get("title", "")
        self.figsize: tuple[float, float] = kwargs.get("figsize", (10, 6))
        self.dpi: int = kwargs.get("dpi", 300)
        self.color_palette: str = kwargs.get("color_palette", "academic")

        # Axis labels
        self.x_label: str = kwargs.get("x_label", "Generation")
        self.y_label: str = kwargs.get("y_label", "Value")

        # Data filtering
        self.start_step: int = kwargs.get("start_step", 0)
        self.end_step: int | None = kwargs.get("end_step", None)

        # Display options
        self.show_confidence_bands: bool = kwargs.get("show_confidence_bands", True)
        self.show_legend: bool = kwargs.get("show_legend", True)
        self.show_grid: bool = kwargs.get("show_grid", True)

        # Statistical options
        self.confidence_level: float = kwargs.get("confidence_level", 0.95)
        self.preprocess_data: bool = kwargs.get("preprocess_data", True)

        # Store any additional custom parameters
        self.custom_params = {k: v for k, v in kwargs.items() if k not in self.__dict__}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration parameter."""
        return getattr(self, key, self.custom_params.get(key, default))

    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_params[key] = value


class ChartError(Exception):
    """Exception raised when chart creation or rendering fails."""

    pass


class DataProcessingError(ChartError):
    """Exception raised when data preprocessing fails."""

    pass
