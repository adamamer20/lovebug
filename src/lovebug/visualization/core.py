"""
Core Visualization Engine for LoveBug

This module provides the main visualization engine and chart factory
for creating publication-quality and interactive visualizations of
sexual selection dynamics.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from .backends.base import BaseBackend
from .charts.base import BaseChart
from .data import DataLoader


class VisualizationEngine:
    """
    Main visualization engine for LoveBug simulations.

    Provides a unified interface for creating charts with different backends
    and managing data loading/caching for efficient visualization generation.

    Example:
        # Load simulation data
        engine = VisualizationEngine('data/simulation_results.parquet')

        # Create publication-quality trajectory plot
        fig = engine.create_chart(
            chart_type='trajectory',
            backend='static',
            config={'title': 'Trait-Preference Coevolution'}
        )

        # Create interactive comparison plot
        interactive_fig = engine.create_chart(
            chart_type='comparison',
            backend='interactive',
            config={'add_parameter_controls': True}
        )
    """

    def __init__(self, data_path: str | Path):
        """
        Initialize the visualization engine with simulation data.

        Args:
            data_path: Path to Parquet file containing simulation results
        """
        self.data_loader = DataLoader(data_path)
        self.chart_factory = ChartFactory()
        self.backend_registry = BackendRegistry()

        # Register default backends
        self._register_default_backends()

    def _register_default_backends(self):
        """Register the default rendering backends."""
        try:
            from .backends.static import StaticBackend

            self.backend_registry.register("static", StaticBackend)
        except ImportError:
            warnings.warn("StaticBackend not available. Install matplotlib and seaborn.", stacklevel=2)

        try:
            from .backends.interactive import InteractiveBackend

            self.backend_registry.register("interactive", InteractiveBackend)
        except ImportError:
            # Plotly is available but InteractiveBackend is not yet implemented
            self.backend_registry.register("interactive", None)
            warnings.warn("InteractiveBackend not yet implemented.", stacklevel=2)

        # Placeholder registrations for future backends
        self.backend_registry.register("animation", None)  # Future: Manim
        self.backend_registry.register("web", None)  # Future: Observable/D3

    def create_chart(self, chart_type: str, backend: str = "static", config: dict[str, Any] | None = None) -> Any:
        """
        Create a chart with the specified type and backend.

        Args:
            chart_type: Type of chart ('trajectory', 'distribution', 'comparison', 'heatmap')
            backend: Rendering backend ('static', 'interactive', 'animation', 'web')
            config: Optional configuration dictionary for the chart

        Returns:
            Rendered chart object (type depends on backend)
        """
        config = config or {}

        # Create chart instance
        chart = self.chart_factory.create_chart(chart_type, self.data_loader, config)

        # Get backend instance
        backend_instance = self.backend_registry.get_backend(backend)

        # Render chart
        return backend_instance.render_chart(chart)

    def list_available_charts(self) -> list[str]:
        """List all available chart types."""
        return self.chart_factory.list_chart_types()

    def list_available_backends(self) -> list[str]:
        """List all available rendering backends."""
        return self.backend_registry.list_backends()

    def get_data_summary(self) -> dict[str, Any]:
        """Get summary statistics of the loaded data."""
        return self.data_loader.get_summary_stats()


class ChartFactory:
    """
    Factory for creating chart instances.

    Manages the registry of available chart types and handles
    dynamic chart creation with configuration.
    """

    def __init__(self):
        self._chart_types: dict[str, type[BaseChart]] = {}
        self._register_default_charts()

    def _register_default_charts(self):
        """Register default chart types."""
        try:
            from .charts.trajectory import TrajectoryChart

            self.register_chart("trajectory", TrajectoryChart)
        except ImportError:
            pass

        try:
            from .charts.distribution import DistributionChart

            self.register_chart("distribution", DistributionChart)
        except ImportError:
            pass

        try:
            from .charts.comparison import ComparisonChart

            self.register_chart("comparison", ComparisonChart)
        except ImportError:
            pass

        try:
            from .charts.heatmap import HeatmapChart

            self.register_chart("heatmap", HeatmapChart)
        except ImportError:
            pass

    def register_chart(self, chart_type: str, chart_class: type[BaseChart]):
        """
        Register a new chart type.

        Args:
            chart_type: Unique identifier for the chart type
            chart_class: Chart class that inherits from BaseChart
        """
        self._chart_types[chart_type] = chart_class

    def create_chart(self, chart_type: str, data_loader: DataLoader, config: dict[str, Any]) -> BaseChart:
        """
        Create a chart instance of the specified type.

        Args:
            chart_type: Type of chart to create
            data_loader: Data loader for accessing simulation data
            config: Configuration dictionary for the chart

        Returns:
            Configured chart instance
        """
        if chart_type not in self._chart_types:
            raise ValueError(f"Unknown chart type: {chart_type}. Available types: {list(self._chart_types.keys())}")

        chart_class = self._chart_types[chart_type]
        return chart_class(data_loader, config)

    def list_chart_types(self) -> list[str]:
        """List all registered chart types."""
        return list(self._chart_types.keys())


class BackendRegistry:
    """
    Registry for managing visualization backends.

    Handles registration and instantiation of different rendering
    backends (static, interactive, animation, web).
    """

    def __init__(self):
        self._backends: dict[str, type[BaseBackend] | None] = {}
        self._instances: dict[str, BaseBackend | None] = {}

    def register(self, backend_name: str, backend_class: type[BaseBackend] | None):
        """
        Register a backend class.

        Args:
            backend_name: Unique identifier for the backend
            backend_class: Backend class or None for placeholder
        """
        self._backends[backend_name] = backend_class
        self._instances[backend_name] = None

    def get_backend(self, backend_name: str) -> BaseBackend:
        """
        Get a backend instance, creating it if necessary.

        Args:
            backend_name: Name of the backend to retrieve

        Returns:
            Backend instance ready for rendering
        """
        if backend_name not in self._backends:
            raise ValueError(f"Unknown backend: {backend_name}. Available backends: {list(self._backends.keys())}")

        backend_class = self._backends[backend_name]
        if backend_class is None:
            raise ValueError(f"Backend '{backend_name}' is not yet implemented")

        # Create instance if it doesn't exist
        if self._instances[backend_name] is None:
            self._instances[backend_name] = backend_class()

        instance = self._instances[backend_name]
        if instance is None:
            raise RuntimeError(f"Failed to create backend instance: {backend_name}")
        return instance

    def list_backends(self) -> list[str]:
        """List all registered backends."""
        return list(self._backends.keys())


class ChartConfig:
    """
    Configuration object for chart creation.

    Provides a typed interface for chart configuration with
    validation and default values.
    """

    def __init__(self, **kwargs):
        """Initialize with configuration parameters."""
        # Default configuration
        self.title: str = kwargs.get("title", "")
        self.figsize: tuple[float, float] = kwargs.get("figsize", (10, 6))
        self.dpi: int = kwargs.get("dpi", 300)
        self.style: str = kwargs.get("style", "academic")
        self.color_palette: str = kwargs.get("color_palette", "viridis")
        self.export_format: str = kwargs.get("export_format", "png")
        self.show_confidence_bands: bool = kwargs.get("show_confidence_bands", True)
        self.add_parameter_controls: bool = kwargs.get("add_parameter_controls", False)

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


# Convenience function for quick chart creation
def create_chart(data_path: str | Path, chart_type: str, backend: str = "static", **config) -> Any:
    """
    Convenience function for quick chart creation.

    Args:
        data_path: Path to simulation data
        chart_type: Type of chart to create
        backend: Rendering backend to use
        **config: Chart configuration parameters

    Returns:
        Rendered chart object
    """
    engine = VisualizationEngine(data_path)
    return engine.create_chart(chart_type, backend, config)
