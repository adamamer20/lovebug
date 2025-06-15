"""
Base Backend for LoveBug Visualizations

This module defines the abstract base class for all visualization backends,
providing a consistent interface for rendering charts across different
output formats and technologies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..charts.base import BaseChart


class BaseBackend(ABC):
    """
    Abstract base class for visualization backends.

    All backends must implement the render_chart method to convert
    chart objects into their specific output format (static images,
    interactive HTML, animations, web components, etc.).
    """

    def __init__(self):
        """Initialize the backend with default configuration."""
        self.config: dict[str, Any] = {}
        self._setup_backend()

    @abstractmethod
    def _setup_backend(self) -> None:
        """
        Setup backend-specific configuration.

        This method is called during initialization and should handle
        any backend-specific setup like importing libraries, setting
        default styles, configuring output formats, etc.
        """
        pass

    @abstractmethod
    def render_chart(self, chart: BaseChart) -> Any:
        """
        Render a chart using this backend.

        Args:
            chart: Chart instance to render

        Returns:
            Rendered output (type depends on backend)
        """
        pass

    def configure(self, **kwargs) -> None:
        """
        Configure backend-specific settings.

        Args:
            **kwargs: Configuration parameters specific to this backend
        """
        self.config.update(kwargs)

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration parameter.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def supports_interactivity(self) -> bool:
        """
        Check if this backend supports interactive features.

        Returns:
            True if backend supports interactivity, False otherwise
        """
        return False

    def supports_animation(self) -> bool:
        """
        Check if this backend supports animations.

        Returns:
            True if backend supports animations, False otherwise
        """
        return False

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported output formats.

        Returns:
            List of supported file formats/types
        """
        return []

    def save_output(self, rendered_chart: Any, filepath: str, format: str | None = None, **kwargs) -> None:
        """
        Save rendered chart to file.

        Args:
            rendered_chart: The rendered chart object
            filepath: Output file path
            format: Output format (if None, infer from filepath)
            **kwargs: Additional save parameters
        """
        raise NotImplementedError("Subclasses must implement save_output")


class RenderingError(Exception):
    """Exception raised when chart rendering fails."""

    pass


class BackendNotAvailableError(Exception):
    """Exception raised when required backend dependencies are not available."""

    pass
