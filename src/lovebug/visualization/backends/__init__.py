"""
Visualization Backend System

This package provides different rendering backends for LoveBug visualizations:
- StaticBackend: Matplotlib-based publication quality outputs
- InteractiveBackend: Plotly-based interactive visualizations
- AnimationBackend: Manim-based mathematical animations (future)
- WebBackend: Observable/D3-based web components (future)
"""

from .base import BaseBackend

__all__ = ["BaseBackend"]
