"""
LoveBug Post-Hoc Visualization System

A comprehensive visualization architecture for sexual selection research,
designed for post-hoc analysis of agent-based model simulations.

This module provides:
- Data collection and storage systems
- Modular chart types for sexual selection research
- Multiple rendering backends (static, interactive, animation, web)
- Integration with Marimo notebooks and academic workflows

Example usage:
    from lovebug.visualization import VisualizationEngine

    # Load simulation data
    engine = VisualizationEngine('data/simulation_results.parquet')

    # Create publication-quality trajectory plot
    fig = engine.create_chart(
        chart_type='trajectory',
        backend='static',
        config={'title': 'Trait-Preference Coevolution'}
    )
"""

from .core import ChartFactory, VisualizationEngine
from .data import DataCollector, DataLoader

__all__ = ["VisualizationEngine", "ChartFactory", "DataCollector", "DataLoader"]

__version__ = "0.1.0"
