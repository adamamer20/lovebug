"""
Clean Experiment Framework for Evolutionary Simulations

This module provides a type-safe, null-pollution-free experiment framework
for running genetic and cultural evolution simulations.
"""

from .collectors import (
    CulturalResultCollector,
    ExperimentStorage,
    GeneticResultCollector,
    IntegratedResultCollector,
)
from .models import (
    CommonParameters,
    CulturalExperimentResult,
    ExperimentMetadata,
    GeneticExperimentResult,
    IntegratedExperimentResult,
)
from .runner import CleanExperimentRunner, UnifiedConfig, run_experiments

__all__ = [
    # Data models
    "ExperimentMetadata",
    "CommonParameters",
    "GeneticExperimentResult",
    "CulturalExperimentResult",
    "IntegratedExperimentResult",
    # Collectors
    "GeneticResultCollector",
    "CulturalResultCollector",
    "IntegratedResultCollector",
    "ExperimentStorage",
    # Runner
    "CleanExperimentRunner",
    "UnifiedConfig",
    "run_experiments",
]
