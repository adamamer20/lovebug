"""
Vectorized Polars-based Layer 2 implementation.

High-performance, mesa-frames native cultural transmission system that replaces
sequential per-agent loops with bulk DataFrame operations, targeting O(n log n)
complexity instead of O(n²).
"""

from __future__ import annotations

from .cultural_layer import VectorizedCulturalLayer
from .learning_algorithms import (
    CulturalInnovationEngine,
    HorizontalTransmissionEngine,
    LearningEligibilityComputer,
    MemoryDecayEngine,
    ObliqueTransmissionEngine,
)
from .network import NetworkTopology, VectorizedSocialNetwork

__all__ = [
    "NetworkTopology",
    "VectorizedSocialNetwork",
    "VectorizedCulturalLayer",
    "LearningEligibilityComputer",
    "ObliqueTransmissionEngine",
    "HorizontalTransmissionEngine",
    "CulturalInnovationEngine",
    "MemoryDecayEngine",
]
