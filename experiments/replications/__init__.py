"""
Empirical Replication Experiments

This module contains replications of key empirical experiments from the
sexual selection and cultural evolution literature. These experiments
validate that the LoveBug model can reproduce qualitative patterns from
real biological systems.

Available Replications:
- DugatkinReplication: Mate-choice copying experiment
- WitteReplication: Cultural transmission persistence experiment
- RoddReplication: Sensory bias evolution experiment
"""

from .dugatkin_replication import DugatkinReplication
from .rodd_replication import RoddReplication
from .witte_replication import WitteReplication

__all__ = ["DugatkinReplication", "WitteReplication", "RoddReplication"]
