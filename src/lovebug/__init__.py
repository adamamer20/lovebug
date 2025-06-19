"""
LoveBug: An agent‑based model (ABM) of sexual selection and mating‑preference co‑evolution, built with Mesa‑Frames + Polars.
"""

from importlib import metadata as _metadata

from .lande_kirkpatrick import LandeKirkpatrickParams

# Import vectorized components for advanced users
from .layer2 import CulturalLayer
from .layer2.config import Layer2Config

# Import core configuration classes
from .layer_activation import LayerActivationConfig

# Import the enhanced LoveModel as the primary interface
from .unified_mesa_model import LoveAgents, LoveModel

__all__ = [
    "__version__",
    # Enhanced primary interface
    "LoveModel",
    "LoveAgents",
    # Configuration classes
    "LayerActivationConfig",
    "LandeKirkpatrickParams",
    "Layer2Config",
    # Vectorized components
    "CulturalLayer",
]

try:
    __version__: str = _metadata.version(__name__)
except _metadata.PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0+dev"

# -- Development-only runtime type-checking ------------------------------
import os

if os.getenv("DEV_TYPECHECK", "0") == "1":
    try:
        from typeguard.importhook import install_import_hook

        # Check *this* package (children included) on import
        install_import_hook(__name__)
    except ImportError:
        # typeguard not available, skip type checking
        pass
# ------------------------------------------------------------------------
