"""
LoveBug: An agent‑based model (ABM) of sexual selection and mating‑preference co‑evolution, built with Mesa‑Frames + Polars.
"""

from importlib import metadata as _metadata

__all__ = ["__version__"]

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
