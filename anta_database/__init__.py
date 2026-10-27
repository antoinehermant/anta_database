"""
Anta Database Package

Main package for isochrone database access and processing.

The query, indexing and plotting tools only need the core dependencies.
Compiling and PISM plotting need optional extras and are imported lazily:
    pip install "anta_database[compile]"
    pip install "anta_database[pism]"
"""

from typing import TYPE_CHECKING

# Database modules
from anta_database.database.database import Database, MetadataResult

# Indexing modules
from anta_database.indexing.index_database import IndexDatabase

# Plotting modules
from anta_database.plotting.plotting import Plotting

if TYPE_CHECKING:
    from anta_database.compiling.zarr_converter_simple import FlightLineZarrConverter
    from anta_database.compiling.compile import CompileDatabase
    from anta_database.plotting.plotting_pism import PISMPlotting

# Optional classes, imported on first access
_LAZY_IMPORTS = {
    "FlightLineZarrConverter": "anta_database.compiling",
    "CompileDatabase": "anta_database.compiling",
    "PISMPlotting": "anta_database.plotting.plotting_pism",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY_IMPORTS))


__all__ = [
    # Database classes
    "Database",
    "MetadataResult",
    # Compiling classes
    "FlightLineZarrConverter",
    "CompileDatabase",
    # Indexing functions
    "IndexDatabase",
    # Plotting functions
    "Plotting",
    "PISMPlotting",
]
