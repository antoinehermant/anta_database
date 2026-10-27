"""
Compiling module

Contains classes and functions for data compilation and conversion.
Requires the optional dependencies: pip install "anta_database[compile]"
"""

try:
    from anta_database.compiling.zarr_converter_simple import FlightLineZarrConverter
    from anta_database.compiling.compile import CompileDatabase
except ImportError as e:
    raise ImportError(
        f"{e}. The compiling tools need optional dependencies, "
        "install them with: pip install \"anta_database[compile]\""
    ) from e

__all__ = ["FlightLineZarrConverter", "CompileDatabase"]
