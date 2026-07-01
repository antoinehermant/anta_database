"""
Anta Database Package

Main package for isochrone database access and processing.
"""

# Database modules
from .database.database import Database
from .database.cloud_database import CloudDatabase

# Compiling modules
from .compiling.zarr_converter_simple import FlightLineZarrConverter
from .compiling.compile import compile_database

# Indexing modules
from .indexing.index_database import index_database

# Plotting modules
from .plotting.plotting import plot_database

__all__ = [
    # Database classes
    "Database",
    "CloudDatabase",
    # Compiling classes
    "FlightLineZarrConverter",
    "compile_database",
    # Indexing functions
    "index_database",
    # Plotting functions
    "plot_database",
]


def get_database(cloud=False, s3_bucket="anta-database", **kwargs):
    """
    Factory function to get the appropriate database class.

    Args:
        cloud: If True, return CloudDatabase
        s3_bucket: S3 bucket name for cloud database
        **kwargs: Additional arguments for CloudDatabase

    Returns:
        Database instance
    """
    if cloud:
        return CloudDatabase(s3_bucket=s3_bucket, **kwargs)
    else:
        return Database()
