"""
Database module

Contains classes for local and cloud database access.
"""

from .database import Database
from .cloud_database import CloudDatabase

__all__ = ["Database", "CloudDatabase"]
