from .database import *
from .indexing import *
from .compiling import *
from .plotting import *

__all__ = ["Database", "CloudDatabase"]


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
