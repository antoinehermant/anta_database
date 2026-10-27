"""
Cloud distribution of the AntADatabase.

sync() only needs the standard library. publish() is for maintainers and needs
the [publish] extra: pip install "anta_database[publish]"
"""

from typing import TYPE_CHECKING

from anta_database.cloud.sync import default_database_dir, sync

if TYPE_CHECKING:
    from anta_database.cloud.publish import publish


def __getattr__(name):
    if name == "publish":
        from anta_database.cloud.publish import publish

        return publish
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["sync", "publish", "default_database_dir"]
