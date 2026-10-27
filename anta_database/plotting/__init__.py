"""
Plotting module

Contains functions for data visualization and plotting.
PISMPlotting requires the optional dependencies: pip install "anta_database[pism]"
"""

from typing import TYPE_CHECKING

from anta_database.plotting.plotting import Plotting

if TYPE_CHECKING:
    from anta_database.plotting.plotting_pism import PISMPlotting


def __getattr__(name):
    if name == "PISMPlotting":
        from anta_database.plotting.plotting_pism import PISMPlotting

        return PISMPlotting
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Plotting", "PISMPlotting"]
