"""How the files of the database are written: netCDF-4, compact, and quick to read in parts.

Values are float32: depths change by at most 0.1 mm, far below what radar can resolve. Coordinates
and distances stay float64. The shuffle filter and larger chunks make the files much smaller than
gzip alone, and let one IRH be read without the others, also over HTTP.
"""

import os

import numpy as np
import xarray as xr

CHUNK_POINTS = 50_000
FLOAT32 = {"IRH_DEPTH", "ICE_THK", "SURF_ELEV", "BED_ELEV", "BASAL_UNIT", "IRH_FRAC_DEPTH", "BEDMAP_POINT_DIST"}
COMPRESSION = {"zlib": True, "complevel": 4, "shuffle": True}

# Where each value of ICE_THK, BED_ELEV and SURF_ELEV comes from (<VAR>_SOURCE, int8, missing where
# the variable has none): written by the compilation (0, 4) and by the Bedmap fill (1-3,
# anta_database.bedmap.reconnect)
SOURCE_VARIABLES = ["ICE_THK", "BED_ELEV", "SURF_ELEV"]
SOURCES = {
    "published_dataset": 0,
    "bedmap1_survey_point": 1,
    "bedmap2_survey_point": 2,
    "bedmap3_survey_point": 3,
    "computed_from_published": 4,
}
SOURCE_FILL = -1


def source_attrs(var: str) -> dict:
    """The attributes of <var>_SOURCE."""
    return {
        "long_name": f"source of {var}",
        "flag_values": np.array(list(SOURCES.values()), dtype=np.int8),
        "flag_meanings": " ".join(SOURCES),
        "description": f"Where each value of {var} comes from: 0 the published dataset; 1, 2, 3 the nearest "
        "point of the Bedmap1, Bedmap2 or Bedmap3 survey line of the same radar traces (attribute "
        "reconnection, distance in BEDMAP_POINT_DIST), never the gridded Bedmap products; 4 computed from the "
        f"two other variables of the published dataset. Missing where {var} has no value.",
    }


def encoding(ds: xr.Dataset) -> dict:
    """Encoding of each variable of a line."""
    result = {}
    for name in ds.variables:
        var = ds[name]
        if "point" not in var.dims or var.dtype.kind not in "fiu":
            continue
        chunks = tuple(min(CHUNK_POINTS, ds.sizes["point"]) if d == "point" else 1 for d in var.dims)
        result[name] = {**COMPRESSION, "chunksizes": chunks}
        if name in FLOAT32:
            result[name]["dtype"] = "float32"
        elif name.endswith("_SOURCE"):
            result[name].update(dtype="int8", _FillValue=SOURCE_FILL)
        elif name == "IRH_NUM":
            result[name]["dtype"] = "float32" if np.isnan(var.values.astype(float)).any() else "int16"
        elif name == "point":
            result[name]["dtype"] = "int32"
    return result


def write(ds: xr.Dataset, path: str) -> None:
    """Write a line to a new file, then replace: a failure never leaves a half-written file."""
    ds = ds.copy()
    for name in ds.variables:
        ds[name].encoding = {}
    tmp = f"{path}.tmp"
    try:
        ds.to_netcdf(tmp, engine="h5netcdf", mode="w", encoding=encoding(ds))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
