"""How the files of the database are written: netCDF-4, compact, and quick to read in parts.

Values are float32: depths change by at most 0.1 mm, far below what radar can resolve. Coordinates
and distances stay float64. The shuffle filter and larger chunks make the files much smaller than
gzip alone, and let one IRH be read without the others, also over HTTP.
"""

import os

import numpy as np
import xarray as xr

CHUNK_POINTS = 50_000
FLOAT32 = {"IRH_DEPTH", "ICE_THK", "SURF_ELEV", "BED_ELEV", "BASAL_UNIT", "IRH_FRAC_DEPTH", "BEDMAP_DIST"}
COMPRESSION = {"zlib": True, "complevel": 4, "shuffle": True}


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
