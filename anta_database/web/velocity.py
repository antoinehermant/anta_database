"""
Export MEaSUREs ice velocities for the 3D web viewer.

Averages the 450 m MEaSUREs InSAR-based Antarctica ice velocity map (Rignot et
al., NSIDC-0484) in square blocks and writes raw little-endian int16 grids
(m/a, gzipped), plus a velocity.json describing them:

    python -m anta_database.web.velocity antarctica_ice_velocity_450m_v2.nc -o web/public/data/velocity

Grids:
    speed_{res}km      mean speed (m/a), for the colours of the velocity layer
    vx_{res}km, vy_{res}km   mean velocity components (m/a), for the flow lines

Rows are written with y ascending (first row is the southernmost y), x ascending.
"""

import argparse
import gzip
import json
import os
from typing import Dict

import numpy as np

NODATA = -32768
DEFAULT_SPEED_KM = 2
DEFAULT_VECTOR_KM = 4
# A block is only valid if at least this fraction of its cells has data
MIN_VALID_FRACTION = 0.25
CITATION = (
    "Rignot, E., J. Mouginot, and B. Scheuchl. 2017. MEaSUREs InSAR-Based Antarctica Ice "
    "Velocity Map, Version 2. NASA National Snow and Ice Data Center Distributed Active "
    "Archive Center. https://doi.org/10.5067/D7GK8F5J8M8R"
)
DOI = "https://doi.org/10.5067/D7GK8F5J8M8R"


def _write_gz(array: np.ndarray, path: str) -> None:
    tmp = f"{path}.part"
    with gzip.open(tmp, "wb", compresslevel=9) as f:
        f.write(np.ascontiguousarray(array).tobytes())
    os.replace(tmp, path)


def _to_int16(a: np.ndarray) -> np.ndarray:
    out = np.full(a.shape, NODATA, dtype="<i2")
    valid = np.isfinite(a)
    out[valid] = np.clip(np.rint(a[valid]), NODATA + 1, 32767)
    return out


def _grid(x: np.ndarray, y: np.ndarray, res: float) -> Dict:
    """Blocks of res (m) covering the cells of x and y, starting at their first edge."""
    d = abs(float(x[1] - x[0]))
    x_edge = float(x.min()) - d / 2
    y_edge = float(y.min()) - d / 2
    nx = int(np.ceil((float(x.max()) + d / 2 - x_edge) / res))
    ny = int(np.ceil((float(y.max()) + d / 2 - y_edge) / res))
    return {
        "nx": nx,
        "ny": ny,
        "dx": res,
        # Centre of the first block
        "x0": x_edge + res / 2,
        "y0": y_edge + res / 2,
        "cells_per_block": (res / d) ** 2,
    }


def block_means(vx, vy, x: np.ndarray, y: np.ndarray, grid: Dict, rows: int = 512):
    """
    Mean vx, vy and speed of the cells in each block of grid, NaN where less
    than MIN_VALID_FRACTION of the block has data.

    vx and vy are (len(y), len(x)) arrays (e.g. memory mapped), 0 or NaN where
    there is no data, read a few rows at a time.
    """
    nx, ny, res = grid["nx"], grid["ny"], grid["dx"]
    columns = np.floor((x - (grid["x0"] - res / 2)) / res).astype(np.int64)
    sums = np.zeros((4, ny * nx))  # vx, vy, speed, count
    for start in range(0, len(y), rows):
        block_x = np.asarray(vx[start : start + rows], dtype=np.float64)
        block_y = np.asarray(vy[start : start + rows], dtype=np.float64)
        valid = np.isfinite(block_x) & np.isfinite(block_y) & ((block_x != 0) | (block_y != 0))
        r = np.floor((y[start : start + rows] - (grid["y0"] - res / 2)) / res).astype(np.int64)
        cells = (r[:, None] * nx + columns[None, :])[valid]
        u, v = block_x[valid], block_y[valid]
        for k, w in enumerate((u, v, np.hypot(u, v), None)):
            sums[k] += np.bincount(cells, weights=w, minlength=ny * nx)
    count = sums[3]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums[:3] / count
    means[:, count < MIN_VALID_FRACTION * grid["cells_per_block"]] = np.nan
    return [m.reshape(ny, nx) for m in means]


def export_velocity(
    vx,
    vy,
    x: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    speed_km: float = DEFAULT_SPEED_KM,
    vector_km: float = DEFAULT_VECTOR_KM,
) -> Dict:
    """
    Write the speed and velocity grids and velocity.json to output_dir.

    Args:
        vx, vy: Velocity components (m/a) on the (y, x) grid, 0 or NaN without data.
        x, y: Cell centres (m, EPSG:3031), in any order.
        output_dir: Output directory, e.g. web/public/data/velocity.
        speed_km: Resolution of the speed grid (colours).
        vector_km: Resolution of the vx and vy grids (flow lines).

    Returns:
        The velocity.json content.
    """
    os.makedirs(output_dir, exist_ok=True)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    meta = {
        "crs": "EPSG:3031",
        "units": "m/a",
        "nodata": NODATA,
        "dtype": "int16",
        "row_order": "y ascending",
        "source": "MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2 (NSIDC-0484)",
        "citation": CITATION,
        "doi": DOI,
        "grids": {},
    }
    for name, res in (("speed", speed_km), ("vectors", vector_km)):
        grid = _grid(x, y, res * 1000)
        mean_vx, mean_vy, speed = block_means(vx, vy, x, y, grid)
        layers = {"speed": speed} if name == "speed" else {"vx": mean_vx, "vy": mean_vy}
        level = {k: grid[k] for k in ("nx", "ny", "dx", "x0", "y0")}
        level["layers"] = {}
        for layer, data in layers.items():
            file = f"{layer}_{res:g}km.bin.gz"
            _write_gz(_to_int16(data), os.path.join(output_dir, file))
            level["layers"][layer] = {"file": file, "max": float(np.nanmax(np.abs(data)))}
        meta["grids"][name] = level
        print(f"{name}: {res:g} km, {grid['nx']} x {grid['ny']}")

    with open(os.path.join(output_dir, "velocity.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def export_velocity_file(path: str, output_dir: str, **kwargs) -> Dict:
    """export_velocity from a MEaSUREs netCDF file (variables VX, VY, x, y)."""
    import xarray as xr

    with xr.open_dataset(path, mask_and_scale=False) as ds:
        return export_velocity(ds.VX.values, ds.VY.values, ds.x.values, ds.y.values, output_dir, **kwargs)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export MEaSUREs ice velocities for the 3D web viewer.")
    parser.add_argument("velocity", help="MEaSUREs velocity netCDF, e.g. antarctica_ice_velocity_450m_v2.nc")
    parser.add_argument("-o", "--output", default="web/public/data/velocity", help="Output directory")
    parser.add_argument("--speed-km", type=float, default=DEFAULT_SPEED_KM, help="Resolution of the speed grid (km)")
    parser.add_argument("--vector-km", type=float, default=DEFAULT_VECTOR_KM, help="Resolution of the vx, vy grids (km)")
    args = parser.parse_args(argv)
    export_velocity_file(args.velocity, args.output, speed_km=args.speed_km, vector_km=args.vector_km)


if __name__ == "__main__":
    main()
