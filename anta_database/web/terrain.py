"""
Export Bedmap3 terrain for the 3D web viewer.

Downsamples the 500 m Bedmap3 grids by block averaging and writes them as raw
little-endian int16 heightmaps (gzipped), plus a terrain.json describing them:

    python -m anta_database.web.terrain /path/to/bedmap3.nc -o web/public/data/terrain

Layers:
    bed       bedrock and seafloor elevation (topg)
    surface   ice surface elevation (usurf), only where there is ice
    mask      dominant Bedmap3 class per block (0 ocean, 1 grounded, 2 transiently
              grounded shelf, 3 floating shelf, 4 rock), uint8

Rows are written with y ascending (first row is the southernmost y), x ascending.
"""

import argparse
import gzip
import json
import os
from typing import Dict, Iterable

import numpy as np
import xarray as xr

NODATA = -32768
DEFAULT_RESOLUTIONS_KM = (2, 4, 8)
# A block is only valid if at least this fraction of its cells has data
MIN_VALID_FRACTION = 0.5


def _blocks(a: np.ndarray, f: int) -> np.ndarray:
    """View of a 2D array as (ny, nx, f*f) blocks, trimming the edges."""
    ny, nx = a.shape[0] // f, a.shape[1] // f
    a = a[: ny * f, : nx * f]
    return a.reshape(ny, f, nx, f).swapaxes(1, 2).reshape(ny, nx, f * f)


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """NaN-aware block mean, NaN where less than MIN_VALID_FRACTION of the block is valid."""
    b = _blocks(a, f)
    valid = np.isfinite(b)
    count = valid.sum(axis=-1)
    total = np.where(valid, b, 0).sum(axis=-1, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[count < MIN_VALID_FRACTION * f * f] = np.nan
    return mean.astype(np.float32)


def block_mode(mask: np.ndarray, f: int, n_classes: int = 5) -> np.ndarray:
    """Most frequent class per block (classes 0..n_classes-1)."""
    b = _blocks(mask, f)
    counts = np.stack([(b == c).sum(axis=-1) for c in range(n_classes)], axis=-1)
    return counts.argmax(axis=-1).astype(np.uint8)


def _to_int16(a: np.ndarray) -> np.ndarray:
    out = np.full(a.shape, NODATA, dtype="<i2")
    valid = np.isfinite(a)
    out[valid] = np.clip(np.rint(a[valid]), NODATA + 1, 32767)
    return out


def _write_gz(array: np.ndarray, path: str) -> None:
    tmp = f"{path}.part"
    with gzip.open(tmp, "wb", compresslevel=9) as f:
        f.write(np.ascontiguousarray(array).tobytes())
    os.replace(tmp, path)


def _load(ds: xr.Dataset, name: str) -> np.ndarray:
    """Variable as float32 with NaN for missing values, y ascending."""
    a = ds[name].values.astype(np.float32)
    return a[::-1] if ds.y.values[0] > ds.y.values[-1] else a


def export_terrain(
    bedmap_path: str,
    output_dir: str,
    resolutions_km: Iterable[int] = DEFAULT_RESOLUTIONS_KM,
) -> Dict:
    """
    Write the downsampled Bedmap3 layers and terrain.json to output_dir.

    Returns:
        The terrain.json content.
    """
    os.makedirs(output_dir, exist_ok=True)
    ds = xr.open_dataset(bedmap_path)
    x = ds.x.values.astype(np.float64)
    y = np.sort(ds.y.values.astype(np.float64))
    dx = float(abs(x[1] - x[0]))

    print("Loading Bedmap3...")
    thk = _load(ds, "thk")
    layers = {"bed": _load(ds, "topg")}
    surface = _load(ds, "usurf")
    surface[~(thk > 0)] = np.nan
    layers["surface"] = surface
    del thk
    mask = np.nan_to_num(_load(ds, "mask"), nan=0).astype(np.uint8)

    meta = {
        "crs": "EPSG:3031",
        "units": "m",
        "nodata": NODATA,
        "dtype": "int16",
        "row_order": "y ascending",
        "source": ds.attrs.get("title", "BEDMAP3"),
        "citation": ds.attrs.get("citation", "Pritchard et al., BEDMAP3"),
        "doi": ds.attrs.get("metadata_link", "https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2"),
        "licence": "CC-BY 4.0",
        "mask_classes": {
            "0": "ocean",
            "1": "grounded ice",
            "2": "transiently grounded ice shelf",
            "3": "floating ice shelf",
            "4": "rock",
        },
        "resolutions": {},
    }

    for res in sorted(resolutions_km):
        f = int(round(res * 1000 / dx))
        ny, nx = mask.shape[0] // f, mask.shape[1] // f
        level = {
            "nx": nx,
            "ny": ny,
            "dx": dx * f,
            # Centre of the first block
            "x0": float(x[0] + (f - 1) * dx / 2),
            "y0": float(y[0] + (f - 1) * dx / 2),
            "layers": {},
        }
        for name, data in layers.items():
            mean = block_mean(data, f)
            file = f"{name}_{res}km.bin.gz"
            _write_gz(_to_int16(mean), os.path.join(output_dir, file))
            level["layers"][name] = {
                "file": file,
                "min": float(np.nanmin(mean)),
                "max": float(np.nanmax(mean)),
            }
        file = f"mask_{res}km.bin.gz"
        _write_gz(block_mode(mask, f), os.path.join(output_dir, file))
        level["layers"]["mask"] = {"file": file, "dtype": "uint8"}
        meta["resolutions"][f"{res}km"] = level
        print(f"{res} km: {nx} x {ny}")

    with open(os.path.join(output_dir, "terrain.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export Bedmap3 terrain for the 3D web viewer.")
    parser.add_argument("bedmap", help="Path to the 500 m bedmap3.nc")
    parser.add_argument("-o", "--output", default="web/public/data/terrain", help="Output directory")
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS_KM),
        help="Resolutions in km (multiples of 0.5 km)",
    )
    args = parser.parse_args(argv)
    export_terrain(args.bedmap, args.output, args.resolutions)


if __name__ == "__main__":
    main()
