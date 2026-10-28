"""
Export isochrones (IRHs) as 3D polylines for the web viewer.

For every flight line (h5 file) and IRH, the elevation of the IRH is

    IRH_ELEV = SURF_ELEV - IRH_DEPTH

using the SURF_ELEV of the dataset when it has one, and otherwise the Bedmap3
surface elevation at the point (500 m grid). Each (flight line, IRH) is a
feature with its metadata; its points are split into runs wherever the IRH is
missing or consecutive points are far apart, and averaged along-track.

    python -m anta_database.web.irh /path/to/AntADatabase --bedmap bedmap3.nc \\
        --datasets Sanderson_2024 -o web/public/data/irh

Output:
    irh.json                 features (metadata) and per-dataset file names
    {dataset}.bin.gz         float32 vertices (x km, y km, elevation m), packed per run
    {dataset}_runs.bin.gz    uint32 (feature id, first vertex, vertex count) per run
"""

import argparse
import glob
import gzip
import json
import os
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

DEFAULT_SPACING = 1000.0  # m, along-track averaging
DEFAULT_MAX_GAP = 5000.0  # m, larger gaps between points break the line


class Bedmap3Surface:
    """Bedmap3 surface elevation, sampled at the nearest 500 m cell."""

    def __init__(self, path: str):
        ds = xr.open_dataset(path)
        self.x0 = float(ds.x[0])
        self.y0 = float(ds.y[0])
        self.dx = float(ds.x[1] - ds.x[0])
        self.dy = float(ds.y[1] - ds.y[0])
        self.usurf = ds.usurf.values.astype(np.float32)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        finite = np.isfinite(x) & np.isfinite(y)
        i = np.full(x.shape, -1, dtype=np.int64)
        j = np.full(x.shape, -1, dtype=np.int64)
        i[finite] = np.rint((x[finite] - self.x0) / self.dx)
        j[finite] = np.rint((y[finite] - self.y0) / self.dy)
        ny, nx = self.usurf.shape
        inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        out = np.full(x.shape, np.nan, dtype=np.float32)
        out[inside] = self.usurf[j[inside], i[inside]]
        return out


def _runs(valid: np.ndarray, step: np.ndarray, max_gap: float) -> List[slice]:
    """Slices of consecutive valid points without gaps larger than max_gap."""
    runs = []
    start = None
    for k in range(len(valid)):
        if not valid[k]:
            if start is not None:
                runs.append(slice(start, k))
            start = None
            continue
        if start is not None and step[k] > max_gap:
            runs.append(slice(start, k))
            start = None
        if start is None:
            start = k
    if start is not None:
        runs.append(slice(start, len(valid)))
    return [r for r in runs if r.stop - r.start >= 2]


def _average(x: np.ndarray, y: np.ndarray, z: np.ndarray, spacing: float) -> np.ndarray:
    """
    Mean of the points in consecutive along-track bins of length spacing, as (n, 3).

    Runs shorter than one bin keep their two end points. Points sparser than
    spacing are kept as they are.
    """
    points = np.column_stack([x, y, z])
    if spacing <= 0 or len(x) <= 2:
        return points
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    bins = np.floor(dist / spacing).astype(np.int64)
    edges = np.flatnonzero(np.diff(bins)) + 1
    if len(edges) == 0:
        return points[[0, -1]]
    counts = np.diff(np.concatenate([[0], edges, [len(x)]]))
    return np.add.reduceat(points, np.concatenate([[0], edges]), axis=0) / counts[:, None]


def _citations(dataset_dir: str) -> Dict[int, Dict]:
    """Metadata of each IRH from raw_md.json, by IRH number."""
    path = os.path.join(dataset_dir, "raw_md.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        md = json.load(f)
    out = {}
    for entry in md if isinstance(md, list) else [md]:
        try:
            out[int(entry["IRH"])] = entry
        except (KeyError, TypeError, ValueError):
            pass
    return out


def _scalar(value):
    """JSON-friendly scalar (None for NaN)."""
    if value is None:
        return None
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def export_dataset(
    dataset_dir: str,
    surface: Optional[Bedmap3Surface],
    first_feature_id: int,
    spacing: float = DEFAULT_SPACING,
    max_gap: float = DEFAULT_MAX_GAP,
):
    """Vertices, runs and features of one dataset."""
    dataset = os.path.basename(os.path.normpath(dataset_dir))
    irh_md = _citations(dataset_dir)
    features, vertices, runs = [], [], []
    n_vertices = 0

    for path in sorted(glob.glob(os.path.join(dataset_dir, "h5", "*.h5"))):
        with xr.open_dataset(path) as ds:
            if "IRH_DEPTH" not in ds or "IRH" not in ds.dims:
                continue
            x = ds.PSX.values.astype(np.float64)
            y = ds.PSY.values.astype(np.float64)
            if "SURF_ELEV" in ds:
                surf = ds.SURF_ELEV.values.astype(np.float64)
                surface_source = "SURF_ELEV"
            elif surface is not None:
                surf = surface(x, y).astype(np.float64)
                surface_source = "Bedmap3"
            else:
                continue
            step = np.concatenate([[0.0], np.hypot(np.diff(x), np.diff(y))])
            flight_id = ds.attrs.get("flight ID", os.path.splitext(os.path.basename(path))[0])

            for irh in ds.IRH.values:
                sel = ds.sel(IRH=irh)
                depth = sel.IRH_DEPTH.values.astype(np.float64)
                elev = surf - depth
                valid = np.isfinite(elev) & np.isfinite(x) & np.isfinite(y)
                feature_runs = _runs(valid, step, max_gap)
                if not feature_runs:
                    continue

                age = _scalar(sel.IRH_AGE.values) if "IRH_AGE" in sel.coords or "IRH_AGE" in sel else None
                age_unc = _scalar(sel.IRH_AGE_UNC.values) if "IRH_AGE_UNC" in sel else None
                md = irh_md.get(int(irh), {})
                feature_id = first_feature_id + len(features)
                features.append(
                    {
                        "dataset": dataset,
                        "flight_id": flight_id,
                        "irh": int(irh),
                        "age_ka": None if age is None else round(age / 1000, 2),
                        "age_unc_ka": None if age_unc is None else round(age_unc / 1000, 2),
                        "citation": md.get("citation"),
                        "doi_dataset": md.get("DOI dataset", ds.attrs.get("DOI dataset")),
                        "doi_publication": md.get("DOI publication", ds.attrs.get("DOI publication")),
                        "institute": ds.attrs.get("institute"),
                        "surface": surface_source,
                    }
                )
                for run in feature_runs:
                    v = _average(x[run], y[run], elev[run], spacing)
                    v[:, :2] /= 1000  # km
                    v = v.astype(np.float32)
                    vertices.append(v)
                    runs.append((feature_id, n_vertices, len(v)))
                    n_vertices += len(v)

    vertices = np.concatenate(vertices) if vertices else np.zeros((0, 3), np.float32)
    runs = np.asarray(runs, dtype=np.uint32).reshape(-1, 3)
    return vertices, runs, features


def _write_gz(array: np.ndarray, path: str) -> None:
    tmp = f"{path}.part"
    with gzip.open(tmp, "wb", compresslevel=9) as f:
        f.write(np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes())
    os.replace(tmp, path)


def export_irh(
    database_dir: str,
    output_dir: str,
    datasets: Optional[List[str]] = None,
    bedmap_path: Optional[str] = None,
    spacing: float = DEFAULT_SPACING,
    max_gap: float = DEFAULT_MAX_GAP,
) -> Dict:
    """
    Export the IRHs of a database directory to output_dir.

    Args:
        database_dir: Database directory (with one directory per dataset).
        output_dir: Output directory, e.g. web/public/data/irh.
        datasets: Names of the datasets to export. Defaults to all.
        bedmap_path: Bedmap3 netCDF (500 m), for datasets without SURF_ELEV.
            Without it, these datasets are skipped.
        spacing: Along-track averaging distance (m), e.g. the resolution of the terrain.
        max_gap: Distance (m) between consecutive points above which the line is broken.

    Returns:
        The irh.json content.
    """
    os.makedirs(output_dir, exist_ok=True)
    if datasets is None:
        datasets = sorted(
            d for d in os.listdir(database_dir) if os.path.isdir(os.path.join(database_dir, d, "h5"))
        )
    surface = Bedmap3Surface(bedmap_path) if bedmap_path else None

    index = {"format": 1, "datasets": {}, "features": []}
    for dataset in datasets:
        vertices, runs, features = export_dataset(
            os.path.join(database_dir, dataset), surface, len(index["features"]), spacing, max_gap
        )
        if not features:
            print(f"{dataset}: no IRH exported")
            continue
        _write_gz(vertices, os.path.join(output_dir, f"{dataset}.bin.gz"))
        _write_gz(runs, os.path.join(output_dir, f"{dataset}_runs.bin.gz"))
        index["datasets"][dataset] = {
            "vertices": f"{dataset}.bin.gz",
            "runs": f"{dataset}_runs.bin.gz",
            "n_vertices": int(len(vertices)),
            "n_runs": int(len(runs)),
        }
        index["features"].extend(features)
        print(f"{dataset}: {len(features)} lines, {len(runs)} runs, {len(vertices)} vertices")

    with open(os.path.join(output_dir, "irh.json"), "w") as f:
        json.dump(index, f, indent=0)
    return index


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export IRHs as 3D lines for the web viewer.")
    parser.add_argument("database_dir", help="AntADatabase directory")
    parser.add_argument("-o", "--output", default="web/public/data/irh", help="Output directory")
    parser.add_argument("--datasets", nargs="+", help="Datasets to export (default: all)")
    parser.add_argument("--bedmap", help="Bedmap3 netCDF, for datasets without SURF_ELEV")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING, help="Along-track averaging distance (m)")
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP, help="Gap that breaks a line (m)")
    args = parser.parse_args(argv)
    export_irh(args.database_dir, args.output, args.datasets, args.bedmap, args.spacing, args.max_gap)


if __name__ == "__main__":
    main()
