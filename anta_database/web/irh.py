"""
Export isochrones (IRHs) as 3D polylines for the web viewer.

For every line (h5 file) and IRH, the elevation of the IRH is

    IRH_ELEV = SURF_ELEV - IRH_DEPTH

using the SURF_ELEV of the dataset when it has one, and otherwise the Bedmap3
surface elevation at the point (500 m grid). The fractional depth is
IRH_DEPTH / ICE_THK, with the Bedmap3 ice thickness where ICE_THK is missing. Each (line, IRH) is a
feature with its metadata; its points are split into runs wherever the IRH is
missing or consecutive points are far apart, and averaged along-track.

Each line is also exported as a "track" at the ice surface, where at
least one IRH is traced, with the number of IRHs traced (IRH_NUM) and the age
of the oldest one at each point.

The basal unit (BASAL_UNIT, depth below the surface, e.g. the top of the echo-free
zone) is exported like an IRH, as features with "layer": "basal unit", no IRH number
and no age.

    python -m anta_database.web.irh /path/to/AntADatabase --bedmap bedmap3.nc \\
        --datasets Sanderson_2024 -o web/public/data/irh

Output:
    irh.json                 features and tracks (metadata), and per-dataset file names
    {dataset}.bin.gz         float32 vertices (x km, y km, elevation m), packed per run
    {dataset}_depth.bin.gz   uint16 depth (m) of each vertex
    {dataset}_fdepth.bin.gz  uint16 fractional depth × 10000 of each vertex (65535: unknown)
    {dataset}_runs.bin.gz    uint32 (feature id, first vertex, vertex count) per run
    {dataset}_tracks.bin.gz  float32 track vertices (x km, y km, surface elevation m,
                             IRH_NUM, oldest age ka or NaN), packed per run
    {dataset}_track_runs.bin.gz  uint32 (track id, first vertex, vertex count) per run
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
FDEPTH_SCALE = 10000  # fractional depths are stored as uint16 fraction × FDEPTH_SCALE
FDEPTH_NODATA = 65535


class Bedmap3Surface:
    """Bedmap3 surface elevation and ice thickness, sampled at the nearest 500 m cell."""

    def __init__(self, path: str, bbox: Optional[tuple] = None):
        """bbox (xmin, xmax, ymin, ymax, in m): only load that part of the grid, e.g. on a
        server with little memory (the whole grid is 1.4 GB)."""
        with xr.open_dataset(path) as ds:
            if bbox is not None:
                xmin, xmax, ymin, ymax = bbox
                i = np.flatnonzero((ds.x.values >= xmin - 1000) & (ds.x.values <= xmax + 1000))
                j = np.flatnonzero((ds.y.values >= ymin - 1000) & (ds.y.values <= ymax + 1000))
                ds = ds.isel(x=slice(i[0], i[-1] + 1), y=slice(j[0], j[-1] + 1))
            self.x0 = float(ds.x[0])
            self.y0 = float(ds.y[0])
            self.dx = float(ds.x[1] - ds.x[0])
            self.dy = float(ds.y[1] - ds.y[0])
            self.usurf = ds.usurf.values.astype(np.float32)
            self.thk = ds.thk.values.astype(np.float32)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Surface elevation (m)."""
        return self._sample(self.usurf, x, y)

    def thickness(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ice thickness (m)."""
        return self._sample(self.thk, x, y)

    def _sample(self, grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        finite = np.isfinite(x) & np.isfinite(y)
        i = np.full(x.shape, -1, dtype=np.int64)
        j = np.full(x.shape, -1, dtype=np.int64)
        i[finite] = np.rint((x[finite] - self.x0) / self.dx)
        j[finite] = np.rint((y[finite] - self.y0) / self.dy)
        ny, nx = grid.shape
        inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        out = np.full(x.shape, np.nan, dtype=np.float32)
        out[inside] = grid[j[inside], i[inside]]
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


def _average(points: np.ndarray, spacing: float, maximum=()) -> np.ndarray:
    """
    Mean of the points (rows of x, y, ...) in consecutive along-track bins of
    length spacing, or their maximum for the columns in maximum.

    Runs shorter than one bin keep their two end points. Points sparser than
    spacing are kept as they are.
    """
    if spacing <= 0 or len(points) <= 2:
        return points
    x, y = points[:, 0], points[:, 1]
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    bins = np.floor(dist / spacing).astype(np.int64)
    edges = np.flatnonzero(np.diff(bins)) + 1
    if len(edges) == 0:
        return points[[0, -1]]
    starts = np.concatenate([[0], edges])
    counts = np.diff(np.append(starts, len(points)))
    out = np.add.reduceat(points, starts, axis=0) / counts[:, None]
    for k in maximum:
        out[:, k] = np.fmax.reduceat(points[:, k], starts)
    return out


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
    """IRH lines (vertices, depths, fractional depths, runs, features) and tracks of one dataset."""
    dataset = os.path.basename(os.path.normpath(dataset_dir))
    irh_md = _citations(dataset_dir)
    features, vertices, depths, fdepths, runs = [], [], [], [], []
    n_vertices = 0
    tracks, track_vertices, track_runs = [], [], []
    n_track_vertices = 0

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
            # Ice thickness: of the dataset, completed with Bedmap3
            thk = ds.ICE_THK.values.astype(np.float64) if "ICE_THK" in ds else np.full(len(x), np.nan)
            thickness_source = "ICE_THK" if "ICE_THK" in ds else "Bedmap3"
            if surface is not None and not np.isfinite(thk).all():
                missing = ~np.isfinite(thk)
                thk[missing] = surface.thickness(x[missing], y[missing])
            thk[thk <= 0] = np.nan
            step = np.concatenate([[0.0], np.hypot(np.diff(x), np.diff(y))])
            flight_id = ds.attrs.get("flight ID", os.path.splitext(os.path.basename(path))[0])
            # IRHs traced at each point, and age of the oldest (ka)
            irh_num = np.zeros(len(x))
            oldest = np.full(len(x), np.nan)
            citation = None

            def add_feature(depth: np.ndarray, layer: Dict, md: Dict) -> None:
                """A (line, layer) feature, and its runs of vertices."""
                nonlocal n_vertices
                elev = surf - depth
                feature_runs = _runs(np.isfinite(elev) & np.isfinite(x) & np.isfinite(y), step, max_gap)
                if not feature_runs:
                    return
                feature_id = first_feature_id + len(features)
                features.append(
                    {
                        "dataset": dataset,
                        "flight_id": flight_id,
                        **layer,
                        "citation": md.get("citation"),
                        "doi_dataset": md.get("DOI dataset", ds.attrs.get("DOI dataset")),
                        "doi_publication": md.get("DOI publication", ds.attrs.get("DOI publication")),
                        "institute": ds.attrs.get("institute"),
                        "surface": surface_source,
                        "thickness": thickness_source,
                    }
                )
                for run in feature_runs:
                    v = _average(np.column_stack([x[run], y[run], elev[run], depth[run], depth[run] / thk[run]]), spacing)
                    v[:, :2] /= 1000  # km
                    vertices.append(v[:, :3].astype(np.float32))
                    depths.append(np.clip(np.rint(v[:, 3]), 0, 65535).astype(np.uint16))
                    fdepth = np.clip(np.rint(v[:, 4] * FDEPTH_SCALE), 0, FDEPTH_NODATA - 1)
                    fdepths.append(np.where(np.isfinite(fdepth), fdepth, FDEPTH_NODATA).astype(np.uint16))
                    runs.append((feature_id, n_vertices, len(v)))
                    n_vertices += len(v)

            for irh in ds.IRH.values:
                sel = ds.sel(IRH=irh)
                depth = sel.IRH_DEPTH.values.astype(np.float64)
                valid = np.isfinite(surf - depth) & np.isfinite(x) & np.isfinite(y)
                age = _scalar(sel.IRH_AGE.values) if "IRH_AGE" in sel.coords or "IRH_AGE" in sel else None
                irh_num += valid
                if age is not None:
                    oldest[valid] = np.fmax(oldest[valid], age / 1000)
                if not valid.any():
                    continue
                age_unc = _scalar(sel.IRH_AGE_UNC.values) if "IRH_AGE_UNC" in sel else None
                md = irh_md.get(int(irh), {})
                citation = citation or md.get("citation")
                add_feature(depth, {
                    "irh": int(irh),
                    "age_ka": None if age is None else round(age / 1000, 2),
                    "age_unc_ka": None if age_unc is None else round(age_unc / 1000, 2),
                }, md)

            # Basal unit: a layer without age (not an IRH, nor in IRH_NUM), same metadata as the IRHs
            if "BASAL_UNIT" in ds and ds.BASAL_UNIT.dims == ("point",):
                md = next(iter(irh_md.values()), {})
                citation = citation or md.get("citation")
                add_feature(
                    ds.BASAL_UNIT.values.astype(np.float64),
                    {"irh": None, "layer": "basal unit", "age_ka": None, "age_unc_ka": None},
                    md,
                )

            # Track at the surface, where IRHs are traced
            on_track = (irh_num > 0) & np.isfinite(surf) & np.isfinite(x) & np.isfinite(y)
            line_runs = _runs(on_track, step, max_gap)
            if not line_runs:
                continue
            track_id = len(tracks)
            tracks.append(
                {
                    "dataset": dataset,
                    "flight_id": flight_id,
                    "citation": citation,
                    "surface": surface_source,
                }
            )
            for run in line_runs:
                points = np.column_stack([x[run], y[run], surf[run], irh_num[run], oldest[run]])
                v = _average(points, spacing, maximum=(3, 4))
                v[:, :2] /= 1000  # km
                track_vertices.append(v.astype(np.float32))
                track_runs.append((track_id, n_track_vertices, len(v)))
                n_track_vertices += len(v)

    return {
        "vertices": np.concatenate(vertices) if vertices else np.zeros((0, 3), np.float32),
        "depths": np.concatenate(depths) if depths else np.zeros(0, np.uint16),
        "fdepths": np.concatenate(fdepths) if fdepths else np.zeros(0, np.uint16),
        "runs": np.asarray(runs, dtype=np.uint32).reshape(-1, 3),
        "features": features,
        "track_vertices": np.concatenate(track_vertices) if track_vertices else np.zeros((0, 5), np.float32),
        "track_runs": np.asarray(track_runs, dtype=np.uint32).reshape(-1, 3),
        "tracks": tracks,
    }


def _write_gz(array: np.ndarray, path: str) -> None:
    tmp = f"{path}.part"
    with gzip.open(tmp, "wb", compresslevel=9) as f:
        f.write(np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes())
    os.replace(tmp, path)


def _bbox(dataset_dirs: List[str]) -> Optional[tuple]:
    """Extent (xmin, xmax, ymin, ymax, in m) of the lines of datasets."""
    xs, ys = [], []
    for dataset_dir in dataset_dirs:
        for path in glob.glob(os.path.join(dataset_dir, "h5", "*.h5")):
            with xr.open_dataset(path) as ds:
                xs += [float(ds.PSX.min()), float(ds.PSX.max())]
                ys += [float(ds.PSY.min()), float(ds.PSY.max())]
    xs, ys = [v for v in xs if np.isfinite(v)], [v for v in ys if np.isfinite(v)]
    return (min(xs), max(xs), min(ys), max(ys)) if xs and ys else None


def export_irh(
    database_dir: str,
    output_dir: str,
    datasets: Optional[List[str]] = None,
    bedmap_path: Optional[str] = None,
    spacing: float = DEFAULT_SPACING,
    max_gap: float = DEFAULT_MAX_GAP,
    crop: bool = False,
) -> Dict:
    """
    Export the IRHs of a database directory to output_dir.

    Args:
        database_dir: Database directory (with one directory per dataset).
        output_dir: Output directory, e.g. web/public/data/irh.
        datasets: Names of the datasets to export. Defaults to all.
        bedmap_path: Bedmap3 netCDF (500 m), for datasets without SURF_ELEV
            (skipped without it) and points without ICE_THK.
        spacing: Along-track averaging distance (m), e.g. the resolution of the terrain.
        max_gap: Distance (m) between consecutive points above which the line is broken.
        crop: Only load Bedmap3 around the lines, to save memory (a few datasets).

    Returns:
        The irh.json content.
    """
    os.makedirs(output_dir, exist_ok=True)
    if datasets is None:
        datasets = sorted(
            d for d in os.listdir(database_dir) if os.path.isdir(os.path.join(database_dir, d, "h5"))
        )
    surface = None
    if bedmap_path:
        bbox = _bbox([os.path.join(database_dir, d) for d in datasets]) if crop else None
        surface = Bedmap3Surface(bedmap_path, bbox)

    index = {"format": 1, "datasets": {}, "features": [], "tracks": []}
    for dataset in datasets:
        out = export_dataset(
            os.path.join(database_dir, dataset), surface, len(index["features"]), spacing, max_gap
        )
        if not out["features"]:
            print(f"{dataset}: no IRH exported")
            continue
        # Track ids are global, like feature ids
        out["track_runs"][:, 0] += len(index["tracks"])
        files = {
            "vertices": (f"{dataset}.bin.gz", out["vertices"]),
            "depths": (f"{dataset}_depth.bin.gz", out["depths"]),
            "fdepths": (f"{dataset}_fdepth.bin.gz", out["fdepths"]),
            "runs": (f"{dataset}_runs.bin.gz", out["runs"]),
            "tracks": (f"{dataset}_tracks.bin.gz", out["track_vertices"]),
            "track_runs": (f"{dataset}_track_runs.bin.gz", out["track_runs"]),
        }
        for file, array in files.values():
            _write_gz(array, os.path.join(output_dir, file))
        index["datasets"][dataset] = {key: file for key, (file, _) in files.items()}
        index["datasets"][dataset].update(
            n_vertices=int(len(out["vertices"])),
            n_runs=int(len(out["runs"])),
            n_track_vertices=int(len(out["track_vertices"])),
        )
        index["features"].extend(out["features"])
        index["tracks"].extend(out["tracks"])
        print(
            f"{dataset}: {len(out['features'])} lines, {len(out['runs'])} runs, "
            f"{len(out['vertices'])} vertices, {len(out['tracks'])} tracks"
        )

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
