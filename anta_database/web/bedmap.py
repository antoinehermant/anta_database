"""
Export the Bedmap survey lines (anta_database.bedmap.convert) as tracks for the web viewer: the
radar data, traced or not, at the ice surface, with the bed under them.

Each line is cut into runs where its points are further apart than max_gap, averaged along
track in bins of spacing (NaN-aware: a bin keeps the values its points have), then simplified:
a vertex is only kept where leaving it out would move the line more than SIMPLIFY_HORIZONTAL, or
its surface or bed more than SIMPLIFY_VERTICAL (Douglas-Peucker). Bedmap lines are mostly straight
flights over smooth ice: this leaves a fraction of the vertices, for a viewer that stays fluid. The surface is the
measured SURF_ELEV, else the Bedmap3 surface at the point; the bed is the measured BED_ELEV,
else SURF_ELEV - ICE_THK where both are measured (for the view only: never stored).

    python -m anta_database.web.bedmap ~/data/isochrones/BEDMAP-build --bedmap bedmap3.nc \\
        -o web/public/data/bedmap

Lines with their own line ID (line ID flag "original") also get a profile, for the stratigraphy
window of the viewer: the line along Distance with the surface, bed and ice thickness, at full
resolution when it fits in PROFILE_MAX_BYTES. Surveys stored as one line (no line IDs) have none:
they are many profiles one after the other.

Output:
    bedmap.json                 surveys, tracks (survey, line ID, profile) and per-dataset file names
    {dataset}_tracks.bin.gz     float32 vertices (x km, y km, surface m, bed m or NaN,
                                ice thickness m or NaN, 1 where the surface is measured),
                                packed per run
    {dataset}_track_runs.bin.gz uint32 (track id, first vertex, vertex count) per run
    profiles/{dataset}/{n}.bin.gz  float32 profile of track n: rows of (x km, y km, distance km,
                                surface m, bed m, ice thickness m), NaN where missing, and a
                                row of NaN (but the distance) at each gap
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from anta_database.web.irh import PROFILE_COLUMNS, Bedmap3Surface, _bbox, _profile, _runs, _write_gz

DATASETS = ["BEDMAP1", "BEDMAP2", "BEDMAP3"]
DEFAULT_SPACING = 2000.0  # m: Bedmap is dense, the terrain of the viewer is 2 km
DEFAULT_MAX_GAP = 5000.0  # m
STRIDE = 6
SIMPLIFY_HORIZONTAL = 250.0  # m
SIMPLIFY_VERTICAL = 50.0  # m
# Bedmap lines are long and dense: smaller profiles than the isochrones (most at full resolution)
PROFILE_MAX_BYTES = 1_000_000


def _nan_average(points: np.ndarray, spacing: float) -> np.ndarray:
    """Mean of the finite values of the points (rows of x, y, ...) in along-track bins of spacing."""
    if spacing <= 0 or len(points) <= 2:
        return points
    dist = np.r_[0.0, np.cumsum(np.hypot(*np.diff(points[:, :2], axis=0).T))]
    bins = np.floor(dist / spacing).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
    if len(starts) == 1:
        return points[[0, -1]]
    finite = np.isfinite(points)
    sums = np.add.reduceat(np.where(finite, points, 0), starts, axis=0)
    counts = np.add.reduceat(finite.astype(np.int64), starts, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _simplify(v: np.ndarray, horizontal: float = SIMPLIFY_HORIZONTAL,
              vertical: float = SIMPLIFY_VERTICAL) -> np.ndarray:
    """Vertices (rows of x m, y m, surface m, bed m, ...) kept by Douglas-Peucker, with an error
    in the horizontal and in the surface and bed; a vertex where the bed starts or stops being
    known is always kept."""
    n = len(v)
    if n <= 2 or horizontal <= 0:
        return v
    keep = np.zeros(n, bool)
    keep[[0, -1]] = True
    known = np.isfinite(v[:, 3])
    change = np.flatnonzero(known[1:] != known[:-1])
    keep[change] = keep[change + 1] = True
    anchors = np.flatnonzero(keep)
    stack = list(zip(anchors[:-1], anchors[1:]))
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        inner = v[i + 1:j]
        d = v[j, :2] - v[i, :2]
        length = np.hypot(*d)
        rel = inner[:, :2] - v[i, :2]
        t = np.clip(rel @ d / length**2, 0, 1) if length > 0 else np.zeros(len(inner))
        error = np.hypot(*(rel - t[:, None] * d).T) / horizontal
        for k in (2, 3):
            line = v[i, k] + t * (v[j, k] - v[i, k])
            off = np.abs(inner[:, k] - line) / vertical
            error = np.fmax(error, np.where(np.isfinite(off), off, 0))
        worst = int(np.argmax(error))
        if error[worst] > 1:
            m = i + 1 + worst
            keep[m] = True
            stack += [(i, m), (m, j)]
    return v[keep]


def _values(ds: xr.Dataset, name: str) -> Optional[np.ndarray]:
    return ds[name].values.astype(np.float64) if name in ds else None


def export_dataset(dataset_dir: str, surface: Optional[Bedmap3Surface], first_track: int,
                   spacing: float, max_gap: float) -> Dict:
    """Tracks of the lines of one Bedmap dataset; track ids from first_track."""
    with open(os.path.join(dataset_dir, "raw_md.json"), encoding="utf-8") as f:
        surveys = {row["survey"]: row for row in json.load(f)}
    dataset = os.path.basename(dataset_dir.rstrip("/"))
    tracks, vertices, runs, profiles = [], [], [], []
    n_vertices = 0
    for path in sorted(glob.glob(os.path.join(dataset_dir, "nc", "*.nc"))):
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            x, y = _values(ds, "PSX"), _values(ds, "PSY")
            surf, thk, bed = _values(ds, "SURF_ELEV"), _values(ds, "ICE_THK"), _values(ds, "BED_ELEV")
            survey, line_id = ds.attrs["survey"], ds.attrs["line ID"]
            has_id = ds.attrs.get("line ID flag") == "original"
            distance = _values(ds, "Distance")
        n = len(x)
        nan = np.full(n, np.nan)
        surf = nan.copy() if surf is None else surf
        thk = nan.copy() if thk is None else thk
        bed = nan.copy() if bed is None else bed
        bed = np.where(np.isfinite(bed), bed, surf - thk)
        measured = np.isfinite(surf).astype(np.float64)
        if surface is not None:
            surf = np.where(np.isfinite(surf), surf, surface(x, y))
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(surf)
        step = np.r_[0.0, np.hypot(np.diff(x), np.diff(y))]
        line_runs = _runs(valid, step, max_gap)
        if not line_runs:
            continue
        track_id = first_track + len(tracks)
        track = {
            "survey": survey,
            "line_id": line_id,
            "citation": surveys.get(survey, {}).get("citation"),
            "surface": "SURF_ELEV" if measured[valid].all() else "SURF_ELEV, else Bedmap3",
        }
        if has_id:
            rows, resolution = _profile(distance, np.column_stack([x, y, surf, bed, thk]), max_gap, PROFILE_MAX_BYTES)
            track["profile"] = {
                "file": f"profiles/{dataset}/{track_id}.bin.gz",
                "columns": PROFILE_COLUMNS,
                "layers": [],
                "bed": "BED_ELEV, else SURF_ELEV - ICE_THK",
                "thickness": "ICE_THK measured",
                "resolution": resolution,
            }
            profiles.append((track["profile"]["file"], rows))
        tracks.append(track)
        for run in line_runs:
            v = _nan_average(np.column_stack([x[run], y[run], surf[run], bed[run], thk[run], measured[run]]), spacing)
            v = _simplify(v)
            v[:, :2] /= 1000  # km
            vertices.append(v.astype(np.float32))
            runs.append((track_id, n_vertices, len(v)))
            n_vertices += len(v)
    keys = ["survey", "institute", "project", "acquisition year", "radar instrument", "citation",
            "DOI dataset", "DOI article"]
    return {
        "surveys": {s: {k: row.get(k) for k in keys} for s, row in surveys.items()},
        "tracks": tracks,
        "vertices": np.concatenate(vertices) if vertices else np.zeros((0, STRIDE), np.float32),
        "runs": np.asarray(runs, dtype=np.uint32).reshape(-1, 3),
        "profiles": profiles,
    }


def export_bedmap(build_dir: str, output_dir: str, datasets: Optional[List[str]] = None,
                  bedmap_path: Optional[str] = None, spacing: float = DEFAULT_SPACING,
                  max_gap: float = DEFAULT_MAX_GAP) -> Dict:
    """Export the Bedmap datasets of build_dir (one directory per dataset) to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    datasets = [d for d in (datasets or DATASETS) if os.path.isdir(os.path.join(build_dir, d, "nc"))]
    surface = None
    if bedmap_path:
        surface = Bedmap3Surface(bedmap_path, _bbox([os.path.join(build_dir, d) for d in datasets]))
    index = {"format": 1, "datasets": {}, "surveys": {}, "tracks": []}
    for dataset in datasets:
        out = export_dataset(os.path.join(build_dir, dataset), surface, len(index["tracks"]), spacing, max_gap)
        files = {"tracks": f"{dataset}_tracks.bin.gz", "track_runs": f"{dataset}_track_runs.bin.gz"}
        _write_gz(out["vertices"], os.path.join(output_dir, files["tracks"]))
        _write_gz(out["runs"], os.path.join(output_dir, files["track_runs"]))
        os.makedirs(os.path.join(output_dir, "profiles", dataset), exist_ok=True)
        for file, rows in out["profiles"]:
            _write_gz(rows, os.path.join(output_dir, file))
        index["datasets"][dataset] = {**files, "n_vertices": int(len(out["vertices"]))}
        index["surveys"].update(out["surveys"])
        index["tracks"].extend({"dataset": dataset, **t} for t in out["tracks"])
        print(f"{dataset}: {len(out['tracks'])} tracks, {len(out['runs'])} runs, {len(out['vertices'])} vertices, "
              f"{len(out['profiles'])} profiles")
    with open(os.path.join(output_dir, "bedmap.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"), ensure_ascii=False)
    return index


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export the Bedmap survey lines for the web viewer.")
    parser.add_argument("build_dir", help="directory with BEDMAP1/, BEDMAP2/, BEDMAP3/ (anta_database.bedmap.convert)")
    parser.add_argument("-o", "--output", default="web/public/data/bedmap", help="Output directory")
    parser.add_argument("--datasets", nargs="+", help="Datasets to export (default: all)")
    parser.add_argument("--bedmap", help="Bedmap3 netCDF, for the surface where it is not measured")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING, help="Along-track averaging distance (m)")
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP, help="Gap that breaks a line (m)")
    args = parser.parse_args(argv)
    export_bedmap(args.build_dir, args.output, args.datasets, args.bedmap, args.spacing, args.max_gap)


if __name__ == "__main__":
    main()
