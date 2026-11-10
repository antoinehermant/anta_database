"""
Reconnect the isochrone lines to the Bedmap survey lines (anta_database.bedmap.convert): the
Bedmap line that has the same radar traces as an isochrone line, its ice thickness, bed and
surface elevation, and its line ID (docs/bedmap/README.md, section 6).

For an isochrone line:
1. candidates: the Bedmap lines of the same platform (airborne or ground-based, "platform" of the
   line; _AIR_, or _GRN_ and _SEI_ in the name of the Bedmap survey) whose extent overlaps that of
   the line (index(), built once);
2. each point gets the nearest point of each candidate within max_dist;
3. the matched Bedmap survey is the one whose lines have the most points within max_dist (its
   coverage): a flight is often cut into several Bedmap lines. Of surveys with almost the same
   coverage (MATCH_SHARE of the best), the newest generation, as a survey is often in several:
   BEDMAP3 over BEDMAP2 over BEDMAP1;
4. values are only taken from that one survey, each from the nearest point of its lines: never
   mixed from two surveys or generations;
5. the survey is only taken if its points are within MAX_MEDIAN of those of the line, on median:
   the same radar traces are 0-7 m apart, other flights over the same place 25-42 m.

Each point also keeps the Bedmap line nearest to it ("segments"), for the lines without an ID,
whose points may be several Bedmap lines one after the other.

fill() writes the values of the matched survey where a line has none (the values published stay),
only on the lines it fills:
- <VAR>_SOURCE (int8, per point, anta_database.netcdf.SOURCES): 1, 2 or 3 where the value comes from
  Bedmap1, 2 or 3; the compilation wrote 0 (published) and 4 (computed from the published values);
- BEDMAP_POINT_DIST (float32, m, per point): distance to the Bedmap point the values come from, NaN
  where none;
- the attribute "reconnection": the Bedmap survey and lines, how many points they match and fill.

    python -m anta_database.bedmap.reconnect BEDMAP_BUILD ISOCHRONE_BUILD --datasets A B -o report.csv
    python -m anta_database.bedmap.reconnect BEDMAP_BUILD ISOCHRONE_BUILD --datasets A B --fill
"""

import argparse
import glob
import os
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from anta_database.netcdf import SOURCE_VARIABLES as VARIABLES, SOURCES, source_attrs

GENERATIONS = {"BEDMAP3": 3, "BEDMAP2": 2, "BEDMAP1": 1}
DEFAULT_MAX_DIST = 50.0  # m, per point
MAX_MEDIAN = 10.0  # m, of the points of a line: the same traces
MATCH_SHARE = 0.9
INDEX_FILE = "lines_index.parquet"
# Platform of a Bedmap survey, from its name (seismic soundings are on the ground)
SURVEY_PLATFORMS = {"AIR": "airborne", "GRN": "ground-based", "SEI": "ground-based"}
DIST_ATTRS = {
    "units": "m",
    "long_name": "distance to the Bedmap survey point",
    "description": "Distance to the point of the Bedmap survey line (attribute reconnection) the values of "
    "source 1, 2 or 3 (<VAR>_SOURCE) come from, never the gridded Bedmap products; NaN where none",
}


def _extent(path: str) -> dict:
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        x, y = ds.PSX.values, ds.PSY.values
        return {
            "path": path,
            "generation": ds.attrs["dataset"],
            "survey": ds.attrs["survey"],
            "line_id": str(ds.attrs["line ID"]),
            "line_id_flag": ds.attrs.get("line ID flag"),
            "institute": str(ds.attrs.get("institute")),
            "year": str(ds.attrs.get("acquisition year")),
            "points": len(x),
            "xmin": float(np.nanmin(x)), "xmax": float(np.nanmax(x)),
            "ymin": float(np.nanmin(y)), "ymax": float(np.nanmax(y)),
        }


def index(bedmap_build: str, cpus: int = 4, rebuild: bool = False) -> pd.DataFrame:
    """The extent of every Bedmap line (one row per file), kept in bedmap_build/INDEX_FILE."""
    cache = os.path.join(bedmap_build, INDEX_FILE)
    if os.path.exists(cache) and not rebuild:
        return pd.read_parquet(cache)
    paths = sorted(glob.glob(os.path.join(bedmap_build, "BEDMAP*", "nc", "*.nc")))
    with Pool(cpus) as pool:
        rows = pool.map(_extent, paths, chunksize=50)
    df = pd.DataFrame(rows)
    df.to_parquet(cache)
    return df


def _with_platform(lines: pd.DataFrame) -> pd.DataFrame:
    platform = lines.survey.str.extract(r"_(AIR|GRN|SEI)_", expand=False).map(SURVEY_PLATFORMS)
    return lines.assign(platform=platform)


def _platforms(attrs: dict) -> List[str]:
    """The platforms of an isochrone line (attribute platform: a string, or a list for both)."""
    value = attrs.get("platform", [])
    return [value] if isinstance(value, str) else [str(v) for v in value]


def _values(path: str) -> Dict[str, np.ndarray]:
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        out = {"xy": np.c_[ds.PSX.values, ds.PSY.values]}
        for var in VARIABLES:
            out[var] = ds[var].values.astype(np.float64) if var in ds else np.full(len(ds.PSX), np.nan)
    return out


def match(xy: np.ndarray, lines: pd.DataFrame, max_dist: float = DEFAULT_MAX_DIST) -> dict:
    """The Bedmap survey of the points xy (n, 2) in m, among lines (rows of index()), and per point
    the distance to the nearest point of its lines and its values; plus the nearest Bedmap line of
    each point."""
    n = len(xy)
    located = np.isfinite(xy).all(axis=1)  # points without coordinates match nothing
    lo, hi = np.nanmin(xy, axis=0) - max_dist, np.nanmax(xy, axis=0) + max_dist
    cand = lines[(lines.xmax >= lo[0]) & (lines.xmin <= hi[0]) & (lines.ymax >= lo[1]) & (lines.ymin <= hi[1])]
    nearest = np.full(n, -1)
    nearest_dist = np.full(n, np.inf)
    per_line = {}
    for k, row in cand.iterrows():
        v = _values(row.path)
        dist, idx = np.full(n, np.inf), np.full(n, len(v["xy"]))
        dist[located], idx[located] = cKDTree(v["xy"]).query(xy[located], distance_upper_bound=max_dist)
        within = np.isfinite(dist)
        if not within.any():
            continue
        per_line[k] = (dist, np.where(within, idx, 0), within, v)
        closer = dist < nearest_dist
        nearest[closer], nearest_dist[closer] = k, dist[closer]
    result = {"candidates": len(cand), "nearest": nearest, "nearest_dist": nearest_dist}
    if not per_line:
        return {**result, "line": None}
    # By survey: a flight is often cut into several Bedmap lines, all of the same traces
    surveys: Dict[tuple, dict] = {}
    for k, (dist, idx, within, v) in per_line.items():
        key = (lines.loc[k, "generation"], lines.loc[k, "survey"])
        s = surveys.setdefault(key, {"dist": np.full(n, np.inf), "line": np.full(n, -1),
                                     "values": {var: np.full(n, np.nan) for var in VARIABLES}})
        closer = within & (dist < s["dist"])
        s["dist"][closer], s["line"][closer] = dist[closer], k
        for var in VARIABLES:
            s["values"][var][closer] = v[var][idx[closer]]
    coverage = {key: int(np.isfinite(s["dist"]).sum()) for key, s in surveys.items()}
    best = max(coverage.values())
    close = [key for key, c in coverage.items() if c >= MATCH_SHARE * best]
    chosen = max(close, key=lambda key: (GENERATIONS.get(key[0], 0), coverage[key]))
    s = surveys[chosen]
    within = np.isfinite(s["dist"])
    used = pd.Series(s["line"][within]).value_counts()
    return {**result, "line": int(used.index[0]), "lines": [int(k) for k in used.index], "survey": chosen[1],
            "generation": chosen[0], "coverage": coverage[chosen] / n, "distance": np.where(within, s["dist"], np.nan),
            "values": s["values"], "lines_matched": len(per_line)}


def _study_line(job) -> Optional[dict]:
    path, lines, max_dist = job
    with xr.open_dataset(path) as ds:
        xy = np.c_[ds.PSX.values, ds.PSY.values]
        own = {var: ds[var].values.astype(np.float64) if var in ds else np.full(len(xy), np.nan) for var in VARIABLES}
        attrs = dict(ds.attrs)
    m = match(xy, lines[lines.platform.isin(_platforms(attrs))], max_dist)
    row = {"dataset": attrs.get("dataset"), "line_id": str(attrs.get("line ID")), "points": len(xy),
           "platform": ", ".join(_platforms(attrs)), "candidates": m["candidates"],
           "lines_matched": m.get("lines_matched", 0)}
    if m["line"] is None:
        return row
    b = lines.loc[m["line"]]
    row.update({"generation": b.generation, "survey": b.survey, "bedmap_line": b.line_id, "bedmap_lines": len(m["lines"]),
                "bedmap_line_flag": b.line_id_flag, "same_id": b.line_id.lower() == row["line_id"].lower(),
                "coverage": round(m["coverage"], 4), "median_dist": round(float(np.nanmedian(m["distance"])), 2),
                "p95_dist": round(float(np.nanpercentile(m["distance"], 95)), 2)})
    for var in VARIABLES:
        has, got = np.isfinite(own[var]), np.isfinite(m["values"][var])
        row[f"{var}_missing"] = int((~has).sum())
        row[f"{var}_fill"] = int((~has & got).sum())
        both = has & got
        row[f"{var}_diff_median"] = round(float(np.median(m["values"][var][both] - own[var][both])), 2) if both.any() else None
        row[f"{var}_diff_p95abs"] = round(float(np.percentile(np.abs(m["values"][var][both] - own[var][both]), 95)), 2) if both.any() else None
    # Lines without an ID: the Bedmap lines its points are nearest to, in order
    near = m["nearest"][np.isfinite(m["nearest_dist"])]
    if len(near):
        changes = np.flatnonzero(np.diff(near)) + 1
        segments = [int(near[s]) for s in np.r_[0, changes]]
        row["segments"] = len(segments)
        row["segment_lines"] = len(set(segments))
    return row


def _fill_line(job) -> Optional[dict]:
    path, lines, max_dist, max_median = job
    from anta_database.netcdf import write

    with xr.open_dataset(path) as ds:
        ds = ds.load()
    n = ds.sizes["point"]
    row = {"dataset": ds.attrs.get("dataset"), "line_id": str(ds.attrs.get("line ID")), "points": n,
           "platform": ", ".join(_platforms(ds.attrs))}
    if not any(np.isnan(ds[var].values).any() if var in ds else True for var in VARIABLES):
        return None  # nothing missing
    m = match(np.c_[ds.PSX.values, ds.PSY.values], lines[lines.platform.isin(_platforms(ds.attrs))], max_dist)
    if m["line"] is None:
        return {**row, "status": "no Bedmap survey of the same platform within the distance"}
    b = lines.loc[m["line"]]
    median = float(np.nanmedian(m["distance"]))
    row.update({"generation": b.generation, "survey": b.survey, "coverage": round(m["coverage"], 4),
                "median_dist": round(median, 2)})
    if median > max_median:
        return {**row, "status": f"not the same traces: more than {max_median:g} m apart on median"}
    code = SOURCES[f"{b.generation.lower()}_survey_point"]
    taken = np.zeros(n, bool)
    filled = {}
    for var in VARIABLES:
        own = ds[var].values.astype(np.float64) if var in ds else np.full(n, np.nan)
        take = ~np.isfinite(own) & np.isfinite(m["values"][var])
        if not take.any():
            continue
        source = (ds[f"{var}_SOURCE"].values.astype(np.float64) if f"{var}_SOURCE" in ds
                  else np.where(np.isfinite(own), SOURCES["published_dataset"], np.nan))
        own[take] = m["values"][var][take]
        source[take] = code
        ds[var] = ("point", own, ds[var].attrs if var in ds else {})
        ds[f"{var}_SOURCE"] = ("point", source, source_attrs(var))
        taken |= take
        filled[var] = int(take.sum())
    if not filled:
        return {**row, "status": "nothing to take: Bedmap has no value where the line has none"}
    ds["BEDMAP_POINT_DIST"] = ("point", np.where(taken, np.round(m["distance"], 1), np.nan), DIST_ATTRS)
    ds.attrs["reconnection"] = (
        f"{b.generation}, survey {b.survey}, "
        + (f"line {b.line_id}" if len(m["lines"]) == 1 else f"{len(m['lines'])} lines ({', '.join(lines.loc[m['lines'], 'line_id'][:5])}"
           + (", ..." if len(m["lines"]) > 5 else "") + ")")
        + f": {m['coverage']:.1%} of the points within "
        f"{max_dist:g} m (median {median:.1f} m). Taken from its survey points where the dataset has no value, "
        "never from the gridded Bedmap products: "
        + ", ".join(f"{var} at {k} points" for var, k in filled.items())
    )
    sentence = "Missing values taken from the Bedmap survey points of the same traces (<VAR>_SOURCE, BEDMAP_POINT_DIST, attribute reconnection)."
    if sentence not in str(ds.attrs.get("processing", "")):
        ds.attrs["processing"] = f"{ds.attrs.get('processing', '')} {sentence}".strip()
    write(ds, path)
    return {**row, "status": "filled", **filled}


def fill(bedmap_build: str, isochrone_build: str, datasets: List[str], max_dist: float = DEFAULT_MAX_DIST,
         cpus: int = 4, max_median: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Write the Bedmap values where the lines of datasets have none; one row per line with values
    missing, with what was done (status). max_median: limits other than MAX_MEDIAN, by dataset."""
    lines = _with_platform(index(bedmap_build, cpus))
    limits = {d: (max_median or {}).get(d, MAX_MEDIAN) for d in datasets}
    jobs = [(p, lines, max_dist, limits[d]) for d in datasets
            for p in sorted(glob.glob(os.path.join(isochrone_build, d, "nc", "*.nc")))]
    with Pool(cpus) as pool:
        rows = [r for r in pool.imap(_fill_line, jobs, chunksize=4) if r]
    return pd.DataFrame(rows)


def study(bedmap_build: str, isochrone_build: str, datasets: List[str], max_dist: float = DEFAULT_MAX_DIST,
          cpus: int = 4) -> pd.DataFrame:
    """One row per isochrone line of datasets: the Bedmap line matched, and what it would fill."""
    lines = _with_platform(index(bedmap_build, cpus))
    jobs = [(p, lines, max_dist) for d in datasets for p in sorted(glob.glob(os.path.join(isochrone_build, d, "nc", "*.nc")))]
    with Pool(cpus) as pool:
        rows = [r for r in pool.imap(_study_line, jobs, chunksize=4) if r]
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Match isochrone lines to Bedmap lines (study, writes a table).")
    parser.add_argument("bedmap_build")
    parser.add_argument("isochrone_build")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--max-dist", type=float, default=DEFAULT_MAX_DIST)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("-o", "--output", default="reconnect_study.csv")
    parser.add_argument("--fill", action="store_true", help="Write the values into the isochrone files (not a study)")
    args = parser.parse_args(argv)
    run = fill if args.fill else study
    df = run(args.bedmap_build, args.isochrone_build, args.datasets, args.max_dist, args.cpus)
    df.to_csv(args.output, index=False)
    print(df.to_string(max_rows=60))


if __name__ == "__main__":
    main()
