"""
Convert the standardised CSV files of Bedmap (one per survey) to the format of the database:
one nc per line, written by anta_database.netcdf, with the attributes of the isochrone lines.

Lines (see docs/bedmap/README.md):
- the trajectory_id of a survey names its lines when they are names (one run of points per ID);
  they are kept as published, also with "/" in them ("CBG/CBG03b"), and named "line ID flag":
  "original";
- some surveys have no IDs (-9999), or number their points (one ID per point, as BEDMAP1 and
  most of BEDMAP2): the survey is then one line in the order of the file, named after the
  survey (raw_md.json "line ID", else the project and year) with "line ID flag": "not_provided";
  with split_at_jump_km, it is cut where consecutive points are further apart, as _1, _2...
Points stay in the order of the file (the order of the traces). A file is named after its
survey and line, so lines of two surveys with the same ID (C9 in the Lake Ellsworth surveys of
2007 and 2008) never meet.

Values are the ones published: surface, ice thickness and bed elevation, none computed from the
two others. Points without any of them are dropped.

    python -m anta_database.bedmap.convert ~/data/isochrones/database_project/temp \\
        ~/data/isochrones/BEDMAP-build --datasets BEDMAP3 --surveys KOPRI_2017_KRT1_AIR_BM3
"""

import argparse
import json
import os
import re
from importlib.resources import files
from multiprocessing import Pool
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from anta_database.netcdf import write

DATASETS = ["BEDMAP1", "BEDMAP2", "BEDMAP3"]
COLUMNS = {
    "trajectory_id": "Line_ID",
    "longitude (degree_east)": "lon",
    "latitude (degree_north)": "lat",
    "surface_altitude (m)": "SURF_ELEV",
    "land_ice_thickness (m)": "ICE_THK",
    "bedrock_altitude (m)": "BED_ELEV",
}
VALUES = ["SURF_ELEV", "ICE_THK", "BED_ELEV"]
NODATA = ["-9999", "-9999.0", "-9999.00"]
# A trajectory_id names lines when its IDs have at least this many points each (median):
# otherwise the IDs number the points
MIN_POINTS_PER_ID = 10
# Surveys without line IDs stay one line for now; a number of km cuts them at the jumps
SPLIT_AT_JUMP_KM: Optional[float] = None
# Escapes left in the raw_md.json of the old compilation
FIXES = {"Fru00e9mand": "Frémand"}
# Values of the published files found wrong (docs/bedmap/README.md, section 4): left out, the
# reason recorded in the attribute "processing". In these three surveys of BEDMAP2, the ice
# thickness agrees with the Bedmap3 grid, but the "surface" falls as the real surface rises: it is
# the distance from the aircraft down to the surface (and the "bed", surface plus thickness, down to
# the bed), and the aircraft altitude is not in the files to convert it back
DROPPED = {
    survey: (["SURF_ELEV", "BED_ELEV"], "Surface and bed elevation of the published file left out: they "
             "are the distances from the aircraft down to the surface and the bed (the surface falls where the "
             "Bedmap3 surface rises), not elevations; the ice thickness agrees with Bedmap3.")
    for survey in ["NASA_2002_ICEBRIDGE_AIR_BM2", "NASA_2004_ICEBRIDGE_AIR_BM2", "NASA_2010_ICEBRIDGE_AIR_BM2"]
}
# Parts of surveys left out, all their values: a profile (the points between two jumps of more than
# PROFILE_GAP_KM) by its first point, in km (EPSG:3031), or the end of a line from a distance along
# it, in km; with the reason (docs/bedmap/README.md, section 4)
PROFILE_GAP_KM = 5.0
DROPPED_PARTS = {
    "PRIC_2016_CHA2_AIR_BM3": {
        "profiles": [(1829, -151), (1056, 228), (2197, 527)],
        "reason": "Three of its profiles left out (starting at x, y = 1829, -151; 1056, 228; 2197, 527 km): "
        "their surface is 700 to 2700 m below Bedmap3 and noisy from point to point, their thickness 700 to "
        "1200 m off, unlike the other 110 profiles (within 30 m).",
    },
}
VAR_ATTRS = {
    "PSX": ("m", "Polar Stereographic X", "Polar Stereographic X coordinates (metres; EPSG: 3031)"),
    "PSY": ("m", "Polar Stereographic Y", "Polar Stereographic Y coordinates (metres; EPSG: 3031)"),
    "ICE_THK": ("m", "Ice thickness", "Thickness of the ice in meters"),
    "SURF_ELEV": ("m a.s.l.", "Surface elevation", "Elevation of the surface above sea level in meters"),
    "BED_ELEV": ("m a.s.l.", "Bedrock elevation", "Elevation of the bedrock above sea level in meters"),
    "Distance": ("m", "Along_track distance", "Along-track distance along radar transect"),
}
TO_POLAR = Transformer.from_proj(
    "EPSG:4326",
    "+proj=stere +lon_0=0 +lat_0=-90 +lat_ts=-71 +datum=WGS84 +units=m +no_defs",
    always_xy=True,
)

_basins = None


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value) in ("nan", "none", "")


def raw_header(path: str) -> Dict[str, str]:
    """The "#name: value" lines at the top of a standardised Bedmap CSV."""
    header = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            name, _, value = line[1:].partition(":")
            header[name.strip()] = value.strip()
    return header


def _number(text: Optional[str]) -> Optional[float]:
    match = re.match(r"\s*(-?[\d.]+)", text or "")
    return float(match.group(1)) if match and match.group(1) != "-9999" else None


def survey_metadata(dataset_dir: str) -> List[dict]:
    """The rows of raw_md.json with the current keys, completed from the headers of the raw files."""
    with open(os.path.join(dataset_dir, "raw_md.json"), encoding="utf-8") as f:
        rows = json.load(f)
    result = []
    for row in rows:
        header = raw_header(os.path.join(dataset_dir, "raw", row["raw file"]))
        instrument = row.get("radar instrument")
        if _missing(instrument) and header.get("instrument"):
            instrument = header["instrument"].rstrip(".")
        citation = row["citation"]
        for old, new in FIXES.items():
            citation = citation.replace(old, new)
        new = {
            "raw file": row["raw file"],
            "survey": os.path.splitext(row["raw file"])[0],
            "file structure": "survey",
            "dataset": row["dataset"],
            "institute": row["institute"],
            "project": row["project"],
            "acquisition year": row["acquisition year"],
            # The name of the lines of a survey without line IDs
            "line ID": row.get("flight ID", row.get("line ID")),
            "radar instrument": instrument,
            "wave speed": _number(header.get("electromagnetic_wave_speed_in_ice")),
            "firn correction": _number(header.get("firn_correction")),
            "citation": citation,
            "DOI dataset": row["DOI dataset"],
            "DOI article": row.get("DOI article", row.get("DOI publication")),
        }
        result.append({k: (None if _missing(v) else v) for k, v in new.items()})
    return result


def _id(value: str) -> str:
    """A line ID as text: 3.0 -> 3 (IDs read as floats somewhere before Bedmap)."""
    value = str(value).strip()
    return value[:-2] if re.fullmatch(r"-?\d+\.0", value) else value


def read_survey(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", usecols=list(COLUMNS), dtype={"trajectory_id": str},
                     na_values=NODATA, keep_default_na=True, low_memory=False)
    df = df.rename(columns=COLUMNS)
    df = df.dropna(subset=VALUES, how="all").dropna(subset=["lon", "lat"])
    df["Line_ID"] = df["Line_ID"].fillna("-9999").map(_id)
    return df.reset_index(drop=True)


def has_line_ids(ids: pd.Series) -> bool:
    """Whether the trajectory_id of a survey names lines (and does not number points)."""
    named = ids[ids != "-9999"]
    return len(named) > 0 and named.value_counts().median() >= MIN_POINTS_PER_ID


def _cuts(x: np.ndarray, y: np.ndarray, km: Optional[float]) -> List[slice]:
    if not km or len(x) < 2:
        return [slice(0, len(x))]
    jumps = np.flatnonzero(np.hypot(np.diff(x), np.diff(y)) > km * 1000) + 1
    bounds = np.r_[0, jumps, len(x)]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def lines(df: pd.DataFrame, name: str, km: Optional[float]):
    """(line ID, flag, points) of a survey, in the order of the file."""
    if has_line_ids(df["Line_ID"]):
        for line_id, points in df.groupby("Line_ID", sort=False):
            yield line_id, "original", points
        return
    parts = _cuts(df["PSX"].values, df["PSY"].values, km)
    for n, part in enumerate(parts, start=1):
        yield (f"{name}_{n}" if len(parts) > 1 else name), "not_provided", df.iloc[part]


def drop_parts(df: pd.DataFrame, survey: str) -> pd.DataFrame:
    """The points of a survey without its parts of DROPPED_PARTS (PSX, PSY in m)."""
    rule = DROPPED_PARTS.get(survey)
    if not rule:
        return df
    keep = np.ones(len(df), bool)
    if "profiles" in rule:
        x, y = df["PSX"].values, df["PSY"].values
        starts = np.r_[0, np.flatnonzero(np.hypot(np.diff(x), np.diff(y)) > PROFILE_GAP_KM * 1000) + 1]
        bounds = np.r_[starts, len(df)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if (round(x[a] / 1000), round(y[a] / 1000)) in rule["profiles"]:
                keep[a:b] = False
    if "line" in rule:
        line = (df["Line_ID"] == rule["line"]).values
        idx = np.flatnonzero(line)
        steps = np.hypot(np.diff(df["PSX"].values[idx]), np.diff(df["PSY"].values[idx]))
        along = np.r_[0.0, np.cumsum(steps)]
        keep[idx[along > rule["after_km"] * 1000]] = False
    if keep.all():
        raise ValueError(f"{survey}: the parts of DROPPED_PARTS were not found")
    return df[keep].reset_index(drop=True)


def _file_name(survey: str, line_id: str) -> str:
    return f"{survey}__{re.sub(r'[^A-Za-z0-9._()-]+', '_', line_id)}.nc"


def imbie_basins(x: np.ndarray, y: np.ndarray) -> str:
    """The IMBIE basins a line crosses, as in compile (Islands left out)."""
    global _basins
    if _basins is None:
        _basins = gpd.read_parquet(files("anta_database.data").joinpath("ANT_Basins_IMBIE2_v1.6.parquet"))
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=_basins.crs)
    joined = gpd.sjoin(points, _basins, how="inner", predicate="within")
    joined = joined[joined["Regions"] != "Islands"][["Subregion", "Regions"]].drop_duplicates()
    return json.dumps(dict(zip(joined["Subregion"], joined["Regions"])))


def _attr(value):
    return "nan" if value is None else value


def convert_survey(task: tuple) -> dict:
    """Write the lines of one survey; returns a summary of them."""
    dataset_dir, out_dir, md, km = task
    df = read_survey(os.path.join(dataset_dir, "raw", md["raw file"]))
    dropped, reason = DROPPED.get(md["survey"], ([], ""))
    df[dropped] = np.nan
    df = df.dropna(subset=VALUES, how="all").reset_index(drop=True)
    df["PSX"], df["PSY"] = TO_POLAR.transform(df["lon"].values, df["lat"].values)
    df = drop_parts(df, md["survey"])
    if md["survey"] in DROPPED_PARTS:
        reason = f"{reason} {DROPPED_PARTS[md['survey']]['reason']}".strip()
    name = md["line ID"] or f"{md['project']}_{md['acquisition year']}"
    summary = {"survey": md["survey"], "points": len(df), "line IDs": has_line_ids(df["Line_ID"]),
               "lines": 0, "back and forth": 0, "files": []}
    for line_id, flag, points in lines(df, name, km):
        x, y = points["PSX"].values, points["PSY"].values
        steps = np.hypot(np.diff(x), np.diff(y))
        distance = np.r_[0.0, np.cumsum(steps)]
        ds = xr.Dataset(
            {
                "PSX": ("point", x),
                "PSY": ("point", y),
                **{v: ("point", points[v].values.astype("float32")) for v in VALUES if points[v].notna().any()},
                "Distance": ("point", distance),
            },
            coords={"point": np.arange(len(points))},
            attrs={
                "dataset": md["dataset"],
                "survey": md["survey"],
                "institute": _attr(md["institute"]),
                "project": _attr(md["project"]),
                "radar instrument": _attr(md["radar instrument"]),
                "acquisition year": str(md["acquisition year"]),
                "line ID": line_id,
                "line ID flag": flag,
                "DOI dataset": _attr(md["DOI dataset"]),
                "DOI article": _attr(md["DOI article"]),
                "processing": (
                    "Converted from the standardised CSV file of the survey, values as published."
                    + (f" {reason}" if reason else "")
                    + ("" if flag == "original" else
                       " The survey has no line IDs: lines named after it"
                       + (f", cut at jumps of more than {km:g} km." if km else "."))
                ),
                "IMBIE_basins": imbie_basins(x, y),
            },
        )
        for var, (units, long_name, description) in VAR_ATTRS.items():
            if var in ds:
                ds[var].attrs = {"units": units, "long_name": long_name, "description": description}
        path = os.path.join(out_dir, _file_name(md["survey"], line_id))
        write(ds, path)
        summary["lines"] += 1
        # More than 5 steps of over 50 times the usual spacing (and 2 km): points maybe out of order
        median = np.median(steps) if len(steps) else 0
        summary["back and forth"] += int((steps > max(50 * median, 2000)).sum() > 5)
        summary["files"].append(os.path.basename(path))
    return summary


def convert_dataset(src_dir: str, out_root: str, surveys: Optional[List[str]] = None,
                    cpus: int = 2, km: Optional[float] = SPLIT_AT_JUMP_KM) -> List[dict]:
    """Convert one Bedmap generation (BEDMAP1, 2 or 3) from src_dir into out_root/<dataset>."""
    dataset = os.path.basename(src_dir.rstrip("/"))
    metadata = survey_metadata(src_dir)
    if surveys:
        metadata = [md for md in metadata if md["survey"] in surveys]
    out_dir = os.path.join(out_root, dataset)
    nc_dir = os.path.join(out_dir, "nc")
    os.makedirs(nc_dir, exist_ok=True)
    tasks = [(src_dir, nc_dir, md, km) for md in metadata]
    # Surveys are large (up to 2 GB of CSV): few at a time
    with Pool(max(1, min(cpus, len(tasks)))) as pool:
        summaries = list(pool.imap_unordered(convert_survey, tasks))

    md_path = os.path.join(out_dir, "raw_md.json")
    kept = {}
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as f:
            kept = {row["survey"]: row for row in json.load(f)}
    kept.update({md["survey"]: md for md in metadata})
    with open(md_path, "w", encoding="utf-8") as f:
        json.dump(sorted(kept.values(), key=lambda r: r["survey"]), f, indent=1, ensure_ascii=False)
    for s in sorted(summaries, key=lambda s: s["survey"]):
        print(f"{s['survey']}: {s['points']} points, {s['lines']} lines "
              f"({'line IDs' if s['line IDs'] else 'no line IDs'}), {s['back and forth']} out of order")
    return summaries


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Convert the Bedmap CSV files to the database format.")
    parser.add_argument("src", help="directory with BEDMAP1/, BEDMAP2/, BEDMAP3/ (raw/, raw_md.json)")
    parser.add_argument("out", help="output directory (one directory per dataset)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--surveys", nargs="+", help="only these surveys (raw file names without .csv)")
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--split-at-jump-km", type=float, default=SPLIT_AT_JUMP_KM,
                        help="cut surveys without line IDs at jumps larger than this")
    args = parser.parse_args(argv)
    for dataset in args.datasets:
        convert_dataset(os.path.join(args.src, dataset), args.out, args.surveys, args.cpus, args.split_at_jump_km)


if __name__ == "__main__":
    main()
