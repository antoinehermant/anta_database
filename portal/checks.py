"""Checks of a submission on the server, before and after its compilation (pipeline.py).

The form (web/src/submit/files.ts) checks the files in the browser, but the API can be called
without it: the raw files are checked again here with the same rules before they are compiled
(errors stop the compilation). After the compilation, the checks need the whole dataset and
Bedmap3, and only give warnings for the review:

- points lost by the compilation (rows that could not be read);
- IRHs deeper than the ice thickness (of the dataset, or Bedmap3);
- IRHs crossing each other, an older one above a younger one (a wrong IRH number or age);
- sudden jumps of an IRH between neighbouring points (spikes, a picking problem);
- points off the ice sheet and the ice shelves of Bedmap3.

Each check gives a Report: errors, warnings, and a summary of the dataset.
"""

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COORDINATES = ("PSX", "PSY", "lon", "lat")
VARIABLES = ("ICE_THK", "SURF_ELEV", "BED_ELEV", "BASAL_UNIT")
MISSING = ["", "-9999", "-9999.0", "NaN", "nan"]
SEPARATORS = {".csv": ",", ".tab": "\t"}
IRH_COLUMN = re.compile(r"^IRH(\d+)$")
# Column of the line of each point; Flight_ID is its name before ground-based radar
LINE_ID_COLUMNS = ("Line_ID", "Flight_ID")
MAX_DEPTH = 5000  # m, deeper than any ice in Antarctica
PS_EXTENT = 3.4e6  # m, Antarctica in EPSG:3031
CHUNK = 1_000_000  # rows read at once
# Thresholds of the warnings after the compilation
THICKNESS_TOLERANCE = 1.02  # depth / thickness
CROSSING_SHARE = 0.01  # of the points where two IRHs are both traced
SPIKE_DEPTH = 150.0  # m, between points
SPIKE_DISTANCE = 500.0  # m apart at most
SPIKE_SHARE = 0.005


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"errors": self.errors, "warnings": self.warnings, "summary": self.summary}


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def are(n: int) -> str:
    return "is" if n == 1 else "are"


def share(n: int, total: int) -> str:
    return f"{n / total:.1%}" if total else "-"


# --- Raw files, before the compilation


def _header_line(path: Path) -> int:
    """Number of description lines (starting with #) at the top of a file."""
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            n += 1
    return n


def check_raw(dataset_dir: Path, record: dict) -> Report:
    """Check the raw files of a submission against what the form described (record: the
    submission.json); also counts the points of each IRH, for check_compiled()."""
    report = Report()
    points: dict[int, int] = {}
    structure = record["structure"]
    for f in record["files"]:
        name = f["name"]
        path = dataset_dir / "raw" / name
        if not path.is_file():
            report.errors.append(f"{name}: missing")
            continue
        sep = SEPARATORS.get(path.suffix.lower())
        if sep is None:
            report.errors.append(f"{name}: only .csv and .tab files")
            continue
        skip = _header_line(path)
        try:
            header = list(pd.read_csv(path, sep=sep, skiprows=skip, nrows=0).columns)
        except Exception as e:
            report.errors.append(f"{name}: unreadable ({e})")
            continue
        if len(header) < 2:
            report.errors.append(f"{name}: one column only: the values must be separated by {'tabs' if sep == chr(9) else 'commas'}")
            continue
        coords = ("PSX", "PSY") if {"PSX", "PSY"} <= set(header) else ("lon", "lat") if {"lon", "lat"} <= set(header) else None
        if coords is None:
            report.errors.append(f"{name}: no coordinates (PSX and PSY, or lon and lat)")
            continue

        kind = f["kind"]
        if kind == "irh":
            values = ["IRH_DEPTH"]
        elif kind == "variable":
            values = [f["variable"]]
        else:
            values = [f"IRH{n}" for n in f.get("irh_columns") or []] + [v for v in VARIABLES if v in header]
        line_column = next((c for c in LINE_ID_COLUMNS if c in header), None)
        required = values
        missing = [c for c in required if c not in header]
        if structure == "layer" and line_column is None:
            missing.insert(0, "Line_ID")
        if missing:
            report.errors.append(f"{name}: no {', '.join(missing)} column")
            continue
        known = set(COORDINATES) | set(VARIABLES) | {*LINE_ID_COLUMNS, "IRH_DEPTH", "acq_year"}
        unknown = [c for c in header if c not in known and not IRH_COLUMN.match(c)]
        if unknown:
            report.warnings.append(f"{name}: columns ignored: {', '.join(unknown[:10])}")

        rows = invalid = outside = negative = deep = 0
        line_ids: set[str] = set()
        counts = {c: 0 for c in values}
        numeric = [*coords, *values]
        try:
            for chunk in pd.read_csv(
                path, sep=sep, skiprows=skip, dtype=str, keep_default_na=False, chunksize=CHUNK,
                usecols=list(dict.fromkeys(numeric + ([line_column] if line_column else []))),
            ):
                rows += len(chunk)
                if line_column:
                    line_ids.update(chunk[line_column].str.strip().unique())
                table = {}
                for c in numeric:
                    text = chunk[c].str.strip()
                    number = pd.to_numeric(text.where(~text.isin(MISSING)), errors="coerce")
                    invalid += int((number.isna() & ~text.isin(MISSING)).sum())
                    table[c] = number.to_numpy(dtype=float)
                x, y = table[coords[0]], table[coords[1]]
                if coords == ("lon", "lat"):
                    out = (y >= -60) | (y < -90) | (np.abs(x) > 360)
                else:
                    out = (np.abs(x) > PS_EXTENT) | (np.abs(y) > PS_EXTENT)
                outside += int(out.sum())
                for c in values:
                    v = table[c]
                    counts[c] += int(np.isfinite(v).sum())
                    if c in ("SURF_ELEV", "BED_ELEV"):
                        continue
                    negative += int((v < 0).sum())
                    deep += int((v > MAX_DEPTH).sum())
        except Exception as e:
            report.errors.append(f"{name}: unreadable ({e})")
            continue

        if invalid:
            report.errors.append(f"{name}: {plural(invalid, 'value')} {are(invalid)} not numbers")
        if outside:
            report.errors.append(f"{name}: {plural(outside, 'point')} {are(outside)} outside Antarctica")
        if negative:
            report.errors.append(f"{name}: {plural(negative, 'value')} {are(negative)} negative (depths and thicknesses are in metres below the surface)")
        if deep:
            report.warnings.append(f"{name}: {plural(deep, 'value')} deeper than {MAX_DEPTH} m")
        if f.get("rows") and rows != f["rows"]:
            report.warnings.append(f"{name}: {rows:,} rows, the form counted {f['rows']:,}")
        if kind == "irh" and f.get("irh") is not None:
            points[int(f["irh"])] = points.get(int(f["irh"]), 0) + counts["IRH_DEPTH"]
        elif kind == "line":
            for n in f.get("irh_columns") or []:
                points[n] = points.get(n, 0) + counts[f"IRH{n}"]
    report.summary["raw points"] = points
    return report


# --- The compiled dataset


def check_compiled(dataset_dir: Path, raw_points: dict[int, int], surface=None) -> Report:
    """Check the compiled h5 files of a dataset (after IndexDatabase assigned the ages).
    surface: anta_database.web.irh.Bedmap3Surface around the dataset, or None."""
    import xarray as xr

    report = Report()
    paths = sorted(glob.glob(os.path.join(dataset_dir, "h5", "*.h5")))
    if not paths:
        report.errors.append("The compilation made no line")
        return report

    points: dict[int, int] = {}
    ages: dict[int, float] = {}
    too_deep: dict[int, int] = {}
    spikes: dict[int, int] = {}
    crossings: dict[tuple[int, int], list[int]] = {}  # (older, younger): [crossing, shared]
    off_ice = total_points = 0
    xmin = ymin = np.inf
    xmax = ymax = -np.inf

    for path in paths:
        with xr.open_dataset(path) as ds:
            x = ds.PSX.values.astype(float)
            y = ds.PSY.values.astype(float)
            total_points += len(x)
            if np.isfinite(x).any():
                xmin, xmax = min(xmin, np.nanmin(x)), max(xmax, np.nanmax(x))
                ymin, ymax = min(ymin, np.nanmin(y)), max(ymax, np.nanmax(y))
            if "IRH_DEPTH" not in ds:
                continue
            thk = ds.ICE_THK.values.astype(float) if "ICE_THK" in ds else np.full(len(x), np.nan)
            if surface is not None:
                bedmap_thk = surface.thickness(x, y)
                off_ice += int((~(bedmap_thk > 0)).sum())
                thk = np.where(np.isfinite(thk), thk, bedmap_thk)
            step = np.hypot(np.diff(x), np.diff(y))
            depth = ds.IRH_DEPTH.transpose("point", "IRH").values.astype(float)
            irhs = [int(i) for i in ds.IRH.values]
            if "IRH_AGE" in ds.coords or "IRH_AGE" in ds:
                for i, age in zip(irhs, ds.IRH_AGE.values.astype(float)):
                    if np.isfinite(age):
                        ages[i] = age
            for k, i in enumerate(irhs):
                d = depth[:, k]
                valid = np.isfinite(d)
                points[i] = points.get(i, 0) + int(valid.sum())
                too_deep[i] = too_deep.get(i, 0) + int((d > thk * THICKNESS_TOLERANCE + 10).sum())
                jump = np.abs(np.diff(d)) > SPIKE_DEPTH
                spikes[i] = spikes.get(i, 0) + int((jump & (step < SPIKE_DISTANCE)).sum())
            # An older IRH must be deeper than a younger one
            dated = [(k, i) for k, i in enumerate(irhs) if i in ages]
            for a, (ka, ia) in enumerate(dated):
                for kb, ib in dated[a + 1:]:
                    if ages[ia] == ages[ib]:
                        continue
                    (ko, io), (ky, iy) = ((ka, ia), (kb, ib)) if ages[ia] > ages[ib] else ((kb, ib), (ka, ia))
                    both = np.isfinite(depth[:, ko]) & np.isfinite(depth[:, ky])
                    if not both.any():
                        continue
                    pair = crossings.setdefault((io, iy), [0, 0])
                    pair[0] += int((depth[both, ko] < depth[both, ky] - 1).sum())
                    pair[1] += int(both.sum())

    for i in sorted(points):
        lost = raw_points.get(i, 0) - points[i]
        if lost > max(0.005 * raw_points.get(i, 0), 0):
            report.warnings.append(f"IRH {i}: {plural(lost, 'point')} of the files were not compiled ({share(lost, raw_points[i])})")
        if too_deep.get(i):
            report.warnings.append(
                f"IRH {i}: {plural(too_deep[i], 'point')} deeper than the ice thickness ({share(too_deep[i], points[i])}): "
                "check the depths (and the firn correction)"
            )
        if spikes.get(i, 0) > SPIKE_SHARE * points[i]:
            report.warnings.append(
                f"IRH {i}: {plural(spikes[i], 'jump')} of more than {SPIKE_DEPTH:g} m between points less than "
                f"{SPIKE_DISTANCE:g} m apart"
            )
    for i in sorted(set(raw_points) - set(points)):
        report.warnings.append(f"IRH {i}: not in the compiled dataset")
    # The IRHs out of order: the one crossing the most others, then the next one among those left
    crossing = {pair for pair, (n, shared) in crossings.items() if n > 10 and n > CROSSING_SHARE * shared}
    while crossing:
        count: dict[int, int] = {}
        for pair in crossing:
            for i in pair:
                count[i] = count.get(i, 0) + 1
        worst = max(sorted(count), key=count.get)
        others = sorted(j for pair in crossing if worst in pair for j in pair if j != worst)
        report.warnings.append(
            f"IRH {worst} ({ages[worst] / 1000:g} ka) crosses {plural(len(others), 'IRH')} of other ages "
            f"(IRH {', '.join(map(str, others))}): check its number and age"
        )
        crossing = {pair for pair in crossing if worst not in pair}
    if off_ice:
        report.warnings.append(f"{plural(off_ice, 'point')} off the ice of Bedmap3 ({share(off_ice, total_points)})")

    report.summary.update({
        "lines": len(paths),
        "IRHs": len(points),
        "points": total_points,
        "IRH points": sum(points.values()),
        "undated IRHs": sorted(set(points) - set(ages)),
    })
    if np.isfinite(xmin):
        report.summary["bbox"] = [round(float(v)) for v in (xmin, xmax, ymin, ymax)]
    return report
