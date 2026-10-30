"""
Export the catalogue of datasets for the download page of the website.

For every dataset: its citation and DOIs, institutes, projects, radar instruments,
acquisition years, regions and IMBIE basins (from the SQLite index, with the same
logic as Database.query), the IRHs with their ages, the variables, the number of
lines and points, and its coverage on a coarse grid (for a small map).

The files to download are not listed here: the download page reads them from the
manifest of the published database, so they always match the bucket.

    python -m anta_database.web.catalog /path/to/AntADatabase -o web/public/data/catalog

Output:
    catalog.json
"""

import argparse
import glob
import json
import os
from typing import Dict, List

import h5py
import numpy as np

from anta_database.database.database import Database

# Coverage grid of the map (EPSG:3031)
CELL_KM = 10
EXTENT_KM = 3335
NO_VALUES = {"", "nan", "none", "None"}


def _clean(values) -> List[str]:
    return sorted({str(v) for v in values if v is not None and str(v) not in NO_VALUES})


def _years(values: List[str]) -> str:
    """Span of acquisition years, e.g. "2008-2018"."""
    years = []
    for v in values:
        years += [int(y) for y in str(v).split("-") if y.strip().isdigit()]
    if not years:
        return ""
    return str(min(years)) if min(years) == max(years) else f"{min(years)}-{max(years)}"


def _scan_files(files: List[str]) -> Dict:
    """IRHs, variables, points and coverage cells of the h5 files of a dataset."""
    n = 2 * EXTENT_KM // CELL_KM
    cells = np.zeros(n * n, dtype=bool)
    irhs: Dict[int, Dict] = {}
    variables = set()
    n_points = 0
    for path in files:
        with h5py.File(path, "r") as f:
            variables.update(k for k in f.keys() if k not in ("point", "IRH", "PSX", "PSY", "Distance", "IRH_AGE", "IRH_AGE_UNC"))
            x = f["PSX"][:] / 1000
            y = f["PSY"][:] / 1000
            ok = np.isfinite(x) & np.isfinite(y)
            n_points += int(ok.sum())
            ix = ((x[ok] + EXTENT_KM) // CELL_KM).astype(int).clip(0, n - 1)
            iy = ((y[ok] + EXTENT_KM) // CELL_KM).astype(int).clip(0, n - 1)
            cells[iy * n + ix] = True
            ages = f["IRH_AGE"][:] if "IRH_AGE" in f else [np.nan] * len(f["IRH"])
            uncs = f["IRH_AGE_UNC"][:] if "IRH_AGE_UNC" in f else [np.nan] * len(f["IRH"])
            for irh, age, unc in zip(f["IRH"][:], ages, uncs):
                irhs.setdefault(int(irh), {
                    "irh": int(irh),
                    "age": float(age) if np.isfinite(age) else None,
                    "age_unc": float(unc) if np.isfinite(unc) else None,
                })
    # Occupied cells, as gaps between their flat indices (smaller json)
    flat = np.flatnonzero(cells)
    return {
        "irhs": [irhs[k] for k in sorted(irhs)],
        "variables": sorted(variables),
        "n_points": n_points,
        "coverage": np.diff(flat, prepend=0).tolist(),
    }


def export_catalog(database_dir: str, output_dir: str) -> Dict:
    db = Database(database_dir, check_updates=False)
    datasets = []
    for d in sorted(glob.glob(os.path.join(database_dir, "*", "h5"))):
        name = os.path.basename(os.path.dirname(d))
        files = sorted(glob.glob(os.path.join(d, "*.h5")))
        if not files:
            continue
        md = db.query(dataset=name)
        entry = {
            "name": name,
            "citation": ", ".join(md["reference"]),
            "doi_dataset": _clean(md["DOI_dataset"]),
            "doi_publication": _clean(md["DOI_publication"]),
            "institutes": _clean(md["institute"]),
            "projects": _clean(md["project"]),
            "radar_instruments": _clean(md["radar_instrument"]),
            "acquisition_years": _years(md["acquisition_year"]),
            "regions": _clean(md["region"]),
            "imbie_basins": _clean(md["IMBIE_basin"]),
            "n_lines": len(files),
            **_scan_files(files),
        }
        datasets.append(entry)
        print(f"{name}: {len(files)} lines, {len(entry['irhs'])} IRHs")

    catalog = {
        "format": 1,
        "coverage_grid": {"cell_km": CELL_KM, "x0_km": -EXTENT_KM, "y0_km": -EXTENT_KM, "n": 2 * EXTENT_KM // CELL_KM},
        "datasets": datasets,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "catalog.json"), "w", encoding="utf-8") as f:
        json.dump(catalog, f, separators=(",", ":"))
    return catalog


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export the dataset catalogue for the download page.")
    parser.add_argument("database_dir", help="AntADatabase directory")
    parser.add_argument("-o", "--output", default="web/public/data/catalog", help="Output directory")
    args = parser.parse_args(argv)
    export_catalog(args.database_dir, args.output)


if __name__ == "__main__":
    main()
