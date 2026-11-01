"""Build the published v1 of the database: the h5 files of the maintainer's working copy, converted
to .nc with the encoding of the database (anta_database/netcdf.py), without compiling again.

The h5 files are identical to the publication 2026.10.30-1403, but for the ages of Cavitte_2020,
which come from older metadata: the metadata is the published one (the ages of the papers), with the
fixes below. Every change to the published data is here, so it can be read and reported.

    python docs/harmonisation/build_v1.py [--datasets A B]
"""
import argparse
import ast
import glob
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from importlib.resources import files
from pyproj import Transformer

from anta_database import IndexDatabase
from anta_database.netcdf import write

HOME = Path.home() / "data" / "isochrones"
WORKING = HOME / "database_project" / "myAntADatabase"  # h5 files, raw files: only read
PUBLISHED = HOME / "AntADatabase-release-test"  # the metadata as published
BUILD = HOME / "AntADatabase-v1-build"
VAR_ATTRS = json.loads((Path(__file__).resolve().parents[2] / "portal" / "var_attrs.json").read_text())

DATASETS = """Ashmore_2020 Beem_2021 Bodart_2021 Bodart_2025a Bodart_2025b Cavitte_2020 Chung_2023
Franke_2025 Jacobel_2005 Leysinger_Vieli_2011 Muldoon_2023 Mulvaney_2023 Napoleoni_2026
Sanderson_2024 Siegert_2004 Wang_2023 Winter_2018 Yan_2025""".split()

BODART_PAPER = "https://doi.org/10.5194/tc-20-1379-2026"
NAPOLEONI_DATA = "https://doi.org/10.5194/tc-20-2793-2026-supplement"
LEYSINGER_JUMP_KM = 50

# One name for each institute, project and radar. A name that joined several becomes a list
VOCABULARY = {
    "institute": {
        "UTIG-AAD": ["UTIG", "AAD"],
        "PRIC-UTIG-AAD": ["PRIC", "UTIG", "AAD"],
        "NASA-CRESIS": ["CReSIS"],
        "SPRI-NSF-DTU": ["SPRI-NSF-TUD"],
    },
    "project": {
        "ICECAP-OIA": ["ICECAP", "OIA"],
        "ICECAP2": ["ICECAP"],
        "AGAP-NORTH": ["AGAP"],
        "AGAP-SOUTH-GAMBIT": ["AGAP"],
        "WISE_ISODYN": ["WISE-ISODYN"],
    },
    "radar instrument": {"SPRI-NSF-DTU-radar": ["SPRI-NSF-TUD-radar"]},
}

# What was done to the raw files, shown by db.query(dataset=...) for a single dataset
PROCESSING = {
    "Cavitte_2020": "Firn correction of +14.6 m added to IRH_DEPTH: the published depths do not have "
    "it (value of Cavitte et al., 2021; with it, the depths match their Table 3 at EPICA Dome C).",
    "Wang_2023": "Two-way travel times converted to depths with 0.08425 m/ns (half of 168.5 m/us), "
    "and a firn correction of +15.5 m added (IRH_DEPTH, ICE_THK), as in Wang et al. (2023).",
    "Leysinger_Vieli_2011": "The raw file has no line IDs: lines named after the dataset, and cut where "
    f"consecutive points are more than {LEYSINGER_JUMP_KM} km apart.",
    "Siegert_2004": "The raw file has no line ID: the line is named after the dataset. Age uncertainties "
    "from Ashmore et al. (2020), for all layers but the 0.8 ka one; not in the published files.",
    "Winter_2018": "Lines separated with the column Line of the published files.",
    "Jacobel_2005": "Age uncertainty of 0.5 ka from Bodart et al. (2021); not in the published files.",
    "Yan_2025": "Line IDs are the names of the raw files, without their ending _depth.",
}


# Age uncertainties not in the published files, by IRH (years): see PROCESSING
AGE_UNCERTAINTIES = {"Siegert_2004": {7: 160, 10: 175, 11: 181, 12: 324}, "Jacobel_2005": {1: 500}}


def one_name(key: str, value):
    """The value of an institute, project or radar, with the names of VOCABULARY."""
    names = [value] if isinstance(value, str) else value
    new = list(dict.fromkeys(n for name in names for n in VOCABULARY[key].get(name, [name])))
    return new[0] if len(new) == 1 else new


def fix_metadata(name: str, rows: list[dict]) -> list[dict]:
    for row in rows:
        row.pop("data type", None)
        row["DOI article"] = row.pop("DOI publication")
        if row["IRH"] in AGE_UNCERTAINTIES.get(name, {}):
            row["age uncertainty"] = AGE_UNCERTAINTIES[name][row["IRH"]]
        if "radar_instrument" in row:
            row["radar instrument"] = row.pop("radar_instrument")
        if name == "Siegert_2004":
            row["citation"] = "Siegert and Payne (2004)"
        if name in ("Bodart_2025a", "Bodart_2025b"):
            row["DOI article"] = BODART_PAPER
        if name == "Bodart_2025a":
            for key in ("project", "radar instrument"):
                if isinstance(row[key], str) and "/" in row[key]:
                    row[key] = row[key].split("/")
        if name == "Cavitte_2020":
            # IRH 7 pointed to the file of IRH 6 (a typo in the metadata, not in the data)
            row["raw file"] = f"OIA_EDC_IRH{row['IRH']}.csv"
        if name == "Mulvaney_2023" and row.get("institute") is None:
            row["institute"] = "BAS"
        if name == "Napoleoni_2026":
            row["DOI dataset"] = NAPOLEONI_DATA
        if name == "Beem_2021":
            # The two flights of the SPC survey (Beem et al., 2021), as in the files
            row["acquisition year"] = "2016-2017"
        for key in VOCABULARY:
            if row.get(key):
                row[key] = one_name(key, row[key])
    return rows


def missing(value) -> bool:
    return str(value) in ("nan", "none", "None", "")


def fill_attrs(ds: xr.Dataset, rows: list[dict], name: str) -> None:
    """The metadata the file lacks, from raw_md.json. What a file has of its own is kept."""
    for attr in ("radar instrument", "acquisition year", "project", "institute"):
        values = {json.dumps(r[attr]) for r in rows if r.get(attr) is not None}
        if missing(ds.attrs.get(attr)) and len(values) == 1:
            value = next(r[attr] for r in rows if r.get(attr) is not None)
            ds.attrs[attr] = str(value)
    for key in VOCABULARY:
        value = ds.attrs.get(key)
        if isinstance(value, str) and not missing(value):
            value = ast.literal_eval(value) if value.startswith("[") else value
            ds.attrs[key] = str(one_name(key, value))
        elif value is not None and not isinstance(value, str):
            ds.attrs[key] = one_name(key, np.asarray(value).tolist())
    ds.attrs.pop("DOI publication", None)
    ds.attrs["DOI article"] = rows[0]["DOI article"]
    ds.attrs["DOI dataset"] = rows[0]["DOI dataset"]
    ds.attrs["processing"] = PROCESSING.get(name, "")


def set_ages(ds: xr.Dataset, rows: list[dict]) -> xr.Dataset:
    ids, ages, uncs = IndexDatabase._target_ages(ds.IRH.values, pd.DataFrame(rows), None)
    from anta_database.indexing.index_database import AGE_ATTRS

    ds["IRH_AGE"] = ("IRH", IndexDatabase._stored(ages), AGE_ATTRS["IRH_AGE"])
    ds["IRH_AGE_UNC"] = ("IRH", IndexDatabase._stored(uncs), AGE_ATTRS["IRH_AGE_UNC"])
    return ds.set_coords("IRH_AGE")


def set_var_attrs(ds: xr.Dataset) -> None:
    for item in VAR_ATTRS:
        if item["variable"] in ds.variables:
            ds[item["variable"]].attrs.update({k: item[k] for k in ("units", "long_name", "description")})


def recompute(ds: xr.Dataset, basins: gpd.GeoDataFrame) -> xr.Dataset:
    """Distance and basins of a part of a former line."""
    ds = ds.assign_coords(point=np.arange(ds.sizes["point"]))
    steps = np.hypot(np.diff(ds.PSX.values), np.diff(ds.PSY.values))
    ds["Distance"] = ("point", np.r_[0, np.cumsum(steps)], ds["Distance"].attrs)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(ds.PSX.values, ds.PSY.values), crs=basins.crs)
    joined = gpd.sjoin(points, basins, how="inner", predicate="within")
    lookup = joined[joined["Regions"] != "Islands"][["Subregion", "Regions"]].drop_duplicates()
    ds.attrs["IMBIE_basins"] = json.dumps(dict(zip(lookup["Subregion"], lookup["Regions"])))
    return ds


def winter_lines() -> dict:
    """(event, PSX, PSY) of each point of Winter_2018 -> its line, from the column Line of the raw
    files. The event is the h5 file: some positions are in several flights."""
    transformer = Transformer.from_proj(
        "EPSG:4326", "+proj=stere +lon_0=0 +lat_0=-90 +lat_ts=-71 +datum=WGS84 +units=m +no_defs", always_xy=True
    )
    lookup = {}
    for path in sorted(glob.glob(str(WORKING / "Winter_2018" / "raw" / "*.tab"))):
        df = pd.read_csv(path, sep="\t", comment="#", usecols=["Event", "Longitude", "Latitude", "Line"], dtype={"Line": str})
        x, y = transformer.transform(df["Longitude"].astype(float).values, df["Latitude"].astype(float).values)
        lookup.update(zip(zip(df["Event"].str.replace("/", "_"), x, y), df["Line"]))
    return lookup


def parts_of(name: str, ds: xr.Dataset, stem: str, winter: dict, basins) -> list[tuple[str, xr.Dataset]]:
    """The lines (file name, data) in a file."""
    if name == "Siegert_2004":
        ds.attrs.update({"line ID": name, "line ID flag": "not_provided"})
        return [(name, ds)]
    if name == "Leysinger_Vieli_2011":
        jumps = np.hypot(np.diff(ds.PSX.values), np.diff(ds.PSY.values)) > LEYSINGER_JUMP_KM * 1000
        groups = np.r_[0, np.cumsum(jumps)]
    elif name == "Winter_2018":
        lines = np.array([winter.get((stem, x, y)) for x, y in zip(ds.PSX.values, ds.PSY.values)], dtype=object)
        if any(line is None for line in lines):
            raise ValueError(f"Winter_2018 {stem}: {sum(l is None for l in lines)} points without a line")
        names, groups = np.unique(lines.astype(str), return_inverse=True)
    elif name == "Yan_2025":
        line = stem.removesuffix("_depth")
        ds.attrs["line ID"] = line
        return [(line, ds)]
    else:
        return [(stem, ds)]
    parts = []
    for g in np.unique(groups):
        part = recompute(ds.isel(point=np.where(groups == g)[0]).copy(), basins)
        line = f"{name}_{g + 1}" if name == "Leysinger_Vieli_2011" else str(names[g])
        part.attrs["line ID"] = line
        part.attrs["line ID flag"] = "not_provided" if name == "Leysinger_Vieli_2011" else "original"
        parts.append((line, part))
    return parts


def convert(job: tuple) -> int:
    name, path, rows, winter = job
    basins = gpd.read_parquet(files("anta_database.data").joinpath("ANT_Basins_IMBIE2_v1.6.parquet"))
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        ds = ds.load()
    if name == "Cavitte_2020":
        # The compilation also added the firn correction to these two. An elevation must not
        # change, and the thickness of the raw files already matches Bedmap3 without it
        for var in ("ICE_THK", "BED_ELEV"):
            ds[var] = ds[var] - 14.6
    ds = ds.drop_vars([v for v in ("IRH_AGE", "IRH_AGE_UNC") if v in ds.variables])
    ds = set_ages(ds, rows)
    fill_attrs(ds, rows, name)
    set_var_attrs(ds)
    written = 0
    for stem, part in parts_of(name, ds, Path(path).stem, winter, basins):
        write(part, str(BUILD / name / "nc" / f"{stem}.nc"))
        written += 1
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--cpus", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()
    winter = winter_lines() if "Winter_2018" in args.datasets else {}
    for name in args.datasets:
        target = BUILD / name / "nc"
        target.mkdir(parents=True, exist_ok=True)
        for old in target.glob("*.nc"):
            old.unlink()
        rows = fix_metadata(name, json.loads((PUBLISHED / name / "raw_md.json").read_text()))
        (BUILD / name / "raw_md.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        if name in PROCESSING:
            (BUILD / name / "PROCESSING.txt").write_text(f"{name}: what was done to the raw files\n\n{PROCESSING[name]}\n")
        jobs = [(name, p, rows, winter if name == "Winter_2018" else {}) for p in sorted(glob.glob(str(WORKING / name / "h5" / "*.h5")))]
        with Pool(min(args.cpus, len(jobs))) as pool:
            print(name, len(jobs), "->", sum(pool.imap_unordered(convert, jobs)), "files", flush=True)
    if all(any((BUILD / n / "nc").glob("*.nc")) for n in DATASETS if (BUILD / n).exists()) and set(DATASETS) <= {p.name for p in BUILD.iterdir()}:
        IndexDatabase(str(BUILD), assign_ages=True, cpus=args.cpus).index_database()


if __name__ == "__main__":
    main()
