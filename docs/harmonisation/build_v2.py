"""Build v2 of the database: compile the raw files again, with the metadata of v1.

v1 (build_v1.py) converted the h5 files of an earlier compilation. v2 compiles the raw files of
the working copy (only read), so that:
- the points of every line are in order along it (compiling/lines.py, order_line);
- no line is cut: Leysinger_Vieli_2011 and Siegert_2004, without line IDs, are one line each,
  until Bedmap gives them their lines (docs/bedmap/README.md);
- the corrected files of AntADatabase-v2-source are used (fix_cavitte.py);
- a line only lists the layers it has values of (docs/harmonisation/metadata-issues.md);
- ice thickness, bed and surface elevation are taken from Bedmap where a line has none
  (anta_database/bedmap/reconnect.py; report in BUILD/bedmap_fill.csv).

Each dataset of BUILD gets links to its raw files, the metadata of v1 (raw_md.json, with the
fixes of build_v1.py), the column names, and compile_settings.json with what the compilation
does to the raw files (SETTINGS: the processing of v1).

    python docs/harmonisation/build_v2.py [--datasets A B] [--cpus N]
"""
import argparse
import json
import os
import shutil
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd

from anta_database import IndexDatabase
from anta_database.bedmap.reconnect import fill
from anta_database.compiling import CompileDatabase

HOME = Path.home() / "data" / "isochrones"
WORKING = HOME / "database_project" / "myAntADatabase"  # raw files: only read
SOURCES = HOME / "AntADatabase-v2-source"  # corrected raw files, used instead of those of WORKING
V1 = HOME / "AntADatabase-v1-build"  # the metadata of v1
BUILD = HOME / "AntADatabase-v2-build"
BEDMAP = HOME / "BEDMAP-build"  # anta_database.bedmap.convert
VAR_ATTRS = Path(__file__).resolve().parents[2] / "portal" / "var_attrs.json"

DATASETS = """Ashmore_2020 Beem_2021 Bodart_2021 Bodart_2025a Bodart_2025b Cavitte_2020 Chung_2023
Franke_2025 Jacobel_2005 Leysinger_Vieli_2011 Muldoon_2023 Mulvaney_2023 Napoleoni_2026
Sanderson_2024 Siegert_2004 Wang_2023 Winter_2018 Yan_2025""".split()

# What the compilation does to the raw files (compile.py, SETTINGS_FILE), as in v1
SETTINGS = {
    "Cavitte_2020": {
        "firn_correction": 14.6,
        "firn_correction_vars": ["IRH_DEPTH"],
        "note": "The published depths do not have the firn correction (value of Cavitte et al., 2021; with it, "
        "the depths match their Table 3 at EPICA Dome C). Depths interpolated across the gaps of the HiCARS "
        "lines (OIA_JKB2n_*, IRH 1-20) removed, with the picks of M. Cavitte (2025).",
    },
    "Wang_2023": {
        "wave_speed": 0.08425,
        "firn_correction": 15.5,
        "note": "Wave speed of 168.5 m/us and firn correction as in Wang et al. (2023).",
    },
    "Bodart_2021": {
        "drop_vars": {"R2_OIB_20181022_01.csv": ["ICE_THK"], "R3_OIB_20181022_01.csv": ["ICE_THK"]},
        "note": "Ice thickness of R2_OIB_20181022_01.csv and R3_OIB_20181022_01.csv left out: it repeats the same "
        "value at almost every point and is about 400 m below Bedmap3 (median), unlike the other files.",
    },
    "Leysinger_Vieli_2011": {"line_ids": "dataset"},
    "Siegert_2004": {
        "line_ids": "dataset",
        "note": "Age uncertainties from Ashmore et al. (2020), for all layers but the 0.8 ka one; not in the "
        "published files.",
    },
    "Jacobel_2005": {"note": "Age uncertainty of 0.5 ka from Bodart et al. (2021); not in the published files."},
    "Winter_2018": {"note": "Lines from the column Line of the published files."},
    "Yan_2025": {"note": "Line IDs are the names of the raw files, without their ending _depth."},
}
# Columns of the raw files taken as other variables than in the working copy
COLUMNS = {"Winter_2018": {"Event": "Line"}}  # the lines, not the events (flights)


def raw_files(name: str) -> dict:
    """Name in BUILD -> file: those of SOURCES instead of those of WORKING."""
    files = {}
    for src in sorted((WORKING / name / "raw").iterdir()):
        if src.is_file() and "." in src.name:
            files[src.name] = src
    corrected = SOURCES / name / "raw"
    if corrected.is_dir():
        for src in corrected.iterdir():
            if src.is_file() and not src.is_symlink():
                files[src.name] = src
    if name == "Yan_2025":
        files = {k.replace("_depth.", "."): v for k, v in files.items()}
    return files


def prepare(name: str) -> Path:
    """The directory of a dataset in BUILD, without the results of an earlier compilation."""
    out = BUILD / name
    for sub in ("raw", "nc", "pkl"):
        shutil.rmtree(out / sub, ignore_errors=True)
    (out / "raw").mkdir(parents=True)
    for file, src in raw_files(name).items():
        os.symlink(src, out / "raw" / file)
    shutil.copy2(V1 / name / "raw_md.json", out / "raw_md.json")
    columns = (WORKING / name / "original_new_column_names.csv").read_text().splitlines()
    header = columns[0].split(",")
    for old, new in COLUMNS.get(name, {}).items():
        header[header.index(old)] = new
    (out / "original_new_column_names.csv").write_text("\n".join([",".join(header), *columns[1:]]) + "\n")
    settings = out / "compile_settings.json"
    if name in SETTINGS:
        settings.write_text(json.dumps(SETTINGS[name], indent=1, ensure_ascii=False) + "\n")
    elif settings.exists():
        settings.unlink()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--cpus", type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()
    dirs = [str(prepare(name)) for name in args.datasets]
    CompileDatabase(dirs, var_attrs_json=str(VAR_ATTRS)).compile(cpus=args.cpus)
    # Missing ice thickness, bed and surface elevation from the Bedmap line of the same traces
    filled = fill(str(BEDMAP), str(BUILD), args.datasets, cpus=args.cpus)
    report = BUILD / "bedmap_fill.csv"
    if report.exists() and set(args.datasets) != set(DATASETS):
        filled = pd.concat([pd.read_csv(report).query("dataset not in @args.datasets"), filled])
    filled.to_csv(report, index=False)
    if set(DATASETS) <= {p.name for p in BUILD.iterdir() if (p / "nc").is_dir()}:
        IndexDatabase(str(BUILD), assign_ages=True, cpus=args.cpus).index_database()


if __name__ == "__main__":
    main()
