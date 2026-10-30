"""Fake submissions to test the portal, made from a compiled dataset (Chung_2023 by default).

The h5 files of the dataset are turned back into raw files with the standard column names, as a
contributor would upload them, under the dataset name Fake_2026:

    per_layer/  one .tab file per IRH (Line_ID, PSX, PSY, IRH_DEPTH) and one per variable
                (ICE_THK, BASAL_UNIT)
    per_line/   one .csv file per line (lon, lat, ICE_THK, BASAL_UNIT, IRH1, IRH2, ...)
    broken/     files with the mistakes the form should catch

The form of the submission page can be filled from fake_submission.json in dev mode, written to
web/public/dev/ (published with the site, for the development version /submit/?dev).

    python portal/dev/make_fake_data.py [--dataset DIR] [--output DIR]
"""

import argparse
import json
import os
import shutil
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NODATA = -9999

# Inverse of the polar stereographic projection of EPSG:3031 (WGS84, true scale at 71°S)
A = 6378137.0
E = 0.0818191908426215


def to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi_c = np.radians(71)
    m_c = np.cos(phi_c) / np.sqrt(1 - E**2 * np.sin(phi_c) ** 2)
    t_c = np.tan(np.pi / 4 - phi_c / 2) / ((1 - E * np.sin(phi_c)) / (1 + E * np.sin(phi_c))) ** (E / 2)
    rho = np.hypot(x, y)
    t = rho * t_c / (A * m_c)
    phi = np.pi / 2 - 2 * np.arctan(t)
    for _ in range(10):
        phi = np.pi / 2 - 2 * np.arctan(t * ((1 - E * np.sin(phi)) / (1 + E * np.sin(phi))) ** (E / 2))
    return np.degrees(np.arctan2(x, y)), -np.degrees(phi)


def read_dataset(path: str) -> list[xr.Dataset]:
    files = sorted(glob(os.path.join(path, "h5", "*.h5")))
    if not files:
        raise SystemExit(f"No h5 files in {path}/h5")
    return [xr.load_dataset(f, engine="h5netcdf") for f in files]


def all_irhs(lines: list[xr.Dataset]) -> dict[int, tuple[float, float]]:
    """Age and uncertainty of every IRH of the dataset, some are not in every line."""
    irhs = {}
    for ds in lines:
        for irh, age, unc in zip(ds.IRH.values, ds.IRH_AGE.values, ds.IRH_AGE_UNC.values):
            irhs.setdefault(int(irh), (float(age), float(unc)))
    return dict(sorted(irhs.items()))


def per_layer(lines: list[xr.Dataset], out: str) -> None:
    os.makedirs(out)
    for irh in all_irhs(lines):
        rows = []
        for ds in lines:
            if irh not in ds.IRH:
                continue
            depth = ds.IRH_DEPTH.sel(IRH=irh).values
            ok = np.isfinite(depth)
            rows.append(pd.DataFrame({
                "Line_ID": ds.attrs["line ID"],
                "PSX": ds.PSX.values[ok].round(1),
                "PSY": ds.PSY.values[ok].round(1),
                "IRH_DEPTH": depth[ok].round(2),
            }))
        path = os.path.join(out, f"IRH{irh}.tab")
        with open(path, "w") as f:
            if irh == 1:
                # A description at the top, commented out
                f.write("# Fake dataset made from Chung et al. (2023)\n# Depth of the IRH in m below the ice surface\n")
            pd.concat(rows).to_csv(f, sep="\t", index=False)
    for var in ("ICE_THK", "BASAL_UNIT"):
        rows = []
        for ds in lines:
            if var not in ds:
                continue
            values = ds[var].values
            ok = np.isfinite(values)
            rows.append(pd.DataFrame({
                "Line_ID": ds.attrs["line ID"],
                "PSX": ds.PSX.values[ok].round(1),
                "PSY": ds.PSY.values[ok].round(1),
                var: values[ok].round(2),
            }))
        pd.concat(rows).to_csv(os.path.join(out, f"{var}.tab"), sep="\t", index=False)


def per_line(lines: list[xr.Dataset], out: str) -> None:
    os.makedirs(out)
    for ds in lines:
        lon, lat = to_lonlat(ds.PSX.values, ds.PSY.values)
        df = pd.DataFrame({"lon": lon.round(6), "lat": lat.round(6)})
        for var in ("ICE_THK", "BASAL_UNIT"):
            if var in ds:
                df[var] = ds[var].values.round(2)
        for irh in ds.IRH.values:
            df[f"IRH{irh}"] = ds.IRH_DEPTH.sel(IRH=irh).values.round(2)
        df.fillna(NODATA).to_csv(os.path.join(out, f"{ds.attrs['line ID']}.csv"), index=False)


def broken(good: str, out: str) -> None:
    """Mistakes the form should catch, from the per-layer files."""
    os.makedirs(out)
    # Not an accepted extension
    shutil.copy(os.path.join(good, "IRH1.tab"), os.path.join(out, "IRH1.txt"))
    # Tab separated, but named .csv
    shutil.copy(os.path.join(good, "IRH2.tab"), os.path.join(out, "IRH2.csv"))
    # Non-standard column names
    df = pd.read_csv(os.path.join(good, "IRH3.tab"), sep="\t")
    df.rename(columns={"IRH_DEPTH": "Depth (m)", "Line_ID": "line_id"}).to_csv(
        os.path.join(out, "IRH3.tab"), sep="\t", index=False)
    # Elevations instead of depths, and points north of Antarctica
    df = pd.read_csv(os.path.join(good, "IRH4.tab"), sep="\t")
    df["IRH_DEPTH"] = 3200 - df["IRH_DEPTH"] - 4000
    df.loc[:9, "PSY"] = 9e6
    df.to_csv(os.path.join(out, "IRH4.tab"), sep="\t", index=False)
    # Variables in a layer file, which should be separate files
    df = pd.read_csv(os.path.join(good, "IRH5.tab"), sep="\t")
    df["ICE_THK"] = 3000.0
    df.to_csv(os.path.join(out, "IRH5.tab"), sep="\t", index=False)
    # A description at the top that is not commented out
    df = pd.read_csv(os.path.join(good, "IRH6.tab"), sep="\t")
    with open(os.path.join(out, "IRH6.tab"), "w") as f:
        f.write("Fake dataset made from Chung et al. (2023)\n")
        df.to_csv(f, sep="\t", index=False)
    # A "#" in the data, which would cut the line
    lines = open(os.path.join(good, "IRH7.tab")).read().splitlines()
    lines[5] += "  # checked"
    with open(os.path.join(out, "IRH7.tab"), "w") as f:
        f.write("\n".join(lines) + "\n")


def form_data(path: str, lines: list[xr.Dataset]) -> dict:
    """Metadata of the original dataset, as the form would be filled."""
    with open(os.path.join(path, "raw_md.json")) as f:
        md = json.load(f)[0]
    listed = lambda v: v if isinstance(v, list) else ([v] if v else [])
    return {
        "contributor": {"name": "Test Contributor", "email": "test@example.org",
                        "affiliation": "University of Bern", "orcid": ""},
        "licence": "yes",
        "publication": {"first_author": "Fake", "et_al": True, "year": "2026",
                        "doi_publication": "10.5194/fake-2026", "doi_dataset": md.get("DOI dataset") or ""},
        "survey": {"institutes": listed(md.get("institute")), "projects": listed(md.get("project")),
                   "radar_instruments": listed(md.get("radar instrument")),
                   "acquisition_year": str(md.get("acquisition year") or "")},
        "layers": [{"irh": irh, "age": age, "age_unc": unc} for irh, (age, unc) in all_irhs(lines).items()],
        "nodata": str(NODATA),
        "notes": "Fake submission made from Chung_2023 by portal/dev/make_fake_data.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=os.path.expanduser("~/data/isochrones/AntADatabase/Chung_2023"))
    parser.add_argument("--output", default=os.path.join(HERE, "fake_data"))
    args = parser.parse_args()

    lines = read_dataset(args.dataset)
    shutil.rmtree(args.output, ignore_errors=True)
    per_layer(lines, os.path.join(args.output, "per_layer"))
    per_line(lines, os.path.join(args.output, "per_line"))
    broken(os.path.join(args.output, "per_layer"), os.path.join(args.output, "broken"))

    dev = os.path.join(REPO, "web", "public", "dev")
    os.makedirs(dev, exist_ok=True)
    with open(os.path.join(dev, "fake_submission.json"), "w") as f:
        json.dump(form_data(args.dataset, lines), f, indent=2)

    n = sum(ds.sizes["point"] for ds in lines)
    print(f"{len(lines)} lines, {len(all_irhs(lines))} IRHs, {n} points -> {args.output}")
    print(f"Form data -> {dev}/fake_submission.json")


if __name__ == "__main__":
    main()
