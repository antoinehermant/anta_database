"""Remove the interpolated depths of the HiCARS lines of Cavitte_2020, for v2.

The published files (OIA_EDC_IRH1-26.csv, https://doi.org/10.15784/601411) fill the gaps of the
HiCARS picks (lines OIA_JKB2n_*) with interpolated depths. Marie Cavitte's picks of these lines
(HiCARS_Bed.txt, HiCARS_IRH_1-19.txt, 2025-12-28; one file per IRH, shifted by one: HiCARS_Bed is
IRH 1, HiCARS_IRH_n is IRH n + 1) have the same points without the fills. At every point in both,
their depth is the published one + 10 m (their firn correction), so the published values stay:
only the depths of the HiCARS lines with no pick at the same position (1 m) become nan. IRH 21-26
and the other lines (MCoRDS, DELORES) are not in the picks and stay as published.

The other lines have such fills too, without picks to compare (anta_json #10). In a fill the depth
follows a straight line, rounded to 0.1 m: from one point to the next its change is the same but
for the rounding (second difference at most 0.1 m), when real picks vary by about 1 m. With
--min-km, the fills of all the lines and IRHs of at least that length along the line become nan
too, but for their two ends (picks, which the straight line joins). Shorter ones can be smooth picks.

Writes the corrected files, byte for byte the published ones but for these depths, to the v2
sources (or --out); the files left as published are links to the working copy (only read).

    python docs/harmonisation/fix_cavitte.py [--min-km 2] [--out DIR]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

HOME = Path.home() / "data" / "isochrones"
WORKING = HOME / "database_project" / "myAntADatabase" / "Cavitte_2020"  # only read
PICKS = HOME / "Cavitte_fixed_data"
OUT = HOME / "AntADatabase-v2-source" / "Cavitte_2020"
LAYERS = 20  # in the picks
ALL_LAYERS = 26
SAME_POINT = 1.0  # m
FLAT = 0.1 + 1e-6  # m: largest second difference of a straight line rounded to 0.1 m
DEPTH_COLUMN = "IRHdepth_m"


def picks(irh: int) -> pd.DataFrame:
    name = "HiCARS_Bed.txt" if irh == 1 else f"HiCARS_IRH_{irh - 1}.txt"
    df = pd.read_csv(PICKS / name, sep="\t")
    return df[df["depth(m)"].notna()]


def rewrite(src: Path, out: Path, remove: np.ndarray) -> None:
    """src byte for byte into out, but for the depth of the rows removed (nan). out may be src, or a
    link to the working copy: then replaced by a file, never written through."""
    tmp = out.with_suffix(".tmp")
    row, header, column = -1, None, None
    with open(src, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8", newline="") as fout:
        for text in fin:
            if text.startswith("#") or not text.strip():
                fout.write(text)
                continue
            if header is None:
                header = text.rstrip("\r\n").split(",")
                column = header.index(DEPTH_COLUMN)
                fout.write(text)
                continue
            row += 1
            if remove[row]:
                end = "\r\n" if text.endswith("\r\n") else "\n"
                fields = text.rstrip("\r\n").split(",")
                fields[column] = "nan"
                text = ",".join(fields) + end
            fout.write(text)
    assert row + 1 == len(remove), f"{src.name}: {row + 1} rows written, {len(remove)} read"
    if out.is_symlink():
        out.unlink()
    os.replace(tmp, out)


def fills(raw: pd.DataFrame, min_km: float) -> np.ndarray:
    """The rows of the straight stretches of at least min_km along their line, but for their ends."""
    remove = np.zeros(len(raw), bool)
    depth = raw[DEPTH_COLUMN].to_numpy()
    for _, rows in raw.groupby("line", sort=False).indices.items():
        # Runs of points with a depth, in the order of the file (along the line)
        ok = np.isfinite(depth[rows])
        breaks = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        for start, stop in zip(breaks[::2], breaks[1::2]):
            run = rows[start:stop]
            if len(run) < 3:
                continue
            z = depth[run]
            flat = np.abs(z[2:] - 2 * z[1:-1] + z[:-2]) <= FLAT
            along = np.r_[0, np.cumsum(np.hypot(np.diff(raw["x"].to_numpy()[run]), np.diff(raw["y"].to_numpy()[run])))]
            edges = np.flatnonzero(np.diff(np.r_[0, flat.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                # second differences a..b-1: points a..b+1 on one straight line, a and b+1 its ends
                if along[b + 1] - along[a] >= min_km * 1000:
                    remove[run[a + 1 : b + 1]] = True
    return remove


def remove_fills(irh: int, min_km: float) -> dict:
    out = OUT / "raw" / f"OIA_EDC_IRH{irh}.csv"
    raw = pd.read_csv(out, comment="#", usecols=["x", "y", DEPTH_COLUMN, "line"])
    remove = fills(raw, min_km)
    if remove.any():
        rewrite(out.resolve() if out.is_symlink() else out, out, remove)
    lines = raw.loc[remove, "line"].value_counts()
    return {"IRH": irh, "removed": int(remove.sum()), "lines": lines.to_dict()}


def fix_layer(irh: int, transformer: Transformer) -> dict:
    src = WORKING / "raw" / f"OIA_EDC_IRH{irh}.csv"
    raw = pd.read_csv(src, comment="#", usecols=["x", "y", DEPTH_COLUMN, "line"])
    new = picks(irh)
    x, y = transformer.transform(new["lon"].values, new["lat"].values)
    on_lines = raw["line"].isin(set(new["line"])) & raw[DEPTH_COLUMN].notna()
    distance, _ = cKDTree(np.c_[x, y]).query(raw.loc[on_lines, ["x", "y"]].to_numpy())
    remove = np.zeros(len(raw), bool)
    remove[np.flatnonzero(on_lines)[distance >= SAME_POINT]] = True
    # The published values where both have a pick: the picks are deeper by their firn correction
    kept = np.flatnonzero(on_lines)[distance < SAME_POINT]
    nearest = cKDTree(np.c_[x, y]).query(raw.loc[kept, ["x", "y"]].to_numpy())[1]
    offset = raw.loc[kept, DEPTH_COLUMN].to_numpy() - new["depth(m)"].to_numpy()[nearest]
    assert np.allclose(offset, -10.0, atol=0.05), f"IRH {irh}: picks not the published depths + 10 m"

    rewrite(src, OUT / "raw" / src.name, remove)
    lines = raw.loc[remove, "line"].value_counts()
    return {"IRH": irh, "removed": int(remove.sum()), "kept": len(kept), "lines": lines.to_dict()}


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-km", type=float, help="also remove the straight stretches of at least this length")
    parser.add_argument("--out", type=Path, default=OUT, help="the dataset directory written")
    args = parser.parse_args()
    OUT = args.out
    transformer = Transformer.from_crs(
        "EPSG:4326", "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +datum=WGS84 +units=m", always_xy=True
    )
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    report = [fix_layer(irh, transformer) for irh in range(1, LAYERS + 1)]
    # The rest as published: links to the working copy
    for src in sorted((WORKING / "raw").iterdir()):
        dst = OUT / "raw" / src.name
        if not dst.exists():
            os.symlink(src, dst)
    # The metadata comes from anta_json (build_v2.py): only the column names here
    shutil.copy2(WORKING / "original_new_column_names.csv", OUT / "original_new_column_names.csv")
    for r in report:
        print(f"IRH {r['IRH']:2d}: {r['removed']:6d} depths removed, {r['kept']:6d} kept, on {len(r['lines'])} lines")
    print("total removed (HiCARS picks):", sum(r["removed"] for r in report))
    result = {"hicars_picks": report}
    if args.min_km:
        stretches = [remove_fills(irh, args.min_km) for irh in range(1, ALL_LAYERS + 1)]
        for r in stretches:
            print(f"IRH {r['IRH']:2d}: {r['removed']:6d} depths of straight stretches of {args.min_km:g} km or more, on {len(r['lines'])} lines")
        print("total removed (straight stretches):", sum(r["removed"] for r in stretches))
        result["straight_stretches"] = {"min_km": args.min_km, "layers": stretches}
    (OUT / "fix_cavitte.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
