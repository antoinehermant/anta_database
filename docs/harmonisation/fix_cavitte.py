"""Remove the interpolated depths of the HiCARS lines of Cavitte_2020, for v2.

The published files (OIA_EDC_IRH1-26.csv, https://doi.org/10.15784/601411) fill the gaps of the
HiCARS picks (lines OIA_JKB2n_*) with interpolated depths. Marie Cavitte's picks of these lines
(HiCARS_Bed.txt, HiCARS_IRH_1-19.txt, 2025-12-28; one file per IRH, shifted by one: HiCARS_Bed is
IRH 1, HiCARS_IRH_n is IRH n + 1) have the same points without the fills. At every point in both,
their depth is the published one + 10 m (their firn correction), so the published values stay:
only the depths of the HiCARS lines with no pick at the same position (1 m) become nan. IRH 21-26
and the other lines (MCoRDS, DELORES) are not in the picks and stay as published.

Writes the corrected files, byte for byte the published ones but for these depths, to the v2
sources; the files left as published are links to the working copy (only read).

    python docs/harmonisation/fix_cavitte.py
"""
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
LAYERS = 20
SAME_POINT = 1.0  # m
DEPTH_COLUMN = "IRHdepth_m"


def picks(irh: int) -> pd.DataFrame:
    name = "HiCARS_Bed.txt" if irh == 1 else f"HiCARS_IRH_{irh - 1}.txt"
    df = pd.read_csv(PICKS / name, sep="\t")
    return df[df["depth(m)"].notna()]


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

    # Byte for byte, but for the depth of the rows removed
    out = OUT / "raw" / src.name
    row, header, column = -1, None, None
    with open(src, encoding="utf-8") as fin, open(out, "w", encoding="utf-8", newline="") as fout:
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
    assert row + 1 == len(raw), f"IRH {irh}: {row + 1} rows written, {len(raw)} read"
    lines = raw.loc[remove, "line"].value_counts()
    return {"IRH": irh, "removed": int(remove.sum()), "kept": len(kept), "lines": lines.to_dict()}


def main() -> None:
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
    for name in ("raw_md.json", "original_new_column_names.csv"):
        shutil.copy2(WORKING / name, OUT / name)
    (OUT / "fix_cavitte.json").write_text(json.dumps(report, indent=1))
    for r in report:
        print(f"IRH {r['IRH']:2d}: {r['removed']:6d} depths removed, {r['kept']:6d} kept, on {len(r['lines'])} lines")
    print("total removed:", sum(r["removed"] for r in report))


if __name__ == "__main__":
    main()
