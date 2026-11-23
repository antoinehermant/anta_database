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
too, but for their two ends (picks, which the straight line joins). Shorter ones can be smooth picks; but
where the other picks of the layer on the line are noisy (less than SMOOTH_SHARE of their second differences
within the rounding), the stretches of NOISY_MIN_KM or more are fills too (VCD_JKB1a_X08a, anta_json #10).

Then, on all the files (with --min-km):
- CUTS: depths removed by hand, by line, IRH and km along the line (from its first point in the files,
  column distance_m, in km): where a layer crosses the bed (anta_json #16, #22).
- Spikes of the surface: the published files have no surface, the compilation takes thickness + bed. Where
  both were filled across a gap with straight lines of different ends, their sum jumps by hundreds of metres
  (ASB_JKB1a_R09Ea, anta_json #22). Where thickness + bed is more than SPIKE m off its running median over
  SPIKE_WINDOW_KM along the line, and around it while more than SPIKE_EDGE m off, thickness and bed are
  removed (Bedmap may fill them, from its own line of the same traces).

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
# A layer on a line is smooth when at least this share of the second differences of its picks are within FLAT
# (the ground lines, DC_LDC_DIVIDE...): there only the stretches of --min-km are taken as fills; on the others,
# from NOISY_MIN_KM
SMOOTH_SHARE, NOISY_MIN_KM = 0.6, 1.0
DEPTH_COLUMN = "IRHdepth_m"
THICKNESS, BED, DISTANCE = "thk_m", "bedelv_mABSL", "distance_m"  # distance_m is in km
# (line, IRHs, first km, last km, why): km from the first point of the line in the files
CUTS = [
    ("ASB_JKB1a_R09Ea", [11], 0.0, 5.0, "the 160.4 ka layer crosses the bed in the first 5 km (anta_json #16, #22)"),
]
SPIKE, SPIKE_EDGE, SPIKE_WINDOW_KM = 50.0, 10.0, 10.0  # m, m, km


def picks(irh: int) -> pd.DataFrame:
    name = "HiCARS_Bed.txt" if irh == 1 else f"HiCARS_IRH_{irh - 1}.txt"
    df = pd.read_csv(PICKS / name, sep="\t")
    return df[df["depth(m)"].notna()]


def rewrite(src: Path, out: Path, remove) -> None:
    """src byte for byte into out, but for the values removed (nan): remove is the rows of the depth, or
    {column: rows}. out may be src, or a link to the working copy: then replaced by a file, never written
    through."""
    if not isinstance(remove, dict):
        remove = {DEPTH_COLUMN: remove}
    tmp = out.with_suffix(".tmp")
    row, header, columns = -1, None, None
    with open(src, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8", newline="") as fout:
        for text in fin:
            if text.startswith("#") or not text.strip():
                fout.write(text)
                continue
            if header is None:
                header = text.rstrip("\r\n").split(",")
                columns = [(header.index(name), rows) for name, rows in remove.items()]
                fout.write(text)
                continue
            row += 1
            hit = [k for k, rows in columns if rows[row]]
            if hit:
                end = "\r\n" if text.endswith("\r\n") else "\n"
                fields = text.rstrip("\r\n").split(",")
                for k in hit:
                    fields[k] = "nan"
                text = ",".join(fields) + end
            fout.write(text)
    for name, rows in remove.items():
        assert row + 1 == len(rows), f"{src.name}: {row + 1} rows written, {len(rows)} read"
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
        second = np.abs(np.diff(depth[rows][ok], 2))
        second = second[second < 20]  # not across the gaps
        noisy = len(second) >= 50 and np.mean(second <= FLAT) < SMOOTH_SHARE
        length = min(min_km, NOISY_MIN_KM) if noisy else min_km
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
                if along[b + 1] - along[a] >= length * 1000:
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


def cuts(raw: pd.DataFrame, irh: int) -> np.ndarray:
    """The rows of the depths removed by hand (CUTS)."""
    remove = np.zeros(len(raw), bool)
    for line, irhs, first, last, _ in CUTS:
        if irh not in irhs:
            continue
        on = (raw["line"] == line).to_numpy()
        if on.any():
            km = raw[DISTANCE].to_numpy() - raw.loc[on, DISTANCE].min()
            remove |= on & (km >= first) & (km <= last) & raw[DEPTH_COLUMN].notna().to_numpy()
    return remove


def spikes(raw: pd.DataFrame) -> np.ndarray:
    """The rows where thickness + bed (the surface of the compilation) jumps: more than SPIKE m off its
    running median over SPIKE_WINDOW_KM, and around them while more than SPIKE_EDGE m off."""
    from scipy.ndimage import median_filter

    remove = np.zeros(len(raw), bool)
    surface = (raw[THICKNESS] + raw[BED]).to_numpy()
    km = raw[DISTANCE].to_numpy()
    for _, rows in raw.groupby("line", sort=False).indices.items():
        rows = rows[np.isfinite(surface[rows])]
        if len(rows) < 5:
            continue
        step = np.median(np.abs(np.diff(km[rows]))) or 0.01
        size = max(5, int(SPIKE_WINDOW_KM / step) | 1)
        off = np.abs(surface[rows] - median_filter(surface[rows], size=size, mode="nearest"))
        edge = off > SPIKE_EDGE
        # The runs over SPIKE_EDGE that reach SPIKE somewhere
        bounds = np.flatnonzero(np.diff(np.r_[0, edge.astype(int), 0]))
        for a, b in zip(bounds[::2], bounds[1::2]):
            if off[a:b].max() > SPIKE:
                remove[rows[a:b]] = True
    return remove


def fix_all(irh: int) -> dict:
    """CUTS (depths) and the spikes of the surface (thickness and bed) of the file of one IRH."""
    out = OUT / "raw" / f"OIA_EDC_IRH{irh}.csv"
    raw = pd.read_csv(out, comment="#", usecols=["line", DISTANCE, THICKNESS, BED, DEPTH_COLUMN])
    cut, spike = cuts(raw, irh), spikes(raw)
    if cut.any() or spike.any():
        rewrite(out.resolve() if out.is_symlink() else out, out, {DEPTH_COLUMN: cut, THICKNESS: spike, BED: spike})
    return {"IRH": irh, "depths cut": int(cut.sum()), "cut lines": raw.loc[cut, "line"].value_counts().to_dict(),
            "thickness and bed removed": int(spike.sum()),
            "spike lines": raw.loc[spike, "line"].value_counts().to_dict()}


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
        fixed = [fix_all(irh) for irh in range(1, ALL_LAYERS + 1)]
        for r in fixed:
            print(f"IRH {r['IRH']:2d}: {r['depths cut']:5d} depths cut {r['cut lines']}, thickness and bed removed at "
                  f"{r['thickness and bed removed']} points {r['spike lines']}")
        result["cuts"] = {"cuts": [list(c[:1]) + [c[1], c[2], c[3], c[4]] for c in CUTS], "layers": fixed}
        result["surface_spikes"] = {"spike_m": SPIKE, "edge_m": SPIKE_EDGE, "window_km": SPIKE_WINDOW_KM}
    (OUT / "fix_cavitte.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
