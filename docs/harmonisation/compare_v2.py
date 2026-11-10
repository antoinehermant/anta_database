"""Compare v2 (build_v2.py) with v1, dataset by dataset: what changed, and only what should.

For each dataset: the lines in v1 only, in v2 only, and in both (by line ID); for these, the
points (by position, to the cm) in one version only; the largest difference of each variable,
and of each layer (by IRH), at the points in both; the lines out of order (more than 5 steps over
50 times the median step and over 2 km) in each version.

    python docs/harmonisation/compare_v2.py [--datasets A B]
"""
import argparse
import glob
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from anta_database.compiling.lines import out_of_order

HOME = Path.home() / "data" / "isochrones"
V1 = HOME / "AntADatabase-v1-build"
V2 = HOME / "AntADatabase-v2-build"
VARS = ["ICE_THK", "BED_ELEV", "SURF_ELEV", "BASAL_UNIT"]


def lines(root: Path, name: str) -> dict:
    out = {}
    for path in glob.glob(str(root / name / "nc" / "*.nc")):
        with xr.open_dataset(path) as ds:
            out[str(ds.attrs["line ID"])] = path
    return out


def table(ds: xr.Dataset) -> pd.DataFrame:
    df = pd.DataFrame({"x": np.round(ds.PSX.values, 2), "y": np.round(ds.PSY.values, 2)})
    for var in VARS:
        if var in ds:
            df[var] = ds[var].values.astype(float)
    if "IRH_DEPTH" in ds:
        for k, irh in enumerate(ds.IRH.values):
            df[f"IRH {irh}"] = ds.IRH_DEPTH.values[:, k].astype(float)
    return df.drop_duplicates(["x", "y"]).set_index(["x", "y"])


def compare_line(job) -> dict:
    p1, p2 = job
    with xr.open_dataset(p1) as a, xr.open_dataset(p2) as b:
        t1, t2 = table(a), table(b)
        order = (out_of_order(np.c_[a.PSX.values, a.PSY.values]), out_of_order(np.c_[b.PSX.values, b.PSY.values]))
        layers = (set(int(i) for i in a.IRH.values) if "IRH" in a.dims else set(),
                  set(int(i) for i in b.IRH.values) if "IRH" in b.dims else set())
    both = t1.index.intersection(t2.index)
    diff = {}
    for col in sorted(set(t1.columns) | set(t2.columns)):
        if col in t1 and col in t2:
            u, v = t1.loc[both, col].to_numpy(), t2.loc[both, col].to_numpy()
            values = np.isfinite(u) & np.isfinite(v)
            gained = int((~np.isfinite(u) & np.isfinite(v)).sum())
            lost = int((np.isfinite(u) & ~np.isfinite(v)).sum())
            diff[col] = (float(np.max(np.abs(u - v)[values])) if values.any() else 0.0, gained, lost)
        else:
            diff[col] = ("v1 only" if col in t1 else "v2 only", int(np.isfinite((t1 if col in t1 else t2)[col]).sum()), 0)
    return {"only_v1": len(t1.index.difference(t2.index)), "only_v2": len(t2.index.difference(t1.index)),
            "points": len(both), "diff": diff, "order": order, "empty_v1": layers[0] - layers[1]}


def compare(name: str, cpus: int) -> None:
    l1, l2 = lines(V1, name), lines(V2, name)
    common = sorted(set(l1) & set(l2))
    print(f"\n== {name}: {len(l1)} lines in v1, {len(l2)} in v2, {len(common)} in both")
    if set(l1) - set(l2):
        print("   v1 only:", sorted(set(l1) - set(l2))[:12])
    if set(l2) - set(l1):
        print("   v2 only:", sorted(set(l2) - set(l1))[:12])
    with Pool(cpus) as pool:
        results = pool.map(compare_line, [(l1[k], l2[k]) for k in common])
    print(f"   points: {sum(r['points'] for r in results)} in both, {sum(r['only_v1'] for r in results)} in v1 only, "
          f"{sum(r['only_v2'] for r in results)} in v2 only")
    print(f"   out of order: {sum(r['order'][0] for r in results)} lines in v1, {sum(r['order'][1] for r in results)} in v2")
    empty = sum(len(r["empty_v1"]) for r in results)
    if empty:
        print(f"   layers of v1 not in v2: {empty}")
    worst = defaultdict(lambda: [0.0, 0, 0])
    for r in results:
        for col, (d, gained, lost) in r["diff"].items():
            w = worst[col]
            if isinstance(d, str):
                w[0] = d
            elif not isinstance(w[0], str):
                w[0] = max(w[0], d)
            w[1] += gained
            w[2] += lost
    for col, (d, gained, lost) in sorted(worst.items()):
        if (isinstance(d, str)) or d > 0.01 or gained or lost:
            print(f"   {col}: max |v2 - v1| {d if isinstance(d, str) else round(d, 3)}, values gained {gained}, lost {lost}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--cpus", type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()
    names = args.datasets or sorted(p.name for p in V2.iterdir() if (p / "nc").is_dir())
    for name in names:
        compare(name, args.cpus)


if __name__ == "__main__":
    main()
