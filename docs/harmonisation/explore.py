"""First look at the harmonisation of the layers (plan.md, section 1): the distance from the ice cores to
the radar lines, and the layers of two datasets that are closest in depth where their lines meet. On the
.nc files of a compiled database, after crossovers.py. Exploratory, not part of the package.

Each pair of layers is compared at every pair of points of crossovers.py (crossings and coincident tracks),
then summarised per cluster (the median per cluster, then the median of the clusters), so that one shared
flight or two lines meeting again and again count once.

    python docs/harmonisation/explore.py [database dir] [-o crossovers output dir]

Writes irh_pairs.json (for check_depth_correction.py) into the output directory of crossovers.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(__file__))
from load import DEFAULT, load_lines  # noqa: E402

# As in the 3D viewer (web/src/icecores.ts)
CORES = {"EDC": (-75.1017, 123.3482), "BELDC": (-75.2992, 122.4451), "Dome Fuji": (-77.3167, 39.7), "Vostok": (-78.4644, 106.8372),
         "EDML": (-75.0025, 0.0684), "TALDICE": (-72.8167, 159.1833), "Taylor Dome": (-77.7964, 158.7239), "Law Dome": (-66.7725, 112.8113),
         "SPICEcore": (-89.99, -98.16), "WDC": (-79.467, -112.085), "Byrd": (-80.0167, -119.5167), "Siple Dome": (-81.65, -148.81),
         "Skytrain": (-79.741, -78.5448), "ABN": (-71.17, 111.37), "RICE": (-79.36, -161.71), "Fletcher": (-77.9, -82.61),
         "Berkner": (-79.55, -45.68), "JRI": (-64.2017, -57.685)}


def cores(lines):
    """Nearest point with an IRH of each dataset within 20 km of each core."""
    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    print("Ice cores: nearest radar point with an IRH, per dataset (within 20 km)")
    for core, (lat, lon) in CORES.items():
        cx, cy = to_xy.transform(lon, lat)
        best = {}
        for L in lines:
            d = np.hypot(L["x"] - cx, L["y"] - cy)
            i = int(np.argmin(d))
            if d[i] < best.get(L["dataset"], (np.inf,))[0]:
                best[L["dataset"]] = (d[i], int(np.isfinite(L["depth"][i]).sum()))
        near = sorted((d, name, n) for name, (d, n) in best.items() if d < 20000)
        print(f"  {core:12s}", "; ".join(f"{name} {d / 1000:.2f} km ({n} IRHs)" for d, name, n in near) or "none")


def layer_pairs(out: Path) -> pd.DataFrame:
    """Every pair of layers (A, IRH, B, IRH) seen together: depth difference B - A per cluster, then over clusters."""
    pairs = pd.read_parquet(out / "pairs.parquet")
    depths = pd.read_parquet(out / "pair_depths.parquet")
    a = depths[depths.side == "a"].drop(columns="side")
    b = depths[depths.side == "b"].drop(columns="side")
    m = a.merge(b, on="pair", suffixes=("_a", "_b")).merge(
        pairs[["kind", "dataset_a", "dataset_b", "cluster"]], left_on="pair", right_index=True)
    m["dz"] = m.depth_b - m.depth_a
    keys = ["dataset_a", "IRH_a", "age_a", "age_unc_a", "dataset_b", "IRH_b", "age_b", "age_unc_b"]
    per_cluster = m.groupby(keys + ["cluster"]).agg(dz=("dz", "median"), depth=("depth_a", "median"), n=("dz", "size"),
                                                    parallel=("kind", lambda k: bool((k == "parallel").any())))
    out_rows = per_cluster.groupby(keys).agg(
        clusters=("dz", "size"), points=("n", "sum"), coincident=("parallel", "sum"), depth=("depth", "median"),
        dz_med=("dz", "median"), dz_mad=("dz", lambda d: float(np.median(np.abs(d - np.median(d)))))).reset_index()
    return out_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("database", nargs="?", default=DEFAULT)
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    out = Path(args.output or Path.home() / "data" / "isochrones" / "harmonisation" / Path(args.database).name)

    lines = load_lines(args.database)
    print(f"{len(lines)} lines\n")
    cores(lines)

    rows = layer_pairs(out)
    # For each layer of A, the layer of B closest in depth
    best = rows.loc[rows.groupby(["dataset_a", "IRH_a", "dataset_b"]).dz_med.apply(lambda d: d.abs().idxmin())]
    match = best[best.dz_med.abs() < 0.02 * best.depth].sort_values(["dataset_a", "dataset_b", "depth"])
    print("\nClosest-in-depth layer pairs (|median depth difference| < 2 % of the depth), with their ages:")
    print("  clusters: places compared (crossings of one pair of lines, or one coincident track); coin.: of them coincident")
    for r in match.itertuples():
        print(f"  {r.dataset_a:20s} IRH{r.IRH_a:<3d} {r.age_a / 1e3:7.1f} ka | {r.dataset_b:20s} IRH{r.IRH_b:<3d} {r.age_b / 1e3:7.1f} ka | "
              f"{r.clusters:4d} clusters ({r.coincident:3d} coin.) depth {r.depth:6.0f} m  dz {r.dz_med:+6.1f} ± {r.dz_mad:4.1f} m  "
              f"dAge {(r.age_b - r.age_a) / 1e3:+6.1f} ka")
    rows.to_json(out / "irh_pairs.json", orient="records")
    print(f"\n{len(rows)} pairs of layers written to {out / 'irh_pairs.json'}")


if __name__ == "__main__":
    main()
