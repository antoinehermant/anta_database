"""First look at the reconciliation (H5) around EDC: which layers of the datasets near the core are the same.

1. The crossings of every pair of datasets within RADIUS of EDC, with the depths of all the layers of both lines
   there (transfer.crossings).
2. For each pair of layers, the median and the spread (1.4826 MAD) of their depth difference over the crossings.
3. For each pair of datasets, a common offset (the median of the closest pairs of layers, within 15 m), then each
   layer matched to the layer of the other dataset closest to that offset: within max(5 m, 2 MAD), MAD at most 10 m,
   at least 3 crossings, and the order of the layers kept. Only depths and order are used: the published ages are
   kept aside, to check the matches.
4. Horizons: the matched pairs joined across datasets (union-find); a horizon with two layers of the same dataset
   would be a conflict. The loops (three datasets) say whether the pair offsets close.
5. The depth difference of each matched pair at the core (ties.csv, Gaussian surfaces) against the crossings.

Writes into the output directory of ties.py: crossings_EDC.parquet, matches_EDC.csv, horizons_EDC.csv.

    python docs/harmonisation/scripts/edc_horizons.py
"""
import itertools
import os

import numpy as np
import pandas as pd

from anta_database.harmonisation.cores import ice_cores
from anta_database.harmonisation.ties import OUTPUT
from anta_database.harmonisation.transfer import crossings, load

DATASETS = ["Cavitte_2020", "Mulvaney_2023", "Winter_2018", "Chung_2023", "Bodart_2025b"]
RADIUS = 30e3


def mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def pair_crossings(core="EDC"):
    c = ice_cores().loc[core]
    data = {}
    for d in DATASETS:
        p = load(d)
        data[d] = p[np.hypot(p.x - c.x, p.y - c.y) <= RADIUS]
    rows = []
    for a, b in itertools.combinations(DATASETS, 2):
        for r in crossings(data[a], data[b], target_gap=1e3, ref_gap=1e3).itertuples():
            for ia, za in r.target.items():
                for ib, zb in r.ref.items():
                    rows.append((a, b, r.target_file, r.ref_file, r.x, r.y, ia, za, ib, zb))
    d = pd.DataFrame(rows, columns=["a", "b", "fa", "fb", "x", "y", "ia", "za", "ib", "zb"])
    d["dz"] = d.zb - d.za
    return d


def match(d, ties):
    g = d.groupby(["a", "b", "ia", "ib"]).dz
    P = pd.DataFrame({"n": g.size(), "dz": g.median(), "mad": g.agg(mad)}).reset_index()
    out = []
    for (a, b), q in P.groupby(["a", "b"]):
        q = q[q.n >= 3]
        close = q.loc[q.groupby("ia").dz.apply(lambda s: s.abs().idxmin())]
        off = close[close.dz.abs() < 15].dz.median() if (close.dz.abs() < 15).any() else 0.0
        q = q.assign(res=q.dz - off)
        best = q.loc[q.groupby("ia").res.apply(lambda s: s.abs().idxmin())]
        best = best[(best.res.abs() <= np.maximum(5, 2 * best.mad)) & (best.mad <= 10)].sort_values("ia")
        keep = []
        for r in best.itertuples():
            if keep and r.ib <= keep[-1].ib:  # the order of the layers: keep the smaller residual
                if abs(r.res) < abs(keep[-1].res):
                    keep[-1] = r
                continue
            keep.append(r)
        for r in keep:
            out.append(dict(a=a, b=b, ia=r.ia, ib=r.ib, n=r.n, dz=r.dz, mad=r.mad, offset=off,
                            dz_core=ties.depth.get((b, r.ib), np.nan) - ties.depth.get((a, r.ia), np.nan),
                            age_a=ties.age_published.get((a, r.ia), np.nan),
                            age_b=ties.age_published.get((b, r.ib), np.nan)))
    return pd.DataFrame(out)


def horizons(m, ties):
    parent = {}

    def find(k):
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for r in m.itertuples():
        parent[find((r.a, r.ia))] = find((r.b, r.ib))
    groups = {}
    for k in list(parent):
        groups.setdefault(find(k), []).append(k)
    rows = []
    for members in groups.values():
        ds = [x[0] for x in members]
        ref = [ties.age_ref.get(x, np.nan) for x in members]
        row = {"depth": np.nanmedian([ties.depth.get(x, np.nan) for x in members]), "datasets": len(members),
               "conflict": len(ds) != len(set(ds)), "members": " ".join(f"{d}:{i}" for d, i in sorted(members)),
               "age_ref_cavitte": dict(zip(ds, ref)).get("Cavitte_2020", np.nan),
               "age_ref_spread": np.nanmax(ref) - np.nanmin(ref)}
        for d, i in members:
            row[f"published_{d}"] = ties.age_published.get((d, i), np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("depth")


def loops(m):
    off = m.groupby(["a", "b"]).offset.first()

    def o(a, b):
        return off.get((a, b), -off.get((b, a), np.nan))

    names = sorted(set(m.a) | set(m.b))
    return {(a, b, c): o(a, b) + o(b, c) + o(c, a) for a, b, c in itertools.combinations(names, 3)
            if np.isfinite(o(a, b) + o(b, c) + o(c, a))}


def main():
    ties = pd.read_csv(os.path.join(OUTPUT, "ties.csv"))
    ties = ties[ties.core == "EDC"].set_index(["dataset", "irh"])
    d = pair_crossings()
    d.to_parquet(os.path.join(OUTPUT, "crossings_EDC.parquet"))
    m = match(d, ties)
    m.to_csv(os.path.join(OUTPUT, "matches_EDC.csv"), index=False)
    h = horizons(m, ties)
    h.to_csv(os.path.join(OUTPUT, "horizons_EDC.csv"), index=False)
    print(d.drop_duplicates(["a", "b", "x", "y"]).groupby(["a", "b"]).size().rename("crossings").to_string())
    print(m.round(1).to_string(index=False))
    print(h.drop(columns="members").round(1).to_string(index=False))
    print("conflicts:", int(h.conflict.sum()))
    for k, v in loops(m).items():
        print("loop", " -> ".join(k), f"{v:+.1f} m")
    print("core minus crossings, median |.|:", round(float((m.dz_core - m.dz).abs().median()), 1), "m")


if __name__ == "__main__":
    main()
