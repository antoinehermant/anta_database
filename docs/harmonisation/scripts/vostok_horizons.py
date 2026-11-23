"""The same reflections across the datasets at Vostok (notes.md, H5 at Vostok), for the ladder figure.

At Vostok only Winter_2018 (IRH 1-4), Cavitte_2020 (3, 5, 7) and Leysinger_Vieli_2011 (1-7) reach the core (ties.csv).
Two more datasets join through them, far from the core:
- Sanderson_2024 (460 km away) through Winter_2018, along the same flight (notes.md, SPICEcore): 1 = Winter 1 (sure),
  2 = Winter 3 (likely), 3 = Winter 4 (doubtful).
- Yan_2025 (124 km away) through Cavitte_2020, along Yan's line 20131127_01, which runs along a Cavitte line (60 m).

Pairs, by depth and order only (published ages kept aside):
1. Winter and Cavitte: crossings within 30 km of Vostok, edc_horizons.match.
2. Leysinger against Winter and Cavitte: its points are sparse, so its line 009 (within 110 km of Vostok) against the
   points of the others within 1 km. A pair when the median difference is within max(25 m, 2 MAD), MAD <= 20 m.
3. Yan against Cavitte: points within 300 m along the shared track; a pair when within max(5 m, 2 MAD) of the pair
   offset... here each layer is checked against the closest Cavitte layer (|median| <= 20 m, MAD <= 10 m). Yan
   layers without a Cavitte layer there are placed by their position between the Cavitte layers around them.

Placing the layers that don't reach Vostok: a layer on the shared track is carried to Vostok by its position between
the two Cavitte layers around it that are tied at Vostok (3, 5, 7; the surface above 3), linear in depth.

Writes into the output directory of ties.py: matches_Vostok.csv, placed_Vostok.csv.

    python docs/harmonisation/scripts/vostok_horizons.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edc_horizons import mad, match  # noqa: E402

from anta_database.harmonisation.cores import ice_cores  # noqa: E402
from anta_database.harmonisation.ties import OUTPUT  # noqa: E402
from anta_database.harmonisation.transfer import crossings, load  # noqa: E402

RADIUS = 30e3
LEYSINGER_LINE, LEYSINGER_KM = "009.nc", 110
YAN_LINE = "20131127_01.nc"
SANDERSON = [(1, 1, "sure"), (3, 2, "likely"), (4, 3, "doubtful")]  # Winter IRH, Sanderson IRH


def pairs_at_crossings(A, B, na, nb):
    rows = []
    for r in crossings(A, B, target_gap=1e3, ref_gap=1e3).itertuples():
        for ia, za in r.target.items():
            for ib, zb in r.ref.items():
                rows.append((na, nb, r.target_file, r.ref_file, r.x, r.y, ia, za, ib, zb))
    d = pd.DataFrame(rows, columns=["a", "b", "fa", "fb", "x", "y", "ia", "za", "ib", "zb"])
    d["dz"] = d.zb - d.za
    return d


def near_points(A, B, max_dist):
    """Median and MAD of B minus A for every pair of layers, at the points of A with a point of B within max_dist."""
    rows = []
    for ia, a in A.groupby("irh"):
        for ib, b in B.groupby("irh"):
            d, i = cKDTree(b[["x", "y"]].values).query(a[["x", "y"]].values, distance_upper_bound=max_dist)
            ok = np.isfinite(d)
            if ok.sum() < 5:
                continue
            dz = b.depth.values[i[ok]] - a.depth.values[ok]
            rows.append(dict(ia=ia, ib=ib, n=int(ok.sum()), dz=np.median(dz), mad=mad(dz),
                             za=np.median(a.depth.values[ok]), zb=np.median(b.depth.values[i[ok]])))
    return pd.DataFrame(rows)


def main():
    ties = pd.read_csv(os.path.join(OUTPUT, "ties.csv"))
    ties = ties[ties.core == "Vostok"].set_index(["dataset", "irh"])
    c = ice_cores().loc["Vostok"]
    near = lambda p, km: p[np.hypot(p.x - c.x, p.y - c.y) <= km * 1e3]  # noqa: E731
    C, W, L, Y = (load(d) for d in ["Cavitte_2020", "Winter_2018", "Leysinger_Vieli_2011", "Yan_2025"])
    out = []

    # 1. Winter and Cavitte at their crossings near Vostok
    m = match(pairs_at_crossings(near(C, 30), near(W, 30), "Cavitte_2020", "Winter_2018"), ties)
    for r in m.itertuples():
        out.append(dict(a=r.a, ia=r.ia, b=r.b, ib=r.ib, n=r.n, dz=r.dz, mad=r.mad, how="sure", method="crossings"))

    # 2. Leysinger line 009 against Winter and Cavitte
    l009 = near(L[L.file == LEYSINGER_LINE], LEYSINGER_KM)
    for name, P in [("Winter_2018", W), ("Cavitte_2020", C)]:
        # the points of the denser dataset, each with the nearest Leysinger point within 1 km
        q = near_points(near(P, LEYSINGER_KM), l009, 1000)
        q = q.loc[q.groupby("ib").dz.apply(lambda s: s.abs().idxmin())]
        for r in q.itertuples():
            if abs(r.dz) <= max(25, 2 * r.mad) and r.mad <= 20:
                out.append(dict(a=name, ia=r.ia, b="Leysinger_Vieli_2011", ib=r.ib, n=r.n, dz=r.dz, mad=r.mad,
                                how="likely", method="points within 1 km, Leysinger line 009"))

    # 3. Yan against Cavitte along the shared track
    y = Y[Y.file == YAN_LINE]
    cy = C[np.hypot(*(C[["x", "y"]].values - y[["x", "y"]].mean().values).T) < 400e3]
    q = near_points(cy, y, 300)
    best = q.loc[q.groupby("ib").dz.apply(lambda s: s.abs().idxmin())]
    for r in best.itertuples():
        if abs(r.dz) <= 20 and r.mad <= 10:
            out.append(dict(a="Cavitte_2020", ia=r.ia, b="Yan_2025", ib=r.ib, n=r.n, dz=r.dz, mad=r.mad,
                            how="sure", method=f"points within 300 m, Yan line {YAN_LINE}"))

    # 4. Sanderson through Winter (notes.md, SPICEcore)
    for iw, isd, how in SANDERSON:
        out.append(dict(a="Winter_2018", ia=iw, b="Sanderson_2024", ib=isd, how=how,
                        method="points within 300 m along the same flight (notes.md)"))
    matches = pd.DataFrame(out)
    matches.to_csv(os.path.join(OUTPUT, "matches_Vostok.csv"), index=False, float_format="%.4g")

    # Placing the layers of the shared track at Vostok: at each point, the position of the layer between the two
    # Cavitte layers tied at Vostok just above and below it there (the surface above IRH 3), carried to their
    # depths at Vostok; the median over the points
    tied = [0] + [i for i in (3, 5, 7) if ("Cavitte_2020", i) in ties.index]
    core = {0: 0.0, **{i: ties.loc[("Cavitte_2020", i)].depth for i in tied[1:]}}
    wide = cy.pivot_table(index=["file", "x", "y"], columns="irh", values="depth").reset_index()
    wide[0] = 0.0
    tree = cKDTree(wide[["x", "y"]].values)

    def place(z, row):
        for up, lo in zip(tied[:-1], tied[1:]):
            zu, zl = row.get(up, np.nan), row.get(lo, np.nan)
            if np.isfinite(zu) and np.isfinite(zl) and zu <= z < zl:
                return core[up] + (z - zu) / (zl - zu) * (core[lo] - core[up])
        return np.nan

    placed = []
    yan_wide = y.pivot_table(index=["x", "y"], columns="irh", values="depth").reset_index()
    d, i = tree.query(yan_wide[["x", "y"]].values, distance_upper_bound=300)
    ok = np.isfinite(d)
    rows_c = wide.iloc[i[ok]].to_dict("records")
    for irh in [k for k in yan_wide.columns if isinstance(k, (int, np.integer))]:
        v = np.array([place(z, r) for z, r in zip(yan_wide[irh].values[ok], rows_c)])
        if np.isfinite(v).sum() >= 5:
            placed.append(dict(dataset="Yan_2025", irh=irh, n=int(np.isfinite(v).sum()), depth=np.nanmedian(v),
                               depth_sd=mad(v[np.isfinite(v)])))
    near_track = wide.iloc[np.unique(i[ok])]  # the Cavitte points along the track
    for irh in [k for k in wide.columns if isinstance(k, (int, np.integer)) and k not in tied and k <= 8]:
        v = np.array([place(r.get(irh, np.nan), r) for r in near_track.to_dict("records")])
        if np.isfinite(v).sum() >= 5:
            placed.append(dict(dataset="Cavitte_2020", irh=irh, n=int(np.isfinite(v).sum()), depth=np.nanmedian(v),
                               depth_sd=mad(v[np.isfinite(v)])))
    placed = pd.DataFrame(placed)

    # Yan layers without a Cavitte layer at the same points: the Cavitte layer placed closest at Vostok, within 30 m
    done = set(matches[matches.b == "Yan_2025"].ib)
    cav = placed[placed.dataset == "Cavitte_2020"].set_index("irh").depth
    extra = []
    for r in placed[(placed.dataset == "Yan_2025") & ~placed.irh.isin(done)].itertuples():
        k = (cav - r.depth).abs().idxmin()
        if abs(cav[k] - r.depth) <= 30:
            extra.append(dict(a="Cavitte_2020", ia=k, b="Yan_2025", ib=r.irh, dz=r.depth - cav[k], how="likely",
                              method="placed at Vostok between the Cavitte layers around them"))
    matches = pd.concat([matches, pd.DataFrame(extra)], ignore_index=True)
    matches.to_csv(os.path.join(OUTPUT, "matches_Vostok.csv"), index=False, float_format="%.4g")
    placed.to_csv(os.path.join(OUTPUT, "placed_Vostok.csv"), index=False, float_format="%.5g")

    print(matches.round(1).to_string(index=False))
    print(placed.round(1).to_string(index=False))
    print("Yan layers on the track, closest Cavitte layer:")
    print(best.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
