"""Figure of Yan_2025's two 10 ka layers (notes.md, H5 at Vostok): IRH 4 (10 ka, modelled) and IRH 10 (10 ka, from
EDC through Cavitte), along the lines that have IRH 10, with the Cavitte layers within 1 km.

    python docs/harmonisation/figures/yan_10ka.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from anta_database.harmonisation.transfer import load

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
SHOWN = {4: ("#eb6834", "Yan 4 (10 ka, modelled)"), 6: ("#e87ba4", "Yan 6 (13.5 ka, modelled)"),
         10: ("#1baf7a", "Yan 10 (10 ka, from EDC)"), 11: ("#2a78d6", "Yan 11 (29.4 ka, from EDC)")}
LINES = ["PEL_GCX0d_R42a.nc", "PEL_GCX0d_R43a.nc", "20131127_01.nc"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def main():
    Y, C = load("Yan_2025"), load("Cavitte_2020")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, f in zip(axes, LINES):
        q = Y[Y.file == f]
        xy = q.drop_duplicates(["x", "y"])[["x", "y"]].values
        km = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))] / 1e3
        pos = dict(zip(map(tuple, xy), km))
        for irh, g in q.groupby("irh"):
            s = [pos[(a, b)] for a, b in zip(g.x, g.y)]
            col, label = SHOWN.get(irh, ("#c9c8c2", None))
            ax.scatter(s, g.depth, s=2, color=col, lw=0, label=label, zorder=3 if irh in SHOWN else 2)
        ax.scatter([], [], s=2, color="#c9c8c2", label="Other Yan layers")
        t = cKDTree(xy)
        for ic, mk in [(1, "x"), (2, "+")]:
            g = C[C.irh == ic]
            d, i = t.query(g[["x", "y"]].values, distance_upper_bound=1000)
            ok = np.isfinite(d)
            if ok.any():
                ax.scatter(km[i[ok]], g.depth.values[ok], s=6, marker=mk, color=INK, lw=0.6, zorder=4,
                           label=f"Cavitte {ic} ({'10.0' if ic == 1 else '29.4'} ka), within 1 km")
        ax.set_ylim(1100, 0)
        ax.grid(color=GRID, lw=0.5)
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Yan_2025 line {f.removesuffix('.nc')}", loc="left", color=INK, fontsize=9.5)
        ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower right", markerscale=4)
    axes[-1].set_xlabel("km along the line")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "yan_10ka.png"), dpi=150)


if __name__ == "__main__":
    main()
