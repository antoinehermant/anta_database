"""Figure of Leysinger_Vieli_2011 against Sanderson_2024 (notes.md, H3, SPICEcore): the layers of the Leysinger lines
121 and 134 (between SPICEcore and Vostok) along the line, and the Sanderson layers within 2 km of them.

    python docs/harmonisation/figures/leysinger_sanderson.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from anta_database.harmonisation.cores import ice_cores
from anta_database.harmonisation.transfer import load

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
LEYSINGER = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4"}
SANDERSON = {1: (INK, "38.5 ka"), 2: ("#4d4c47", "90.4 ka"), 3: ("#8a8983", "161.9 ka")}
PUBLISHED = {1: 17.5, 2: 36.6, 3: 62.5, 4: 72.9, 5: 118.3}
MAX_KM = 2000  # line 121 ends with a jump of 1200 km to one point near Vostok: left out
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def main():
    c = ice_cores()
    L, S = load("Leysinger_Vieli_2011"), load("Sanderson_2024")
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for ax, f in zip(axes, ["121.nc", "134.nc"]):
        q = L[L.file == f]
        xy = q.drop_duplicates(["x", "y"])[["x", "y"]].values
        km = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))] / 1e3
        keep = km <= MAX_KM
        xy, km = xy[keep], km[keep]
        pos = dict(zip(map(tuple, xy), km))
        for irh, g in q.groupby("irh"):
            if irh not in LEYSINGER:
                continue
            s = np.array([pos.get((a, b), np.nan) for a, b in zip(g.x, g.y)])
            ax.plot(s, g.depth, lw=0.9, color=LEYSINGER[irh], label=f"Leysinger {irh} ({PUBLISHED[irh]:g} ka)")
        t = cKDTree(xy)
        for si, (col, age) in SANDERSON.items():
            g = S[S.irh == si]
            d, i = t.query(g[["x", "y"]].values, distance_upper_bound=2000)
            ok = np.isfinite(d)
            ax.scatter(km[i[ok]], g.depth.values[ok], s=5, color=col, marker="x", lw=0.8, zorder=3,
                       label=f"Sanderson {si} ({age}), within 2 km")
        for k in ["SPICEcore", "Vostok"]:
            dd = np.hypot(xy[:, 0] - c.loc[k].x, xy[:, 1] - c.loc[k].y)
            j = dd.argmin()
            if dd[j] < 100e3:
                ax.axvline(km[j], color=MUTED, lw=0.8, ls=":")
                ax.annotate(f"{k} ({dd[j] / 1e3:.0f} km)", (km[j], 0.97), xycoords=("data", "axes fraction"),
                            ha="left", va="top", fontsize=7.5, color=MUTED, xytext=(4, 0), textcoords="offset points")
        ax.invert_yaxis()
        ax.grid(color=GRID, lw=0.5)
        ax.set_ylabel("Depth (m)")
        ax.set_title(f"Leysinger_Vieli_2011 line {f.removesuffix('.nc')}", loc="left", color=INK, fontsize=9.5)
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower left")
    axes[1].set_xlabel("km along the line")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "leysinger_sanderson.png"), dpi=160)


if __name__ == "__main__":
    main()
