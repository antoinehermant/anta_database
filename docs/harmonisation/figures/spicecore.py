"""Figure of the layers dated at SPICEcore (notes.md, H3): the lines within 20 km of the core, and our depths of the
layers at the core on SP19 against the published ages.

Inputs (output directory of ties.py): ties_SPICEcore.csv and points_SPICEcore.parquet, from
anta_database.harmonisation.ties.core_ties("SPICEcore").

    python docs/harmonisation/figures/spicecore.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anta_database.harmonisation import chronologies
from anta_database.harmonisation.cores import ice_cores
from anta_database.harmonisation.ties import OUTPUT

HERE = os.path.dirname(os.path.abspath(__file__))
COLOURS = {"Sanderson_2024": "#2a78d6", "Beem_2021": "#eb6834", "Leysinger_Vieli_2011": "#6b6a64"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def main():
    t = pd.read_csv(os.path.join(OUTPUT, "ties_SPICEcore.csv"))
    p = pd.read_parquet(os.path.join(OUTPUT, "points_SPICEcore.parquet"))
    c = ice_cores().loc["SPICEcore"]
    sp19 = chronologies.table("SP19", "SPICEcore")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})

    for d, g in p.drop_duplicates(["dataset", "file", "x", "y"]).groupby("dataset"):
        a.scatter((g.x - c.x) / 1e3, (g.y - c.y) / 1e3, s=3, color=COLOURS[d], label=d, lw=0)
    a.scatter([0], [0], marker="*", s=160, color=INK, zorder=3)
    a.annotate("SPICEcore", (0, 0), xytext=(8, 6), textcoords="offset points", fontsize=8, color=INK)
    a.add_patch(plt.Circle((0, 0), 20, fill=False, color=MUTED, ls=":", lw=0.8))
    a.set_aspect("equal")
    a.set_xlabel("km from the core (polar stereographic x)")
    a.set_ylabel("km from the core (y)")
    a.legend(frameon=False, fontsize=7.5, loc="lower left", markerscale=4)
    a.set_title("Lines with layers within 20 km of SPICEcore", loc="left", color=INK, fontsize=9.5)

    b.plot(sp19.age, sp19.depth, color=INK, lw=1.2, label="SP19 (ends at 1751 m, 54.3 ka)")
    for d, g in t.groupby("dataset"):
        col = COLOURS[d]
        on = g[np.isfinite(g.age_ref)]
        b.errorbar(on.age_ref, on.depth, yerr=on.depth_sd, fmt="o", ms=5, color=col, elinewidth=1, capsize=0,
                   label=f"{d}: our depth at the core, SP19 age")
        b.scatter(g.age_published, g.depth, s=30, facecolors="none", edgecolors=col, lw=1,
                  label=f"{d}: published age at our depth")
        for r in g.itertuples():
            b.annotate(f"{r.irh}", (r.age_published, r.depth), xytext=(5, 2), textcoords="offset points",
                       fontsize=7, color=MUTED)
    b.axhline(1751, color=MUTED, lw=0.8, ls=":")
    b.invert_yaxis()
    b.set_xlim(0, 100)
    b.set_xlabel("Age (ka BP1950)")
    b.set_ylabel("Depth at SPICEcore (m), ±1 sd of the Gaussian surface")
    b.grid(color=GRID, lw=0.5)
    b.legend(frameon=False, fontsize=7, loc="lower left")
    b.set_title("Layers at SPICEcore: SP19 age at our depth (filled), published age (open)", loc="left",
                color=INK, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "spicecore_ties.png"), dpi=160)


if __name__ == "__main__":
    main()
