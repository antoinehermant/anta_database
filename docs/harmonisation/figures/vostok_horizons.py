"""Figure of the layers matched between the datasets at Vostok (notes.md, H5 at Vostok): the ladder of the layers at
the core and the lines joining the same reflections, as edc_horizons.py for EDC.

Inputs (output directory of ties.py): ties.csv, and matches_Vostok.csv and placed_Vostok.csv of
docs/harmonisation/scripts/vostok_horizons.py.

    python docs/harmonisation/figures/vostok_horizons.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anta_database.harmonisation.ties import ANTA_JSON, OUTPUT

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["Sanderson_2024", "Leysinger_Vieli_2011", "Winter_2018", "Cavitte_2020", "Yan_2025"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
COLOURS = {"Cavitte_2020": "#2a78d6", "Yan_2025": "#1baf7a", "Winter_2018": "#e87ba4",
           "Leysinger_Vieli_2011": MUTED, "Sanderson_2024": "#3d3c38"}
STYLE = {"sure": dict(ls="-", alpha=0.75), "likely": dict(ls=":", alpha=0.9), "doubtful": dict(ls=":", alpha=0.35)}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def published(dataset):
    return {r["IRH"]: r["age"] / 1000 for r in json.load(open(os.path.join(ANTA_JSON, dataset, "raw_md.json")))
            if r.get("age")}


def main():
    t = pd.read_csv(os.path.join(OUTPUT, "ties.csv"))
    t = t[t.core == "Vostok"]
    m = pd.read_csv(os.path.join(OUTPUT, "matches_Vostok.csv"))
    p = pd.read_csv(os.path.join(OUTPUT, "placed_Vostok.csv"))
    # The depth at Vostok of each layer: tied at the core (filled), or placed (open)
    depth = {(r.dataset, r.irh): (r.depth, r.depth_sd, True) for r in t.itertuples()}
    for r in p.itertuples():
        depth.setdefault((r.dataset, r.irh), (r.depth, r.depth_sd, False))
    for r in m[m.b == "Sanderson_2024"].itertuples():
        z = depth[(r.a, r.ia)][0]
        depth[(r.b, r.ib)] = (z, np.nan, False)

    col = {d: i for i, d in enumerate(ORDER)}
    fig, ax = plt.subplots(figsize=(11, 9))
    tied_ages = t.set_index(["dataset", "irh"]).age_published
    for d in ORDER:
        ages = published(d)
        ages.update({i: a for (dd, i), a in tied_ages.items() if dd == d})
        x = col[d]
        side = -1 if x < col["Cavitte_2020"] else 1
        for (dd, irh), (z, sd, tied) in depth.items():
            if dd != d:
                continue
            doubtful = d == "Sanderson_2024" and irh == 3
            kw = dict(color=COLOURS[d], alpha=0.4 if doubtful else 1)
            if tied:
                ax.errorbar([x], [z], yerr=[sd], fmt="_", ms=18, mew=2, elinewidth=1, ecolor=COLOURS[d], zorder=3,
                            **kw)
            else:
                ax.scatter([x], [z], s=150, marker="_", lw=1, zorder=3, **kw)
                ax.scatter([x], [z], s=22, facecolors="white", edgecolors=COLOURS[d], lw=1, zorder=4,
                           alpha=kw["alpha"])
            ax.annotate(f"{round(ages.get(irh, np.nan), 1):g}" + ("?" if doubtful else ""), (x, z),
                        xytext=(side * 14, -3), textcoords="offset points", ha="right" if side < 0 else "left",
                        fontsize=7, color=MUTED)
    for r in m.itertuples():
        a, b = depth.get((r.a, r.ia)), depth.get((r.b, r.ib))
        if a is None or b is None:
            continue
        ax.plot([col[r.a], col[r.b]], [a[0], b[0]], color=INK, lw=0.9, zorder=2, **STYLE[r.how])
    ax.plot([], [], color=INK, lw=0.9, label="The same reflection (at crossings, or along the same track)")
    ax.plot([], [], color=INK, lw=0.9, ls=":", label="Likely the same (faint: doubtful)")
    ax.scatter([], [], s=22, facecolors="white", edgecolors=INK, label="Not traced to Vostok: placed there (see notes)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_xticks(range(len(ORDER)), [d.replace("_Vieli", " Vieli") for d in ORDER], fontsize=8)
    ax.set_xlim(-0.8, len(ORDER) - 0.2)
    ax.invert_yaxis()
    ax.set_ylabel("Depth at Vostok (m): the Gaussian surface of each layer at the core, ±1 sd")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("Layers at Vostok, and the reflections they share (matched by depth and order only).\n"
                 "Numbers: published ages (ka), not used for the matching. Open: layers that don't reach Vostok "
                 "(Sanderson 460 km, Yan 124 km;\nCavitte 1, 2, 4, 6), placed between the Cavitte layers around them",
                 loc="left", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "vostok_horizons_ladder.png"), dpi=160)


if __name__ == "__main__":
    main()
