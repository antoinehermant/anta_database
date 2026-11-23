"""Figures of the layers matched between datasets near EDC (notes.md, H5 first look): the ladder of the layers at the
core with the horizons joining them, and the depth difference of each pair of datasets at the crossings.

Inputs (output directory of ties.py): ties.csv, and matches_EDC.csv and crossings_EDC.parquet of
docs/harmonisation/scripts/edc_horizons.py.

    python docs/harmonisation/figures/edc_horizons.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anta_database.harmonisation.ties import OUTPUT

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["Winter_2018", "Bodart_2025b", "Cavitte_2020", "Leysinger_Vieli_2011", "Chung_2023", "Mulvaney_2023"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
COLOURS = dict(zip(["Cavitte_2020", "Mulvaney_2023", "Bodart_2025b", "Chung_2023", "Winter_2018"],
                   ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]), Leysinger_Vieli_2011=MUTED)
# Leysinger_Vieli_2011 is matched to Cavitte_2020 by the sequence of its layers, not by depth (notes.md, H3): its
# layers are 50 to 60 m shallower. IRH 2 and 4 are sure, IRH 6 and 8 likely.
LEYSINGER = {2: (3, True), 4: (5, True), 6: (10, False), 8: (14, False)}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def ladder(t, m):
    fig, ax = plt.subplots(figsize=(11, 9))
    col = {d: i for i, d in enumerate(ORDER)}
    for d in ORDER:
        g = t[t.dataset == d]
        x = col[d]
        ax.errorbar(np.full(len(g), x), g.depth, yerr=g.depth_sd, fmt="_", ms=18, mew=2, color=COLOURS[d],
                    ecolor=COLOURS[d], elinewidth=1, zorder=3)
        side = -1 if x < 2 else 1
        for r in g.itertuples():
            ax.annotate(f"{round(r.age_published, 1):g}", (x, r.depth), xytext=(side * 14, -3), textcoords="offset points",
                        ha="right" if side < 0 else "left", fontsize=7, color=MUTED)
    depth = t.set_index(["dataset", "irh"]).depth
    for r in m.itertuples():
        xa, xb = col[r.a], col[r.b]
        za, zb = depth.get((r.a, r.ia)), depth.get((r.b, r.ib))
        if za is None or zb is None:
            continue
        ax.plot([xa, xb], [za, zb], color=INK, lw=0.8, alpha=0.55 if abs(xa - xb) == 1 else 0.25, zorder=2)
    xa, xb = col["Cavitte_2020"], col["Leysinger_Vieli_2011"]
    for ib, (ia, sure) in LEYSINGER.items():
        ax.plot([xa, xb], [depth[("Cavitte_2020", ia)], depth[("Leysinger_Vieli_2011", ib)]], color=MUTED, lw=1,
                ls="--" if sure else ":", zorder=2)
    ax.plot([], [], color=INK, lw=0.8, alpha=0.55, label="Matched at the crossings (depth and order)")
    ax.plot([], [], color=MUTED, lw=1, ls="--", label="Leysinger_Vieli_2011: matched by the sequence of its layers")
    ax.plot([], [], color=MUTED, lw=1, ls=":", label="Leysinger_Vieli_2011: the same, likely")
    ax.legend(loc="lower left", frameon=False, fontsize=7.5, labelcolor=INK)
    ax.set_xticks(range(len(ORDER)), [d.replace("_Vieli", " Vieli") for d in ORDER], fontsize=8)
    ax.set_xlim(-0.8, len(ORDER) - 0.2)
    ax.invert_yaxis()
    ax.set_ylabel("Depth at EDC (m): the Gaussian surface of each layer at the core, ±1 sd")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title("Layers at EDC, and the horizons joining them (matched at the crossings by depth and order only).\n"
                 "Numbers: published ages (ka), not used for the matching. Leysinger: its nearest line is 13 km away", loc="left", color=INK, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "edc_horizons_ladder.png"), dpi=160)


def offsets(d, m):
    pairs = m.groupby(["a", "b"]).size().index.tolist()
    cols = 5
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.9 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, (a, b) in zip(axes.flat, pairs):
        q = m[(m.a == a) & (m.b == b)]
        for r in q.itertuples():
            v = d[(d.a == a) & (d.b == b) & (d.ia == r.ia) & (d.ib == r.ib)]
            ax.scatter(v.za, v.dz, s=5, color=COLOURS[b], alpha=0.35, lw=0, zorder=2)
            ax.scatter([v.za.median()], [r.dz], s=26, color=INK, zorder=3)
        ax.axhline(q.offset.iloc[0], color=COLOURS[b], lw=1.5, zorder=1)
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"{b.split('_')[0]} − {a.split('_')[0]}: {q.offset.iloc[0]:+.1f} m", loc="left", fontsize=8.5,
                     color=INK)
        ax.grid(color=GRID, lw=0.5)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Depth (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Depth difference (m)")
    axes[0, 0].set_ylim(-30, 40)
    fig.suptitle("Matched layers at the crossings within 30 km of EDC: depth difference of each pair of datasets "
                 "(dots: crossings; black: median per layer; line: the offset of the pair)", x=0.01, ha="left",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, "edc_pair_offsets.png"), dpi=160)


def main():
    t = pd.read_csv(os.path.join(OUTPUT, "ties.csv"))
    t = t[t.core == "EDC"]
    m = pd.read_csv(os.path.join(OUTPUT, "matches_EDC.csv"))
    d = pd.read_parquet(os.path.join(OUTPUT, "crossings_EDC.parquet"))
    ladder(t, m)
    offsets(d, m)


if __name__ == "__main__":
    main()
