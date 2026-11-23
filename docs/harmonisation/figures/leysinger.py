"""Figures of the dating of Leysinger_Vieli_2011 through the other datasets (notes.md, H3).

    python docs/harmonisation/figures/leysinger.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from anta_database.harmonisation.cores import ice_cores
from anta_database.harmonisation.transfer import _wide, crossings, load, reference_ages
from anta_database.harmonisation.ties import OUTPUT

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "Leysinger_Vieli_2011"
REFS = ["Cavitte_2020", "Mulvaney_2023", "Bodart_2025a", "Chung_2023", "Winter_2018"]
# Categorical slots in fixed order (dataviz reference palette)
COLOURS = dict(zip(REFS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
# The two clusters of Leysinger Vieli et al. (2011), paragraph 17
VOSTOK_CLUSTER = {"009.nc", "121.nc", "123.nc", "133.nc", "134.nc"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
PUBLISHED = {1: 17.5, 2: 36.6, 3: 62.5, 4: 72.9, 5: 118.3, 6: 129.8, 7: 170.5, 8: 244.7, 9: 352.4}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def cluster(file):
    return "Vostok cluster" if file in VOSTOK_CLUSTER else "Dome C cluster"


def _draw_map(ax, target, refs, cross, box=None):
    tree = cKDTree(target[["x", "y"]].values)
    for name in REFS:
        r = refs[name]
        d, _ = tree.query(r[["x", "y"]].values, distance_upper_bound=30e3)
        for _, g in r[np.isfinite(d)].drop_duplicates(["file", "x", "y"]).groupby("file"):
            ax.plot(g.x / 1e3, g.y / 1e3, color=COLOURS[name], lw=0.6, alpha=0.35, zorder=1)
    for file, g in target.drop_duplicates(["file", "x", "y"]).groupby("file"):
        vostok = file in VOSTOK_CLUSTER
        ax.plot(g.x / 1e3, g.y / 1e3, color=INK, lw=2.2 if vostok else 1.4, ls="-" if vostok else "--", zorder=3)
        inside = g if box is None else g[(g.x / 1e3).between(*box[:2]) & (g.y / 1e3).between(*box[2:])]
        if len(inside):
            m = inside.iloc[len(inside) // 3]
            ax.annotate(file[:-3], (m.x / 1e3, m.y / 1e3), xytext=(4, 6), textcoords="offset points", color=INK,
                        fontsize=8, zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for name in REFS:
        c = cross[cross.reference == name].drop_duplicates(["x", "y"])
        ax.scatter(c.x / 1e3, c.y / 1e3, s=30, color=COLOURS[name], edgecolor="white", lw=0.8, zorder=4,
                   label=f"{name} ({len(c)} crossings)")
    cores = ice_cores()
    for core in ["EDC", "Vostok", "SPICEcore"]:
        c = cores.loc[core]
        ax.scatter(c.x / 1e3, c.y / 1e3, marker="^", s=70, color=INK, zorder=6)
        ax.annotate(core, (c.x / 1e3, c.y / 1e3), xytext=(6, -12), textcoords="offset points", fontsize=9,
                    color=INK, weight="bold", zorder=6)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.5)
    ax.set_xlabel("x (km, EPSG:3031)")


def fig_map(target, refs, cross):
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    _draw_map(ax, target, refs, cross)
    xy = target[["x", "y"]].values / 1e3
    ax.set_xlim(xy[:, 0].min() - 60, xy[:, 0].max() + 60)
    ax.set_ylim(xy[:, 1].min() - 60, xy[:, 1].max() + 60)
    ax.set_ylabel("y (km)")
    box = (1180, 1540, -1000, -800)
    ax.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec=MUTED, lw=1))
    ax.set_title("Leysinger_Vieli_2011 lines and the newer lines crossing them", loc="left", color=INK)
    ax.plot([], [], color=INK, lw=2.2, label="Leysinger, Vostok cluster")
    ax.plot([], [], color=INK, lw=1.4, ls="--", label="Leysinger, Dome C cluster")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    _draw_map(zoom, target, refs, cross, box)
    zoom.set_xlim(*box[:2])
    zoom.set_ylim(*box[2:])
    zoom.set_title("Around Dome C", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "leysinger_crossings_map.png"), dpi=160)


def fig_ages(cross):
    c = cross[cross.firn == 14].copy()
    c["cluster"] = c.file.map(cluster)
    irhs = [2, 4, 6, 8]
    fig, axes = plt.subplots(1, len(irhs), figsize=(11, 4.2), sharey=False)
    order = sorted(c.file.unique(), key=lambda f: (f in VOSTOK_CLUSTER, f))
    for ax, irh in zip(axes, irhs):
        g = c[c.irh == irh]
        files = [f for f in order if f in set(g.file)]
        for i, f in enumerate(files):
            for k, name in enumerate(REFS):
                h = g[(g.file == f) & (g.reference == name)]
                jitter = (k - 2) * 0.08
                ax.scatter(np.full(len(h), i + jitter), h.age, s=14, color=COLOURS[name], edgecolor="white", lw=0.5,
                           zorder=3, label=name if (irh == irhs[0] and i == 0) else None)
            ax.hlines(g[g.file == f].age.median(), i - 0.3, i + 0.3, color=INK, lw=2, zorder=4)
        ax.axhline(PUBLISHED[irh], color=MUTED, lw=1, ls=":", zorder=2)
        ax.annotate(f"published {PUBLISHED[irh]:g} ka", (len(files) - 0.5, PUBLISHED[irh]), xytext=(0, 3),
                    textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)
        split = sum(f not in VOSTOK_CLUSTER for f in files)
        if 0 < split < len(files):
            ax.axvline(split - 0.5, color=GRID, lw=1)
        ax.set_xticks(range(len(files)), [f[:-3] for f in files])
        ax.set_title(f"IRH {irh}", loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.5)
    axes[0].set_ylabel("Age at the crossing (ka, AICC2023, via the crossing dataset)")
    fig.suptitle("Ages of Leysinger layers at the crossings, per line: Dome C cluster (left of the line), "
                 "Vostok cluster (right). Bar: median", x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(os.path.join(HERE, "leysinger_crossing_ages.png"), dpi=160)


def fig_profile(target, refs, ages):
    """One crossing on each cluster with Cavitte: its layers (depth, age) and the Leysinger layers read on them."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    a = ages[ages.dataset == "Cavitte_2020"].set_index("irh").age
    cross = crossings(target, refs["Cavitte_2020"])
    for ax, file in zip(axes, ["108.nc", "009.nc"]):
        cc = cross[cross.target_file == file]
        if cc.empty:
            continue
        c = cc.iloc[np.argmax(cc.target.map(len))]
        prof = sorted((z, a[i]) for i, z in c.ref.items() if i in a.index)
        z, t = np.array(prof).T
        ax.plot(t, z, color=COLOURS["Cavitte_2020"], lw=2, zorder=2)
        ax.scatter(t, z, s=30, color=COLOURS["Cavitte_2020"], edgecolor="white", lw=0.8, zorder=3)
        for irh, zt in sorted(c.target.items()):
            zt += 14
            if not z[0] < zt < z[-1]:
                continue
            age = np.interp(zt, z, t)
            ax.plot([0, age], [zt, zt], color=MUTED, lw=0.8, ls=":")
            ax.plot([age, age], [zt, z.max() * 1.05], color=MUTED, lw=0.8, ls=":")
            ax.scatter([age], [zt], marker="D", s=36, color=INK, zorder=4)
            ax.annotate(f"IRH {irh}: {age:.0f} ka (published {PUBLISHED[irh]:g})", (age, zt), xytext=(6, -3),
                        textcoords="offset points", fontsize=7.5, color=INK)
        ax.set_title(f"Line {file[:-3]} ({cluster(file)}) × Cavitte {c.ref_file[:-3]}", loc="left", color=INK)
        ax.set_xlabel("Age (ka)")
        ax.grid(color=GRID, lw=0.5)
        ax.set_xlim(0, t.max() * 1.05)
    axes[0].set_ylim(z.max() * 1.05, 0)
    axes[0].set_ylabel("Depth (m)")
    axes[1].scatter([], [], s=30, color=COLOURS["Cavitte_2020"], label="Cavitte_2020 layers (our AICC2023 ages at EDC)")
    axes[1].scatter([], [], marker="D", color=INK, label="Leysinger layer, read on the profile")
    axes[1].legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle("Reading the age of a Leysinger layer at a crossing (depth +14 m, firn)", x=0.01, ha="left",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, "leysinger_crossing_profile.png"), dpi=160)


def fig_offsets():
    """Leysinger (Dome C cluster) against the Cavitte layers they match, at the crossings: depth against depth, and
    the offset against depth (constant: a common shift; growing with depth: a wave speed or thickness scale)."""
    from anta_database.harmonisation.transfer import crossings as _crossings

    T, C = load(TARGET), load("Cavitte_2020")
    c = _crossings(T, C)
    c = c[~c.target_file.isin(VOSTOK_CLUSTER)]
    pairs = [(2, 3, "38.1"), (4, 5, "73.4"), (6, 10, "127.8"), (8, 14, "220.1")]
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))
    allz = []
    for (l, r, age), col in zip(pairs, colours):
        zl = np.array([x.target.get(l, np.nan) for x in c.itertuples()])
        zr = np.array([x.ref.get(r, np.nan) for x in c.itertuples()])
        ok = np.isfinite(zl) & np.isfinite(zr)
        allz += list(zr[ok])
        label = f"Leysinger {l} vs Cavitte {r} ({age} ka), n={ok.sum()}"
        a1.scatter(zr[ok], zl[ok], s=16, color=col, edgecolor="white", lw=0.5, label=label, zorder=3)
        a2.scatter(zr[ok], zr[ok] - zl[ok], s=16, color=col, edgecolor="white", lw=0.5, zorder=3)
        a2.hlines(np.median(zr[ok] - zl[ok]), np.min(zr[ok]), np.max(zr[ok]), color=col, lw=2, zorder=4)
    lim = (min(allz) - 100, max(allz) + 100)
    a1.plot(lim, lim, color=MUTED, lw=1, ls=":")
    a1.set_xlim(lim), a1.set_ylim(lim)
    a1.invert_yaxis()
    a1.set_xlabel("Cavitte_2020 depth at the crossing (m)")
    a1.set_ylabel("Leysinger_Vieli_2011 depth (m, as published)")
    a1.set_title("Depths at the crossings (Dome C cluster); dotted: same depth", loc="left", color=INK)
    a1.legend(loc="lower right", frameon=False, fontsize=7.5)
    a1.grid(color=GRID, lw=0.5)
    a2.axhline(0, color=MUTED, lw=1, ls=":")
    a2.axhspan(-14, 14, color=GRID, alpha=0.6, lw=0)
    a2.annotate("±14 m: a firn correction", (min(allz), -14), xytext=(0, -11), textcoords="offset points",
                fontsize=7.5, color=MUTED)
    zz = np.linspace(min(allz), max(allz), 2)
    a2.plot(zz, 0.03 * zz, color=MUTED, lw=1, ls="--")
    a2.annotate("3 percent of the depth (wave speed or thickness)", (1250, 0.03 * 1250), xytext=(0, -14),
                textcoords="offset points", fontsize=7.5, color=MUTED)
    a2.set_xlabel("Cavitte_2020 depth (m)")
    a2.set_ylabel("Cavitte − Leysinger (m); bar: median")
    a2.set_title("Offset at each crossing", loc="left", color=INK)
    a2.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "leysinger_cavitte_offsets.png"), dpi=160)


def main():
    target = load(TARGET)
    refs = {name: load(name) for name in REFS}
    cross = pd.read_csv(os.path.join(OUTPUT, f"crossings_{TARGET}.csv"))
    fig_map(target, refs, cross[cross.firn == 14])
    fig_ages(cross)
    fig_profile(target, refs, reference_ages())
    fig_offsets()


if __name__ == "__main__":
    main()
