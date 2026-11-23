"""Figure of the ages of the Leysinger_Vieli_2011 layers at the three cores its lines reach (notes.md, H3, SPICEcore):
our depth at each core on that core's chronology, against the published ages.

Inputs (output directory of ties.py): ties.csv (EDC, Vostok) and ties_SPICEcore.csv.

    python docs/harmonisation/figures/leysinger_cores.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anta_database.harmonisation import chronologies
from anta_database.harmonisation.ties import OUTPUT

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
# Near SPICEcore, the depths of line 121 vary by about 50 m (up to 100 m) within 5 km of the core: the error bar there
SPICE_SPREAD = 50


def main():
    t = pd.concat([pd.read_csv(os.path.join(OUTPUT, "ties.csv")),
                   pd.read_csv(os.path.join(OUTPUT, "ties_SPICEcore.csv"))])
    t = t[t.dataset == "Leysinger_Vieli_2011"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    off = {"Vostok": -0.2, "SPICEcore": 0.0, "EDC": 0.2}
    for irh in sorted(t.irh.unique()):
        g = t[t.irh == irh]
        ax.scatter(g.age_published.iloc[0], irh, marker="D", s=40, facecolors="none", edgecolors=INK, lw=1.2,
                   zorder=4, label="Published" if irh == 1 else None)
        for r in g.itertuples():
            y = irh + off[r.core]
            if r.core == "Vostok":
                ax.errorbar(r.age_ref, y, xerr=np.hypot(r.age_ref_unc_chrono, r.age_ref_unc_depth), fmt="o", ms=6,
                            color="#2a78d6", label="Vostok, AICC2023 (0.6 km)" if irh == 1 else None, zorder=3)
                ax.scatter(r.age_authors, y, s=30, facecolors="none", edgecolors="#2a78d6", lw=1, zorder=3,
                           label="Vostok, RR03 (the authors' chronology)" if irh == 1 else None)
            elif r.core == "SPICEcore" and np.isfinite(r.age_ref):
                lo, _ = chronologies.age_at("SP19", "SPICEcore", [r.depth - SPICE_SPREAD])
                hi, _ = chronologies.age_at("SP19", "SPICEcore", [min(r.depth + SPICE_SPREAD, 1751)])
                ax.errorbar(r.age_ref, y, xerr=[[r.age_ref - lo[0]], [hi[0] - r.age_ref]], fmt="o", ms=6,
                            color="#eb6834", label="SPICEcore, SP19 (0.75 km; bar: ±50 m)" if irh == 1 else None,
                            zorder=3)
            elif r.core == "EDC":
                ax.errorbar(r.age_ref, y, xerr=np.hypot(r.age_ref_unc_chrono, r.age_ref_unc_depth), fmt="o", ms=5,
                            color=MUTED, alpha=0.7, label="EDC, AICC2023 (13 km)" if irh == 2 else None, zorder=3)
    ax.annotate("below SP19", (80, 4.0), fontsize=7, color="#eb6834", va="center")
    ax.set_yticks(sorted(t.irh.unique()))
    ax.set_ylim(8.6, 0.4)
    ax.set_xlim(0, 260)
    ax.set_ylabel("Leysinger_Vieli_2011 IRH")
    ax.set_xlabel("Age (ka BP1950) at our depth of the layer at each core")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("The same Leysinger layers dated at Vostok, SPICEcore and EDC", loc="left", color=INK,
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "leysinger_cores.png"), dpi=160)


if __name__ == "__main__":
    main()
