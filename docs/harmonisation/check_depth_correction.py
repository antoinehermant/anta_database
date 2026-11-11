"""Do the published wave speeds and firn corrections explain the depth offsets between datasets?
(plan.md, 3.2). Run crossovers.py and explore.py first: they write the pairs of layers where the lines meet.

The firn correction in the published depths comes from the attribute 'firn correction' of the files (0 for
'none', datasets with 'not known' left out), the wave speed from datasets_methods.csv. Layer pairs of about
the same age (3 %), close in depth (5 %), seen in at least 3 clusters.

    python docs/harmonisation/check_depth_correction.py [database dir] [-o crossovers output dir]
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from load import DEFAULT, firn_correction, load_lines  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("database", nargs="?", default=DEFAULT)
parser.add_argument("-o", "--output")
args = parser.parse_args()
out = Path(args.output or Path.home() / "data" / "isochrones" / "harmonisation" / Path(args.database).name)

speed = {}
for row in csv.DictReader(open(Path(__file__).with_name("datasets_methods.csv"))):
    number = re.match(r"[\d.]+", row["wave_speed_m_per_us"])
    # A range (Muldoon: 168-169.5) is not one value
    if number and "-" not in row["wave_speed_m_per_us"].split()[0]:
        speed[row["dataset"]] = float(number.group())
firn = {}
for L in load_lines(args.database):
    firn.setdefault(L["dataset"], firn_correction(L["attrs"]))
P = {d: (speed[d], firn[d]) for d in firn if d in speed and firn[d] is not None}
print("Used (wave speed m/us, firn m):", {d: P[d] for d in sorted(P)})
print("Left out:", sorted(set(firn) - set(P)), "\n")

rows = json.load(open(out / "irh_pairs.json"))
best = {}
for r in rows:
    if r["dataset_a"] not in P or r["dataset_b"] not in P or r["clusters"] < 3:
        continue
    if abs(r["age_b"] - r["age_a"]) > 0.03 * max(r["age_a"], r["age_b"]):
        continue
    if abs(r["dz_med"]) > 0.05 * r["depth"]:
        continue
    k = (r["dataset_a"], r["IRH_a"], r["dataset_b"])
    if k not in best or abs(r["dz_med"]) < abs(best[k]["dz_med"]):
        best[k] = r
o = []
print(f"{'A':15s} {'B':15s} {'age ka':>7s} {'depth':>6s} {'clus.':>5s} {'measured B-A':>12s} {'predicted':>9s} {'residual':>8s}")
for k, r in sorted(best.items()):
    va, fa = P[r["dataset_a"]]
    vb, fb = P[r["dataset_b"]]
    z = r["depth"]
    pred = (vb * (z - fa) / va + fb) - z
    o.append((r["dz_med"], pred))
    print(f"{r['dataset_a']:15s} {r['dataset_b']:15s} {r['age_a'] / 1e3:7.1f} {z:6.0f} {r['clusters']:5d} "
          f"{r['dz_med']:+12.1f} {pred:+9.1f} {r['dz_med'] - pred:+8.1f}")
d, p = np.array(o).T
print(f"\nlayer pairs {len(o)} | median |measured| {np.median(abs(d)):.1f} | median |residual| {np.median(abs(d - p)):.1f} | "
      f"RMS measured {np.sqrt((d ** 2).mean()):.1f} | RMS residual {np.sqrt(((d - p) ** 2).mean()):.1f} | "
      f"correlation {np.corrcoef(d, p)[0, 1]:.2f}")
