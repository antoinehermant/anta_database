"""First look at the harmonisation of the layers (2026-11-01, see plan.md): crossovers between
datasets, the depths and ages of their IRHs there, and the distance from the ice cores to the radar
lines. Exploratory, not part of the package.

    python docs/harmonisation/explore.py [database dir]    (default: ~/data/isochrones/AntADatabase)
"""
import sys, collections

import glob, os, json, pickle, numpy as np, h5py
D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/data/isochrones/AntADatabase")
names = sorted(json.load(open(f"{D}/manifest.json"))["datasets"])
lines = []
for n in names:
    for f in sorted(glob.glob(f"{D}/{n}/h5/*.h5")):
        with h5py.File(f, "r") as h:
            g = lambda k: h[k][...].astype(float) if k in h else None
            lines.append(dict(dataset=n, file=os.path.basename(f), x=g("PSX"), y=g("PSY"), depth=g("IRH_DEPTH"),
                              irh=h["IRH"][...].astype(int), age=g("IRH_AGE"), unc=g("IRH_AGE_UNC"),
                              thk=g("ICE_THK"), surf=g("SURF_ELEV")))
print(len(lines), "lines")

# --- Ice cores: nearest radar point with an IRH, per dataset

import pickle, numpy as np
from pyproj import Transformer
cores = {"EDC": (-75.1017, 123.3482), "BELDC": (-75.2992, 122.4451), "Dome Fuji": (-77.3167, 39.7), "Vostok": (-78.4644, 106.8372),
 "EDML": (-75.0025, 0.0684), "TALDICE": (-72.8167, 159.1833), "Taylor Dome": (-77.7964, 158.7239), "Law Dome": (-66.7725, 112.8113),
 "SPICEcore": (-89.99, -98.16), "WDC": (-79.467, -112.085), "Byrd": (-80.0167, -119.5167), "Siple Dome": (-81.65, -148.81),
 "Skytrain": (-79.741, -78.5448), "ABN": (-71.17, 111.37), "RICE": (-79.36, -161.71), "Fletcher": (-77.9, -82.61),
 "Berkner": (-79.55, -45.68), "JRI": (-64.2017, -57.685)}
t = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
for c, (lat, lon) in cores.items():
    cx, cy = t.transform(lon, lat)
    best = {}
    for L in lines:
        ok = np.isfinite(L["depth"]).any(axis=1) & np.isfinite(L["x"])
        if not ok.any(): continue
        d = np.hypot(L["x"][ok] - cx, L["y"][ok] - cy)
        i = np.argmin(d)
        if d[i] < best.get(L["dataset"], (np.inf,))[0]:
            n_irh = int(np.isfinite(L["depth"][ok][i]).sum())
            best[L["dataset"]] = (d[i], n_irh, L["file"])
    near = sorted((v[0], k, v[1]) for k, v in best.items() if v[0] < 20000)
    print(f"{c:12s}", "; ".join(f"{k} {d/1000:.2f} km ({n} IRHs)" for d, k, n in near) or "no dataset within 20 km")

# --- Crossovers between datasets

import pickle, numpy as np, collections, json
STEP = 100.0   # m, decimation of the lines for the intersection search
CELL = 5000.0
segs = []      # (line index, i0, i1) indices into original arrays
grid = collections.defaultdict(list)
for li, L in enumerate(lines):
    has = np.isfinite(L["depth"]).any(axis=1) & np.isfinite(L["x"]) & np.isfinite(L["y"])
    idx = np.nonzero(has)[0]
    if len(idx) < 2: continue
    keep = [idx[0]]
    for i in idx[1:]:
        if np.hypot(L["x"][i] - L["x"][keep[-1]], L["y"][i] - L["y"][keep[-1]]) >= STEP: keep.append(i)
    for a, b in zip(keep[:-1], keep[1:]):
        if np.hypot(L["x"][b] - L["x"][a], L["y"][b] - L["y"][a]) > 2000: continue   # gap, not a segment
        s = len(segs); segs.append((li, a, b))
        xs = (L["x"][a], L["x"][b]); ys = (L["y"][a], L["y"][b])
        for cx in range(int(min(xs) // CELL), int(max(xs) // CELL) + 1):
            for cy in range(int(min(ys) // CELL), int(max(ys) // CELL) + 1):
                grid[(cx, cy)].append(s)

def inter(p, p2, q, q2):
    r = p2 - p; s = q2 - q; d = r[0] * s[1] - r[1] * s[0]
    if d == 0: return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / d; u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / d
    return (t, u) if 0 <= t <= 1 and 0 <= u <= 1 else None

def depth_at(L, a, b, t):
    """Depth of each IRH at fraction t between points a and b: nearest original point (lines are dense)."""
    i = a + int(round(t * (b - a)))
    return L["depth"][i]

xovers = []; seen = set()
for cell, ss in grid.items():
    for k, s1 in enumerate(ss):
        l1, a1, b1 = segs[s1]
        for s2 in ss[k + 1:]:
            l2, a2, b2 = segs[s2]
            if lines[l1]["dataset"] == lines[l2]["dataset"] or (s1, s2) in seen: continue
            seen.add((s1, s2))
            L1, L2 = lines[l1], lines[l2]
            r = inter(np.array([L1["x"][a1], L1["y"][a1]]), np.array([L1["x"][b1], L1["y"][b1]]),
                      np.array([L2["x"][a2], L2["y"][a2]]), np.array([L2["x"][b2], L2["y"][b2]]))
            if r is None: continue
            x = L1["x"][a1] + r[0] * (L1["x"][b1] - L1["x"][a1]); y = L1["y"][a1] + r[0] * (L1["y"][b1] - L1["y"][a1])
            xovers.append(dict(l1=l1, l2=l2, x=x, y=y, d1=depth_at(L1, a1, b1, r[0]), d2=depth_at(L2, a2, b2, r[1])))
print(len(xovers), "crossovers between lines of different datasets")

# Pairs of IRHs at the crossovers
pairs = collections.defaultdict(list)   # (dsA, irhA, dsB, irhB) -> list of (depthA, depthB)
dspairs = collections.Counter()
for c in xovers:
    L1, L2 = lines[c["l1"]], lines[c["l2"]]
    if L1["dataset"] > L2["dataset"]: L1, L2, d1, d2 = L2, L1, c["d2"], c["d1"]
    else: d1, d2 = c["d1"], c["d2"]
    dspairs[(L1["dataset"], L2["dataset"])] += 1
    for i, (ia, za) in enumerate(zip(L1["irh"], d1)):
        for j, (ib, zb) in enumerate(zip(L2["irh"], d2)):
            if np.isfinite(za) and np.isfinite(zb):
                pairs[(L1["dataset"], int(ia), float(L1["age"][i]), float(L1["unc"][i]), L2["dataset"], int(ib), float(L2["age"][j]), float(L2["unc"][j]))].append((za, zb))
print("\nCrossovers per pair of datasets:")
for (a, b), n in dspairs.most_common(): print(f"  {a:22s} x {b:22s} {n}")
# For each IRH of A at its crossovers with B: the IRH of B closest in depth (median over crossovers)
rows = []
for key, v in pairs.items():
    v = np.array(v); dz = v[:, 1] - v[:, 0]
    rows.append(dict(A=key[0], irhA=key[1], ageA=key[2], uncA=key[3], B=key[4], irhB=key[5], ageB=key[6], uncB=key[7],
                     n=len(v), depth=float(np.median(v[:, 0])), dz_med=float(np.median(dz)), dz_mad=float(np.median(abs(dz - np.median(dz))))))
best = {}
for r in rows:
    k = (r["A"], r["irhA"], r["B"])
    if k not in best or abs(r["dz_med"]) < abs(best[k]["dz_med"]): best[k] = r
print("\nClosest-in-depth IRH pairs (|median depth difference| < 2% of depth), with their ages:")
match = [r for r in best.values() if abs(r["dz_med"]) < 0.02 * r["depth"]]
for r in sorted(match, key=lambda r: (r["A"], r["B"], r["depth"])):
    da = r["ageB"] - r["ageA"]
    print(f"  {r['A']:20s} IRH{r['irhA']:<3d} {r['ageA']/1e3:7.1f} ka | {r['B']:20s} IRH{r['irhB']:<3d} {r['ageB']/1e3:7.1f} ka | "
          f"n={r['n']:4d} depth {r['depth']:6.0f} m  dz {r['dz_med']:+6.1f} ± {r['dz_mad']:4.1f} m  dAge {da/1e3:+6.1f} ka")

# For check_depth_correction.py
json.dump(rows, open("/tmp/anta_pairs.json", "w"))
