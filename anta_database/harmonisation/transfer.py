"""Dating the layers of one dataset through the layers of others, where their lines meet or pass near (H3, H5).

At each point of the target dataset's lines, each layer of a reference dataset is interpolated there by a Gaussian
process: the reference points within NEIGHBOUR_KM, with the covariance fitted at the cores (ties.py; not refitted at
each point). That gives the reference layers' depths at the point, with their uncertainty, and their ages are our own
ties at a core (ties.csv: the reference age at our depth, AICC2023 at EDC and Vostok). A target layer at that point
then gets its age from this local depth-age profile, linear in depth between the two reference layers around it.

Done at every target point and for every reference dataset: the spread of the ages along the lines and between the
reference datasets says how well it works. The depths of the target are shifted by `firn` (m) first, for a dataset
without a firn correction.

    python -m anta_database.harmonisation.transfer Leysinger_Vieli_2011 [--refs Cavitte_2020 ...] [--firn 0 13]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from anta_database.harmonisation.ties import BIN, BUILD, OUTPUT, _bin_line

NEIGHBOUR_KM = 10.0
MAX_SD = 10.0  # m: reference layers less certain than this at a point are not used there
LENGTH_KM = 8.0  # Matern 3/2 length scale, about those fitted at the cores (6 to 11 km)
NOISE = 2.0  # m, the scatter of the binned picks at the cores (1 to 2 m)
# Where the reference ages come from: our ties at these cores, the first that has the layer
CORES = ["EDC", "Vostok", "Dome Fuji"]


def load(dataset: str, build: str = BUILD) -> pd.DataFrame:
    """All the layer points of a dataset, binned along its lines: file, x, y, irh, depth."""
    import netCDF4

    rows = []
    for path in sorted(glob.glob(os.path.join(build, dataset, "nc", "*.nc"))):
        with netCDF4.Dataset(path) as ds:
            x, y = ds["PSX"][:].filled(np.nan), ds["PSY"][:].filled(np.nan)
            z = ds["IRH_DEPTH"][:].filled(np.nan).astype(float)
            z = z[:, None] if z.ndim == 1 else z
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 2:
                continue
            bx, by, bz = _bin_line(x[ok], y[ok], z[ok])
            for k, irh in enumerate(ds["IRH"][:]):
                m = np.isfinite(bz[:, k])
                rows.append(pd.DataFrame({"file": os.path.basename(path), "x": bx[m], "y": by[m], "irh": int(irh),
                                          "depth": bz[m, k]}))
    return pd.concat(rows, ignore_index=True)


def reference_ages(ties_dir: str = OUTPUT) -> pd.DataFrame:
    """The age (ka, AICC2023 or the reference of the core) of each layer we tied at a core: dataset, irh, age,
    age_unc (the depth part only), core."""
    t = pd.read_csv(os.path.join(ties_dir, "ties.csv"))
    t = t[t.core.isin(CORES)].dropna(subset=["age_ref"])
    t["order"] = t.core.map({c: i for i, c in enumerate(CORES)})
    t = t.sort_values(["order", "depth_sd"]).drop_duplicates(["dataset", "irh"])
    return t[["dataset", "irh", "core", "reference", "age_ref", "age_ref_unc_depth", "depth_sd"]].rename(
        columns={"age_ref": "age", "age_ref_unc_depth": "age_unc"})


def _gp_at(xy: np.ndarray, z: np.ndarray, at: np.ndarray):
    """Kriging with a fixed covariance (LENGTH_KM, NOISE, the variance of z about its mean): mean and sd at `at`."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    var = max(np.var(z), 25.0)
    k = ConstantKernel(var, "fixed") * Matern(LENGTH_KM, "fixed", nu=1.5) + WhiteKernel(NOISE ** 2, "fixed")
    gp = GaussianProcessRegressor(k, optimizer=None, normalize_y=False)
    mean = z.mean()
    gp.fit(xy, z - mean)
    m, s = gp.predict(at, return_std=True)
    return m + mean, np.sqrt(np.maximum(s ** 2 - NOISE ** 2, 0))


def reference_at(target: pd.DataFrame, ref: pd.DataFrame, ages: pd.DataFrame) -> pd.DataFrame:
    """Depth (and sd) of each dated reference layer at each target point: point, irh, depth, sd, age, age_unc."""
    pts = target[["x", "y"]].drop_duplicates().reset_index(drop=True)
    tree = cKDTree(pts.values)
    out = []
    for irh, layer in ref.groupby("irh"):
        a = ages[ages.irh == irh]
        if a.empty:
            continue
        lxy = layer[["x", "y"]].values
        ltree = cKDTree(lxy)
        # Target points with reference points of this layer within NEIGHBOUR_KM
        near = [i for i, n in enumerate(ltree.query_ball_point(pts.values, NEIGHBOUR_KM * 1e3, return_length=True)) if n >= 3]
        for i in near:
            idx = ltree.query_ball_point(pts.values[i], NEIGHBOUR_KM * 1e3)
            if len(idx) > 200:
                d = np.hypot(*(lxy[idx] - pts.values[i]).T)
                idx = np.asarray(idx)[np.argsort(d)[:200]]
            xy_km = (lxy[idx] - pts.values[i]) / 1e3
            m, s = _gp_at(xy_km, layer.depth.values[idx], np.zeros((1, 2)))
            out.append((i, irh, m[0], s[0], a.age.iloc[0], a.age_unc.iloc[0]))
    df = pd.DataFrame(out, columns=["point", "irh", "depth", "sd", "age", "age_unc"])
    return pts.join(df.set_index("point"), how="inner").rename_axis("point").reset_index()


def date_target(target: pd.DataFrame, at: pd.DataFrame, firn: float) -> pd.DataFrame:
    """Age of each target layer at each point from the local depth-age profile of a reference dataset."""
    rows = []
    key = target[["x", "y"]].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={"index": "point"})
    t = target.merge(key, on=["x", "y"])
    for point, prof in at[at.sd <= MAX_SD].groupby("point"):
        prof = prof.sort_values("depth")
        if len(prof) < 2 or not (np.diff(prof.age.values) > 0).all():
            continue  # a profile out of order: the reference layers cross there, or are wrongly dated
        z, a, s = prof.depth.values, prof.age.values, prof.sd.values
        for r in t[t.point == point].itertuples():
            zt = r.depth + firn
            j = np.searchsorted(z, zt)
            if j == 0 or j == len(z):
                continue  # above or below all the reference layers: no extrapolation
            w = (zt - z[j - 1]) / (z[j] - z[j - 1])
            age = a[j - 1] + w * (a[j] - a[j - 1])
            # From the depth sd of the two reference layers, through the slope of the profile
            unc = abs(a[j] - a[j - 1]) / (z[j] - z[j - 1]) * np.hypot(s[j - 1], s[j])
            rows.append((point, r.file, r.irh, zt, age, unc, z[j] - z[j - 1], min(zt - z[j - 1], z[j] - zt)))
    return pd.DataFrame(rows, columns=["point", "file", "irh", "depth", "age", "age_unc", "gap", "to_nearest"])


def _along(line: pd.DataFrame, irh_cols: List[int], at: float, max_gap: float) -> Dict[int, float]:
    """Depth of each layer of a line at a distance along it (m): linear between its two points around, if they are
    within max_gap of each other and both have the layer."""
    d, out = line["along"].values, {}
    j = np.searchsorted(d, at)
    if j == 0 or j == len(d) or d[j] - d[j - 1] > max_gap:
        return out
    w = (at - d[j - 1]) / (d[j] - d[j - 1])
    for irh in irh_cols:
        z0, z1 = line[irh].values[j - 1], line[irh].values[j]
        if np.isfinite(z0) and np.isfinite(z1):
            out[irh] = z0 + w * (z1 - z0)
    return out


def _wide(points: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Per line: x, y, along (m), and one column of depth per layer."""
    out = {}
    for file, g in points.groupby("file"):
        w = g.pivot_table(index=["x", "y"], columns="irh", values="depth", sort=False).reset_index()
        xy = w[["x", "y"]].values
        w["along"] = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        out[file] = w
    return out


def crossings(target: pd.DataFrame, ref: pd.DataFrame, target_gap: float = 6e3, ref_gap: float = 1e3) -> pd.DataFrame:
    """Where the lines of the target cross those of a reference dataset: one row per crossing, with the depths of all
    layers of both lines there (linear along each line; gaps up to target_gap and ref_gap m), and the angle."""
    from shapely.geometry import LineString, Point

    tl, rl = _wide(target), _wide(ref)
    rows = []
    rgeom = {f: LineString(w[["x", "y"]].values) for f, w in rl.items() if len(w) > 1}
    for tf, tw in tl.items():
        if len(tw) < 2:
            continue
        tline = LineString(tw[["x", "y"]].values)
        tcols = [c for c in tw.columns if isinstance(c, (int, np.integer))]
        for rf, rline in rgeom.items():
            if not tline.envelope.intersects(rline.envelope):
                continue
            hit = tline.intersection(rline)
            pts = [hit] if hit.geom_type == "Point" else [g for g in getattr(hit, "geoms", []) if g.geom_type == "Point"]
            rw = rl[rf]
            rcols = [c for c in rw.columns if isinstance(c, (int, np.integer))]
            for p in pts:
                td = _along(tw, tcols, tline.project(p), target_gap)
                rd = _along(rw, rcols, rline.project(p), ref_gap)
                if not td or not rd:
                    continue
                rows.append({"target_file": tf, "ref_file": rf, "x": p.x, "y": p.y, "target": td, "ref": rd})
    return pd.DataFrame(rows)


def date_at_crossings(cross: pd.DataFrame, ages: pd.DataFrame, firn: float) -> pd.DataFrame:
    """Age of each target layer at each crossing, from the reference layers there (as date_target, without the GP)."""
    age = dict(zip(ages.irh, ages.age))
    rows = []
    for c in cross.itertuples():
        prof = sorted((z, age[i]) for i, z in c.ref.items() if i in age)
        if len(prof) < 2 or not all(b[1] > a[1] for a, b in zip(prof, prof[1:])):
            continue
        z, a = np.array(prof).T
        for irh, zt in c.target.items():
            zt = zt + firn
            j = np.searchsorted(z, zt)
            if j == 0 or j == len(z):
                continue
            w = (zt - z[j - 1]) / (z[j] - z[j - 1])
            rows.append((c.target_file, c.ref_file, c.x, c.y, irh, zt, a[j - 1] + w * (a[j] - a[j - 1]), z[j] - z[j - 1]))
    return pd.DataFrame(rows, columns=["file", "ref_file", "x", "y", "irh", "depth", "age", "gap"])


def run(target_name: str, refs: List[str], firns: List[float], build: str = BUILD, out: str = OUTPUT) -> pd.DataFrame:
    target = load(target_name, build)
    ages = reference_ages(out)
    results, crossed = [], []
    for ref_name in refs:
        ref = load(ref_name, build)
        # Only the reference points near the target lines
        tree = cKDTree(target[["x", "y"]].drop_duplicates().values)
        d, _ = tree.query(ref[["x", "y"]].values, distance_upper_bound=NEIGHBOUR_KM * 1e3)
        ref = ref[np.isfinite(d)]
        at = reference_at(target, ref, ages[ages.dataset == ref_name])
        cross = crossings(target, ref)
        a_ref = ages[ages.dataset == ref_name]
        for firn in firns:
            results.append(date_target(target, at, firn).assign(reference=ref_name, firn=firn))
            if len(cross):
                crossed.append(date_at_crossings(cross, a_ref, firn).assign(reference=ref_name, firn=firn))
    df = pd.concat(results, ignore_index=True)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"transfer_{target_name}.csv"), index=False, float_format="%.6g")
    if crossed:
        pd.concat(crossed, ignore_index=True).to_csv(os.path.join(out, f"crossings_{target_name}.csv"), index=False,
                                                     float_format="%.6g")
    return df


def summary(df: pd.DataFrame, published: Dict[int, float]) -> pd.DataFrame:
    """Per target layer, reference dataset and firn shift: points, median age, spread (the 16th to 84th percentile)."""
    g = df.groupby(["irh", "firn", "reference"]).age
    s = pd.DataFrame({"points": g.size(), "age": g.median(), "p16": g.quantile(0.16), "p84": g.quantile(0.84)})
    s["published"] = s.index.get_level_values("irh").map(published)
    return s.reset_index()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("target")
    p.add_argument("--refs", nargs="+", default=["Cavitte_2020", "Mulvaney_2023", "Winter_2018", "Chung_2023",
                                                   "Bodart_2025b"])
    p.add_argument("--firn", nargs="+", type=float, default=[0.0])
    p.add_argument("--build", default=BUILD)
    p.add_argument("-o", "--out", default=OUTPUT)
    a = p.parse_args(argv)
    df = run(a.target, a.refs, a.firn, a.build, a.out)
    import netCDF4

    published = {}
    for path in glob.glob(os.path.join(a.build, a.target, "nc", "*.nc")):
        with netCDF4.Dataset(path) as ds:
            published.update(zip(ds["IRH"][:].tolist(), (ds["IRH_AGE"][:] / 1000).tolist()))
    print(summary(df, published).to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main()
