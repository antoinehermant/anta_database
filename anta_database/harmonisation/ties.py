"""The depth of each published layer at each ice core its lines reach (H3), and its age there.

For one core and one layer of one dataset: all its points within RADIUS of the core (binned along the lines to
BIN), and a Gaussian process fitted to them (a Matern 3/2 covariance and a white noise, both fitted to the data),
which gives the depth at the core and its uncertainty: small where a line passes over the core, growing with the
distance to the nearest line.

It is tested by leaving one line out: every line passing within LOLO_KM of the core is removed in turn, and the
depth predicted at its point nearest to the core is compared with its own depth there. (The fractional depth, depth /
ice thickness, was tried too and did worse, 12.3 m against 2.7 m: notes.md, H3.)

The age at the core then comes from the chronologies (chronologies.py): the authors' one (raw_md.json of anta_json,
"ice core" and "chronology"), which should give their published age back, and the reference of the core (AICC2023,
or DF2021, WD2014, SP19 where AICC2023 has none).

    python -m anta_database.harmonisation.ties [--cores EDC Vostok "Dome Fuji"] [--build dir] [-o dir] [-j 3]
"""
import os

# One thread of linear algebra per process (the cores run in parallel, -j)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from anta_database.harmonisation import chronologies as chrono
from anta_database.harmonisation.cores import ice_cores

BUILD = os.path.expanduser("~/data/isochrones/AntADatabase-v2-build")
ANTA_JSON = os.path.expanduser("~/projects/phd/python/anta_json")
OUTPUT = os.path.expanduser("~/data/isochrones/harmonisation/ties")
RADIUS = 20e3  # m
BIN = 250.0  # m along the lines
LOLO_KM = 3.0
MAX_POINTS = 800  # per fit: all within NEAR_KM of the core, a random sample of the others
NEAR_KM = 3.0
# The reference chronology of each core: AICC2023 where it has one
REFERENCE = {"EDC": "AICC2023", "EDML": "AICC2023", "TALDICE": "AICC2023", "Vostok": "AICC2023",
             "Dome Fuji": "DF2021", "WDC": "WD2014", "SPICEcore": "SP19", "Byrd": "Byrd-BB2001"}


# --- The points near a core ---------------------------------------------------------------------------------------

def _bin_line(x, y, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median of the points in bins of BIN m along a line: x, y, values (n, k)."""
    along = np.r_[0, np.cumsum(np.nan_to_num(np.hypot(np.diff(x), np.diff(y))))]
    df = pd.DataFrame(np.c_[x, y, values]).assign(b=np.floor(along / BIN).astype(int))
    g = df.groupby("b")
    xy, v = g[[0, 1]].mean().values, g[list(range(2, 2 + values.shape[1]))].median().values
    return xy[:, 0], xy[:, 1], v


def points_near(core: str, build: str = BUILD, radius: float = RADIUS) -> pd.DataFrame:
    """Every layer point (binned) within radius of a core: dataset, file, line, x, y (m), dist (m), irh, age (yr),
    depth (m), thk (m), firn (the attribute 'firn correction' of the file)."""
    import netCDF4

    c = ice_cores().loc[core]
    rows = []
    for path in sorted(glob.glob(os.path.join(build, "*", "nc", "*.nc"))):
        with netCDF4.Dataset(path) as ds:
            if "IRH_DEPTH" not in ds.variables:
                continue
            x, y = ds["PSX"][:].filled(np.nan), ds["PSY"][:].filled(np.nan)
            d = np.hypot(x - c.x, y - c.y)
            near = d <= radius
            if not near.any():
                continue
            # The whole line between its first and last point near the core, so the bins are along it
            i0, i1 = np.flatnonzero(near)[[0, -1]]
            sl = slice(i0, i1 + 1)
            depth = ds["IRH_DEPTH"][sl].filled(np.nan).astype(float)
            if depth.ndim == 1:
                depth = depth[:, None]
            thk = ds["ICE_THK"][sl].filled(np.nan).astype(float) if "ICE_THK" in ds.variables else np.full(len(depth), np.nan)
            bx, by, bv = _bin_line(x[sl], y[sl], np.c_[thk, depth])
            bd = np.hypot(bx - c.x, by - c.y)
            keep = bd <= radius
            irh, age = ds["IRH"][:], ds["IRH_AGE"][:]
            for k in range(depth.shape[1]):
                ok = keep & np.isfinite(bv[:, k + 1])
                if not ok.any():
                    continue
                rows.append(pd.DataFrame({
                    "dataset": ds.dataset, "file": os.path.basename(path), "line": str(getattr(ds, "line ID", "")),
                    "x": bx[ok], "y": by[ok], "dist": bd[ok], "irh": int(irh[k]), "age": float(age[k]),
                    "depth": bv[ok, k + 1], "thk": bv[ok, 0], "firn": str(getattr(ds, "firn correction", ""))}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --- Gaussian processes -------------------------------------------------------------------------------------------

def _gp(xy_km: np.ndarray, v: np.ndarray, noise: float):
    """A GP fitted to v at xy_km: constant mean, Matern 3/2 (length scale fitted, 0.3 to 300 km), white noise
    (fitted, from noise/10 to noise*10, in the units of v)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=5.0, length_scale_bounds=(0.3, 300), nu=1.5)
              + WhiteKernel((noise / np.std(v)) ** 2 if np.std(v) > 0 else 1e-2, (1e-6, 1e1)))
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(xy_km, v)
    return gp


def _predict(gp, at_km: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the layer itself (without the white noise of the points)."""
    mean, sd = gp.predict(at_km, return_std=True)
    k = gp.kernel_
    white = k.k2.noise_level * float(np.ravel(gp._y_train_std)[0]) ** 2
    return mean, np.sqrt(np.maximum(sd ** 2 - white, 0))


def _params(gp) -> Dict[str, float]:
    k = gp.kernel_
    scale = float(np.ravel(getattr(gp, "_y_train_std", 1.0))[0])
    return {"length_km": float(k.k1.k2.length_scale), "noise": float(np.sqrt(k.k2.noise_level) * scale)}


def _km(df: pd.DataFrame, core_xy) -> np.ndarray:
    return np.c_[(df.x.values - core_xy[0]) / 1e3, (df.y.values - core_xy[1]) / 1e3]


def _thin(layer: pd.DataFrame) -> pd.DataFrame:
    if len(layer) <= MAX_POINTS:
        return layer
    near = layer.dist <= NEAR_KM * 1e3
    far = layer[~near]
    n = max(MAX_POINTS - int(near.sum()), min(len(far), MAX_POINTS // 2))
    return pd.concat([layer[near], far.sample(min(n, len(far)), random_state=0)])


def tie(layer: pd.DataFrame, core_xy) -> dict:
    """Depth of one layer of one dataset at the core, and the errors of leaving one line out."""
    at = np.zeros((1, 2))
    layer = _thin(layer)
    xy = _km(layer, core_xy)
    out = {"lines": layer.file.nunique(), "points": len(layer), "nearest_km": layer.dist.min() / 1e3}
    gp = _gp(xy, layer.depth.values, 5.0)
    m, s = _predict(gp, at)
    out.update(depth=m[0], depth_sd=s[0], **{f"depth_{k}": v for k, v in _params(gp).items()})
    # Leave one line out: the lines passing within LOLO_KM
    errors = []
    for file, line in layer.groupby("file"):
        if line.dist.min() > LOLO_KM * 1e3 or layer.file.nunique() < 2:
            continue
        rest = layer[layer.file != file]
        p = line.loc[[line.dist.idxmin()]]
        g = _gp(_km(rest, core_xy), rest.depth.values, 5.0)
        errors.append(float(_predict(g, _km(p, core_xy))[0][0] - p.depth.iloc[0]))
    out["lolo_n"] = len(errors)
    out["lolo_rms"] = float(np.sqrt(np.mean(np.square(errors)))) if errors else np.nan
    out["lolo_errors"] = json.dumps([round(v, 1) for v in errors])
    return out


# --- Metadata and ages --------------------------------------------------------------------------------------------

def dating(anta_json: str = ANTA_JSON) -> Dict[Tuple[str, int], Dict[str, str]]:
    """The chronology the authors used at each core, by (dataset, IRH): {core: chronology}."""
    out = {}
    for path in glob.glob(os.path.join(anta_json, "*", "raw_md.json")):
        with open(path) as f:
            md = json.load(f)
        for r in md if isinstance(md, list) else [md]:
            if not str(r.get("IRH", "")).isdigit():  # none, or the basal unit
                continue
            cores, chronos = r.get("ice core"), r.get("chronology")
            cores = cores if isinstance(cores, list) else [cores]
            chronos = chronos if isinstance(chronos, list) else [chronos] * len(cores)
            out[(r["dataset"], int(r["IRH"]))] = {c: ch for c, ch in zip(cores, chronos) if c and ch}
    return out


def _age(name: Optional[str], core: str, depth: float) -> Tuple[float, float]:
    try:
        a, u = chrono.age_at(name, core, [depth])
        return float(a[0]), float(u[0])
    except FileNotFoundError:
        return np.nan, np.nan


def ages(row: dict, core: str, authors: Optional[str]) -> dict:
    """The ages (ka) at the depths of a tie: authors' chronology and reference chronology."""
    ref = REFERENCE.get(core, "")
    d, sd = row["depth"], row["depth_sd"]
    a, _ = _age(authors, core, d) if authors else (np.nan, np.nan)
    r, ru = _age(ref, core, d) if ref else (np.nan, np.nan)
    # The uncertainty of the reference age from the depth: the slope of the chronology times the depth sd
    r2, _ = _age(ref, core, d + sd) if ref else (np.nan, np.nan)
    return {"chronology": authors or "", "reference": ref, "age_authors": a, "age_ref": r, "age_ref_unc_chrono": ru,
            "age_ref_unc_depth": abs(r2 - r)}


# --- All of it ----------------------------------------------------------------------------------------------------

def _layer(args) -> dict:
    core, xy, dataset, irh, layer, authors = args
    row = {"core": core, "dataset": dataset, "irh": irh, "age_published": layer.age.iloc[0] / 1000,
           "firn": layer.firn.iloc[0]}
    row.update(tie(layer, xy))
    row.update(ages(row, core, authors))
    return row


def core_ties(core: str, build: str = BUILD, out: str = OUTPUT, jobs: int = 3) -> pd.DataFrame:
    """The ties of every layer near a core (ties_<core>.csv); the points near it are cached (points_<core>.parquet)."""
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, f"points_{core.replace(' ', '_')}.parquet")
    points = pd.read_parquet(cache) if os.path.exists(cache) else points_near(core, build)
    if not os.path.exists(cache) and len(points):
        points.to_parquet(cache)
    if not len(points):
        return pd.DataFrame()
    c = ice_cores().loc[core]
    xy = (c.x, c.y)
    meta = dating()
    tasks = [(core, xy, dataset, irh, layer, meta.get((dataset, int(irh)), {}).get(core))
             for (dataset, irh), layer in points.groupby(["dataset", "irh"]) if len(layer) >= 3]
    with ProcessPoolExecutor(jobs) as ex:
        df = pd.DataFrame(list(ex.map(_layer, tasks)))
    df.to_csv(os.path.join(out, f"ties_{core.replace(' ', '_')}.csv"), index=False, float_format="%.6g")
    return df


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cores", nargs="+", default=["EDC", "Vostok", "Dome Fuji"])
    p.add_argument("--build", default=BUILD)
    p.add_argument("-o", "--out", default=OUTPUT)
    p.add_argument("-j", "--jobs", type=int, default=3)
    a = p.parse_args(argv)
    dfs = [core_ties(core, a.build, a.out, a.jobs) for core in a.cores]
    df = pd.concat([d for d in dfs if len(d)], ignore_index=True)
    df.to_csv(os.path.join(a.out, "ties.csv"), index=False, float_format="%.6g")
    cols = ["core", "dataset", "irh", "age_published", "lines", "nearest_km", "depth", "depth_sd", "lolo_rms",
            "chronology", "age_authors", "age_ref"]
    print(df[[c for c in cols if c in df]].to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
