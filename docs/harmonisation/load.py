"""Lines of a compiled database (the .nc files of build_v2.py), for the harmonisation scripts.

    from load import load_lines
    lines = load_lines("~/data/isochrones/AntADatabase-v2-build")

Each line: dataset, file, line_id, x, y (m, EPSG:3031), depth (points x IRHs, m), irh, age, unc (years),
thk, surf (m), and the attributes of the file (attrs). Only the points with at least one IRH and a position.
"""
import glob
import os
import re

import numpy as np
import xarray as xr

DEFAULT = os.path.expanduser("~/data/isochrones/AntADatabase-v2-build")


def load_lines(root: str = DEFAULT) -> list:
    root = os.path.expanduser(root)
    lines = []
    for path in sorted(glob.glob(os.path.join(root, "*", "nc", "*.nc"))):
        with xr.open_dataset(path) as ds:
            if "IRH_DEPTH" not in ds:
                continue
            depth = ds.IRH_DEPTH.values.astype(float)
            x, y = ds.PSX.values.astype(float), ds.PSY.values.astype(float)
            keep = np.isfinite(depth).any(axis=1) & np.isfinite(x) & np.isfinite(y)
            if keep.sum() < 2:
                continue
            get = lambda k: ds[k].values.astype(float)[keep] if k in ds else np.full(keep.sum(), np.nan)
            lines.append(dict(
                dataset=ds.attrs["dataset"], file=os.path.basename(path),
                line_id=str(ds.attrs.get("line ID", os.path.splitext(os.path.basename(path))[0])),
                x=x[keep], y=y[keep], depth=depth[keep], irh=ds.IRH.values.astype(int),
                age=ds.IRH_AGE.values.astype(float), unc=ds.IRH_AGE_UNC.values.astype(float) if "IRH_AGE_UNC" in ds else
                np.full(ds.sizes["IRH"], np.nan), thk=get("ICE_THK"), surf=get("SURF_ELEV"), attrs=dict(ds.attrs)))
    return lines


def firn_correction(attrs: dict):
    """Firn correction in the published depths (m) from the attribute 'firn correction' of a file: 0 for
    'none', None when not known; the first number otherwise (some datasets give one per site)."""
    text = str(attrs.get("firn correction", "")).strip().lower()
    if text.startswith("none"):
        return 0.0
    number = re.match(r"[\d.]+", text)
    return float(number.group()) if number else None
