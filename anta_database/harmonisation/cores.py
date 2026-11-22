"""
The positions of the ice cores, from the primary sources (drilling and logging reports, the data
records of the cores): data/ice_cores.csv, with the source, the precision (m) and a note for each.
The one table of the database: the regions of the queries, the harmonisation, and the 3D viewer and
the submission form (web/src/icecore-sites.ts, checked against it with `python -m
anta_database.harmonisation.cores`).
"""

import re
import sys
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import pandas as pd
from pyproj import Transformer

TABLE = "ice_cores.csv"


@lru_cache(maxsize=1)
def ice_cores() -> pd.DataFrame:
    """The cores by name (as in chronologies.json): site, lat, lon, elevation_m, elevation_reference,
    precision_m, source, doi, note, and x, y (m, EPSG:3031)."""
    cores = pd.read_csv(files("anta_database.data").joinpath(TABLE), keep_default_na=False, na_values={"lat": [""], "lon": [""]})
    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    cores["x"], cores["y"] = to_xy.transform(cores["lon"].to_numpy(), cores["lat"].to_numpy())
    return cores.set_index("name")


def check_viewer(path: Path) -> list:
    """The cores of icecore-sites.ts whose position or elevation differ from the table."""
    text = path.read_text(encoding="utf-8")
    cores = ice_cores()
    problems = []
    for name, lat, lon, elevation in re.findall(
        r'name: "([^"]+)",[^}]*?lat: (-?[\d.]+), lon: (-?[\d.]+), elevation: (-?[\d.]+)', text
    ):
        if name not in cores.index:
            problems.append(f"{name}: not in {TABLE}")
            continue
        core = cores.loc[name]
        if (float(lat), float(lon), float(elevation)) != (core.lat, core.lon, float(core.elevation_m)):
            problems.append(f"{name}: {lat}, {lon}, {elevation} m in the viewer, {core.lat}, {core.lon}, {core.elevation_m} m in {TABLE}")
    return problems


if __name__ == "__main__":
    viewer = Path(__file__).resolve().parents[2] / "web" / "src" / "icecore-sites.ts"
    problems = check_viewer(viewer)
    print("\n".join(problems) or f"{viewer.name}: the same positions as {TABLE}")
    sys.exit(1 if problems else 0)
