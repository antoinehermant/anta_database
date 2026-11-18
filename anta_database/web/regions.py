"""
The regions of the viewer's filter (regions.json next to irh.json): the regions of the package
(anta_database.regions) that have lines, with the tracks of each. The ice sheets and IMBIE basins by
their polygons, the places (short names, then every place of the SCAR gazetteer) by their radius.

    {"source": "...", "regions": [
     {"key": "DC", "name": "Dome C", "kind": "place", "aliases": ["EDC", ...], "type": "Dome",
      "radius_km": 75, "box": [xmin, xmax, ymin, ymax] (m), "tracks": [3, 4, ...]},
     ...]}
"""

import json
from importlib.resources import files
from typing import Dict, List

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from anta_database import regions as R


def _box(xy: np.ndarray) -> List[float]:
    return [round(float(v)) for v in (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())]


def track_regions(track_xy: Dict[int, np.ndarray]) -> List[dict]:
    """The regions with lines; track_xy: the vertices (m, (n, 2)) of each track, by its index."""
    ids = np.concatenate([np.full(len(xy), t) for t, xy in track_xy.items()])
    xy = np.concatenate(list(track_xy.values())).astype(np.float64)
    out = []

    # Ice sheets and basins: the tracks with a vertex in their polygons
    basins = gpd.read_parquet(files("anta_database.data").joinpath("ANT_Basins_IMBIE2_v1.6.parquet"))
    points = gpd.GeoDataFrame({"track": ids}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=basins.crs)
    joined = gpd.sjoin(points, basins, predicate="within")
    named = R.named()
    for r in named.values():
        if r.kind == "place":
            continue
        column = "Regions" if r.kind == "ice sheet" else "Subregion"
        inside = joined[joined[column] == r.code]
        if len(inside):
            tracks = sorted(set(int(t) for t in inside.track))
            out.append({"key": r.key, "name": r.name, "kind": r.kind, "aliases": list(r.aliases),
                        "box": _box(xy[np.isin(ids, tracks)]), "tracks": tracks})

    # Places: the tracks with a vertex within the radius
    tree = cKDTree(xy)

    def at(region: R.Region) -> List[int]:
        hits = tree.query_ball_point([region.x, region.y], region.radius_km * 1000)
        return sorted(set(int(t) for t in ids[hits]))

    def place(region: R.Region, key, aliases) -> dict:
        r = region.radius_km * 1000
        return {"key": key, "name": region.name, "kind": "place", "aliases": aliases, "type": region.feature_type,
                "radius_km": region.radius_km, "box": [round(region.x - r), round(region.x + r), round(region.y - r), round(region.y + r)]}

    gaz = R.gazetteer()
    names_of = dict(zip(gaz.scar_id, gaz.names))
    with_key = set()
    for r in named.values():
        if r.kind == "place":
            tracks = at(r)
            other = [n for n in names_of.get(r.scar_id, "").split("|") if n and n != r.name]
            if tracks:
                out.append({**place(r, r.key, [*r.aliases, *other]), "tracks": tracks})
            with_key.add(r.scar_id)
    for row in gaz.itertuples():
        if row.scar_id in with_key:
            continue
        region = R._place(gaz.loc[row.Index])
        tracks = at(region)
        if tracks:
            names = row.names.split("|")
            out.append({**place(region, None, [n for n in names if n != row.name]), "tracks": tracks})
    return out


def export_regions(track_xy: Dict[int, np.ndarray], path: str) -> int:
    """Write regions.json; the number of regions."""
    regions = track_regions(track_xy)
    with open(path, "w") as f:
        json.dump({"source": R._config()["sources"], "regions": regions}, f, separators=(",", ":"))
    return len(regions)
