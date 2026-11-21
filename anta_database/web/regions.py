"""
The regions of the viewer's filter (regions.json next to irh.json): the regions of the package
(anta_database.regions) that have lines, with the tracks of each. The ice sheets and IMBIE basins by
their polygons, the places (short names, then every place of the SCAR gazetteer) by their radius.

    {"source": "...", "empty": "regions_empty.json.gz", "outlines": "regions_outlines.json.gz", "regions": [
     {"key": "DC", "name": "Dome C", "kind": "place", "aliases": ["EDC", ...], "type": "Dome",
      "radius_km": 75, "box": [xmin, xmax, ymin, ymax] (m), "tracks": [3, 4, ...]},
     ...]}

The viewer draws the regions chosen on the ice surface: a place as the circle of its radius around
the centre of its box, an ice sheet or basin by its outline, in regions_outlines.json.gz (loaded
when one is chosen; the bucket does not compress regions.json): {"WSB": [[x0, y0, x1, y1, ...], ...]}
by short name, the outer rings of its parts (km, simplified, without the islands).

The regions without lines (the places of the gazetteer no line passes near yet), so that the
viewer can still find them and tell there is no line: regions_empty.json.gz, loaded on the first
search, {"fields": ["key", "name", "aliases", "type", "radius_km", "x", "y"], "places": [[...], ...]}
(aliases "|" between them, x and y in km), and the ice sheets and basins without lines, as in
regions.json.

Rewritten from an export (its tracks), without exporting again:

    python -m anta_database.web.regions web/public/data/v2/irh
"""

import argparse
import gzip
import json
import os
from importlib.resources import files
from typing import Dict, List, Optional, Sequence

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from anta_database import regions as R

EMPTY = "regions_empty.json.gz"
OUTLINES = "regions_outlines.json.gz"
OUTLINE_TOLERANCE = 2000.0  # m, of the simplified outlines of the ice sheets and basins
OUTLINE_MIN_AREA = 500e6  # m², the smaller parts (islands) not drawn


def _box(xy: np.ndarray) -> List[float]:
    return [round(float(v)) for v in (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())]


def _outline(geometry) -> List[List[float]]:
    """The outer rings of the parts of a polygon (km, x0, y0, x1, y1...), simplified, without the small parts."""
    geometry = geometry.simplify(OUTLINE_TOLERANCE)
    parts = getattr(geometry, "geoms", [geometry])
    return [[round(float(v) / 1000, 1) for xy in p.exterior.coords for v in xy] for p in parts if p.area >= OUTLINE_MIN_AREA]


def track_regions(track_xy: Dict[int, np.ndarray], empty: Optional[list] = None, outlines: Optional[dict] = None) -> List[dict]:
    """The regions with lines; track_xy: the vertices (m, (n, 2)) of each track, by its index.
    empty: a list to which the regions without lines are added (places as rows of EMPTY);
    outlines: a dict to which the outlines of the ice sheets and basins are added, by short name."""
    ids = np.concatenate([np.full(len(xy), t) for t, xy in track_xy.items()])
    xy = np.concatenate(list(track_xy.values())).astype(np.float64)
    out = []
    empty = [] if empty is None else empty
    outlines = {} if outlines is None else outlines

    # Ice sheets and basins: the tracks with a vertex in their polygons
    basins = gpd.read_parquet(files("anta_database.data").joinpath("ANT_Basins_IMBIE2_v1.6.parquet"))
    basins = basins[basins["Regions"] != "Islands"]
    points = gpd.GeoDataFrame({"track": ids}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=basins.crs)
    joined = gpd.sjoin(points, basins, predicate="within")
    named = R.named()
    for r in named.values():
        if r.kind == "place":
            continue
        column = "Regions" if r.kind == "ice sheet" else "Subregion"
        inside = joined[joined[column] == r.code]
        polygon = basins[basins[column] == r.code].geometry.union_all()
        region = {"key": r.key, "name": r.name, "kind": r.kind, "aliases": list(r.aliases)}
        outlines[r.key] = _outline(polygon)
        if len(inside):
            tracks = sorted(set(int(t) for t in inside.track))
            out.append({**region, "box": _box(xy[np.isin(ids, tracks)]), "tracks": tracks})
        else:
            box = polygon.bounds  # xmin, ymin, xmax, ymax
            empty.append({**region, "box": [round(box[0]), round(box[2]), round(box[1]), round(box[3])], "tracks": []})

    # Places: the tracks with a vertex within the radius
    tree = cKDTree(xy)

    def at(region: R.Region) -> List[int]:
        hits = tree.query_ball_point([region.x, region.y], region.radius_km * 1000)
        return sorted(set(int(t) for t in ids[hits]))

    def place(region: R.Region, key, aliases) -> dict:
        r = region.radius_km * 1000
        # Places without a feature type in the gazetteer: NaN there, not valid JSON
        kind = region.feature_type if isinstance(region.feature_type, str) else None
        return {"key": key, "name": region.name, "kind": "place", "aliases": aliases, "type": kind,
                "radius_km": region.radius_km, "box": [round(region.x - r), round(region.x + r), round(region.y - r), round(region.y + r)]}

    def row(region: R.Region, key, aliases) -> list:
        kind = region.feature_type if isinstance(region.feature_type, str) else None
        return [key, region.name, "|".join(aliases), kind, region.radius_km,
                round(region.x / 1000, 2), round(region.y / 1000, 2)]

    gaz = R.gazetteer()
    names_of = dict(zip(gaz.scar_id, gaz.names))
    with_key = set()
    for r in named.values():
        if r.kind == "place":
            tracks = at(r)
            other = [n for n in names_of.get(r.scar_id, "").split("|") if n and n != r.name]
            if tracks:
                out.append({**place(r, r.key, [*r.aliases, *other]), "tracks": tracks})
            else:
                empty.append(row(r, r.key, [*r.aliases, *other]))
            with_key.add(r.scar_id)
    for row_ in gaz.itertuples():
        if row_.scar_id in with_key:
            continue
        region = R._place(gaz.loc[row_.Index])
        tracks = at(region)
        names = [n for n in row_.names.split("|") if n != row_.name]
        if tracks:
            out.append({**place(region, None, names), "tracks": tracks})
        else:
            empty.append(row(region, None, names))
    return out


def _write_json_gz(text: str, path: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write(text)


def export_regions(track_xy: Dict[int, np.ndarray], path: str) -> int:
    """Write regions.json, and next to it the regions without lines (EMPTY) and the outlines (OUTLINES); the
    number of regions with lines."""
    empty: list = []
    outlines: dict = {}
    regions = track_regions(track_xy, empty, outlines)
    with open(path, "w") as f:
        json.dump({"source": R._config()["sources"], "empty": EMPTY, "outlines": OUTLINES, "regions": regions}, f,
                  separators=(",", ":"), allow_nan=False)
    places = [e for e in empty if isinstance(e, list)]
    others = [e for e in empty if isinstance(e, dict)]
    text = json.dumps({"fields": ["key", "name", "aliases", "type", "radius_km", "x", "y"], "places": places,
                       "regions": others}, separators=(",", ":"), allow_nan=False)
    _write_json_gz(text, os.path.join(os.path.dirname(path), EMPTY))
    _write_json_gz(json.dumps(outlines, separators=(",", ":")), os.path.join(os.path.dirname(path), OUTLINES))
    return len(regions)


def export_tracks(export_dir: str) -> Dict[int, np.ndarray]:
    """The vertices (m) of each track of an export of anta_database.web.irh, by its index."""
    with open(os.path.join(export_dir, "irh.json")) as f:
        index = json.load(f)
    track_xy: Dict[int, List[np.ndarray]] = {}
    for files_ in index["datasets"].values():
        with gzip.open(os.path.join(export_dir, files_["tracks"])) as f:
            vertices = np.frombuffer(f.read(), np.float32).reshape(-1, 5)  # x km, y km, surface, IRH_NUM, age
        with gzip.open(os.path.join(export_dir, files_["track_runs"])) as f:
            runs = np.frombuffer(f.read(), np.uint32).reshape(-1, 3)  # global track ids
        for track, start, count in runs:
            track_xy.setdefault(int(track), []).append(vertices[start : start + count, :2] * 1000.0)
    return {t: np.concatenate(v) for t, v in track_xy.items()}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Rewrite regions.json (and the regions without lines) of an export")
    parser.add_argument("export_dir", help="the directory of irh.json")
    args = parser.parse_args(argv)
    n = export_regions(export_tracks(args.export_dir), os.path.join(args.export_dir, "regions.json"))
    print(f"{n} regions with lines")


if __name__ == "__main__":
    main()
