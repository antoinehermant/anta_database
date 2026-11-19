"""
The regions of the queries (db.query(region=...)), by any of their names, without case or accents:
- the ice sheets: EAST, WEST, AP (or EAIS, WAIS, APIS), the IMBIE regions;
- the IMBIE basins, by a short name (DML, WL, ASE...) or their code (A-Ap, Cp-D, G-H...);
- every place of the SCAR Composite Gazetteer of Antarctica (about 20,000: Dome C, Wilkes Subglacial
  Basin, Recovery Glacier...), and short names for some (DC, WSB, EDML...). A line is at a place when it
  passes within a radius of its point: by the feature type of the place (75 km for a dome, 300 km for a
  subglacial basin...), or the radius of the query.

The ice sheets and basins of a line are in the index (assigned by the compilation, attribute
IMBIE_basins); the places are found from its track, thinned to TRACK_STEP and kept in the index (table
line_tracks).

The list is anta_database/data/regions.json; the gazetteer, one row per place, is
data/scar_gazetteer.parquet, built from the CSV of the Australian Antarctic Data Centre with:

    python -m anta_database.regions SCAR_CGA_PLACE_NAMES.csv
"""

import argparse
import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

GAZETTEER = "scar_gazetteer.parquet"
GAZETTEER_URL = (
    "https://data.aad.gov.au/geoserver/ows?service=wfs&version=2.0.0&request=GetFeature"
    "&typeName=aadc:SCAR_CGA_PLACE_NAMES&outputFormat=csv"
)
TRACK_STEP = 1000.0  # m, between the points of a thinned track
KINDS = ("ice sheet", "basin", "place")
# The gazetteer whose name of a place is taken first, when they differ
NAME_ORDER = ("GBR", "USA", "AUS", "NZL")


@dataclass(frozen=True)
class Region:
    key: str  # the short name, or the SCAR place ID ("SCAR 2550")
    name: str
    kind: str  # one of KINDS
    code: Optional[str] = None  # of an ice sheet or basin in the index: "EAIS", "Cp-D"
    x: Optional[float] = None  # of a place, m (EPSG:3031)
    y: Optional[float] = None
    radius_km: Optional[float] = None
    feature_type: Optional[str] = None
    scar_id: Optional[int] = None
    aliases: tuple = field(default=(), compare=False)


def normalise(name: str) -> str:
    """For matching: no case, no accents, single spaces."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return " ".join(text.casefold().replace("_", " ").split())


@lru_cache(maxsize=1)
def _config() -> dict:
    return json.loads(files("anta_database.data").joinpath("regions.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def gazetteer() -> pd.DataFrame:
    """The places of the SCAR Composite Gazetteer: scar_id, name, names (all, "|"), feature_type,
    lat, lon, x, y (m, EPSG:3031); feature_type None where the gazetteer gives none."""
    gaz = pd.read_parquet(files("anta_database.data").joinpath(GAZETTEER))
    gaz["feature_type"] = gaz["feature_type"].astype(object).where(gaz["feature_type"].notna(), None)
    return gaz


def place_radius_km(feature_type: Optional[str]) -> float:
    radii = _config()["place radius km"]
    return float(radii.get(feature_type or "", radii["default"]))


@lru_cache(maxsize=1)
def named() -> Dict[str, Region]:
    """The regions with a short name: ice sheets, basins and places, by short name."""
    cfg = _config()
    out = {}
    for key, r in cfg["ice sheets"].items():
        out[key] = Region(key, r["name"], "ice sheet", code=r["code"], aliases=tuple(r.get("aliases", ())))
    for key, r in cfg["basins"].items():
        out[key] = Region(key, r["name"], "basin", code=r["code"], aliases=(r["code"], *r.get("aliases", ())))
    gaz = gazetteer().set_index("scar_id")
    cores = pd.read_csv(files("anta_database.data").joinpath("ice-core-coordinates.csv")).set_index("site")
    for key, r in cfg["places"].items():
        if "scar" in r:
            p = gaz.loc[r["scar"]]
            name, x, y, ftype, scar_id = p["name"], p.x, p.y, p.feature_type, int(r["scar"])
        else:
            name, ftype, scar_id = r["name"], "Ice core", None
        if "ice core" in r:  # the site, more precise than the gazetteer
            x, y = cores.loc[r["ice core"], ["x", "y"]]
        radius = r.get("radius km", place_radius_km(ftype))
        out[key] = Region(key, r.get("name", name), "place", x=float(x), y=float(y), radius_km=float(radius),
                          feature_type=ftype, scar_id=scar_id, aliases=tuple(r.get("aliases", ())))
    return out


def _place(row, radius_km: Optional[float] = None) -> Region:
    return Region(f"SCAR {int(row.scar_id)}", row["name"], "place", x=float(row.x), y=float(row.y),
                  radius_km=float(radius_km if radius_km is not None else place_radius_km(row.feature_type)),
                  feature_type=row.feature_type, scar_id=int(row.scar_id))


@lru_cache(maxsize=1)
def _lookup() -> Dict[str, List]:
    """Normalised name -> the short names and SCAR IDs it can be."""
    table: Dict[str, List] = {}
    for key, r in named().items():
        for n in (key, r.name, *r.aliases):
            table.setdefault(normalise(n), []).append(key)
    for scar_id, names in zip(gazetteer().scar_id, gazetteer().names):
        for n in names.split("|"):
            entries = table.setdefault(normalise(n), [])
            if int(scar_id) not in entries:
                entries.append(int(scar_id))
    return table


def resolve(name, radius_km: Optional[float] = None) -> Region:
    """The region of a name: a short name, a name, an alias, an IMBIE code, a SCAR place name or ID
    (2550 or "SCAR 2550"). radius_km: for a place, instead of the radius of its feature type."""
    text = str(name).strip()
    scar_id = text[5:].strip() if normalise(text).startswith("scar ") else text
    if scar_id.isdigit():
        rows = gazetteer()[gazetteer().scar_id == int(scar_id)]
        if len(rows):
            return _place(rows.iloc[0], radius_km)
    found = _lookup().get(normalise(text))
    if not found:  # the same name without its hyphens or spaces: "JJpp" for "J-Jpp", "Dome-C"
        bare = re.sub(r"[^0-9a-z]", "", normalise(text))
        found = next((v for k, v in _lookup().items() if bare and re.sub(r"[^0-9a-z]", "", k) == bare), None)
    if not found:
        close = difflib.get_close_matches(normalise(text), list(_lookup()), n=5, cutoff=0.75)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ValueError(
            f"Unknown region {name!r}.{hint} The named regions are in db.regions(), and the places of the SCAR "
            "gazetteer in db.places('<part of a name>')."
        )
    # A place of the gazetteer that has a short name is that region
    by_scar = {r.scar_id: key for key, r in named().items() if r.scar_id is not None}
    found = list(dict.fromkeys(by_scar.get(k, k) for k in found))
    keys = [k for k in found if isinstance(k, str)]
    if keys:  # a short name, or a name of one, comes first
        region = named()[keys[0]]
        if radius_km is not None and region.kind == "place":
            region = Region(**{**region.__dict__, "radius_km": float(radius_km)})
        return region
    if len(found) > 1:
        rows = gazetteer().set_index("scar_id").loc[found]
        listed = "; ".join(f"SCAR {i}: {r['name']} ({r.feature_type}, {r.lat:.2f}, {r.lon:.2f})" for i, r in rows.iterrows())
        raise ValueError(f"{len(found)} places are named {name!r}, give the one by its ID, e.g. 'SCAR {found[0]}': {listed}")
    return _place(gazetteer()[gazetteer().scar_id == found[0]].iloc[0], radius_km)


def resolve_all(names, radius_km: Optional[float] = None) -> List[Region]:
    names = [names] if isinstance(names, (str, int)) else list(names)
    return [resolve(n, radius_km) for n in names]


def regions_table(kind: Optional[str] = None) -> pd.DataFrame:
    """The regions with a short name (db.regions())."""
    rows = [{"short name": r.key, "name": r.name, "kind": r.kind, "aliases": ", ".join(r.aliases),
             "IMBIE code": r.code, "radius km": r.radius_km, "SCAR ID": r.scar_id} for r in named().values()]
    df = pd.DataFrame(rows)
    if kind is not None:
        if kind not in KINDS:
            raise ValueError(f"kind is one of {KINDS}")
        df = df[df.kind == kind]
    return df.dropna(axis=1, how="all").reset_index(drop=True)


def places_table(search: Optional[str] = None, feature_type: Optional[str] = None) -> pd.DataFrame:
    """The places of the SCAR gazetteer whose names contain search (db.places())."""
    df = gazetteer()
    if search:
        key = normalise(search)
        df = df[df.names.map(lambda names: any(key in normalise(n) for n in names.split("|")))]
    if feature_type:
        df = df[df.feature_type.fillna("").str.casefold() == feature_type.casefold()]
    out = df.assign(radius_km=df.feature_type.map(place_radius_km), names=df.names.str.replace("|", ", "))
    return out[["scar_id", "name", "feature_type", "lat", "lon", "radius_km", "names"]].reset_index(drop=True)


def thin_track(x: np.ndarray, y: np.ndarray, step: float = TRACK_STEP) -> np.ndarray:
    """The points of a line about step apart along it (and its first and last), float32 (n, 2): enough
    to find the places it passes."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, np.float64)[ok], np.asarray(y, np.float64)[ok]
    if not len(x):
        return np.zeros((0, 2), np.float32)
    along = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    keep = np.r_[True, np.diff(np.floor(along / step)) > 0]
    keep[-1] = True
    return np.c_[x[keep], y[keep]].astype(np.float32)


def lines_at(region: Region, tracks: Dict[int, np.ndarray]) -> List[int]:
    """The lines (keys of tracks, thinned) that pass within the radius of a place."""
    radius = region.radius_km * 1000 + TRACK_STEP / 2  # a point of the line between two kept
    r2 = radius * radius
    out = []
    for key, xy in tracks.items():
        if len(xy) and (((xy[:, 0] - region.x) ** 2 + (xy[:, 1] - region.y) ** 2) <= r2).any():
            out.append(key)
    return out


def build_gazetteer(csv_path: str, output: Optional[str] = None) -> pd.DataFrame:
    """One row per place of the SCAR Composite Gazetteer (CSV of GAZETTEER_URL), south of 60°S: the
    name of the first gazetteer of NAME_ORDER that has it, all its names (one per gazetteer), the
    feature type most given (the longest of those given as often: "Subglacial mountains" rather than
    "Mountain"), and the median of the positions given."""
    from pyproj import Transformer

    df = pd.read_csv(csv_path, low_memory=False)
    df = df[(df.latitude <= -60) & ~df.is_deleted.astype(str).str.upper().eq("Y") & df.scar_common_id.notna()]
    # Not places: the continent, territories (at the pole in the gazetteer)
    continent = df.place_name_mapping.map(normalise).isin(["antarctica", "the antarctic"])
    df = df[~df.feature_type_name.isin(["Continent", "Territory"]) & ~df.scar_common_id.isin(df.scar_common_id[continent])]
    rank = {g: i for i, g in enumerate(NAME_ORDER)}
    df = df.assign(rank=df.gazetteer.map(rank).fillna(len(rank)))
    rows = []
    for scar_id, g in df.groupby("scar_common_id"):
        g = g.sort_values("rank")
        # The names as written on maps; place_name_gazetteer has them inverted for sorting ("C, Dome")
        names = list(dict.fromkeys(g.place_name_mapping.dropna())) or list(g.place_name_gazetteer.dropna())
        ftype = g.feature_type_name.dropna()
        rows.append({"scar_id": int(scar_id), "name": names[0],
                     "names": "|".join(names), "feature_type": max(ftype.mode(), key=len) if len(ftype) else None,
                     "lat": float(g.latitude.median()), "lon": float(g.longitude.median())})
    out = pd.DataFrame(rows)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True).transform(out.lon.values, out.lat.values)
    out["x"], out["y"] = np.round(x, 1), np.round(y, 1)
    out.to_parquet(output or str(files("anta_database.data").joinpath(GAZETTEER)), index=False)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the SCAR gazetteer of the package from its CSV.")
    parser.add_argument("csv", help=f"SCAR_CGA_PLACE_NAMES as CSV ({GAZETTEER_URL})")
    parser.add_argument("-o", "--output", help="Default: in the package")
    args = parser.parse_args(argv)
    out = build_gazetteer(args.csv, args.output)
    print(f"{len(out)} places, {out.names.str.count('[|]').add(1).sum()} names")


if __name__ == "__main__":
    main()
