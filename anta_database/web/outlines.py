"""
Export grounding lines and ice divides for the web viewer.

Grounding lines:
    - the RAISED reconstruction of the grounding line at 20, 15, 10 and 5 ka (Bentley et
      al., 2014), with the confidence of each line (measured, inferred or speculative)
      and its scenarios (A and/or B, which differ in the Weddell Sea sector);
    - the present grounding line, traced on the Bedmap3 mask (grounded ice and rock
      against floating ice and ocean) at 1 km.
Ice divides: the boundaries shared by two IMBIE2 drainage basins (Rignot et al.).

The lines are 2D (x, y): the viewer lays them on the bed, and builds the walls of the ice
divides from the bed to the ice surface, with the heights of the terrain it draws.

    python -m anta_database.web.outlines --raised RAISED_PaleoIceExtents.shp \\
        --bedmap bedmap3.nc -o web/public/data/outlines

Output:
    outlines.json                   features (metadata) of each layer, and the file names
    grounding_lines.bin.gz          float32 vertices (x km, y km), packed per run
    grounding_lines_runs.bin.gz     uint32 (feature id, first vertex, vertex count) per run
    divides.bin.gz, divides_runs.bin.gz   the same for the ice divides
"""

import argparse
import gzip
import json
import os
from importlib.resources import files
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

RAISED = {
    "citation": "RAISED Consortium, Bentley et al. (2014)",
    "doi": "https://doi.org/10.1016/j.quascirev.2014.06.025",
}
BEDMAP3_DOI = "https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2"
IMBIE = {
    "citation": "IMBIE2 basins (Rignot et al.), v1.6",
    "doi": "https://doi.org/10.1126/science.1208336",
}
SEGMENT = 1000.0  # m, longest segment: the lines follow the terrain between their vertices
MIN_LENGTH = 5000.0  # m, shorter present grounding lines (small islands, noise) are left out
GROUNDED = (1, 4)  # Bedmap3 mask: grounded ice and rock


def _lines(geometry) -> List[np.ndarray]:
    """Coordinates (n × 2, m) of the lines of a geometry, split at most every SEGMENT."""
    import shapely

    out = []
    for part in shapely.get_parts(shapely.segmentize(geometry, SEGMENT)):
        if part.geom_type in ("LineString", "LinearRing") and len(part.coords) >= 2:
            out.append(np.asarray(part.coords)[:, :2])
    return out


def raised_grounding_lines(path: str) -> List[Tuple[Dict, List[np.ndarray]]]:
    """Features and lines of the RAISED shapefile (EPSG:3031)."""
    import geopandas as gpd

    data = gpd.read_file(path).to_crs(3031)
    out = []
    for row in data.itertuples():
        scenarios = [s for s, column in (("A", "ScenarioA"), ("B", "ScenarioB")) if getattr(row, column) == "Y"]
        feature = {
            "source": "RAISED",
            "age_ka": int(str(row.Age).removesuffix("ka")),
            "confidence": str(row.Confidence),
            "scenarios": scenarios,
        }
        out.append((feature, _lines(row.geometry)))
    return out


def _grounded_fraction(path: str, factor: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Share of grounded cells of the Bedmap3 mask in blocks of factor × factor, and the
    block centres (m), read in bands of rows to keep the memory low."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        x = ds.x.values
        y = ds.y.values
        ny = len(y) // factor * factor
        nx = len(x) // factor * factor
        bands = []
        rows = 512 * factor
        for start in range(0, ny, rows):
            mask = ds.mask[start : min(start + rows, ny), :nx].values
            grounded = np.isin(mask, GROUNDED).astype(np.float32)
            bands.append(grounded.reshape(-1, factor, nx // factor, factor).mean(axis=(1, 3)))
    fraction = np.concatenate(bands)
    xc = x[:nx].reshape(-1, factor).mean(axis=1)
    yc = y[:ny].reshape(-1, factor).mean(axis=1)
    return fraction, xc, yc


def present_grounding_line(bedmap_path: str, factor: int = 2) -> List[Tuple[Dict, List[np.ndarray]]]:
    """The edge of grounded ice and rock on the Bedmap3 mask, at factor × 500 m."""
    import contourpy

    fraction, x, y = _grounded_fraction(bedmap_path, factor)
    lines = contourpy.contour_generator(x, y, fraction, line_type="Separate").lines(0.5)
    lines = [np.asarray(line) for line in lines]
    lines = [line for line in lines if np.hypot(*np.diff(line, axis=0).T).sum() >= MIN_LENGTH]
    feature = {"source": "Bedmap3", "age_ka": 0, "confidence": "Measured", "scenarios": ["A", "B"]}
    return [(feature, lines)]


def ice_divides(basins_path: Optional[str] = None) -> List[Tuple[Dict, List[np.ndarray]]]:
    """Boundaries shared by two IMBIE2 basins (the islands are not a drainage basin)."""
    import pandas as pd
    import shapely

    path = basins_path or str(files("anta_database.data").joinpath("ANT_Basins_IMBIE2_v1.6.parquet"))
    basins = pd.read_parquet(path)
    basins = basins[basins["Regions"] != "Islands"].reset_index(drop=True)
    geometries = shapely.from_wkb(basins["geometry"])
    out = []
    for i, j in combinations(range(len(basins)), 2):
        a, b = geometries[i], geometries[j]
        if not a.intersects(b):
            continue
        shared = shapely.line_merge(shapely.intersection(a.boundary, b.boundary))
        lines = [line for line in _lines(shared) if np.hypot(*np.diff(line, axis=0).T).sum() >= SEGMENT]
        if not lines:
            continue
        feature = {
            "basins": [basins["Subregion"][i], basins["Subregion"][j]],
            "regions": [basins["Regions"][i], basins["Regions"][j]],
        }
        out.append((feature, lines))
    return out


def _pack(items: List[Tuple[Dict, List[np.ndarray]]]) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Features, vertices (x km, y km) and runs (feature, first vertex, count)."""
    features, vertices, runs = [], [], []
    n = 0
    for feature, lines in items:
        if not lines:
            continue
        feature_id = len(features)
        features.append(feature)
        for line in lines:
            vertices.append((line / 1000).astype(np.float32))
            runs.append((feature_id, n, len(line)))
            n += len(line)
    return (
        features,
        np.concatenate(vertices) if vertices else np.zeros((0, 2), np.float32),
        np.asarray(runs, dtype=np.uint32).reshape(-1, 3),
    )


def _write_gz(array: np.ndarray, path: str) -> None:
    tmp = f"{path}.part"
    with gzip.open(tmp, "wb", compresslevel=9) as f:
        f.write(np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes())
    os.replace(tmp, path)


def export_outlines(output_dir: str, raised_path: str, bedmap_path: str, basins_path: Optional[str] = None) -> Dict:
    """Export the grounding lines and ice divides to output_dir. Returns the outlines.json content."""
    os.makedirs(output_dir, exist_ok=True)
    index = {"format": 1}
    layers = {
        "grounding_lines": (
            raised_grounding_lines(raised_path) + present_grounding_line(bedmap_path),
            {"RAISED": RAISED, "Bedmap3": {"citation": "Bedmap3 mask (Pritchard et al., 2025)", "doi": BEDMAP3_DOI}},
        ),
        "divides": (ice_divides(basins_path), {"IMBIE": IMBIE}),
    }
    for name, (items, sources) in layers.items():
        features, vertices, runs = _pack(items)
        _write_gz(vertices, os.path.join(output_dir, f"{name}.bin.gz"))
        _write_gz(runs, os.path.join(output_dir, f"{name}_runs.bin.gz"))
        index[name] = {
            "vertices": f"{name}.bin.gz",
            "runs": f"{name}_runs.bin.gz",
            "features": features,
            "sources": sources,
        }
        print(f"{name}: {len(features)} features, {len(runs)} lines, {len(vertices)} vertices")
    with open(os.path.join(output_dir, "outlines.json"), "w") as f:
        json.dump(index, f, indent=0)
    return index


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export grounding lines and ice divides for the web viewer.")
    parser.add_argument("--raised", required=True, help="RAISED_PaleoIceExtents.shp (Bentley et al., 2014)")
    parser.add_argument("--bedmap", required=True, help="Bedmap3 netCDF, for the present grounding line")
    parser.add_argument("--basins", help="IMBIE2 basins (parquet; default: the copy in the package)")
    parser.add_argument("-o", "--output", default="web/public/data/outlines", help="Output directory")
    args = parser.parse_args(argv)
    export_outlines(args.output, args.raised, args.bedmap, args.basins)


if __name__ == "__main__":
    main()
