#!/usr/bin/env python
"""
Visual test script for IMBIE basins with HoloViews.
This script creates HTML files for visual inspection.
"""

import numpy as np
import holoviews as hv
from holoviews import opts
import panel as pn
import geopandas as gpd
from importlib.resources import files
import os

# Load panel extension
pn.extension()

# Path to shapefile
data_dir = files("anta_database.data")
shapefile_path = str(data_dir.joinpath("ANT_Basins_IMBIE2_v1.6.shp"))

# Load with GeoPandas
print(f"Loading shapefile from: {shapefile_path}")
gdf = gpd.read_file(shapefile_path)
print(f"Loaded {len(gdf)} features")
print(f"CRS: {gdf.crs}")
print(f"Geometry types: {gdf.geometry.type.value_counts().to_dict()}")

# Get coordinate ranges
all_coords = []
for geom in gdf.geometry:
    if geom is not None:
        if geom.geom_type == "Polygon":
            all_coords.extend(geom.exterior.coords)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                all_coords.extend(poly.exterior.coords)

coords_array = np.array(all_coords)
x_min, x_max = coords_array[:, 0].min() * 0.001, coords_array[:, 0].max() * 0.001
y_min, y_max = coords_array[:, 1].min() * 0.001, coords_array[:, 1].max() * 0.001

print(f"\nCoordinate ranges (km):")
print(f"  X: {x_min:.0f} to {x_max:.0f}")
print(f"  Y: {y_min:.0f} to {y_max:.0f}")

# Create output directory
output_dir = "/tmp/basins_test"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# TEST 1: Using hv.Path (simplest - basins as lines)
# ============================================================================
print("\n=== TEST 1: Basins as Path (lines) ===")
hv_paths = []
for geom in gdf.geometry:
    if geom is not None:
        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords) * 0.001
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])
            hv_paths.append(hv.Path(coords))
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = np.array(poly.exterior.coords) * 0.001
                if not np.allclose(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[0]])
                hv_paths.append(hv.Path(coords))

# Apply style to each path
styled_paths = [p.opts(opts.Path(color="black", line_width=0.5)) for p in hv_paths]

plot1 = hv.Overlay(styled_paths).opts(
    xlabel="X [km]",
    ylabel="Y [km]",
    aspect="equal",
    width=900,
    height=700,
    tools=["pan", "wheel_zoom", "box_zoom", "reset"],
    xlim=(x_min - 100, x_max + 100),
    ylim=(y_min - 100, y_max + 100),
)

hv.save(plot1, f"{output_dir}/basins_path.html")
print(f"Saved to: {output_dir}/basins_path.html")

# ============================================================================
# TEST 2: Using hv.Polygons (basins as outlined polygons)
# ============================================================================
print("\n=== TEST 2: Basins as Polygons (outlines) ===")
basin_vertices = []
for geom in gdf.geometry:
    if geom is not None:
        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords) * 0.001
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])
            basin_vertices.append(coords)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = np.array(poly.exterior.coords) * 0.001
                if not np.allclose(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[0]])
                basin_vertices.append(coords)

plot2 = (
    hv.Polygons(basin_vertices).opts(
        opts.Polygons(
            fill_color="white",
            fill_alpha=0,
            line_color="black",
            line_width=0.5,
        )
    )
).opts(
    xlabel="X [km]",
    ylabel="Y [km]",
    aspect="equal",
    width=900,
    height=700,
    tools=["pan", "wheel_zoom", "box_zoom", "reset"],
    xlim=(x_min - 100, x_max + 100),
    ylim=(y_min - 100, y_max + 100),
)

hv.save(plot2, f"{output_dir}/basins_polygons.html")
print(f"Saved to: {output_dir}/basins_polygons.html")

print("\n=== TEST COMPLETE ===")
print(f"All HTML files saved to: {output_dir}")
print("\nOpen the following files in a browser to inspect:")
print(f"  1. {output_dir}/basins_path.html (basins as lines)")
print(f"  2. {output_dir}/basins_polygons.html (basins as outlined polygons)")
print("\nIf the basins are visible in these HTML files, then the approach works.")
print("If not, we need to investigate further.")
