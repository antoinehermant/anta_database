#!/usr/bin/env python
"""
Simple test to plot IMBIE basins and save to HTML.
"""

import os
import numpy as np
import holoviews as hv
from holoviews import opts
import panel as pn
import geopandas as gpd
from importlib.resources import files

# Load panel extension
pn.extension()

# Path to shapefile
data_dir = files("anta_database.data")
shapefile_path = str(data_dir.joinpath("ANT_Basins_IMBIE2_v1.6.shp"))

# Load with GeoPandas
gdf = gpd.read_file(shapefile_path)
print(f"Loaded {len(gdf)} features from {shapefile_path}")
print(f"CRS: {gdf.crs}")

# Test 1: Plot just the first polygon as a simple test
print("\n=== Test 1: Single Polygon ===")
first_geom = gdf.geometry[0]
print(f"First geometry type: {first_geom.geom_type}")

if first_geom.geom_type == "MultiPolygon":
    # Get the first part of the multipolygon
    first_poly = first_geom.geoms[0]
    coords = np.array(first_poly.exterior.coords) * 0.001  # Scale to km
    print(f"First sub-polygon has {len(coords)} vertices")
    print(f"First coordinate: {coords[0]}")

    # Create HoloViews Path (simpler than Polygons)
    path = hv.Path(coords).opts(
        opts.Path(
            color="red",
            line_width=2,
            xlabel="X [km]",
            ylabel="Y [km]",
            aspect="equal",
            width=900,
            height=700,
        )
    )

    # Save to HTML
    output_file = "/tmp/test_single_polygon.html"
    hv.save(path, output_file)
    print(f"Saved single polygon to: {output_file}")

# Test 2: Plot all basin boundaries as Path objects
print("\n=== Test 2: All Basin Boundaries as Paths ===")
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

print(f"Created {len(hv_paths)} path objects")

# Overlay all paths
all_paths = hv.Overlay(hv_paths).opts(
    opts.Path(
        color="black",
        line_width=0.5,
        xlabel="X [km]",
        ylabel="Y [km]",
        aspect="equal",
        width=900,
        height=700,
        tools=["pan", "wheel_zoom", "box_zoom", "reset"],
    )
)

output_file = "/tmp/test_all_basins_paths.html"
hv.save(all_paths, output_file)
print(f"Saved all basins as paths to: {output_file}")

# Test 3: Using HoloViews Polygons
print("\n=== Test 3: All Basins as HoloViews Polygons ===")
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

print(f"Created {len(basin_vertices)} polygon vertex arrays")

polygons = hv.Polygons(basin_vertices).opts(
    opts.Polygons(
        fill_color="white",
        fill_alpha=0,
        line_color="black",
        line_width=0.5,
        xlabel="X [km]",
        ylabel="Y [km]",
        aspect="equal",
        width=900,
        height=700,
        tools=["pan", "wheel_zoom", "box_zoom", "reset"],
    )
)

output_file = "/tmp/test_all_basins_polygons.html"
hv.save(polygons, output_file)
print(f"Saved all basins as polygons to: {output_file}")

print("\n=== All tests complete ===")
print(f"HTML files saved to /tmp/ directory")
print("Open them in a browser to inspect the plots")
