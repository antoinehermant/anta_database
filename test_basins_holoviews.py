#!/usr/bin/env python
"""
Simple test script to plot IMBIE basins shapefile using HoloViews.
Run this in a Python REPL to test basin visualization.
"""

import os
import numpy as np
import holoviews as hv
from holoviews import opts
import panel as pn
import geopandas as gpd
from matplotlib.patches import Polygon
from importlib.resources import files

# Load panel extension
pn.extension()

# Path to shapefile
data_dir = files("anta_database.data")
shapefile_path = str(data_dir.joinpath("ANT_Basins_IMBIE2_v1.6.shp"))

print(f"Loading shapefile from: {shapefile_path}")
print(f"File exists: {os.path.exists(shapefile_path)}")

# Method 1: Using GeoPandas directly
print("\n=== Method 1: GeoPandas + HoloViews Polygons ===")
gdf = gpd.read_file(shapefile_path)
print(f"Loaded {len(gdf)} features")
print(f"Geometry types: {gdf.geometry.type.value_counts().to_dict()}")

# Extract coordinates from first few features
basin_vertices = []
for i, geom in enumerate(gdf.geometry[:10]):  # Test with first 10
    if geom is not None:
        if geom.geom_type == "Polygon":
            # Get exterior ring and scale to km
            coords = np.array(geom.exterior.coords) * 0.001
            # Ensure closed
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

# Create HoloViews Polygons
if basin_vertices:
    polygons = hv.Polygons(basin_vertices).opts(
        opts.Polygons(
            fill_color="white",
            fill_alpha=0,
            line_color="black",
            line_width=0.5,
        )
    )

    # Create a simple plot
    plot = (polygons).opts(
        opts.Polygons(
            width=900,
            height=700,
            xlabel="X [km]",
            ylabel="Y [km]",
            aspect="equal",
            tools=["pan", "wheel_zoom", "box_zoom", "reset"],
        )
    )

    print("\n=== Displaying plot with HoloViews ===")
    print(f"Plot type: {type(plot)}")
    print(f"Plot: {plot}")

    # Try to display
    from IPython.display import display

    display(plot)
else:
    print("No polygons created!")

# Method 2: Using HoloViews Path (as polylines instead of polygons)
print("\n=== Method 2: HoloViews Path (polygon boundaries as lines) ===")
hv_paths = []
for i, geom in enumerate(gdf.geometry[:10]):
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

if hv_paths:
    plot2 = (hv.Overlay(hv_paths)).opts(
        opts.Path(
            color="black",
            line_width=0.5,
            width=900,
            height=700,
            xlabel="X [km]",
            ylabel="Y [km]",
            aspect="equal",
            tools=["pan", "wheel_zoom", "box_zoom", "reset"],
        )
    )
    print(f"Created plot with {len(hv_paths)} paths")
    display(plot2)

print("\n=== Method 3: Raw GeoPandas plot ===")
# Try plotting the GeoDataFrame directly with HoloViews
gdf_scaled = gdf.copy()
gdf_scaled.geometry = gdf_scaled.geometry.scale(
    xfact=0.001, yfact=0.001, zfact=1.0, origin=(0, 0, 0)
)
print(f"Scaled GeoDataFrame CRS: {gdf_scaled.crs}")

# Convert to HoloViews
try:
    gdf_hv = hv.Path(gdf_scaled)
    plot3 = gdf_hv.opts(
        opts.Path(
            color="black",
            line_width=0.5,
            width=900,
            height=700,
            xlabel="X [km]",
            ylabel="Y [km]",
            aspect="equal",
        )
    )
    print("GeoPandas -> HoloViews Path conversion successful")
    display(plot3)
except Exception as e:
    print(f"Error converting GeoDataFrame to HoloViews: {e}")

print("\n=== Test Complete ===")
