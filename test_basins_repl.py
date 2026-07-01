#!/usr/bin/env python
"""
Simple REPL-friendly script to test IMBIE basins plotting with HoloViews.

To use: Copy and paste the code blocks below into a Python REPL to test.
"""

# ============================================================================
# BLOCK 1: Setup (run this first)
# ============================================================================
import numpy as np
import holoviews as hv
from holoviews import opts
import panel as pn
import geopandas as gpd
from importlib.resources import files

pn.extension()

# Load the shapefile
data_dir = files("anta_database.data")
shapefile_path = str(data_dir.joinpath("ANT_Basins_IMBIE2_v1.6.shp"))
gdf = gpd.read_file(shapefile_path)

print(f"Loaded {len(gdf)} features")
print(f"CRS: {gdf.crs}")
print(f"Geometry types: {gdf.geometry.type.value_counts().to_dict()}")


# ============================================================================
# BLOCK 2: Test with HoloViews Path (simplest approach - lines only)
# ============================================================================

# Plot basins as Path objects (lines)
hv_paths = []
for geom in gdf.geometry:
    if geom is not None:
        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords) * 0.001  # Scale to km
            if not np.allclose(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])
            hv_paths.append(hv.Path(coords))
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = np.array(poly.exterior.coords) * 0.001
                if not np.allclose(coords[0], coords[-1]):
                    coords = np.vstack([coords, coords[0]])
                hv_paths.append(hv.Path(coords))

print(f"\nCreated {len(hv_paths)} path objects")

# Create the plot
plot = hv.Overlay(hv_paths).opts(
    opts.Path(
        color="black",
        line_width=0.5,
        xlabel="X [km]",
        ylabel="Y [km]",
        aspect="equal",
        width=900,
        height=700,
        tools=["pan", "wheel_zoom", "box_zoom", "reset"],
        xlim=(-2500, 2700),
        ylim=(-2200, 2200),
    )
)

# Display the plot
print("Displaying basins as Path (lines)...")
plot


# ============================================================================
# BLOCK 3: Test with HoloViews Polygons (with outline, no fill)
# ============================================================================

# Plot basins as Polygons (with outline only)
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

print(f"\nCreated {len(basin_vertices)} polygon vertex arrays")

# Create the plot with Polygons
plot_polygons = hv.Polygons(basin_vertices).opts(
    opts.Polygons(
        fill_color="white",
        fill_alpha=0,  # Transparent fill
        line_color="black",
        line_width=0.5,
        xlabel="X [km]",
        ylabel="Y [km]",
        aspect="equal",
        width=900,
        height=700,
        tools=["pan", "wheel_zoom", "box_zoom", "reset"],
        xlim=(-2500, 2700),
        ylim=(-2200, 2200),
    )
)

# Display the plot
print("Displaying basins as Polygons (outlines)...")
plot_polygons


# ============================================================================
# BLOCK 4: Check coordinate ranges
# ============================================================================

# Inspect coordinate ranges
all_coords = []
for geom in gdf.geometry:
    if geom is not None:
        if geom.geom_type == "Polygon":
            all_coords.extend(geom.exterior.coords)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                all_coords.extend(poly.exterior.coords)

coords_array = np.array(all_coords)
print(f"\nCoordinate ranges (meters):")
print(f"  X: {coords_array[:, 0].min():.0f} to {coords_array[:, 0].max():.0f}")
print(f"  Y: {coords_array[:, 1].min():.0f} to {coords_array[:, 1].max():.0f}")
print(f"\nCoordinate ranges (km):")
print(
    f"  X: {coords_array[:, 0].min()*0.001:.0f} to {coords_array[:, 0].max()*0.001:.0f}"
)
print(
    f"  Y: {coords_array[:, 1].min()*0.001:.0f} to {coords_array[:, 1].max()*0.001:.0f}"
)
