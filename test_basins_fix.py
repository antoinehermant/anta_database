#!/usr/bin/env python3
"""
Test script to verify that IMBIE basins are being plotted correctly.
"""

import sys
import os

sys.path.insert(0, ".")

from anta_database.interactive_plotting.interactive_plotter import AntarcticBackground
import numpy as np
import holoviews as hv
from holoviews import opts


def test_basins_loading():
    """Test that basins are loaded correctly."""
    print("Testing basin loading...")

    bg = AntarcticBackground()
    basins = bg.get_basins_polygons()

    print(f"✓ Loaded {len(basins)} basin polygons")

    if len(basins) == 0:
        print("✗ ERROR: No basins loaded!")
        return False

    return True


def test_basins_scaling():
    """Test that basin coordinates are properly scaled to kilometers."""
    print("Testing coordinate scaling...")

    bg = AntarcticBackground()
    basins = bg.get_basins_polygons()

    # Check a few polygons
    for i, poly in enumerate(basins[:5]):
        exterior = poly.get_xy()

        # Coordinates should be in kilometers, so typical Antarctic coordinates
        # should be in the range of thousands of km, not millions of meters
        x_range = exterior[:, 0].max() - exterior[:, 0].min()
        y_range = exterior[:, 1].max() - exterior[:, 1].min()

        print(f"  Basin {i}: x_range={x_range:.2f} km, y_range={y_range:.2f} km")

        # If coordinates were still in meters, the ranges would be 1000x larger
        if (
            x_range > 10000 or y_range > 10000
        ):  # 10,000 km is way too big for Antarctic basins
            print(f"✗ ERROR: Basin {i} coordinates appear to still be in meters!")
            return False

    print("✓ Coordinates are properly scaled to kilometers")
    return True


def test_basins_closure():
    """Test that polygons are properly closed."""
    print("Testing polygon closure...")

    bg = AntarcticBackground()
    basins = bg.get_basins_polygons()

    closed_count = 0
    for i, poly in enumerate(basins[:10]):  # Test first 10
        exterior = poly.get_xy()
        first_point = exterior[0]
        last_point = exterior[-1]

        if np.allclose(first_point, last_point):
            closed_count += 1
        else:
            print(f"✗ ERROR: Basin {i} is not properly closed!")
            print(f"  First: {first_point}, Last: {last_point}")
            return False

    print(f"✓ All {closed_count} tested polygons are properly closed")
    return True


def test_basins_plotting():
    """Test that basins can be converted to HoloViews polygons."""
    print("Testing HoloViews polygon creation...")

    bg = AntarcticBackground()
    basins = bg.get_basins_polygons()

    # Try to create HoloViews polygons (this is what the actual plotting code does)
    basin_vertices = []
    for poly in basins[:10]:  # Test first 10
        exterior = poly.get_xy()
        # Ensure polygon is closed
        if not np.allclose(exterior[0], exterior[-1]):
            exterior = np.vstack([exterior, exterior[0]])
        basin_vertices.append(exterior)

    if len(basin_vertices) == 0:
        print("✗ ERROR: No basin vertices created!")
        return False

    # Try to create the HoloViews element
    try:
        basins_poly = hv.Polygons(basin_vertices).opts(
            opts.Polygons(
                fill_color="white",
                fill_alpha=0,
                line_color="black",
                line_width=0.5,
            )
        )
        print("✓ Successfully created HoloViews Polygons element")
        return True
    except Exception as e:
        print(f"✗ ERROR: Failed to create HoloViews Polygons: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing IMBIE basins plotting fix...\n")

    tests = [
        test_basins_loading,
        test_basins_scaling,
        test_basins_closure,
        test_basins_plotting,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ ERROR in {test.__name__}: {e}\n")

    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! IMBIE basins should now plot correctly.")
        return True
    else:
        print("❌ Some tests failed. Basins may not plot correctly.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
