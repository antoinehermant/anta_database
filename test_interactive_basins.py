#!/usr/bin/env python3
"""
Test script to verify that the interactive plotter can display IMBIE basins correctly.
"""

import sys
import os

sys.path.insert(0, ".")

from anta_database.interactive_plotting.interactive_plotter import (
    InteractivePlotter,
    AntarcticBackground,
)
import numpy as np


def test_interactive_plotter_basins():
    """Test that the interactive plotter can create plots with basins."""
    print("Testing InteractivePlotter basin display...")

    # Create a mock database instance (we don't need real data for this test)
    class MockDatabase:
        def __init__(self):
            self._db_dir = "."

    mock_db = MockDatabase()

    try:
        # Create the interactive plotter
        plotter = InteractivePlotter(database_instance=mock_db)

        # Test that the initial plot with basins is created successfully
        initial_plot = plotter._empty_plot
        print("✓ Successfully created initial plot with basins")

        # Check that the plot contains the expected elements
        if hasattr(initial_plot, "opts") and hasattr(initial_plot, "data"):
            print("✓ Initial plot has correct structure")

        # Test the AntarcticBackground directly
        bg = plotter._bg
        basins = bg.get_basins_polygons()
        print(f"✓ InteractivePlotter loaded {len(basins)} basin polygons")

        return True

    except Exception as e:
        print(f"✗ ERROR: Failed to create InteractivePlotter: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_basins_toggle_functionality():
    """Test that the basins toggle works correctly."""
    print("Testing basins toggle functionality...")

    class MockDatabase:
        def __init__(self):
            self._db_dir = "."

    mock_db = MockDatabase()
    plotter = InteractivePlotter(database_instance=mock_db)

    # Test toggle on/off
    plotter.basins_toggle.value = True
    print("✓ Basins toggle set to True")

    plotter.basins_toggle.value = False
    print("✓ Basins toggle set to False")

    return True


def main():
    """Run all tests."""
    print("Testing InteractivePlotter with IMBIE basins...\n")

    tests = [
        test_interactive_plotter_basins,
        test_basins_toggle_functionality,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ ERROR in {test.__name__}: {e}\n")
            import traceback

            traceback.print_exc()

    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All InteractivePlotter tests passed!")
        return True
    else:
        print("❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
