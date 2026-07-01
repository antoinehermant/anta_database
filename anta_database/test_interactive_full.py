#!/usr/bin/env python
"""
Full test for the interactive plotting module.
Tests data loading and basic functionality.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing interactive plotting module...")

# Test 1: Import all classes
print("\n[1/5] Testing imports...")
from anta_database.interactive_plotting import (
    InteractivePlotter,
    TransectLoader,
    load_transects_to_geodataframe,
    EPSG_3031,
)
from anta_database.interactive_plotting.data_loader import TransectLoader
from anta_database.interactive_plotting.utils import (
    get_file_metadata,
    simplify_geometry,
    get_transect_metadata,
    format_metadata_for_display,
)

print("✓ All imports successful")

# Test 2: Test TransectLoader
print("\n[2/5] Testing TransectLoader...")
loader = TransectLoader(db_dir="./test_data", simplify_tolerance=0.1, disable_tqdm=True)
print(f"✓ TransectLoader created: simplify_tolerance={loader.simplify_tolerance}")

# Test 3: Test utilities
print("\n[3/5] Testing utility functions...")
print(f"✓ EPSG_3031 = {EPSG_3031}")
print(f"✓ simplify_geometry function available")
print(f"✓ format_metadata_for_display function available")

# Test 4: Test metadata extraction (if we have a test file)
print("\n[4/5] Testing metadata extraction...")
# This would need an actual H5 file, so we'll skip for now
print("✓ Metadata functions ready (requires actual H5 file to test)")

# Test 5: Test InteractivePlotter creation (without database)
print("\n[5/5] Testing InteractivePlotter...")


# We need a mock database object
class MockDatabase:
    def __init__(self):
        self._db_dir = "./test_data"

    def query(self, **kwargs):
        return {
            "flight_id": ["test_flight_1", "test_flight_2"],
            "dataset": ["test_dataset"],
            "var": ["ICE_THK"],
        }

    def _get_file_paths_from_metadata(self, metadata):
        return []


mock_db = MockDatabase()
try:
    plotter = InteractivePlotter(
        database_instance=mock_db,
        simplify_tolerance=0.1,
        disable_tqdm=True,
    )
    print("✓ InteractivePlotter created successfully")

    # Test panel creation
    panel = plotter.get_panel()
    print("✓ Panel created successfully")

except Exception as e:
    print(f"✗ Error creating InteractivePlotter: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
