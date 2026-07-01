#!/usr/bin/env python
"""
Test script for the interactive plotting module.
This tests the new module without importing the main package.
"""

import os
import sys

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now we can import with absolute path
from anta_database.interactive_plotting.data_loader import TransectLoader
from anta_database.interactive_plotting.utils import (
    get_file_metadata,
    EPSG_3031,
    simplify_geometry,
    get_transect_metadata,
)
from anta_database.interactive_plotting.interactive_plotter import (
    InteractivePlotter,
    InteractivePlotterFactory,
)

print("✓ All interactive_plotting modules imported successfully!")
print(f"✓ EPSG_3031 = {EPSG_3031}")
print(f"✓ TransectLoader class available")
print(f"✓ InteractivePlotter class available")
print(f"✓ InteractivePlotterFactory class available")

# Test the utilities
print("\n--- Testing utilities ---")
print(f"✓ simplify_geometry function available")
print(f"✓ get_file_metadata function available")
print(f"✓ get_transect_metadata function available")

print("\n✓ All tests passed!")
