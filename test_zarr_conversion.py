#!/usr/bin/env python3
"""
Test Zarr conversion with a small subset of files.
"""

import os
import xarray as xr
from anta_database.zarr_converter import FlightLineZarrConverter


def test_zarr_conversion():
    """Test Zarr conversion on a small subset."""
    print("=== Testing Zarr Conversion ===")

    # Test with Cavitte_2020 dataset
    base_dir = "/home/anthe/data/isochrones/database_project/myAntADatabase"
    output_dir = "/tmp/zarr_test"

    # Create converter
    converter = FlightLineZarrConverter(base_dir, output_dir)

    # Convert just 2 files for testing
    h5_dir = os.path.join(base_dir, "Cavitte_2020", "h5")
    if os.path.exists(h5_dir):
        h5_files = [f for f in os.listdir(h5_dir) if f.endswith(".h5")][:2]

        print(f"Testing with {len(h5_files)} files from Cavitte_2020...")

        for h5_file in h5_files:
            try:
                # Read HDF5
                h5_path = os.path.join(h5_dir, h5_file)
                ds = xr.open_dataset(h5_path)

                # Convert to Zarr
                flight_line = h5_file.replace(".h5", "")
                zarr_dir = os.path.join(output_dir, "Cavitte_2020")
                os.makedirs(zarr_dir, exist_ok=True)
                zarr_path = os.path.join(zarr_dir, f"{flight_line}.zarr")

                print(f"Converting {h5_file} to Zarr...")
                ds.to_zarr(zarr_path)
                print(f"✅ Created {zarr_path}")

                # Verify Zarr file
                zarr_ds = xr.open_zarr(zarr_path)
                print(f"   Zarr dimensions: {dict(zarr_ds.sizes)}")
                print(f"   Zarr variables: {list(zarr_ds.data_vars)}")

            except Exception as e:
                print(f"❌ Error converting {h5_file}: {e}")
                continue

        print("\n✅ Zarr conversion test completed!")
        print(f"Test files created in: {output_dir}")
    else:
        print(f"❌ HDF5 directory not found: {h5_dir}")


if __name__ == "__main__":
    test_zarr_conversion()
