#!/usr/bin/env python3
"""
Simple Zarr conversion test without package dependencies.
"""

import os
import xarray as xr


def test_simple_zarr_conversion():
    """Test Zarr conversion on a single file."""
    print("=== Simple Zarr Conversion Test ===")

    # Test file paths
    base_dir = "/home/anthe/data/isochrones/database_project/myAntADatabase"
    h5_dir = os.path.join(base_dir, "Cavitte_2020", "h5")
    output_dir = "/tmp/zarr_simple_test"

    if not os.path.exists(h5_dir):
        print(f"❌ HDF5 directory not found: {h5_dir}")
        return

    # Get first HDF5 file
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith(".h5")]
    if not h5_files:
        print(f"❌ No HDF5 files found in {h5_dir}")
        return

    h5_file = h5_files[0]
    h5_path = os.path.join(h5_dir, h5_file)

    print(f"Testing conversion of: {h5_file}")

    try:
        # Read HDF5
        print("Reading HDF5 file...")
        ds = xr.open_dataset(h5_path)
        print(f"✅ HDF5 loaded: {dict(ds.sizes)}")

        # Create output directory
        zarr_dir = os.path.join(output_dir, "Cavitte_2020")
        os.makedirs(zarr_dir, exist_ok=True)

        # Convert to Zarr
        flight_line = h5_file.replace(".h5", "")
        zarr_path = os.path.join(zarr_dir, f"{flight_line}.zarr")

        print(f"Converting to Zarr at {zarr_path}...")
        ds.to_zarr(zarr_path)
        print("✅ Zarr conversion completed")

        # Verify Zarr
        print("Verifying Zarr file...")
        zarr_ds = xr.open_zarr(zarr_path)
        print(f"✅ Zarr verified: {dict(zarr_ds.sizes)}")
        print(f"   Variables: {list(zarr_ds.data_vars)}")

        # Test direct access
        print("Testing direct Zarr access...")
        sample_data = zarr_ds.isel(point=0)
        print(f"✅ Direct access works: {dict(sample_data.sizes)}")

        print(f"\n✅ All tests passed!")
        print(f"Zarr file created at: {zarr_path}")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_simple_zarr_conversion()
