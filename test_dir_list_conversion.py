#!/usr/bin/env python3
"""
Test Zarr conversion with directory list input.
"""

import os
import sys

sys.path.insert(0, ".")

from anta_database.zarr_converter import FlightLineZarrConverter


def test_dir_list_conversion():
    """Test conversion using directory list."""
    print("=== Testing Directory List Conversion ===")

    # Example directory list
    dir_path_list = [
        "./myAntADatabase/Napoleoni_2026/",
        "./myAntADatabase/Cavitte_2020/",
    ]

    # Create converter
    base_dir = "/home/anthe/data/isochrones/database_project/myAntADatabase"
    output_dir = "/tmp/zarr_dir_test"

    converter = FlightLineZarrConverter(base_dir, output_dir)

    # Test with directory list
    print(f"Converting directories: {dir_path_list}")

    try:
        # Convert using dir_path_list parameter
        converted = converter.convert_all(dir_path_list=dir_path_list)

        print(f"\n✅ Conversion completed!")
        print(f"Datasets converted: {list(converted.keys())}")

        for dataset, files in converted.items():
            print(f"  {dataset}: {len(files)} flight lines")

        # Create metadata
        metadata_path = converter.create_metadata()
        print(f"\n✅ Metadata created: {metadata_path}")

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_dir_list_conversion()
