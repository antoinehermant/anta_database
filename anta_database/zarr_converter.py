"""
Zarr conversion module for Anta Database.
Converts HDF5 flight line files to individual Zarr datasets.
"""

import os
import xarray as xr
from tqdm import tqdm
import shutil


class FlightLineZarrConverter:
    """
    Convert HDF5 flight line files to Zarr format.
    Creates one Zarr dataset per flight line.
    """

    def __init__(self, base_dir, output_dir=None):
        """
        Initialize converter.

        Args:
            base_dir: Base directory containing dataset directories
            output_dir: Output directory (default: zarr subdirectory in each dataset)
        """
        self.base_dir = base_dir
        if output_dir is None:
            self.output_dir = os.path.join(base_dir, "zarr")
        else:
            self.output_dir = output_dir

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_dataset(self, dataset_name):
        """
        Convert all flight lines in a dataset to Zarr.

        Args:
            dataset_name: Name of dataset (e.g., 'Cavitte_2020')

        Returns:
            List of converted flight lines
        """
        print(f"Converting {dataset_name}...")

        # Input and output paths
        h5_dir = os.path.join(self.base_dir, dataset_name, "h5")
        zarr_dataset_dir = os.path.join(self.output_dir, dataset_name)

        if not os.path.exists(h5_dir):
            print(f"Warning: HDF5 directory not found: {h5_dir}")
            return []

        # Create output directory for this dataset
        os.makedirs(zarr_dataset_dir, exist_ok=True)

        converted_files = []

        # Get all HDF5 files
        h5_files = [f for f in os.listdir(h5_dir) if f.endswith(".h5")]

        for h5_file in tqdm(h5_files, desc=f"Converting {dataset_name}"):
            try:
                # Read HDF5 file
                h5_path = os.path.join(h5_dir, h5_file)
                ds = xr.open_dataset(h5_path)

                # Remove flight_line dimension if present
                if "flight_line" in ds.dims and ds.sizes["flight_line"] == 1:
                    ds = ds.squeeze("flight_line", drop=True)
                elif "flight_line" in ds.coords:
                    ds = ds.drop_vars("flight_line")

                # Create Zarr output path
                flight_line = h5_file.replace(".h5", "")
                zarr_path = os.path.join(zarr_dataset_dir, f"{flight_line}.zarr")

                # Save as Zarr
                ds.to_zarr(zarr_path)

                converted_files.append(
                    {
                        "dataset": dataset_name,
                        "flight_line": flight_line,
                        "h5_path": h5_path,
                        "zarr_path": zarr_path,
                    }
                )

            except Exception as e:
                print(f"Error converting {h5_file}: {e}")
                continue

        print(f"✅ Converted {len(converted_files)}/{len(h5_files)} files")
        return converted_files

    def convert_all(self, datasets=None):
        """
        Convert all datasets.

        Args:
            datasets: List of dataset names to convert (None for all)

        Returns:
            Dictionary of all converted files by dataset
        """
        if datasets is None:
            # Find all dataset directories
            datasets = [
                d
                for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))
            ]

        all_converted = {}

        for dataset in datasets:
            converted = self.convert_dataset(dataset)
            if converted:
                all_converted[dataset] = converted

        return all_converted

    def create_metadata(self, output_file="zarr_metadata.json"):
        """
        Create metadata file for converted Zarr datasets.

        Args:
            output_file: Path to save metadata JSON
        """
        import json

        # Find all converted files
        metadata = []

        for dataset in os.listdir(self.output_dir):
            dataset_dir = os.path.join(self.output_dir, dataset)
            if not os.path.isdir(dataset_dir):
                continue

            for zarr_file in os.listdir(dataset_dir):
                if zarr_file.endswith(".zarr"):
                    flight_line = zarr_file.replace(".zarr", "")
                    zarr_path = os.path.join(dataset_dir, zarr_file)

                    # Get file info
                    try:
                        ds = xr.open_zarr(zarr_path)
                        metadata.append(
                            {
                                "dataset": dataset,
                                "flight_line": flight_line,
                                "zarr_path": zarr_path,
                                "dimensions": dict(ds.sizes),
                                "variables": list(ds.data_vars),
                                "coordinates": list(ds.coords),
                            }
                        )
                    except Exception as e:
                        print(f"Error reading {zarr_path}: {e}")
                        continue

        # Save metadata
        metadata_path = os.path.join(self.output_dir, output_file)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved to {metadata_path}")
        return metadata_path


def convert_to_zarr(base_dir, output_dir=None):
    """
    Convenience function to convert HDF5 files to Zarr.

    Args:
        base_dir: Base directory containing dataset directories
        output_dir: Output directory (default: zarr subdirectory)

    Returns:
        Path to metadata file
    """
    converter = FlightLineZarrConverter(base_dir, output_dir)
    converted = converter.convert_all()
    metadata_path = converter.create_metadata()

    print(f"\n=== Conversion Summary ===")
    print(f"Total datasets converted: {len(converted)}")
    for dataset, files in converted.items():
        print(f"  {dataset}: {len(files)} flight lines")
    print(f"\nMetadata: {metadata_path}")

    return metadata_path


if __name__ == "__main__":
    # Example usage
    base_dir = "/home/anthe/data/isochrones/database_project/myAntADatabase"
    convert_to_zarr(base_dir)
