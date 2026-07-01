"""
Simplified Zarr converter with intuitive directory list interface.
"""

import os
import xarray as xr
from tqdm import tqdm


class FlightLineZarrConverter:
    """
    Convert HDF5 flight line files to Zarr format.

    Usage:
        base_dir = "/path/to/database/"
        dir_path_list = ["Cavitte_2020/", "Chung_2023/"]
        converter = FlightLineZarrConverter(base_dir, dir_path_list)
        converter.convert()
    """

    def __init__(self, base_dir, dir_path_list=None):
        """
        Initialize converter.

        Args:
            base_dir: Base directory containing dataset directories
            dir_path_list: List of subdirectories to convert
        """
        self.base_dir = base_dir
        self.dir_path_list = dir_path_list or []
        self.output_dir = os.path.join(base_dir, "zarr")
        os.makedirs(self.output_dir, exist_ok=True)

    def convert(self):
        """Convert all specified directories."""
        all_converted = {}

        for dir_path in self.dir_path_list:
            dataset_name = os.path.basename(dir_path.rstrip("/"))
            converted = self._convert_dataset(dir_path, dataset_name)
            if converted:
                all_converted[dataset_name] = converted

        # Create metadata
        self._create_metadata()

        return all_converted

    def _convert_dataset(self, dir_path, dataset_name):
        """Convert all flight lines in a dataset."""
        h5_dir = os.path.join(self.base_dir, dir_path, "h5")
        zarr_dataset_dir = os.path.join(self.output_dir, dataset_name)

        if not os.path.exists(h5_dir):
            print(f"Warning: HDF5 directory not found: {h5_dir}")
            return []

        os.makedirs(zarr_dataset_dir, exist_ok=True)

        converted_files = []
        h5_files = [f for f in os.listdir(h5_dir) if f.endswith(".h5")]

        for h5_file in tqdm(h5_files, desc=f"Converting {dataset_name}"):
            try:
                h5_path = os.path.join(h5_dir, h5_file)
                ds = xr.open_dataset(h5_path)

                # Remove flight_line dimension if present
                if "flight_line" in ds.dims and ds.sizes["flight_line"] == 1:
                    ds = ds.squeeze("flight_line", drop=True)
                elif "flight_line" in ds.coords:
                    ds = ds.drop_vars("flight_line")

                flight_line = h5_file.replace(".h5", "")
                zarr_path = os.path.join(zarr_dataset_dir, f"{flight_line}.zarr")

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

        print(
            f"✅ Converted {len(converted_files)}/{len(h5_files)} files in {dataset_name}"
        )
        return converted_files

    def _create_metadata(self):
        """Create metadata file."""
        import json

        metadata = []

        for dataset in os.listdir(self.output_dir):
            dataset_dir = os.path.join(self.output_dir, dataset)
            if not os.path.isdir(dataset_dir):
                continue

            for zarr_file in os.listdir(dataset_dir):
                if zarr_file.endswith(".zarr"):
                    flight_line = zarr_file.replace(".zarr", "")
                    zarr_path = os.path.join(dataset_dir, zarr_file)

                    try:
                        ds = xr.open_zarr(zarr_path)
                        metadata.append(
                            {
                                "dataset": dataset,
                                "flight_line": flight_line,
                                "zarr_path": zarr_path,
                                "dimensions": dict(ds.sizes),
                                "variables": list(ds.data_vars),
                            }
                        )
                    except Exception as e:
                        print(f"Error reading {zarr_path}: {e}")
                        continue

        metadata_path = os.path.join(self.output_dir, "zarr_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved to {metadata_path}")
        return metadata_path


def convert_to_zarr(base_dir, dir_path_list):
    """
    Convenience function for quick conversion.

    Args:
        base_dir: Base directory
        dir_path_list: List of subdirectories to convert
    """
    converter = FlightLineZarrConverter(base_dir, dir_path_list)
    return converter.convert()


if __name__ == "__main__":
    # Example usage
    base_dir = "/home/anthe/data/isochrones/database_project/myAntADatabase"
    dir_path_list = ["Cavitte_2020/", "Chung_2023/"]

    converter = FlightLineZarrConverter(base_dir, dir_path_list)
    converter.convert()
