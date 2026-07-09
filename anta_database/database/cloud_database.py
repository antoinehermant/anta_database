"""
Cloud-enabled database adapter for Anta Database.
Allows connecting to SQLite database hosted on S3 while maintaining
the same interface as the local Database class.
"""

import os
import tempfile
import shutil
import requests
import boto3
import numpy as np
import pandas as pd
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from typing import Union, Dict, Tuple, Optional, Generator
from tqdm import tqdm
from .database import Database


class CloudDatabase(Database):
    """
    Cloud-enabled version of the Database class.
    Downloads the SQLite database from S3 and uses it locally.
    """

    def __init__(
        self, s3_bucket="anta-database", local_cache=None, aws_credentials=None
    ):
        """
        Initialize cloud database connection.

        Args:
            s3_bucket: S3 bucket containing your database
            local_cache: Local directory for caching (default: temp directory)
            aws_credentials: Dict with 'key' and 'secret' for private buckets
        """
        self.s3_bucket = s3_bucket
        self.db_key = "AntADatabase/AntADatabase.db"
        self.aws_credentials = aws_credentials

        # Set up local cache
        if local_cache is None:
            self.local_cache = tempfile.mkdtemp()
        else:
            self.local_cache = local_cache
            os.makedirs(self.local_cache, exist_ok=True)

        # Download SQLite DB to local cache
        self.db_path = self._download_sqlite_db()

        # Initialize parent class with local DB path
        super().__init__(os.path.dirname(self.db_path))

    def _download_sqlite_db(self):
        """Download SQLite database from S3 to local cache."""
        local_path = os.path.join(self.local_cache, "AntADatabase.db")

        # Check if already cached
        if os.path.exists(local_path):
            print(f"Using cached database: {local_path}")
            return local_path

        print(f"Downloading database from s3://{self.s3_bucket}/{self.db_key}...")

        # Try S3 direct download first
        try:
            # Configure S3 client
            if self.aws_credentials:
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=self.aws_credentials["key"],
                    aws_secret_access_key=self.aws_credentials["secret"],
                )
            else:
                s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

            # Generate presigned URL
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.s3_bucket, "Key": self.db_key},
                ExpiresIn=3600,  # 1 hour
            )

            # Download with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Save to local file
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"✅ Database downloaded to {local_path}")
            return local_path

        except Exception as e:
            print(f"S3 download failed: {e}")
            print("Trying HTTP fallback...")

            # Fallback to HTTP URL
            return self._download_from_http()

    def _download_from_http(self):
        """Download file from HTTP URL."""
        local_path = os.path.join(self.local_cache, "AntADatabase.db")
        http_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{self.db_key}"

        print(f"Downloading from {http_url}...")
        response = requests.get(http_url, stream=True)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"✅ Downloaded to {local_path}")
        return local_path

    def close(self):
        """Clean up local cache."""
        if hasattr(self, "local_cache") and self.local_cache:
            print(f"Cleaning up cache: {self.local_cache}")
            shutil.rmtree(self.local_cache, ignore_errors=True)

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()

    def test_connection(self):
        """Test if S3 bucket is accessible."""
        try:
            if self.aws_credentials:
                s3 = boto3.client(
                    "s3",
                    aws_access_key_id=self.aws_credentials["key"],
                    aws_secret_access_key=self.aws_credentials["secret"],
                )
            else:
                s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

            s3.head_object(Bucket=self.s3_bucket, Key=self.db_key)
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

    def _get_s3_zarr_path(self, file_path: str) -> str:
        """
        Convert a local file path to an S3 zarr path.
        Assumes zarr files are stored in the same structure as HDF5 files
        but with .zarr extension instead of .h5.
        """
        # Remove the database directory prefix and convert to zarr path
        # Use os.path.normpath to handle any .. or . in the path
        norm_file_path = os.path.normpath(file_path)
        norm_db_dir = os.path.normpath(self._db_dir)

        # Check if file_path starts with db_dir
        if norm_file_path.startswith(norm_db_dir):
            rel_path = norm_file_path[len(norm_db_dir) :].lstrip("/\\")
        elif os.path.isabs(file_path):
            # If it's an absolute path but doesn't start with db_dir, use basename
            rel_path = os.path.basename(file_path)
        else:
            # If it's a relative path, use it as-is (assuming it's relative to db_dir)
            rel_path = file_path

        # Convert .h5 to .zarr and change h5/ to zarr/ directory
        if rel_path.endswith(".h5"):
            # Replace h5/ with zarr/ in the path
            rel_path = rel_path.replace("/h5/", "/zarr/")
            rel_path = rel_path[:-3] + ".zarr"  # Change extension
        elif not rel_path.endswith(".zarr"):
            rel_path = rel_path + ".zarr"
        else:
            # Already has .zarr extension, just ensure it's in zarr/ directory
            rel_path = rel_path.replace("/h5/", "/zarr/")

        # Construct S3 path
        s3_zarr_path = f"s3://{self.s3_bucket}/AntADatabase/{rel_path}"
        return s3_zarr_path

    def _open_zarr_dataset(self, s3_zarr_path: str):
        """
        Open a zarr dataset from S3 using xarray.
        Handles authentication and storage options.
        """
        try:
            # Use the same approach as direct xr.open_zarr() which works
            try:
                # First try direct xarray open_zarr (this works for public buckets)
                ds = xr.open_zarr(s3_zarr_path, consolidated=True)
                return ds
            except Exception as e1:
                # If that fails, try with fsspec
                try:
                    import fsspec

                    # Configure storage options for S3
                    if self.aws_credentials:
                        storage_options = {
                            "key": self.aws_credentials["key"],
                            "secret": self.aws_credentials["secret"],
                        }
                    else:
                        storage_options = {"anon": True}

                    # Use fsspec for S3 access
                    s3_fs = fsspec.filesystem("s3", **storage_options)
                    mapper = s3_fs.get_mapper(s3_zarr_path)

                    # Open zarr dataset
                    ds = xr.open_zarr(mapper, consolidated=True)
                    return ds
                except Exception as e2:
                    print(
                        f"Failed to open zarr dataset {s3_zarr_path}: {e1} (direct) / {e2} (fsspec)"
                    )
                    return None
        except Exception as e:
            print(f"Failed to open zarr dataset {s3_zarr_path}: {e}")
            return None

    def data_generator(
        self,
        metadata: Union[None, Dict, "MetadataResult"] = None,
        data_dir: Optional[str] = None,
        downsampling_factor: Optional[int] = None,
        rolling_distance: Optional[int] = None,
        disable_tqdm: bool = False,
        fraction_depth: Optional[bool] = False,
    ) -> Generator[Tuple[pd.DataFrame, Dict], None, None]:
        """
        Override the data_generator to use zarr data from S3 instead of local HDF5 files.

        This method delegates to zarr_data_generator which works with zarr data stored on S3.
        """
        # Call the zarr data generator (ignore data_dir parameter as we use S3)
        return self.zarr_data_generator(
            metadata=metadata,
            downsampling_factor=downsampling_factor,
            rolling_distance=rolling_distance,
            disable_tqdm=disable_tqdm,
            fraction_depth=fraction_depth,
        )

    def zarr_data_generator(
        self,
        metadata: Union[None, Dict, "MetadataResult"] = None,
        downsampling_factor: Optional[int] = None,
        rolling_distance: Optional[int] = None,
        disable_tqdm: bool = False,
        fraction_depth: Optional[bool] = False,
    ) -> Generator[Tuple[pd.DataFrame, Dict], None, None]:
        """
        Generates DataFrames from Zarr files on S3, one at a time.

        This method replicates the functionality of the HDF5 data_generator
        but works with zarr data stored on S3.

        Args:
            metadata: Metadata for filtering files.
            downsampling_factor: Downsampling factor for data points.
            rolling_distance: Distance for rolling mean calculation.
            disable_tqdm: Whether to disable progress bars.
            fraction_depth: Whether to compute fraction depth for IRH_DEPTH.

        Yields:
            Tuple[pd.DataFrame, Dict]: DataFrame with data and metadata dictionary.
        """
        # Resolve metadata
        md = metadata or self._md
        if not md:
            print(
                "Please provide metadata of the files you want to generate the data from. Exiting..."
            )
            return

        file_paths = self._get_file_paths_from_metadata(metadata=md)
        file_paths = np.unique(file_paths)

        if disable_tqdm or self._disable_tqdm:
            disable_tqdm = True

        for file_path in tqdm(
            file_paths,
            desc="Generating dataframes from Zarr",
            total=len(file_paths),
            unit="file",
            disable=disable_tqdm,
        ):
            # Convert local file path to S3 zarr path
            s3_zarr_path = self._get_s3_zarr_path(file_path)

            # Open zarr dataset from S3
            ds = self._open_zarr_dataset(s3_zarr_path)
            if ds is None:
                continue

            file_md = self._get_file_metadata(file_path)

            # Create dataframe with coordinates
            df = pd.DataFrame(
                {
                    "PSX": ds["PSX"].values[::downsampling_factor],
                    "PSY": ds["PSY"].values[::downsampling_factor],
                }
            )

            if "Distance" in ds.variables:
                df["Distance"] = ds["Distance"].values[::downsampling_factor]

            var_impl = []
            age_impl = []

            for var in md["var"]:
                if var == "IRH_DEPTH":
                    var_impl.append(var)
                    for age in md["age"]:
                        if "IRH_AGE" in ds.variables:
                            irh_values = ds["IRH_AGE"].values
                            irh_index = np.where(irh_values == int(age))[0]
                            if len(irh_index) == 0:
                                continue
                            irh_index = irh_index[0]
                            age_impl.append(age)

                            df[age] = ds[var].values[::downsampling_factor, irh_index]
                            if fraction_depth:
                                if "ICE_THK" in ds.variables:
                                    df[age] *= (
                                        100
                                        / ds["ICE_THK"].values[::downsampling_factor]
                                    )
                                else:
                                    df[age] = np.nan
                else:
                    if var in ds.variables:
                        var_impl.append(var)
                        df[var] = ds[var].values[::downsampling_factor]

            if 0 < len(age_impl) <= 1:
                age_impl = list(age_impl)[0]

            if rolling_distance:
                if "Distance" not in df.columns:
                    print("Distance not in dataset, cannot do rolling mean on distance")
                else:
                    df["bin"] = (
                        np.floor(df["Distance"] / rolling_distance) * rolling_distance
                    )
                    df = df.groupby("bin").mean().reset_index()
                    df.drop(columns=["bin"], inplace=True)

            metadata = {
                "dataset": file_md["dataset"],
                "var": var_impl,
                "age": age_impl,
                "flight_id": file_md["flight_id"],
                "institute": file_md["institute"],
                "project": file_md["project"],
                "acquisition_year": file_md["acquisition_year"],
                "age_unc": md["age_unc"],
                "reference": file_md["reference"],
                "DOI_dataset": file_md["DOI_dataset"],
                "DOI_publication": file_md["DOI_publication"],
                "flight_id": file_md["flight_id"],
                "region": file_md["region"],
                "IMBIE_basin": file_md["IMBIE_basin"],
                "radar_instrument": file_md["radar_instrument"],
            }

            yield df, metadata

            # Close the dataset
            ds.close()
