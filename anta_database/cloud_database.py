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
from botocore import UNSIGNED
from botocore.config import Config
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
        super().close()
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
