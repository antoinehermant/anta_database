"""Where the portal keeps the files it receives: private buckets when they are set up
(PORTAL_SUBMISSIONS_BUCKET, PORTAL_CONTACT_BUCKET), otherwise the local data directory.

A submission or a contact message is first written to a local directory. With a bucket, its
files are then uploaded under <id>/ and removed from the server, except for the small record
(submission.json, message.json) kept for the status of the submission.
"""

import logging
import shutil
from pathlib import Path

from . import config

log = logging.getLogger("portal.storage")

# Records kept on the server with a bucket
RECORDS = ("submission.json", "message.json")


def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT,
        region_name="us-east-1",
        aws_access_key_id=config.S3_ACCESS_KEY_ID,
        aws_secret_access_key=config.S3_SECRET_ACCESS_KEY,
        # Swift-based S3 APIs (Infomaniak) reject the default aws-chunked checksummed uploads,
        # as in anta_database.cloud.publish
        config=Config(request_checksum_calculation="when_required", response_checksum_validation="when_required"),
    )


def upload(directory: Path, bucket: str, prefix: str) -> None:
    """Upload the files under a directory to <prefix>/ in a bucket."""
    s3 = _client()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        s3.upload_file(str(path), bucket, f"{prefix}/{path.relative_to(directory).as_posix()}")  # multipart for large files


def download(bucket: str, prefix: str, directory: Path) -> int:
    """Download the files under <prefix>/ of a bucket into a directory, but the records (those
    on the server are more recent); their number."""
    s3 = _client()
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            path = directory / item["Key"].removeprefix(f"{prefix}/")
            if ".." in path.relative_to(directory).parts or path.name in RECORDS:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, item["Key"], str(path))
            n += 1
    return n


def clean(directory: Path) -> None:
    """Remove the files of a stored directory from the server, except its record."""
    for path in list(directory.iterdir()):
        if path.name not in RECORDS:
            shutil.rmtree(path) if path.is_dir() else path.unlink()


def store(directory: Path, bucket: str, keep: bool = False) -> str:
    """Upload the files of a directory (named after its id) to a bucket, or leave them on the
    server without one. With keep, they also stay on the server, until clean(): the worker
    compiles the submissions from there. Where they are, for the emails."""
    if not bucket:
        return str(directory)
    upload(directory, bucket, directory.name)
    if not keep:
        clean(directory)
    log.info("Stored %s in bucket %s", directory.name, bucket)
    return f"bucket {bucket}, {directory.name}/"


def delete(bucket: str, prefix: str) -> int:
    """Delete everything under <prefix>/ of a bucket; the number of objects deleted."""
    if not bucket:
        return 0
    s3 = _client()
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=item["Key"])
            n += 1
    return n
