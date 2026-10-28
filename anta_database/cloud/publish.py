"""
Publish a local AntADatabase to an S3-compatible bucket (maintainers only).

Needs the [publish] extra: pip install "anta_database[publish]"

Credentials are read by boto3 from the environment (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY) or from ~/.aws/credentials, never from the package.
The bucket must be publicly readable for sync() to work. For Infomaniak
Public Cloud, the S3 endpoint is https://s3.pub1.infomaniak.cloud.

    anta-publish /path/to/AntADatabase --bucket antadatabase

The prefix defaults to the data schema of this package (v1, v2, ...).
--min-package-version sets the oldest anta_database that supports all features of
the data; users with an older package get a warning. It is kept from the previous
publish when not given.
"""

import argparse
import json
import mimetypes
import os
from typing import Dict, Optional

from tqdm import tqdm

from anta_database.cloud.manifest import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    build_manifest,
    check_manifest,
    schema_version,
)

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError as e:
    raise ImportError(
        "Publishing the AntADatabase needs optional dependencies. "
        'Install them with: pip install "anta_database[publish]"'
    ) from e

DEFAULT_ENDPOINT = "https://s3.pub1.infomaniak.cloud"
DEFAULT_PREFIX = f"v{SCHEMA_VERSION}"


def _key(prefix: str, rel: str) -> str:
    return f"{prefix.strip('/')}/{rel}" if prefix.strip("/") else rel


def _get_remote_manifest(s3, bucket: str, prefix: str) -> Optional[Dict]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=_key(prefix, MANIFEST_NAME))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    return check_manifest(json.load(obj["Body"]))


def publish(
    database_dir: str,
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    endpoint_url: Optional[str] = None,
    db_version: Optional[str] = None,
    min_package_version: Optional[str] = None,
    dry_run: bool = False,
    delete: bool = True,
) -> Dict:
    """
    Upload new and changed files of a local database, then the manifest.

    The manifest is uploaded after all data files, so users never sync a
    manifest that points to missing files. A user syncing during a publish may
    hit a checksum error on a file being replaced and simply has to sync again.
    Files that are no longer part of the database are deleted afterwards.

    Args:
        database_dir: Local database directory (with AntADatabase.db).
        bucket: Bucket name.
        prefix: Key prefix inside the bucket. Defaults to the data schema, e.g. "v1".
        endpoint_url: S3 endpoint. Defaults to $AWS_ENDPOINT_URL or Infomaniak.
        db_version: Version label stored in the manifest. Defaults to the UTC date and time.
        min_package_version: Oldest anta_database supporting all features of the data.
            Defaults to the value of the previous publish.
        dry_run: Only report what would be uploaded and deleted.
        delete: Delete objects that were published before but are no longer in the database.

    Returns:
        A dict with "db_version" and the lists "uploaded" and "deleted".
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url or os.environ.get("AWS_ENDPOINT_URL", DEFAULT_ENDPOINT),
        # Swift-based S3 APIs (Infomaniak) reject the default aws-chunked checksummed uploads
        config=Config(
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )

    remote = _get_remote_manifest(s3, bucket, prefix) or {"files": {}}
    if remote["files"] and schema_version(remote) != SCHEMA_VERSION:
        raise ValueError(
            f"{bucket}/{prefix} holds data schema {schema_version(remote)}, but this package "
            f"writes schema {SCHEMA_VERSION}. Publish to prefix v{SCHEMA_VERSION} instead."
        )
    manifest = build_manifest(
        database_dir, db_version, min_package_version or remote.get("min_package_version")
    )

    to_upload = [
        rel
        for rel, entry in manifest["files"].items()
        if remote["files"].get(rel, {}).get("sha256") != entry["sha256"]
    ]
    to_delete = sorted(set(remote["files"]) - set(manifest["files"])) if delete else []
    result = {"db_version": manifest["db_version"], "uploaded": to_upload, "deleted": to_delete}
    if dry_run:
        return result

    total = sum(manifest["files"][rel]["size"] for rel in to_upload)
    with tqdm(total=total, unit="B", unit_scale=True, desc="Uploading") as progress:
        for rel in to_upload:
            content_type = mimetypes.guess_type(rel)[0] or "application/octet-stream"
            s3.upload_file(
                os.path.join(database_dir, rel),
                bucket,
                _key(prefix, rel),
                ExtraArgs={"ContentType": content_type},
                Callback=progress.update,
            )

    # Clients must never fetch a stale manifest from a cache
    s3.put_object(
        Bucket=bucket,
        Key=_key(prefix, MANIFEST_NAME),
        Body=json.dumps(manifest, indent=1, sort_keys=True).encode(),
        ContentType="application/json",
        CacheControl="no-cache",
    )

    for rel in to_delete:
        s3.delete_object(Bucket=bucket, Key=_key(prefix, rel))

    print(
        f"Published AntADatabase {manifest['db_version']} to {bucket}/{prefix}: "
        f"{len(to_upload)} file(s) uploaded, {len(to_delete)} deleted."
    )
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="anta-publish", description="Publish a local AntADatabase to an S3 bucket."
    )
    parser.add_argument("database_dir", help="Local database directory")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--endpoint-url", help=f"S3 endpoint (default: $AWS_ENDPOINT_URL or {DEFAULT_ENDPOINT})")
    parser.add_argument("--db-version", help="Version label (default: UTC date and time)")
    parser.add_argument(
        "--min-package-version",
        help="Oldest anta_database supporting all features of the data (default: kept from the last publish)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only show what would change")
    parser.add_argument("--no-delete", action="store_true", help="Keep objects that are no longer in the database")
    args = parser.parse_args(argv)

    result = publish(
        args.database_dir,
        args.bucket,
        prefix=args.prefix,
        endpoint_url=args.endpoint_url,
        db_version=args.db_version,
        min_package_version=args.min_package_version,
        dry_run=args.dry_run,
        delete=not args.no_delete,
    )
    if args.dry_run:
        print(f"Would upload {len(result['uploaded'])} file(s), delete {len(result['deleted'])}.")
        for rel in result["uploaded"]:
            print(f"  + {rel}")
        for rel in result["deleted"]:
            print(f"  - {rel}")


if __name__ == "__main__":
    main()
