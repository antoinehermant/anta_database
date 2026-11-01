"""Publish the raw files of the first 18 datasets to raw/<Dataset>/ of the public bucket, as a record.
Same layout as portal/review.py uses for accepted submissions, with PROCESSING.txt where the build
changed something (written by build_v1.py). Files already online with the same size are skipped.

    python docs/harmonisation/publish_raw.py            # what would be uploaded
    AWS_PROFILE=infomaniak python docs/harmonisation/publish_raw.py --upload
"""
import argparse
from pathlib import Path

HOME = Path.home() / "data" / "isochrones"
WORKING = HOME / "database_project" / "myAntADatabase"
BUILD = HOME / "AntADatabase-v1-build"
BUCKET = "antadatabase"
ENDPOINT = "https://s3.pub1.infomaniak.cloud"
CONTENT_TYPES = {".csv": "text/csv", ".tab": "text/tab-separated-values", ".txt": "text/plain", ".json": "application/json"}


def files_of(name: str) -> dict[str, Path]:
    """Key inside raw/<Dataset>/ -> local file. The metadata is that of the build."""
    raw = WORKING / name / "raw"
    found = {f"raw/{p.relative_to(raw).as_posix()}": p for p in sorted(raw.rglob("*")) if p.is_file()}
    found["raw_md.json"] = BUILD / name / "raw_md.json"
    for path in (WORKING / name / "original_new_column_names.csv", BUILD / name / "PROCESSING.txt"):
        if path.exists():
            found[path.name] = path
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    client = None
    if args.upload:
        import boto3
        from botocore.config import Config

        # Swift-based S3 APIs (Infomaniak) reject the default aws-chunked checksummed uploads
        client = boto3.client("s3", endpoint_url=ENDPOINT, config=Config(
            request_checksum_calculation="when_required", response_checksum_validation="when_required"))
    total = 0
    for md in sorted(BUILD.glob("*/raw_md.json")):
        name = md.parent.name
        files = files_of(name)
        size = sum(p.stat().st_size for p in files.values())
        total += size
        print(f"{name}: {len(files)} files, {size / 1e6:.1f} MB", flush=True)
        if not client:
            continue
        online = {}
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"raw/{name}/"):
            online.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
        for rel, path in files.items():
            key = f"raw/{name}/{rel}"
            if online.get(key) != path.stat().st_size:
                client.upload_file(str(path), BUCKET, key, ExtraArgs={
                    "ContentType": CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")})
    print(f"Total {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
