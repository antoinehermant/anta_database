"""
Download the published AntADatabase and keep a local copy up to date.

Only files that are new or changed since the last sync are downloaded. Each
file is downloaded to a ".part" file, checked against the manifest checksum
and only then moved into place. The SQLite database is replaced last, so an
interrupted sync leaves a usable local database.

    from anta_database import sync
    sync()                       # default location, see default_database_dir()
    sync("/path/to/AntADatabase")

or from the command line:
    anta-sync --dir /path/to/AntADatabase
"""

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from typing import Dict, Optional

from tqdm import tqdm

from anta_database.cloud.manifest import (
    DB_FILE,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    check_manifest,
    check_schema,
    package_update_warning,
    read_manifest,
    sha256sum,
    write_manifest,
)

# Public URL of the published database (the bucket prefix holding manifest.json).
# Anonymous reads on Infomaniak go through the Swift URL, not the S3 one.
# Each data schema has its own prefix (v1/, v2/, ...), so this package only syncs
# data it can read. Can be overridden with the ANTA_DATABASE_URL environment variable.
BUCKET_URL = (
    "https://s3.pub1.infomaniak.cloud/object/v1/"
    "AUTH_2030d6d085ec4b16a7c2320f0bb9427d/antadatabase"
)
DEFAULT_URL = f"{BUCKET_URL}/v{SCHEMA_VERSION}"

_TIMEOUT = 60


def default_url() -> str:
    return os.environ.get("ANTA_DATABASE_URL", DEFAULT_URL).rstrip("/")


def default_database_dir() -> str:
    """Local database directory: $ANTA_DATABASE_DIR, or the user data directory."""
    if os.environ.get("ANTA_DATABASE_DIR"):
        return os.environ["ANTA_DATABASE_DIR"]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return os.path.join(base, "anta_database", "AntADatabase")


def _fetch_manifest(url: str, timeout: float = _TIMEOUT) -> Dict:
    try:
        with urllib.request.urlopen(f"{url}/{MANIFEST_NAME}", timeout=timeout) as r:
            return check_manifest(json.load(r))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not fetch the AntADatabase manifest from {url} ({e}). "
            "Check your internet connection or the URL (ANTA_DATABASE_URL)."
        ) from e


def local_version(database_dir: str) -> Optional[str]:
    """Version of a synced local database, or None if it was never synced."""
    manifest = read_manifest(os.path.join(database_dir, MANIFEST_NAME))
    return manifest.get("db_version") if manifest else None


def remote_info(url: Optional[str] = None, timeout: float = 3) -> Optional[Dict]:
    """Version, publication date and total size of the published database, or None if unreachable."""
    try:
        manifest = _fetch_manifest((url or default_url()).rstrip("/"), timeout=timeout)
    except Exception:
        return None
    return {
        "db_version": manifest.get("db_version"),
        "published": (manifest.get("created") or "")[:10],
        "size": sum(entry["size"] for entry in manifest["files"].values()),
    }


def check_for_update(database_dir: str, url: Optional[str] = None, timeout: float = 3) -> Optional[str]:
    """
    Newer published version of a synced local database, or None.

    Only fetches the small manifest, and stays silent (returns None) when offline
    or when the local database was never synced (e.g. compiled locally).
    """
    local = local_version(database_dir)
    if local is None:
        return None
    remote = remote_info(url, timeout)
    if remote is None or remote["db_version"] == local:
        return None
    return remote["db_version"]


def _is_up_to_date(path: str, entry: Dict, local_entry: Optional[Dict]) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) != entry["size"]:
        return False
    if local_entry is not None and local_entry["sha256"] == entry["sha256"]:
        return True
    # Unknown or changed in the local manifest (e.g. an interrupted sync): check the file itself
    return sha256sum(path) == entry["sha256"]


def _download(url: str, path: str, entry: Dict, progress: tqdm) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.part"
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            progress.update(len(chunk))
    if os.path.getsize(tmp) != entry["size"] or sha256sum(tmp) != entry["sha256"]:
        os.remove(tmp)
        raise IOError(f"Checksum mismatch for {url}, please run sync again.")
    os.replace(tmp, path)


def sync(
    database_dir: Optional[str] = None,
    url: Optional[str] = None,
    dry_run: bool = False,
    delete: bool = True,
) -> Dict:
    """
    Make a local AntADatabase identical to the published one.

    Args:
        database_dir: Local database directory, created if needed.
            Defaults to default_database_dir().
        url: URL of the published database. Defaults to ANTA_DATABASE_URL or DEFAULT_URL.
        dry_run: Only report what would be downloaded and deleted.
        delete: Remove local files that were synced before but are no longer published.
            Files that were never synced are never touched.

    Returns:
        A dict with the remote "db_version" and the lists "downloaded" and "deleted".
    """
    database_dir = database_dir or default_database_dir()
    url = (url or default_url()).rstrip("/")
    disable_tqdm = os.getenv("JUPYTER_BOOK_BUILD", "0") not in ("", "0")

    remote = _fetch_manifest(url)
    check_schema(remote, url)
    warning = package_update_warning(remote)
    if warning:
        print(warning)
    local_manifest_path = os.path.join(database_dir, MANIFEST_NAME)
    local = read_manifest(local_manifest_path) or {"files": {}}

    to_download = [
        rel
        for rel, entry in remote["files"].items()
        if not _is_up_to_date(
            os.path.join(database_dir, rel), entry, local["files"].get(rel)
        )
    ]
    # The database references the h5 files, so it must arrive after them
    to_download.sort(key=lambda rel: (rel == DB_FILE, rel))
    to_delete = sorted(set(local["files"]) - set(remote["files"])) if delete else []

    result = {
        "db_version": remote.get("db_version"),
        "downloaded": to_download,
        "deleted": to_delete,
    }
    if dry_run:
        return result

    os.makedirs(database_dir, exist_ok=True)
    total = sum(remote["files"][rel]["size"] for rel in to_download)
    if total:
        free = shutil.disk_usage(database_dir).free
        if free < total:
            raise OSError(
                f"Not enough disk space in {database_dir}: "
                f"{total / 1e6:.0f} MB needed, {free / 1e6:.0f} MB free."
            )

    with tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading", disable=disable_tqdm
    ) as progress:
        for rel in to_download:
            _download(
                f"{url}/{rel}",
                os.path.join(database_dir, rel),
                remote["files"][rel],
                progress,
            )

    for rel in to_delete:
        path = os.path.join(database_dir, rel)
        if os.path.isfile(path):
            os.remove(path)

    write_manifest(remote, local_manifest_path)

    if to_download or to_delete:
        print(
            f"AntADatabase {remote.get('db_version')} synced to {database_dir}: "
            f"{len(to_download)} file(s) downloaded, {len(to_delete)} removed."
        )
    else:
        print(f"AntADatabase {remote.get('db_version')} is up to date in {database_dir}.")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="anta-sync", description="Download or update the AntADatabase."
    )
    parser.add_argument("--dir", help=f"Local database directory (default: {default_database_dir()})")
    parser.add_argument("--url", help=f"URL of the published database (default: {default_url()})")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would change")
    parser.add_argument("--no-delete", action="store_true", help="Keep files that are no longer published")
    args = parser.parse_args(argv)

    result = sync(args.dir, args.url, dry_run=args.dry_run, delete=not args.no_delete)
    if args.dry_run:
        print(f"Remote version: {result['db_version']}")
        print(f"Would download {len(result['downloaded'])} file(s), delete {len(result['deleted'])}.")
        for rel in result["downloaded"]:
            print(f"  + {rel}")
        for rel in result["deleted"]:
            print(f"  - {rel}")


if __name__ == "__main__":
    main()
