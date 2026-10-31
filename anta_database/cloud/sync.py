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

Download statistics: after a sync that downloaded files from the published database, one
anonymous count is sent to the AntADatabase portal: whether it is the first download of the
whole database into the directory or a later sync (an update, a missing dataset), the number
of files and bytes, and the versions of the package and the database. The portal adds it to the
visitor statistics of antadatabase.com (country only, no IP address kept). Nothing else is
sent, and nothing at all with ANTA_NO_STATS=1, sync(stats=False) or anta-sync --no-stats.
See https://antadatabase.com/privacy/.
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

# Download statistics (see the module docstring)
STATS_URL = "https://api.antadatabase.com/api/stats/sync"
PRIVACY_URL = "https://antadatabase.com/privacy/"


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


class SyncResult(dict):
    """
    Result of sync(): a dict with "db_version", the files "downloaded" and "deleted", and
    the dataset names "new", "updated" and "removed". Shown as a one-line summary.
    """

    def __repr__(self) -> str:
        datasets = ", ".join(
            f"{len(self[k])} {k}" for k in ("new", "updated", "removed") if self[k]
        )
        files = f"{len(self['downloaded'])} file(s) downloaded, {len(self['deleted'])} removed"
        return f"SyncResult(db_version={self['db_version']!r}, {datasets + ', ' if datasets else ''}{files})"


def _stats_enabled(stats: bool, url: str) -> bool:
    """Count only syncs of the published database (not of a copy, or of tests), unless turned off."""
    off = os.environ.get("ANTA_NO_STATS", "").strip().lower() not in ("", "0", "false", "no")
    return stats and not off and url.startswith(BUCKET_URL)


def _report_sync(kind: str, n_files: int, n_bytes: int, db_version: Optional[str]) -> None:
    """Send the anonymous count of a sync; never fails, never waits more than a few seconds."""
    try:
        from importlib.metadata import version

        package = version("anta_database")
    except Exception:
        package = "unknown"
    body = json.dumps({
        "kind": kind,
        "files": n_files,
        "bytes": n_bytes,
        "db_version": str(db_version or ""),
        "package_version": package,
    }).encode()
    request = urllib.request.Request(
        STATS_URL, data=body, headers={"Content-Type": "application/json", "User-Agent": f"anta_database/{package}"}
    )
    try:
        urllib.request.urlopen(request, timeout=3).close()
    except Exception:
        pass


def _size(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:.1f} MB" if n_bytes < 1e9 else f"{n_bytes / 1e9:.2f} GB"


def _dataset(rel: str) -> Optional[str]:
    """Dataset directory of a file, None for files at the top (AntADatabase.db)."""
    return rel.split("/", 1)[0] if "/" in rel else None


def _citation(name: str, remote: Dict, database_dir: str) -> str:
    """Citation of a dataset: from the manifest, else the local raw_md.json, else its name."""
    if remote.get("datasets", {}).get(name):
        return remote["datasets"][name]
    try:
        with open(os.path.join(database_dir, name, "raw_md.json"), encoding="utf-8") as f:
            unique = list(dict.fromkeys(i["citation"] for i in json.load(f) if i.get("citation")))
        if unique:
            return "; ".join(unique)
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return name


def _changed_datasets(
    to_download: list, to_delete: list, remote: Dict, local: Dict, database_dir: str
) -> Dict[str, list]:
    """
    Datasets that sync adds, updates or removes: new when nothing of it is in the local
    database yet, removed when it is no longer published, updated otherwise.
    """
    changed = sorted({d for rel in to_download + to_delete if (d := _dataset(rel))})
    published = {d for rel in remote["files"] if (d := _dataset(rel))}
    synced = {d for rel in local["files"] if (d := _dataset(rel))}
    groups = {"new": [], "updated": [], "removed": []}
    for name in changed:
        if name not in published:
            groups["removed"].append(name)
        elif name in synced or os.path.isfile(os.path.join(database_dir, name, "raw_md.json")):
            groups["updated"].append(name)
        else:
            groups["new"].append(name)
    return groups


def _print_plan(
    groups: Dict[str, list], remote: Dict, local: Dict, database_dir: str,
    total: int, index_only: bool, dry_run: bool,
) -> None:
    """Tell which datasets will change, before downloading."""
    version = remote.get("db_version")
    if not local["files"] and not groups["updated"] and not groups["removed"]:
        # First download: the list of all datasets would only be long
        verb = "Would download" if dry_run else "Downloading"
        print(
            f"{verb} the AntADatabase {version} ({len(groups['new'])} datasets, "
            f"{_size(total)}) to {database_dir}"
        )
        return
    old = local.get("db_version")
    since = f" from {old}" if old and old != version else ""
    verb = "Would update" if dry_run else "Updating"
    print(f"{verb} the AntADatabase in {database_dir}{since} to {version} ({_size(total)}):")
    labels = {"new": "New dataset", "updated": "Updated dataset", "removed": "Removed dataset"}
    for key, label in labels.items():
        for name in groups[key]:
            print(f"  {label}: {_citation(name, remote, database_dir)}")
    if index_only:
        print("  Updated index of the database (AntADatabase.db)")


def sync(
    database_dir: Optional[str] = None,
    url: Optional[str] = None,
    dry_run: bool = False,
    delete: bool = True,
    stats: bool = True,
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
        stats: Send one anonymous count of the sync (first download or later sync, with the
            number of files and bytes) for the statistics of the AntADatabase. Also
            turned off by the ANTA_NO_STATS=1 environment variable. See the module docstring.

    Returns:
        A SyncResult: a dict with the remote "db_version", the lists of files "downloaded"
        and "deleted", and the lists of datasets "new", "updated" and "removed".
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

    groups = _changed_datasets(to_download, to_delete, remote, local, database_dir)
    result = SyncResult(
        db_version=remote.get("db_version"),
        downloaded=to_download,
        deleted=to_delete,
        **groups,
    )
    total = sum(remote["files"][rel]["size"] for rel in to_download)
    if to_download or to_delete:
        # The index alone changes when a dataset is fixed without changing its files
        index_only = not any(groups.values()) and DB_FILE in to_download
        _print_plan(groups, remote, local, database_dir, total, index_only, dry_run)
    elif dry_run:
        print(f"AntADatabase {remote.get('db_version')} is up to date in {database_dir}.")
    if dry_run:
        return result

    os.makedirs(database_dir, exist_ok=True)
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

    # The first download of the whole database into a directory, and later syncs, apart
    if to_download and _stats_enabled(stats, url):
        first = not local["files"]
        _report_sync("download" if first else "update", len(to_download), total, remote.get("db_version"))
        if first:
            print(
                "This download was counted, anonymously, in the statistics of the AntADatabase. "
                f"To turn this off: ANTA_NO_STATS=1 or sync(stats=False). See {PRIVACY_URL}"
            )

    if to_download or to_delete:
        print(f"AntADatabase {remote.get('db_version')} synced to {database_dir}.")
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
    parser.add_argument("--files", action="store_true", help="With --dry-run, also list the files")
    parser.add_argument("--no-stats", action="store_true", help=f"Don't count this download in the statistics ({PRIVACY_URL})")
    args = parser.parse_args(argv)

    result = sync(args.dir, args.url, dry_run=args.dry_run, delete=not args.no_delete, stats=not args.no_stats)
    if args.dry_run and args.files:
        for rel in result["downloaded"]:
            print(f"  + {rel}")
        for rel in result["deleted"]:
            print(f"  - {rel}")


if __name__ == "__main__":
    main()
