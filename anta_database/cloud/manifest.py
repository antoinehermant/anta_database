"""
Manifest shared by publish (maintainer) and sync (users).

A manifest lists every file of a published database with its checksum:
    {
        "format": 1,
        "schema_version": 2,
        "min_package_version": "0.6.0",
        "db_version": "2026.10.27-1530",
        "created": "2026-10-27T15:30:00+00:00",
        "files": {"AntADatabase.db": {"sha256": "...", "size": 123}, ...},
        "datasets": {"Cavitte_2020": "Cavitte et al. (2020)", ...}
    }
Paths are relative to the database directory and always use "/".

"schema_version" is the layout of the data (h5 variables, SQLite tables). It only
changes with breaking changes, which are published under a new bucket prefix
(v1/, v2/, ...), so each package version keeps syncing data it can read.
Manifests without it are schema 1.
Schema 2 renamed flight IDs to line IDs (h5 attribute "line ID", SQLite column
datasets.line_id), as some lines come from ground-based radar.

"datasets" (optional) gives the citation of each dataset directory, so sync can tell
which datasets it updates before downloading them.

"min_package_version" (optional) is the oldest anta_database that supports all
features of the published data. Older packages still work, with a warning.
"""

import glob
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, Optional

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = 1
# Data layout this package reads. Bump only for breaking changes, see above.
SCHEMA_VERSION = 2
DB_FILE = "AntADatabase.db"

# Files of a database directory that are published, relative to the directory.
# Raw input data and temporary compile files are not published.
PUBLISHED_PATTERNS = [
    DB_FILE,
    "*/raw_md.json",
    "*/h5/*.h5",
]


def sha256sum(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def list_published_files(database_dir: str) -> list:
    """Relative paths of all files to publish from a local database directory."""
    files = set()
    for pattern in PUBLISHED_PATTERNS:
        for path in glob.glob(os.path.join(database_dir, pattern)):
            if os.path.isfile(path):
                rel = os.path.relpath(path, database_dir)
                files.add(rel.replace(os.sep, "/"))
    return sorted(files)


def dataset_citations(database_dir: str) -> Dict[str, str]:
    """Citation of each dataset directory, from its raw_md.json."""
    citations = {}
    for path in sorted(glob.glob(os.path.join(database_dir, "*", "raw_md.json"))):
        name = os.path.basename(os.path.dirname(path))
        try:
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
        except (OSError, ValueError):
            continue
        unique = list(dict.fromkeys(i["citation"] for i in items if i.get("citation")))
        if unique:
            citations[name] = "; ".join(unique)
    return citations


def build_manifest(
    database_dir: str,
    db_version: Optional[str] = None,
    min_package_version: Optional[str] = None,
) -> Dict:
    """Hash all published files of a local database directory."""
    from tqdm import tqdm

    if not os.path.isfile(os.path.join(database_dir, DB_FILE)):
        raise FileNotFoundError(f"{DB_FILE} not found in {database_dir}")

    now = datetime.now(timezone.utc)
    files = {}
    for rel in tqdm(list_published_files(database_dir), desc="Hashing", unit="file"):
        path = os.path.join(database_dir, rel)
        files[rel] = {"sha256": sha256sum(path), "size": os.path.getsize(path)}

    manifest = {
        "format": MANIFEST_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "db_version": db_version or now.strftime("%Y.%m.%d-%H%M"),
        "created": now.isoformat(timespec="seconds"),
        "files": files,
        "datasets": dataset_citations(database_dir),
    }
    if min_package_version:
        manifest["min_package_version"] = min_package_version
    return manifest


def check_manifest(manifest: Dict) -> Dict:
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"Unsupported manifest format {manifest.get('format')!r}. "
            "Try upgrading anta_database: pip install -U anta_database"
        )
    for rel in manifest["files"]:
        # Never let a manifest write outside the database directory
        if rel.startswith("/") or ".." in rel.split("/") or "\\" in rel:
            raise ValueError(f"Invalid path in manifest: {rel!r}")
    return manifest


def schema_version(manifest: Dict) -> int:
    return int(manifest.get("schema_version", 1))


def check_schema(manifest: Dict, where: str) -> None:
    """Raise if the data at `where` has a layout this package cannot read."""
    schema = schema_version(manifest)
    if schema > SCHEMA_VERSION:
        raise RuntimeError(
            f"The AntADatabase in {where} uses data schema {schema}, but this anta_database "
            f"({package_version() or 'unknown version'}) only reads schema {SCHEMA_VERSION}. "
            "Upgrade it with: pip install -U anta_database"
        )
    if schema < SCHEMA_VERSION:
        raise RuntimeError(
            f"The AntADatabase in {where} uses the old data schema {schema}, but this "
            f"anta_database reads schema {SCHEMA_VERSION}. Download the current data with "
            "sync(), into a new directory if you want to keep the old one."
        )


def package_version() -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    try:
        return version("anta_database")
    except PackageNotFoundError:
        return None


def _version_key(version: str) -> tuple:
    """
    Sortable key of a version string, enough for comparing package versions:
    "0.5.1.dev10" < "0.5.1rc1" < "0.5.1" == "0.5.1.0" < "0.5.1.post1".
    """
    match = re.match(r"v?(\d+(?:\.\d+)*)(.*)", version.split("+")[0].strip().lower())
    if match is None:
        return ((0,), 0)
    release = [int(n) for n in match.group(1).split(".")]
    while len(release) > 1 and release[-1] == 0:
        release.pop()
    suffix = match.group(2)
    if "dev" in suffix:
        stage = 0
    elif re.match(r"[.\-_]?(a|b|c|rc|alpha|beta|pre|preview)\d*", suffix):
        stage = 1
    elif "post" in suffix:
        stage = 3
    else:
        stage = 2
    return (tuple(release), stage)


def package_update_warning(manifest: Dict) -> Optional[str]:
    """Warning if the installed package is older than the manifest's min_package_version."""
    required = manifest.get("min_package_version")
    installed = package_version()
    if not required or not installed or _version_key(installed) >= _version_key(required):
        return None
    return (
        f"WARNING: this AntADatabase version ({manifest.get('db_version')}) works best with "
        f"anta_database >= {required}, you have {installed}. Some features may not work. "
        "Upgrade with: pip install -U anta_database"
    )


def read_manifest(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_manifest(manifest: Dict, path: str) -> None:
    tmp = f"{path}.part"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
