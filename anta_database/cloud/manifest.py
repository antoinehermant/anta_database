"""
Manifest shared by publish (maintainer) and sync (users).

A manifest lists every file of a published database with its checksum:
    {
        "format": 1,
        "db_version": "2026.10.27-1530",
        "created": "2026-10-27T15:30:00+00:00",
        "files": {"AntADatabase.db": {"sha256": "...", "size": 123}, ...}
    }
Paths are relative to the database directory and always use "/".
"""

import glob
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

MANIFEST_NAME = "manifest.json"
MANIFEST_FORMAT = 1
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


def build_manifest(database_dir: str, db_version: Optional[str] = None) -> Dict:
    """Hash all published files of a local database directory."""
    from tqdm import tqdm

    if not os.path.isfile(os.path.join(database_dir, DB_FILE)):
        raise FileNotFoundError(f"{DB_FILE} not found in {database_dir}")

    now = datetime.now(timezone.utc)
    files = {}
    for rel in tqdm(list_published_files(database_dir), desc="Hashing", unit="file"):
        path = os.path.join(database_dir, rel)
        files[rel] = {"sha256": sha256sum(path), "size": os.path.getsize(path)}

    return {
        "format": MANIFEST_FORMAT,
        "db_version": db_version or now.strftime("%Y.%m.%d-%H%M"),
        "created": now.isoformat(timespec="seconds"),
        "files": files,
    }


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
