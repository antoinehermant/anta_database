"""
Rename flight IDs to line IDs in an existing database, without compiling it again.

Lines were called flights before ground-based radar joined the database. Since data
schema 2, the h5 files hold "line ID" and "line ID flag" instead of "flight ID" and
"flight ID flag", and the SQLite index has datasets.line_id instead of flight_id.
Only these names change, not the data, so renaming the h5 attributes and indexing
the database again gives the same files as a new compile.

This also renames the old names in the inputs of the compile, when present:
"flight ID" and "flight ID prefix" in raw_md.json, and Flight_ID in
original_new_column_names.csv. The compile still reads the old names anyway.

Usage:
    python -m anta_database.indexing.migrate_line_id DATABASE_DIR [--index] [--dry-run]
"""

import argparse
import glob
import os
import re
from typing import Dict

import h5py
from tqdm import tqdm

H5_ATTRS = {"flight ID": "line ID", "flight ID flag": "line ID flag"}
RAW_MD_KEY = re.compile(r'"flight ID( prefix)?"(\s*:)')
COLUMN = re.compile(r"(^|,)Flight_ID(?=,|\r?$)")


def _rename_h5_attrs(path: str, dry_run: bool) -> bool:
    """Rename the attributes of one h5 file in place. True if it had old names."""
    with h5py.File(path, "r" if dry_run else "r+") as f:
        old = [name for name in H5_ATTRS if name in f.attrs]
        if dry_run or not old:
            return bool(old)
        for name in old:
            # Same dtype as before (h5netcdf strings), so xarray reads them the same
            dtype = f.attrs.get_id(name).dtype
            f.attrs.create(H5_ATTRS[name], f.attrs[name], dtype=dtype)
            del f.attrs[name]
    return True


def _rename_in_text(path: str, rename, dry_run: bool) -> bool:
    """Rewrite a text file with rename(text), keeping the rest of it as it is."""
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    new = rename(text)
    if new == text:
        return False
    if not dry_run:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new)
    return True


def _rename_column(text: str) -> str:
    # The standard names are on the second line, the original ones on the first
    lines = text.split("\n")
    if len(lines) > 1:
        lines[1] = COLUMN.sub(r"\1Line_ID", lines[1])
    return "\n".join(lines)


def migrate_line_id(database_dir: str, dry_run: bool = False) -> Dict[str, int]:
    """
    Rename flight IDs to line IDs in all datasets of database_dir.

    Files already renamed are left untouched, so it can run again safely.
    Index the database afterwards (IndexDatabase(database_dir).index_database()),
    as the SQLite index has the old column name.

    Returns:
        The number of files renamed (or to rename, with dry_run): "h5", "raw_md"
        and "columns".
    """
    h5_files = sorted(glob.glob(os.path.join(database_dir, "*", "h5", "*.h5")))
    counts = {"h5": 0, "raw_md": 0, "columns": 0}
    for path in tqdm(h5_files, desc="Renaming h5 attributes", unit="file"):
        counts["h5"] += _rename_h5_attrs(path, dry_run)
    for path in glob.glob(os.path.join(database_dir, "*", "raw_md.json")):
        counts["raw_md"] += _rename_in_text(
            path, lambda t: RAW_MD_KEY.sub(r'"line ID\1"\2', t), dry_run
        )
    for path in glob.glob(os.path.join(database_dir, "*", "original_new_column_names.csv")):
        counts["columns"] += _rename_in_text(path, _rename_column, dry_run)

    verb = "To rename" if dry_run else "Renamed"
    print(
        f"{verb}: {counts['h5']} of {len(h5_files)} h5 files, {counts['raw_md']} raw_md.json, "
        f"{counts['columns']} original_new_column_names.csv in {database_dir}"
    )
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m anta_database.indexing.migrate_line_id",
        description="Rename flight IDs to line IDs in a database, without compiling it again.",
    )
    parser.add_argument("database_dir")
    parser.add_argument("--index", action="store_true", help="Index the database afterwards")
    parser.add_argument("--dry-run", action="store_true", help="Only count the files to rename")
    args = parser.parse_args(argv)

    migrate_line_id(args.database_dir, dry_run=args.dry_run)
    if args.index and not args.dry_run:
        from anta_database.indexing.index_database import IndexDatabase

        IndexDatabase(args.database_dir).index_database()
    elif not args.dry_run:
        print("Index the database next: IndexDatabase(database_dir).index_database()")


if __name__ == "__main__":
    main()
