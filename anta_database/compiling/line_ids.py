"""Line IDs set in the metadata: line_ids.json next to raw_md.json, applied to the compiled files.

    {
     "note": "Why (shown nowhere, for the maintainers)",
     "lines": {
      "<line ID of the compilation>": {"line ID": "<new ID>", "line ID flag": "<flag>"},
      ...
     }
    }

Either key may be left out: a new ID keeps the flag, a new flag keeps the ID. The line ID of the
compilation is kept in the attribute "line ID original", and the file is renamed after the new ID.
Applying it again changes nothing (the original ID is looked up too), so it runs after every
compilation and every metadata update, without recompiling.

The values of "line ID flag":
- original: the line ID of the radar data, given with the published files;
- not_provided: none given, named by the database after the dataset (<Dataset>, <Dataset>_<n>);
- project_acq-year_number: none given, named after the project, year and a number;
- article: named after the article (a table or figure), not the published files;
- assigned: named by the authors of the dataset, not the line ID of the radar survey;
- bedmap: taken from the Bedmap survey line of the same traces.
"""

import glob
import json
import os

import xarray as xr

from anta_database.netcdf import write

FILE = "line_ids.json"
FLAGS = ("original", "not_provided", "project_acq-year_number", "article", "assigned", "bedmap")


def load(dataset_dir: str, text: str | None = None) -> dict:
    """{line ID of the compilation: {"line ID": ..., "line ID flag": ...}}, {} without a file.
    text: the content of a line_ids.json to use instead of the file of the dataset."""
    path = os.path.join(dataset_dir, FILE)
    if text is None:
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            text = f.read()
    lines = json.loads(text).get("lines", {})
    for old, new in lines.items():
        unknown = set(new) - {"line ID", "line ID flag"}
        if unknown:
            raise ValueError(f"{path}: {old}: unknown keys {sorted(unknown)}")
        if new.get("line ID flag", "original") not in FLAGS:
            raise ValueError(f"{path}: {old}: line ID flag {new['line ID flag']!r} not one of {FLAGS}")
    return lines


def file_name(line_id: str) -> str:
    """The file of a line, as the compilation names it ("/" would make a directory)."""
    return line_id.replace("/", "_") + ".nc"


def apply(dataset_dir: str, dry_run: bool = False, text: str | None = None) -> list[tuple]:
    """Set the line IDs and flags of line_ids.json in the files of a dataset (or of `text`, the
    content of another line_ids.json). Returns (old file, new file, old ID, new ID, old flag, new
    flag) per file changed, or that would change with dry_run."""
    lines = load(dataset_dir, text)
    if not lines:
        return []
    changes = []
    seen = set()
    for path in sorted(glob.glob(os.path.join(dataset_dir, "nc", "*.nc"))):
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            line_id = str(ds.attrs.get("line ID"))
            flag = str(ds.attrs.get("line ID flag"))
            original = str(ds.attrs.get("line ID original", line_id))
        target = lines.get(original)
        if target is None:
            continue
        seen.add(original)
        new_id = str(target.get("line ID", line_id))
        new_flag = target.get("line ID flag", flag)
        new_path = os.path.join(dataset_dir, "nc", file_name(new_id))
        if (new_id, new_flag) == (line_id, flag) and new_path == path:
            continue
        if new_path != path and os.path.exists(new_path):
            raise ValueError(f"{dataset_dir}: {original} renamed {new_id}, but {new_path} exists")
        changes.append((os.path.basename(path), os.path.basename(new_path), line_id, new_id, flag, new_flag))
        if dry_run:
            continue
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            ds = ds.load()
        ds.attrs["line ID"] = new_id
        ds.attrs["line ID flag"] = new_flag
        if new_id != original:
            ds.attrs["line ID original"] = original
        else:
            ds.attrs.pop("line ID original", None)
        write(ds, new_path)
        if new_path != path:
            os.remove(path)
    missing = sorted(set(lines) - seen)
    if missing:
        raise ValueError(f"{dataset_dir}: lines of {FILE} not in the files: {', '.join(missing)}")
    return changes
