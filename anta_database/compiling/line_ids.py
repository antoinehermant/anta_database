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
Lines given the same new ID are parts of one line, joined in the order of the file (the end of
one where the next starts; e.g. a traverse published in two files).
Applying it again changes nothing (the original ID is looked up too), so it runs after every
compilation and every metadata update, without recompiling.

The values of "line ID flag":
- original: the line ID of the radar survey, given with the published files or in the article;
- not_provided: none given, named by the database after the dataset (<Dataset>, <Dataset>_<n>);
- project_acq-year_number: none given, named after the project, year and a number;
- bedmap: taken from the Bedmap survey line of the same traces.
"""

import glob
import json
import os

import numpy as np
import xarray as xr

from anta_database.netcdf import write

FILE = "line_ids.json"
FLAGS = ("original", "not_provided", "project_acq-year_number", "bedmap")


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


def _join(parts: list[xr.Dataset]) -> xr.Dataset:
    """The parts of one line as one: points one after the other, the distance along the whole line,
    the variables of one part only filled with nan in the others, and the attributes merged."""
    irhs = {tuple(p["IRH"].values.tolist()) for p in parts if "IRH" in p.dims}
    if len(irhs) > 1:
        raise ValueError("parts of a line with other IRHs: not joined")
    names = {v: p[v] for p in parts for v in p.data_vars}
    filled = []
    for p in parts:
        for v, like in names.items():
            if v not in p.data_vars:
                shape = tuple(p.sizes[d] if d in p.dims else like.sizes[d] for d in like.dims)
                p[v] = (like.dims, np.full(shape, np.nan, "float32" if like.dtype.kind == "f" else "float64"), like.attrs)
        filled.append(p)
    ds = xr.concat(filled, dim="point", data_vars="minimal", coords="minimal", compat="override")
    ds = ds.assign_coords(point=np.arange(ds.sizes["point"], dtype="int32"))
    if "DIST" in ds:
        steps = np.hypot(np.diff(ds["PSX"].values), np.diff(ds["PSY"].values))
        ds["DIST"] = ("point", np.r_[0.0, np.cumsum(steps)], ds["DIST"].attrs)
    basins = {}
    for p in parts:
        basins.update(json.loads(p.attrs.get("IMBIE_basins", "{}")))
    if basins:
        ds.attrs["IMBIE_basins"] = json.dumps(dict(sorted(basins.items())))
    for key in ("processing", "reconnection"):
        texts = list(dict.fromkeys(str(p.attrs[key]) for p in parts if p.attrs.get(key)))
        if len(texts) > 1 and key == "processing":
            ds.attrs[key] = max(texts, key=len) if all(t in max(texts, key=len) for t in texts) else " ".join(texts)
        elif texts:
            ds.attrs[key] = "; ".join(texts)
    return ds


def apply(dataset_dir: str, dry_run: bool = False, text: str | None = None) -> list[tuple]:
    """Set the line IDs and flags of line_ids.json in the files of a dataset (or of `text`, the
    content of another line_ids.json). Returns (old file, new file, old ID, new ID, old flag, new
    flag) per file changed, or that would change with dry_run."""
    lines = load(dataset_dir, text)
    if not lines:
        return []
    changes = []
    seen = set()
    groups: dict[str, list] = {}  # new ID -> [(path, line ID, flag, original)], in the order of the file
    order = {old: i for i, old in enumerate(lines)}
    for path in sorted(glob.glob(os.path.join(dataset_dir, "nc", "*.nc"))):
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            line_id = str(ds.attrs.get("line ID"))
            flag = str(ds.attrs.get("line ID flag"))
            originals = str(ds.attrs.get("line ID original", line_id)).split(", ")
        known = [o for o in originals if o in lines]
        if not known:
            continue
        seen.update(known)
        target = lines[known[0]]
        groups.setdefault(str(target.get("line ID", line_id)), []).append((path, line_id, flag, originals))
    for new_id, members in groups.items():
        members.sort(key=lambda m: min(order.get(o, len(order)) for o in m[3]))
        path, line_id, flag, _ = members[0]
        original = ", ".join(o for m in members for o in m[3])
        new_flag = lines[members[0][3][0]].get("line ID flag", flag) if members[0][3][0] in lines else flag
        new_path = os.path.join(dataset_dir, "nc", file_name(new_id))
        if len(members) == 1 and (new_id, new_flag) == (line_id, flag) and new_path == path:
            continue
        if new_path not in [m[0] for m in members] and os.path.exists(new_path):
            raise ValueError(f"{dataset_dir}: {original} renamed {new_id}, but {new_path} exists")
        for m in members:
            changes.append((os.path.basename(m[0]), os.path.basename(new_path), m[1], new_id, m[2], new_flag))
        if dry_run:
            continue
        parts = []
        for m in members:
            with xr.open_dataset(m[0], engine="h5netcdf") as ds:
                parts.append(ds.load())
        ds = parts[0] if len(parts) == 1 else _join(parts)
        ds.attrs["line ID"] = new_id
        ds.attrs["line ID flag"] = new_flag
        if new_id != original:
            ds.attrs["line ID original"] = original
        else:
            ds.attrs.pop("line ID original", None)
        write(ds, new_path)
        for m in members:
            if m[0] != new_path:
                os.remove(m[0])
    missing = sorted(set(lines) - seen)
    if missing:
        raise ValueError(f"{dataset_dir}: lines of {FILE} not in the files: {', '.join(missing)}")
    return changes
