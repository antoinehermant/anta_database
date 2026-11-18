"""Update the metadata of the v2 build from anta_json, without compiling again.

For the changes of anta_json that leave the values of the files as they are: ages and age ranges,
dating (method, ice core, chronology), firn correction, and the line IDs of line_ids.json. Takes
raw_md.json and line_ids.json of each dataset at --metadata-ref, sets the line IDs and flags
(anta_database.compiling.line_ids), then indexes the build again with assign_ages, which rewrites
IRH_AGE, its range and dating attributes and the firn correction of the files that changed, and
rebuilds the processing page. A change of the values (new raw files, compile settings) needs
build_v2.py instead.

    python docs/harmonisation/update_metadata.py --dry-run   # what would change
    python docs/harmonisation/update_metadata.py [--metadata-ref main] [--cpus 3]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v2 import BUILD, DATASETS, git  # noqa: E402

from anta_database.compiling import line_ids  # noqa: E402

FIELDS_SKIPPED = {"raw file"}


def changed_fields(old: list, new: list) -> list[str]:
    """What changed between two raw_md.json, per IRH (or raw file): "IRH 2: age -> age min, age max"."""
    key = lambda row: (row.get("IRH"), row.get("raw file"))  # noqa: E731
    old_rows, new_rows = {key(r): r for r in old}, {key(r): r for r in new}
    out = []
    for k in sorted(set(old_rows) | set(new_rows), key=str):
        a, b = old_rows.get(k, {}), new_rows.get(k, {})
        fields = sorted(f for f in set(a) | set(b) if f not in FIELDS_SKIPPED and a.get(f) != b.get(f))
        if fields:
            out.append(f"IRH {k[0]}{'' if k[1] is None else f' ({k[1]})'}: " + ", ".join(
                f"{f} {a.get(f)!r} -> {b.get(f)!r}" for f in fields))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--metadata-ref", default="main", help="Commit or branch of anta_json")
    parser.add_argument("--cpus", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="List the changes, write nothing")
    args = parser.parse_args()
    commit = git("rev-parse", args.metadata_ref).strip()
    files = set(git("ls-tree", "-r", "--name-only", commit).split())

    for name in args.datasets:
        out = BUILD / name
        new_md = git("show", f"{commit}:{name}/raw_md.json")
        old = json.loads((out / "raw_md.json").read_text(encoding="utf-8"))
        for line in changed_fields(old, json.loads(new_md)):
            print(f"{name}: {line}")
        ids = f"{name}/{line_ids.FILE}"
        ids_text = git("show", f"{commit}:{ids}") if ids in files else '{"lines": {}}'
        if not args.dry_run:
            (out / "raw_md.json").write_text(new_md, encoding="utf-8")
            if ids in files:
                (out / line_ids.FILE).write_text(ids_text, encoding="utf-8")
            else:
                (out / line_ids.FILE).unlink(missing_ok=True)
        changes = line_ids.apply(str(out), dry_run=args.dry_run, text=ids_text)
        for old_file, new_file, old_id, new_id, old_flag, new_flag in changes:
            print(f"{name}: line {old_id} ({old_flag}) -> {new_id} ({new_flag}), {old_file} -> {new_file}")

    if args.dry_run:
        print("Dry run: nothing written. The ages, dating and firn correction are rewritten by the index run.")
        return
    info_path = BUILD / "build_info.json"
    info = json.loads(info_path.read_text())
    info["metadata"] = {"ref": args.metadata_ref, "commit": commit}
    info_path.write_text(json.dumps(info, indent=1) + "\n")

    from anta_database.indexing.index_database import IndexDatabase
    import processing_page

    IndexDatabase(str(BUILD), assign_ages=True, cpus=args.cpus).index_database()
    processing_page.main(BUILD)


if __name__ == "__main__":
    main()
