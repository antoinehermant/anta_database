"""Compilation of one submission (run by worker.py, in a process of its own):

1. check the raw files again (checks.check_raw): errors stop here;
2. compile them with CompileDatabase, then IndexDatabase assigns the ages;
3. check the compiled dataset (checks.check_compiled): warnings for the review;
4. export it for the 3D viewer (anta_database.web.irh) into the private preview
   (<PORTAL_DATA>/previews/<id>/), served by the portal to whoever has the key;
5. store the compiled dataset and the report in the bucket of its purpose: compiled (to add
   to the database after review) or review (kept private).

The progress is written to submission.json (status "processing", "step"), for the preview page.

    python -m portal.pipeline <id>        (from the repository, or in the worker container)
"""

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import checks, config, storage

log = logging.getLogger("portal.pipeline")

# Steps shown on the preview page
STEPS = {
    "checking": "Checking the files",
    "compiling": "Compiling the dataset",
    "indexing": "Assigning the ages",
    "reviewing": "Checking the compiled dataset",
    "exporting": "Building the 3D view",
    "storing": "Storing the compiled dataset",
}


class CheckFailed(Exception):
    """The raw files have errors: nothing is compiled."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_record(directory: Path) -> dict:
    return json.loads((directory / "submission.json").read_text())


def write_record(directory: Path, record: dict) -> None:
    """Replace submission.json at once: the portal may read it at any time."""
    tmp = directory / ".submission.json.tmp"
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(directory / "submission.json")


def update(directory: Path, **changes) -> dict:
    record = read_record(directory)
    record.update(changes)
    write_record(directory, record)
    return record


def raw_files(directory: Path, record: dict) -> Path:
    """The dataset directory of a submission, with its raw files, from the bucket when they are
    no longer on the server (e.g. compiled again)."""
    dataset = directory / record["dataset"]
    if (dataset / "raw").is_dir():
        return dataset
    if not config.SUBMISSIONS_BUCKET:
        raise FileNotFoundError(f"{dataset}/raw: no raw files")
    storage.download(config.SUBMISSIONS_BUCKET, directory.name, directory)
    return dataset


def run(submission_id: str) -> dict:
    """Process a submission; its record, with the report."""
    from anta_database import CompileDatabase, IndexDatabase
    from anta_database.web.irh import Bedmap3Surface, _bbox, export_irh

    directory = config.SUBMISSIONS / submission_id
    record = read_record(directory)
    dataset_name = record["dataset"]
    work = config.WORK / submission_id
    preview = config.PREVIEWS / submission_id
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(preview, ignore_errors=True)
    report = checks.Report()

    def step(name: str) -> None:
        log.info("%s: %s", submission_id, STEPS[name])
        update(directory, status="processing", step=name, steps=list(STEPS), step_started=now())

    step("checking")
    source = raw_files(directory, record)
    raw = checks.check_raw(source, record)
    report.errors += raw.errors
    report.warnings += raw.warnings
    if report.errors:
        raise CheckFailed(report)

    step("compiling")
    dataset = work / dataset_name
    dataset.mkdir(parents=True)
    for name in ("raw", "raw_md.json", "original_new_column_names.csv"):
        (shutil.copytree if name == "raw" else shutil.copy)(source / name, dataset / name)
    CompileDatabase(
        [str(dataset)],
        extract_data=True,
        hdf5=True,
        netcdf=False,
        compute_IRH_density=True,
        compute_distance=True,
        compute_IMBIE_basins=True,
        set_attributes=True,
        shapefiles=False,
        geopackages=False,
        remove_tmp_files=True,
        break_transects=False,
        var_attrs_json=str(config.VAR_ATTRS),
    ).compile(cpus=1)

    step("indexing")
    IndexDatabase(str(work), assign_ages=True, cpus=1).index_database()

    step("reviewing")
    # Bedmap3 around the dataset only: the whole grid does not fit in the memory of the server
    bedmap = str(config.BEDMAP) if config.BEDMAP.exists() else None
    bbox = _bbox([str(dataset)])
    surface = Bedmap3Surface(bedmap, bbox) if bedmap and bbox else None
    compiled = checks.check_compiled(dataset, raw.summary["raw points"], surface)
    report.errors += compiled.errors
    report.warnings += compiled.warnings
    report.summary = compiled.summary
    if report.errors:
        raise CheckFailed(report)
    (work / "report.json").write_text(json.dumps(report.to_dict(), indent=2))

    step("exporting")
    export_irh(str(work), str(preview), [dataset_name], bedmap, crop=True)

    step("storing")
    shutil.rmtree(dataset / "raw")  # already in the bucket of the submissions
    bucket = config.COMPILED_BUCKET if record["licence"] == "redistribute" else config.REVIEW_BUCKET
    if bucket:
        storage.upload(work, bucket, submission_id)
        location = f"bucket {bucket}, {submission_id}/"
    else:
        kept = config.DATA / "compiled" / submission_id
        shutil.rmtree(kept, ignore_errors=True)
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work), kept)
        location = str(kept)
    shutil.rmtree(work, ignore_errors=True)
    if config.SUBMISSIONS_BUCKET:
        storage.clean(directory)

    return update(directory, status="ready", step=None, processed=now(), compiled=location, report=report.to_dict())


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.exit("python -m portal.pipeline <submission id>")
    try:
        run(args[0])
    except CheckFailed as e:
        report = e.args[0]
        update(config.SUBMISSIONS / args[0], status="failed", step=None, processed=now(), report=report.to_dict(),
               error="The files have errors: " + "; ".join(report.errors[:10]))
        sys.exit(2)


if __name__ == "__main__":
    main()
