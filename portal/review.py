#!/usr/bin/env python
"""Review of the submissions, on the maintainer's computer (not on the server, which has no
keys for the public bucket). From any directory, as anta-review (a link to this file in
~/.local/bin, made once with: ln -s "$PWD/portal/review.py" ~/.local/bin/anta-review), or from
the repository as python -m portal.review:

    python -m portal.review init [--release-dir DIR]   set up the release copy of the database (once)
    python -m portal.review list                      the submissions on the server
    python -m portal.review show <id>                 checks report, contributor, notes, preview link
    python -m portal.review pull <id> [--to DIR]      the compiled dataset and its raw files, into ./<id>/
    python -m portal.review accept <id>               add it to the database and publish it (below)
    python -m portal.review reject <id>               refuse it, with a message to the contributor (data kept)
    python -m portal.review done <id>                 end a review-only submission, with a message (data deleted)
    python -m portal.review purge <id>                delete the data of a refused submission
    python -m portal.review retry <id>                compile it again on the server
    python -m portal.review delete <id>               delete a test submission (from /submit/?dev) completely
    python -m portal.review remove <Dataset>          take a published dataset out of the database again

accept:
1. makes the release copy identical to the published database: sync, then a check of every file;
2. adds the compiled dataset to it, and rebuilds the index (the ages are already in the files);
3. exports the data of the 3D viewer and of the download page;
4. shows what changes, and asks before publishing the database and these data to the bucket;
5. checks them online, then publishes the raw files of the submission to raw/<Dataset>/ in
   the bucket (below);
6. closes the submission on the server (preview and compiled copy removed, raw files kept,
   contributor told), and adds the dataset to the docs (book/intro.md, book/changelog.md).
It can be run again after a failure: it goes on from where it stopped.
With --test, it publishes to test/ in the bucket, leaves the submission and the docs alone.

The raw files: raw/<Dataset>/ in the public bucket holds the files as the contributor sent them
(raw/), the inputs of CompileDatabase made from the form (raw_md.json,
original_new_column_names.csv) and provenance.json (the submission, without the contact of
the contributor). It is outside the database prefixes (v1/, v2/...): one copy for all the
versions of the data, never in a manifest, so sync() and the download page leave it alone.
remove deletes it with the dataset.

The release copy (default ~/data/isochrones/AntADatabase-release) is only written by this
script, and made identical to the bucket before each change. Settings in
~/.config/antadatabase/review.json: release_dir, bedmap (bedmap3.nc, for the 3D viewer), host
(of the portal), public_profile and portal_profile (AWS profiles, ~/.aws/credentials).
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path.home() / ".config" / "antadatabase" / "review.json"
DEFAULTS = {
    "release_dir": str(Path.home() / "data" / "isochrones" / "AntADatabase-release"),
    "bedmap": str(Path.home() / "data" / "bedmap" / "bedmap3.nc"),
    "host": "ubuntu@api.antadatabase.com",
    "public_profile": "infomaniak",
    "portal_profile": "antadatabase-portal",
}
ENDPOINT = "https://s3.pub1.infomaniak.cloud"
BUCKET = "antadatabase"
# Marks a dataset of the release copy added from a submission: <Dataset>/.submission
MARKER = ".submission"
# Bucket prefix of the raw files of the datasets, raw/<Dataset>/ (see above)
RAW_PREFIX = "raw"
CONTENT_TYPES = {".csv": "text/csv", ".tab": "text/tab-separated-values", ".json": "application/json"}


def settings() -> dict:
    saved = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
    return DEFAULTS | saved


def s3(profile: str):
    import boto3
    from botocore.config import Config

    return boto3.Session(profile_name=profile).client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name="us-east-1",
        config=Config(request_checksum_calculation="when_required", response_checksum_validation="when_required"),
    )


def step(text: str) -> None:
    print(f"\n== {text}")


def confirm(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


# --- The server (portal/manage.py, over SSH)


def manage(*args: str, stdin: str | None = None, capture: bool = True) -> str:
    cfg = settings()
    command = "cd /opt/portal && docker compose exec -T worker python -m portal.manage " + " ".join(map(shlex.quote, args))
    # ssh would otherwise read the terminal, e.g. the answer to the next question
    result = subprocess.run(["ssh", cfg["host"], command], text=True, capture_output=capture,
                            **({"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}))
    if result.returncode:
        sys.exit(f"On the server: {(result.stderr or result.stdout or '').strip()}")
    return result.stdout if capture else ""


def record(submission_id: str) -> dict:
    return json.loads(manage("show", submission_id))


# --- Commands


def cmd_init(args) -> None:
    cfg = settings()
    if args.release_dir:
        cfg["release_dir"] = str(Path(args.release_dir).expanduser().resolve())
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    print(f"Settings in {CONFIG_FILE}:\n{json.dumps(cfg, indent=2)}")
    for name in ("public_profile", "portal_profile"):
        try:
            s3(cfg[name]).list_objects_v2(Bucket=BUCKET if name == "public_profile" else "antadatabase-submissions", MaxKeys=1)
        except Exception as e:
            sys.exit(f"The AWS profile {cfg[name]} ({name}) does not work: {e}")
    manage("list")
    print("The bucket keys and the SSH access to the portal work.")
    release_copy(Path(cfg["release_dir"]))


def cmd_list(args) -> None:
    rows = json.loads(manage("list", "--json"))
    if not rows:
        print("No submission.")
    for r in rows:
        purpose = "review only" if r["licence"] == "review" else "redistribute"
        note = " [test]" if r.get("test") else ""
        if r["status"] == "rejected" and r.get("rejected"):
            days = (datetime.now(timezone.utc) - datetime.fromisoformat(r["rejected"])).days
            if days >= 90:
                note += f"  (rejected {days} days ago: purge?)"
        print(f"{r['id']}  {r['status']:<10}  {r['dataset']:<16}  {purpose:<12}  {r['email']}{note}")


def cmd_show(args) -> None:
    r = record(args.id)
    contributor = r["contributor"]
    report = r.get("report") or {}
    summary = _summary(report)
    print(f"{r['id']}: {r['citation']} ({r['dataset']}), {r['status']}{' [test]' if r.get('test') else ''}")
    print(f"Purpose:     {'redistribute after review' if r['licence'] == 'redistribute' else 'review only'}")
    print(f"From:        {contributor['name']} <{contributor['email']}>, {contributor.get('affiliation') or '-'}"
          f"{', ORCID ' + contributor['orcid'] if contributor.get('orcid') else ''}")
    print(f"Publication: {r['publication'].get('doi_publication') or '-'}")
    print(f"Data:        {r['publication'].get('doi_dataset') or '-'}")
    print(f"Received:    {r['received']}, {len(r['files'])} files ({r['bytes'] / 1e6:.1f} MB, "
          f"{'one per layer' if r['structure'] == 'layer' else 'one per line'})")
    if summary.get("lines"):
        print(f"Compiled:    {summary['lines']} lines, {summary['IRHs']} IRHs, {summary['IRH points']:,} IRH points")
    print(f"Preview:     {r.get('preview') or '-'}")
    if r.get("error"):
        print(f"Error:       {r['error']}")
    for title, items in (("Errors", report.get("errors")), ("Warnings", report.get("warnings")), ("Form warnings", r.get("warnings"))):
        if items:
            print(f"{title}:")
            for item in items:
                print(f"  - {item}")
    print(f"Notes:\n  {(r.get('notes') or '-').strip()}")


def cmd_pull(args) -> None:
    r = record(args.id)
    cfg = settings()
    target = Path(args.to or ".").expanduser().resolve() / args.id
    client = s3(cfg["portal_profile"])
    n = download(client, r["compiled_bucket"], args.id, target)
    raw = download(client, r["submissions_bucket"], f"{args.id}/{r['dataset']}/raw", target / r["dataset"] / "raw")
    if not n:
        print(f"No compiled data (the submission is {r['status']}): only the raw files")
    print(f"{target}: {n} compiled files, {raw} raw files. In Python: Database({str(target)!r})")


def download(client, bucket: str, prefix: str, target: Path) -> int:
    n = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for item in page.get("Contents", []):
            rel = item["Key"].removeprefix(f"{prefix}/")
            if not rel or ".." in Path(rel).parts or rel == "submission.json":
                continue
            path = target / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, item["Key"], str(path))
            n += 1
    return n


def edit_message(r: dict, template: str) -> str:
    """The message to the contributor: --message, or written in $EDITOR."""
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write(template)
        path = f.name
    editor = shlex.split(os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano")
    if os.path.basename(editor[0]) == "emacsclient":
        # Wait for the message: without -n (--no-wait), finished with C-x #
        editor = [a for a in editor if a not in ("-n", "--no-wait")]
        print("Write the message in Emacs, save it, then C-x # to send it from here.")
    subprocess.run([*editor, path], check=True)
    text = "\n".join(l for l in Path(path).read_text().splitlines() if not l.startswith("#")).strip()
    os.unlink(path)
    return text


def send_decision(args, command: str, template: str) -> None:
    r = record(args.id)
    print(f"{r['id']}: {r['citation']} ({r['dataset']}), {r['status']}, to {r['contributor']['email']}")
    text = args.message or edit_message(r, template.format(**r, name=r["contributor"]["name"]))
    if not text:
        sys.exit("No message: nothing done")
    print(f"\n{text}\n")
    if not confirm(f"Send this and {command} {args.id}?"):
        sys.exit("Nothing done")
    print(manage(command, args.id, stdin=text).strip())


def cmd_reject(args) -> None:
    send_decision(args, "reject", (
        "# Your message to the contributor, sent by email (lines with # are left out).\n"
        "# The data of the submission is kept, until: python -m portal.review purge <id>\n"
        "Thank you for your submission {id} ({dataset}). After review, we cannot add it to the AntADatabase "
        "as it is, because:\n\n- \n\n"
        "You are welcome to send a corrected version as a new submission.\n"
    ))


def cmd_done(args) -> None:
    send_decision(args, "done", (
        "# Your message to the contributor, sent by email (lines with # are left out).\n"
        "# All the data of this review-only submission is then deleted.\n"
        "We reviewed your submission {id} ({dataset}), as you asked:\n\n- \n\n"
        "As agreed, it was not added to the AntADatabase, and all its data is now deleted.\n"
    ))


def cmd_purge(args) -> None:
    r = record(args.id)
    print(f"{r['id']}: {r['citation']} ({r['dataset']}), {r['status']}")
    if confirm(f"Delete all the data of {args.id}?"):
        print(manage("purge", args.id).strip())


def cmd_delete(args) -> None:
    r = record(args.id)
    print(f"{r['id']}: {r['citation']} ({r['dataset']}), {r['status']}{' [test]' if r.get('test') else ''}")
    if not r.get("test"):
        sys.exit("Not a test submission: reject it, then purge it")
    if args.yes or confirm(f"Delete the test submission {args.id} completely?"):
        print(manage("delete", args.id).strip())


def cmd_retry(args) -> None:
    print(manage("retry", args.id).strip())


# --- accept


def public_url(prefix: str) -> str:
    from anta_database.cloud.sync import BUCKET_URL

    return f"{BUCKET_URL}/{prefix}"


def db_prefix() -> str:
    """Bucket prefix of the published database: the data schema of the package (v1, v2...)."""
    from anta_database.cloud.manifest import SCHEMA_VERSION

    return f"v{SCHEMA_VERSION}"


def published_datasets(manifest: dict) -> set[str]:
    return {rel.split("/")[0] for rel in manifest["files"] if "/" in rel}


def release_copy(release: Path, keep: str | None = None) -> dict:
    """Make the release copy identical to the published database: its manifest. Datasets added
    by an earlier, unfinished accept are removed, but `keep` (a submission id)."""
    from anta_database.cloud.manifest import list_published_files, sha256sum
    from anta_database.cloud.sync import _fetch_manifest, sync

    step(f"Release copy {release}: identical to the published database")
    remote = _fetch_manifest(public_url(db_prefix()))
    release.mkdir(parents=True, exist_ok=True)
    for marker in release.glob(f"*/{MARKER}"):
        dataset = marker.parent
        if dataset.name not in published_datasets(remote) and marker.read_text().strip() != keep:
            print(f"Removing {dataset.name}, added by an unfinished accept of {marker.read_text().strip()}")
            shutil.rmtree(dataset)
    extra = sorted(set(list_published_files(str(release))) - set(remote["files"]))
    extra = [rel for rel in extra if not (keep and (release / rel.split("/")[0] / MARKER).exists()
                                          and (release / rel.split("/")[0] / MARKER).read_text().strip() == keep)]
    if extra:
        sys.exit(f"The release copy has files that are not published, remove them first:\n  " + "\n  ".join(extra[:20]))
    sync(str(release), url=public_url(db_prefix()))
    # sync trusts its own record of the files: check them all, a file may have been edited
    changed = [rel for rel, entry in remote["files"].items()
               if not (release / rel).is_file() or sha256sum(str(release / rel)) != entry["sha256"]]
    if changed:
        print(f"{len(changed)} files differ from the published ones: downloaded again")
        for rel in changed:
            (release / rel).unlink(missing_ok=True)
        (release / "manifest.json").unlink(missing_ok=True)
        sync(str(release), url=public_url(db_prefix()))
    print(f"Release copy identical to the published database {remote.get('db_version')} ({len(remote['files'])} files)")
    return remote


def docs_line(r: dict) -> str:
    """The line of the dataset in the list of book/intro.md."""
    citation = re.sub(r"\s*\((\d{4}[a-z]?)\)$", r" \1", r["citation"])
    doi = r["publication"].get("doi_dataset") or r["publication"].get("doi_publication") or ""
    if not doi:
        return f"- {citation}"
    url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    label = re.sub(r"^https?://(dx\.)?doi(\.pangaea)?\.(org|de)/", "", url)
    return f"- {citation}, [{label}]({url})"


def _summary(report: dict) -> dict:
    """Summary of a report; "flight lines" in reports made before lines were renamed."""
    summary = dict(report.get("summary", {}))
    if "flight lines" in summary:
        summary["lines"] = summary.pop("flight lines")
    return summary


def update_docs(r: dict, version: str, summary: dict) -> None:
    step("Docs: list of datasets (book/intro.md) and changelog (book/changelog.md)")
    intro = REPO / "book" / "intro.md"
    text = intro.read_text()
    line = docs_line(r)
    lines = text.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("## Included Datasets"))
    items = [i for i in range(start, len(lines)) if lines[i].startswith("- ")]
    items = items[: next((k for k, i in enumerate(items) if k and i != items[k - 1] + 1), len(items))]  # first list only
    if any(lines[i].startswith(line.split(",")[0] + ",") or lines[i].strip() == line for i in items):
        print("Already in book/intro.md")
    else:
        # Alphabetical place, the other lines as they are
        at = next((i for i in items if lines[i][2:].lower() > line[2:].lower()), items[-1] + 1)
        lines.insert(at, line + "\n")
        intro.write_text("".join(lines))
        print(f"book/intro.md: {line}")

    changelog = REPO / "book" / "changelog.md"
    if not changelog.exists():
        changelog.write_text(
            "# Changelog\n\nVersions of the published database: `anta_database.sync()` downloads the latest.\n"
        )
        toc = REPO / "book" / "_toc.yml"
        toc_text = toc.read_text()
        if "file: changelog" not in toc_text:
            toc.write_text(toc_text.replace("- file: contact", "- file: changelog\n  title: Changelog\n- file: contact"))
    text = changelog.read_text()
    entry = (f"\n## {version}\n\n- New dataset {r['dataset']}: {line[2:]}"
             + (f" ({summary['lines']} lines, {summary['IRHs']} IRHs)" if summary.get("lines") else "")
             + "\n")
    if f"New dataset {r['dataset']}:" in text:
        print("Already in book/changelog.md")
    else:
        head, sep, rest = text.partition("\n## ")
        changelog.write_text(head.rstrip("\n") + "\n" + entry + (sep + rest if sep else ""))
        print(f"book/changelog.md: {version}")


def upload_web(stage: Path, prefix: str, profile: str) -> int:
    """Upload the exported viewer data (irh/, catalog/) under <prefix>/ of the public bucket,
    and delete the files of irh/ that are no longer exported."""
    import mimetypes

    client = s3(profile)
    n = 0
    for part in ("irh", "catalog"):
        local = {p.relative_to(stage / part).as_posix(): p for p in (stage / part).rglob("*") if p.is_file()}
        for rel, path in sorted(local.items()):
            client.upload_file(str(path), BUCKET, f"{prefix}/{part}/{rel}", ExtraArgs={
                "ContentType": mimetypes.guess_type(rel)[0] or "application/octet-stream", "CacheControl": "no-cache"})
            n += 1
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{prefix}/{part}/"):
            for item in page.get("Contents", []):
                if item["Key"].removeprefix(f"{prefix}/{part}/") not in local:
                    client.delete_object(Bucket=BUCKET, Key=item["Key"])
    return n


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Cache-Control": "no-cache"}), timeout=60) as r:
        return json.load(r)


def publish_release(release: Path, cfg: dict, args, name: str, present: bool, command: str) -> str:
    """Index the release copy, export the data of the 3D viewer and of the download page, show
    what changes and ask, publish, then check online that the dataset `name` is there
    (present) or gone. The new database version."""
    from anta_database import IndexDatabase
    from anta_database.cloud.publish import publish
    from anta_database.web.catalog import export_catalog
    from anta_database.web.irh import export_irh

    database_prefix, web_prefix = (f"test/{db_prefix()}", "test/web") if args.test else (db_prefix(), "web")
    step("Index (AntADatabase.db)")
    IndexDatabase(str(release)).index_database()

    step("Data of the 3D viewer and of the download page")
    stage = release.parent / f".{release.name}.web"
    shutil.rmtree(stage, ignore_errors=True)
    index = export_irh(str(release), str(stage / "irh"), None, cfg["bedmap"])
    export_catalog(str(release), str(stage / "catalog"))
    if (name in index["datasets"]) != present:
        sys.exit(f"{name} is {'not ' if present else ''}in the export of the 3D viewer")

    os.environ["AWS_PROFILE"] = cfg["public_profile"]
    plan = publish(str(release), BUCKET, database_prefix, db_version=args.db_version, dry_run=True)
    step(f"To publish to {BUCKET}/{database_prefix} and {BUCKET}/{web_prefix}" + (" (TEST)" if args.test else ""))
    print(f"Database version {plan['db_version']}: {len(plan['uploaded'])} files to upload, {len(plan['deleted'])} to delete")
    for rel in [*plan["uploaded"][:8], *(["..."] if len(plan["uploaded"]) > 8 else [])]:
        print(f"  + {rel}")
    for rel in [*plan["deleted"][:8], *(["..."] if len(plan["deleted"]) > 8 else [])]:
        print(f"  - {rel}")
    print(f"3D viewer and download page: {sum(1 for _ in stage.rglob('*') if _.is_file())} files")
    if not args.yes and not confirm("Publish?"):
        sys.exit(f"Not published. Run {command} again to go on.")

    version = plan["db_version"]
    publish(str(release), BUCKET, database_prefix, db_version=version)
    print(f"Viewer data: {upload_web(stage, web_prefix, cfg['public_profile'])} files uploaded")
    shutil.rmtree(stage, ignore_errors=True)

    step("Online check")
    manifest = fetch_json(f"{public_url(database_prefix)}/manifest.json")
    irh = fetch_json(f"{public_url(web_prefix)}/irh/irh.json")
    catalog = fetch_json(f"{public_url(web_prefix)}/catalog/catalog.json")
    problems = [what for what, there in (
        ("manifest", name in published_datasets(manifest)),
        ("3D viewer", name in irh["datasets"]),
        ("download page", any(d["name"] == name for d in catalog["datasets"])),
    ) if there != present]
    if manifest.get("db_version") != version:
        problems.append("manifest version")
    if problems:
        sys.exit(f"Not online as expected: {', '.join(problems)}. Run {command} again.")
    print(f"{name} is {'online' if present else 'removed'} in database version {version}")
    return version


def raw_prefix(name: str, args) -> str:
    return f"{'test/' if args.test else ''}{RAW_PREFIX}/{name}"


def list_keys(client, prefix: str) -> list[str]:
    return [item["Key"] for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{prefix}/")
            for item in page.get("Contents", [])]


def provenance(r: dict, version: str) -> dict:
    """Where the raw files of a dataset come from: its submission, without the contact of the
    contributor (the bucket is public)."""
    keys = ("name", "kind", "irh", "variable", "irh_columns", "columns", "rows", "acquisition_year", "project")
    return {
        "dataset": r["dataset"],
        "citation": r["citation"],
        "DOI publication": r["publication"].get("doi_publication") or None,
        "DOI dataset": r["publication"].get("doi_dataset") or None,
        "submission": r["id"],
        "received": r["received"],
        "database version": version,
        "file structure": "one file per layer" if r["structure"] == "layer" else "one file per line",
        "files": [{k: f[k] for k in keys if f.get(k) not in (None, [])} for f in r["files"]],
        "layers": r.get("layers") or [],
        "survey": r.get("survey") or {},
        "notes": (r.get("notes") or "").strip(),
    }


def publish_raw(r: dict, cfg: dict, args, version: str) -> None:
    """Publish the raw files of a submission to raw/<Dataset>/ of the public bucket, in place of
    those of an earlier version of the dataset."""
    name = r["dataset"]
    prefix = raw_prefix(name, args)
    step(f"Raw files: {BUCKET}/{prefix}/")
    stage = Path(tempfile.mkdtemp(prefix=f"anta-raw-{r['id']}-"))
    try:
        download(s3(cfg["portal_profile"]), r["submissions_bucket"], f"{r['id']}/{name}", stage)
        if not any((stage / "raw").glob("*")):
            sys.exit(f"No raw files of {r['id']} in {r['submissions_bucket']}")
        (stage / "provenance.json").write_text(json.dumps(provenance(r, version), indent=2, ensure_ascii=False))
        local = {p.relative_to(stage).as_posix(): p for p in sorted(stage.rglob("*")) if p.is_file()}
        client = s3(cfg["public_profile"])
        for rel, path in local.items():
            client.upload_file(str(path), BUCKET, f"{prefix}/{rel}",
                               ExtraArgs={"ContentType": CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")})
        stale = [key for key in list_keys(client, prefix) if key.removeprefix(f"{prefix}/") not in local]
        for key in stale:
            client.delete_object(Bucket=BUCKET, Key=key)
        if set(list_keys(client, prefix)) != {f"{prefix}/{rel}" for rel in local}:
            sys.exit(f"The raw files in {BUCKET}/{prefix}/ are not as uploaded. Run accept again.")
        size = sum(p.stat().st_size for p in local.values())
        print(f"{len(local)} files ({size / 1e6:.1f} MB) in {public_url(prefix)}/"
              + (f", {len(stale)} of the earlier version deleted" if stale else ""))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def delete_raw(name: str, cfg: dict, args) -> None:
    prefix = raw_prefix(name, args)
    client = s3(cfg["public_profile"])
    keys = list_keys(client, prefix)
    for key in keys:
        client.delete_object(Bucket=BUCKET, Key=key)
    if keys:
        print(f"Raw files: {len(keys)} deleted from {BUCKET}/{prefix}/ (still in the bucket of the submissions)")


def cmd_accept(args) -> None:
    cfg = settings()
    release = Path(cfg["release_dir"] + ("-test" if args.test else ""))
    r = record(args.id)
    name = r["dataset"]
    print(f"{r['id']}: {r['citation']} ({name}), {r['status']}")
    if r["status"] == "accepted":
        sys.exit("Already accepted")
    if r["licence"] != "redistribute":
        sys.exit("A review-only submission is never added to the database: python -m portal.review done <id>")
    if r["status"] != "ready":
        sys.exit(f"Only compiled submissions are accepted (this one is {r['status']})")
    if r.get("test") and not args.test:
        sys.exit("A test submission (sent from /submit/?dev): only with --test")
    if not Path(cfg["bedmap"]).exists():
        sys.exit(f"No Bedmap3 at {cfg['bedmap']} (for the 3D viewer): set 'bedmap' in {CONFIG_FILE}")

    remote = release_copy(release, keep=args.id)
    dataset = release / name
    published = name in published_datasets(remote)
    ours = (dataset / MARKER).exists() and (dataset / MARKER).read_text().strip() == args.id
    if published and not ours and not args.replace:
        sys.exit(f"{name} is already in the database: --replace to replace it with this submission")

    step(f"Adding {name} from the bucket {r['compiled_bucket']}")
    tmp = release.parent / f".{release.name}.{args.id}"
    shutil.rmtree(tmp, ignore_errors=True)
    n = download(s3(cfg["portal_profile"]), r["compiled_bucket"], f"{args.id}/{name}", tmp)
    if not n or not list((tmp / "h5").glob("*.h5")):
        sys.exit(f"No compiled data of {args.id} in {r['compiled_bucket']}")
    (tmp / MARKER).write_text(args.id)
    shutil.rmtree(dataset, ignore_errors=True)
    tmp.rename(dataset)
    print(f"{len(list((dataset / 'h5').glob('*.h5')))} lines")

    version = publish_release(release, cfg, args, name, present=True, command="accept")
    publish_raw(r, cfg, args, version)
    if args.test:
        print("\nTest: the submission stays open on the server, the docs are unchanged. Clean up with:\n"
              f"  aws --profile {cfg['public_profile']} --endpoint-url {ENDPOINT} s3 rm --recursive s3://{BUCKET}/test/")
        return
    step("Closing the submission on the server")
    print(manage("accept", args.id, version).strip())
    update_docs(r, version, _summary(r.get("report") or {}))
    print("\nDone. Publish the docs:\n"
          f"  git add book/intro.md book/changelog.md book/_toc.yml && git commit -m 'data: add {name}'\n"
          "  ./publish_site.sh")


def remove_from_docs(name: str, citation: str, version: str, reason: str) -> None:
    step("Docs: list of datasets (book/intro.md) and changelog (book/changelog.md)")
    intro = REPO / "book" / "intro.md"
    lines = intro.read_text().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("## Included Datasets"))
    short = re.sub(r"\s*\((\d{4}[a-z]?)\)$", r" \1", citation)
    found = [i for i in range(start, len(lines)) if lines[i].startswith(f"- {short},") or lines[i].strip() == f"- {short}"]
    if found:
        print(f"book/intro.md: removed {lines[found[0]].strip()}")
        del lines[found[0]]
        intro.write_text("".join(lines))
    else:
        print(f"No line for {short} in book/intro.md: check its list of datasets by hand")
    changelog = REPO / "book" / "changelog.md"
    if changelog.exists():
        text = changelog.read_text()
        head, sep, rest = text.partition("\n## ")
        entry = f"\n## {version}\n\n- Removed dataset {name} ({citation}){': ' + reason if reason else ''}\n"
        changelog.write_text(head.rstrip("\n") + "\n" + entry + (sep + rest if sep else ""))
        print(f"book/changelog.md: {version}")


def cmd_remove(args) -> None:
    cfg = settings()
    release = Path(cfg["release_dir"] + ("-test" if args.test else ""))
    name = args.dataset
    remote = release_copy(release)
    if name not in published_datasets(remote):
        sys.exit(f"{name} is not in the published database")
    md = json.loads((release / name / "raw_md.json").read_text()) if (release / name / "raw_md.json").exists() else []
    citation = (md[0] if isinstance(md, list) and md else md or {}).get("citation", name)
    # The submission that brought it, if it came through the portal
    submissions = [r for r in json.loads(manage("list", "--json")) if r["dataset"] == name and r["status"] == "accepted"]
    print(f"{name}: {citation}" + (f", from the submission {submissions[0]['id']}" if submissions else ""))
    if not args.yes and not confirm(f"Take {name} out of the release copy, to publish without it?"):
        sys.exit("Nothing done")
    shutil.rmtree(release / name)
    version = publish_release(release, cfg, args, name, present=False, command="remove")
    delete_raw(name, cfg, args)
    if args.test:
        print("\nTest: the submission and the docs are unchanged. Clean up with:\n"
              f"  aws --profile {cfg['public_profile']} --endpoint-url {ENDPOINT} s3 rm --recursive s3://{BUCKET}/test/")
        return
    for r in submissions:
        step(f"Submission {r['id']} on the server")
        print(manage("withdraw", r["id"], stdin=args.message or "").strip())
    remove_from_docs(name, citation, version, args.reason or "")
    print("\nDone. Users get the new version with sync(), which removes the dataset from their copy too.\n"
          "Publish the docs:\n"
          f"  git add book/intro.md book/changelog.md && git commit -m 'data: remove {name}'\n"
          "  ./publish_site.sh")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="anta-review", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="set up the release copy of the database")
    p.add_argument("--release-dir", help=f"default: {DEFAULTS['release_dir']}")
    p.set_defaults(run=cmd_init)
    sub.add_parser("list", help="the submissions").set_defaults(run=cmd_list)
    for command, run, text in (
        ("show", cmd_show, "checks report, contributor, notes, preview link"),
        ("pull", cmd_pull, "the compiled dataset and its raw files, into ./<id>/"),
        ("accept", cmd_accept, "add it to the database and publish it"),
        ("reject", cmd_reject, "refuse it (data kept)"),
        ("done", cmd_done, "end a review-only submission (data deleted)"),
        ("purge", cmd_purge, "delete the data of a refused submission"),
        ("retry", cmd_retry, "compile it again on the server"),
        ("delete", cmd_delete, "delete a test submission completely"),
    ):
        p = sub.add_parser(command, help=text)
        p.add_argument("id")
        p.set_defaults(run=run)
        if command == "pull":
            p.add_argument("--to", help="directory (default: the current one)")
        if command == "delete":
            p.add_argument("--yes", action="store_true", help="without asking")
        if command in ("reject", "done"):
            p.add_argument("--message", help="the message to the contributor (default: written in $EDITOR)")
        if command == "accept":
            p.add_argument("--replace", action="store_true", help="replace a dataset of the same name")
            p.add_argument("--db-version", help="version of the database (default: the date and time)")
            p.add_argument("--yes", action="store_true", help="publish without asking")
            p.add_argument("--test", action="store_true", help="publish to test/ in the bucket, leave the submission and the docs")
    p = sub.add_parser("remove", help="take a published dataset out of the database again")
    p.add_argument("dataset")
    p.add_argument("--reason", help="for the changelog of the docs")
    p.add_argument("--message", help="email to the contributor, if it came through the portal (default: none)")
    p.add_argument("--db-version", help="version of the database (default: the date and time)")
    p.add_argument("--yes", action="store_true", help="without asking")
    p.add_argument("--test", action="store_true", help="publish to test/ in the bucket, leave the submission and the docs")
    p.set_defaults(run=cmd_remove)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
