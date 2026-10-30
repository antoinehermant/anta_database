"""The submissions on the server, for the maintainer. Usually run from the laptop through
portal/review.py, which calls it over SSH:

    python -m portal.manage list [--json]          id, status, dataset, purpose, contributor
    python -m portal.manage show <id>              the record (JSON), with the report and the preview link
    python -m portal.manage retry <id>             compile again (e.g. after a fix of the package)
    python -m portal.manage accept <id> <version>  published in the database: preview and compiled copy removed,
                                                   the raw files kept, the contributor told
    python -m portal.manage reject <id>            refused (message on stdin, emailed): everything kept for now
    python -m portal.manage done <id>              review-only submission reviewed (message on stdin, emailed):
                                                   everything deleted, as the contributor was promised
    python -m portal.manage purge <id>             delete the data of a refused (or failed) submission
    python -m portal.manage withdraw <id>          its dataset was removed from the database again (message on
                                                   stdin, emailed if any); the raw files kept
    python -m portal.manage delete <id>            delete a test submission (from /submit/?dev) completely

On the server: cd /opt/portal && docker compose exec worker python -m portal.manage list
"""

import json
import shutil
import sys

from . import config, mail, storage
from .pipeline import now, read_record, update


def records() -> list[dict]:
    out = []
    for path in sorted(config.SUBMISSIONS.glob("*/submission.json")):
        try:
            out.append(json.loads(path.read_text()))
        except ValueError:
            pass
    return out


def preview_url(record: dict) -> str | None:
    if not record.get("preview_key") or not (config.PREVIEWS / record["id"]).exists():
        return None
    return f"{config.SITE_URL}/viewer/?preview={record['id']}&key={record['preview_key']}"


def compiled_bucket(record: dict) -> str:
    return config.COMPILED_BUCKET if record.get("licence") == "redistribute" else config.REVIEW_BUCKET


def delete_data(record: dict, raw: bool) -> None:
    """Remove the preview and the compiled copy of a submission, and with raw its files."""
    submission_id = record["id"]
    shutil.rmtree(config.PREVIEWS / submission_id, ignore_errors=True)
    shutil.rmtree(config.WORK / submission_id, ignore_errors=True)
    shutil.rmtree(config.DATA / "compiled" / submission_id, ignore_errors=True)
    storage.delete(compiled_bucket(record), submission_id)
    if raw:
        storage.delete(config.SUBMISSIONS_BUCKET, submission_id)
        storage.clean(config.SUBMISSIONS / submission_id)


def email_contributor(record: dict, subject: str, body: str) -> None:
    contributor = record["contributor"]
    mail.send(
        contributor["email"],
        f"{'[test] ' if record.get('test') else ''}{subject}",
        f"Dear {contributor['name']},\n\n{body.strip()}\n\n"
        f"For any question, reply to this email or use {config.SITE_URL}/contact/.\n\nAntADatabase\n",
        reply_to=config.MAINTAINER_EMAIL,
    )


def message() -> str:
    """The maintainer's message, from stdin."""
    text = sys.stdin.read().strip()
    if not text:
        sys.exit("No message (on stdin)")
    return text


def main(argv=None) -> None:
    args = sys.argv[1:] if argv is None else argv
    command, *rest = args or ["list"]
    if command == "list":
        if rest == ["--json"]:
            fields = ("id", "status", "dataset", "citation", "licence", "test", "received", "processed", "rejected")
            print(json.dumps([{k: r.get(k) for k in fields} | {"email": r["contributor"]["email"]} for r in records()]))
            return
        for r in records():
            test = " [test]" if r.get("test") else ""
            purpose = "review only" if r.get("licence") == "review" else "redistribute"
            print(f"{r['id']}  {r.get('status', '?'):<10}  {r['dataset']:<16}  {purpose:<12}  {r['contributor']['email']}{test}")
        return
    if not rest:
        sys.exit(__doc__)
    directory = config.SUBMISSIONS / rest[0]
    if not (directory / "submission.json").exists():
        sys.exit(f"No submission {rest[0]}")
    record = read_record(directory)
    status = record.get("status")
    title = f"{record['citation']} ({record['dataset']})"

    if command == "show":
        record["preview"] = preview_url(record)
        record["compiled_bucket"] = compiled_bucket(record)
        record["submissions_bucket"] = config.SUBMISSIONS_BUCKET
        record.pop("preview_key", None)
        print(json.dumps(record, indent=2))
    elif command == "retry":
        update(directory, status="received", step=None, attempts=0, error=None)
        print(f"{rest[0]}: compiled again by the worker within a minute")
    elif command == "accept":
        if len(rest) != 2:
            sys.exit("python -m portal.manage accept <id> <database version>")
        if status not in ("ready", "accepted") or record.get("licence") != "redistribute":
            sys.exit(f"{rest[0]} is {status} ({record.get('licence')}): only compiled submissions to redistribute are accepted")
        delete_data(record, raw=False)
        update(directory, status="accepted", accepted=now(), db_version=rest[1])
        if status != "accepted":
            email_contributor(
                record,
                f"Your dataset is in the AntADatabase: {record['citation']}",
                f"Your submission {record['id']} is now part of the AntADatabase, as the dataset {record['dataset']} "
                f"(database version {rest[1]}). Thank you for sharing it!\n\n"
                f"- See it in the 3D viewer: {config.SITE_URL}/viewer/\n"
                f"- Download it: {config.SITE_URL}/download/\n"
                "- In Python: anta_database.sync() downloads the new version of the database.\n\n"
                "The private preview of your submission is now removed.",
            )
        print(f"{rest[0]}: accepted ({rest[1]}), preview and compiled copy removed, raw files kept")
    elif command == "reject":
        text = message()
        update(directory, status="rejected", rejected=now(), rejected_message=text)
        email_contributor(record, f"Your AntADatabase submission: {record['citation']}", text)
        print(f"{rest[0]}: rejected, the contributor was told; the data is kept until: purge {rest[0]}")
    elif command == "done":
        if record.get("licence") != "review":
            sys.exit(f"{rest[0]} is not a review-only submission: accept or reject it")
        text = message()
        delete_data(record, raw=True)
        update(directory, status="done", done=now(), done_message=text)
        email_contributor(record, f"Review of your AntADatabase submission: {record['citation']}", text)
        print(f"{rest[0]}: review done, the contributor was told, all its data deleted")
    elif command == "purge":
        if status in ("accepted", "received", "processing", "ready"):
            sys.exit(f"{rest[0]} is {status}: only refused or failed submissions are purged")
        delete_data(record, raw=True)
        update(directory, status="purged", purged=now())
        print(f"{rest[0]}: data deleted ({title}), only its record is kept")
    elif command == "withdraw":
        if status != "accepted":
            sys.exit(f"{rest[0]} is {status}: only accepted submissions are withdrawn")
        text = sys.stdin.read().strip()
        update(directory, status="withdrawn", withdrawn=now(), withdrawn_message=text or None)
        if text:
            email_contributor(record, f"Your dataset in the AntADatabase: {record['citation']}", text)
        print(f"{rest[0]}: withdrawn{', the contributor was told' if text else ''}; the raw files are kept")
    elif command == "delete":
        if not record.get("test"):
            sys.exit(f"{rest[0]} is not a test submission: reject it, then purge it")
        delete_data(record, raw=True)
        shutil.rmtree(directory)
        print(f"{rest[0]}: test submission deleted ({title})")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
