"""Worker of the portal: compiles the new submissions one after the other (pipeline.py), each in
a process of its own with a time limit, then emails the result to the maintainer (with the
report). The contributor follows the processing on the 3D preview page (its link is in the
receipt email), and is only emailed when the compilation failed.

A submission is compiled when its status is "received" (set by the portal, or by
`python -m portal.manage retry <id>`). One that was being compiled when the worker stopped is
compiled again, twice at most.

    python -m portal.worker            (from the repository, or the worker container)
"""

import logging
import secrets
import shutil
import subprocess
import sys
import time

from . import config, mail
from .pipeline import STEPS, now, read_record, update

log = logging.getLogger("portal.worker")
POLL_SECONDS = 10
MAX_ATTEMPTS = 2


def preview_url(record: dict) -> str:
    return f"{config.SITE_URL}/viewer/?preview={record['id']}&key={record['preview_key']}"


def report_text(report: dict) -> str:
    lines = []
    if report.get("errors"):
        lines += ["Errors:", *(f"- {e}" for e in report["errors"])]
    if report.get("warnings"):
        lines += ["Warnings:", *(f"- {w}" for w in report["warnings"])]
    summary = report.get("summary", {})
    if "lines" in summary:
        lines.append(
            f"Compiled: {summary['lines']} lines, {summary['IRHs']} IRHs, {summary['IRH points']:,} IRH points"
            + (f"; undated IRHs: {', '.join(map(str, summary['undated IRHs']))}" if summary.get("undated IRHs") else "")
        )
    return "\n".join(lines) or "No problem found."


def notify(record: dict) -> None:
    """Email the result of a compilation: to the maintainer, and to the contributor if it failed
    (the preview page then only says the preview has ended)."""
    test = "[test] " if record.get("test") else ""
    contributor = record["contributor"]
    title = f"{record['citation']} ({record['dataset']})"
    report = report_text(record.get("report") or {})
    if record["status"] == "ready":
        mail.to_maintainer(
            f"{test}Compiled: {title}",
            f"Submission {record['id']} is compiled.\n\n{report}\n\nPreview: {preview_url(record)}\n"
            f"Stored in {record['compiled']}\n",
            reply_to=contributor["email"],
        )
        # The contributor sees it on the preview page (link in the receipt email)
    else:
        mail.to_maintainer(
            f"{test}Compilation failed: {title}",
            f"Submission {record['id']} could not be compiled.\n\n{record.get('error', '')}\n\n{report}\n\n"
            f"Log:\n{record.get('log', '-')}\n\nAfter a fix: python -m portal.manage retry {record['id']}\n",
            reply_to=contributor["email"],
        )
        mail.send(
            contributor["email"],
            f"{test}Your AntADatabase submission: {record['citation']}",
            f"Dear {contributor['name']},\n\nYour submission {record['id']} ({record['dataset']}) was received, but it "
            "could not be compiled automatically"
            + (f":\n\n{report}\n\n" if record.get("report", {}).get("errors") else ". ")
            + "We will look into it and come back to you. You can also send corrected files with a new submission.\n\n"
            f"For any question, reply to this email or use {config.SITE_URL}/contact/.\n\nAntADatabase\n",
            reply_to=config.MAINTAINER_EMAIL,
        )


def process(submission_id: str) -> None:
    directory = config.SUBMISSIONS / submission_id
    record = read_record(directory)
    record = update(
        directory,
        attempts=record.get("attempts", 0) + 1,
        # Submissions from before the previews
        preview_key=record.get("preview_key") or secrets.token_urlsafe(16),
    )
    log.info("Compiling %s (%s), attempt %d", submission_id, record["dataset"], record["attempts"])
    try:
        result = subprocess.run(
            [sys.executable, "-m", "portal.pipeline", submission_id],
            capture_output=True, text=True, timeout=config.COMPILE_TIMEOUT,
        )
        output, code = result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired as e:
        output, code = f"{e.stdout or ''}{e.stderr or ''}", None
    tail = "\n".join(output.splitlines()[-40:])
    record = read_record(directory)
    if code == 0:
        log.info("%s compiled", submission_id)
    elif code == 2:  # errors in the files, already in the record
        update(directory, log=tail)
    else:
        error = (
            f"Longer than {config.COMPILE_TIMEOUT} s" if code is None
            else "Out of memory" if code in (-9, 137)
            else f"Failed at the step: {STEPS.get(record.get('step'), record.get('step'))}"
        )
        update(directory, status="failed", step=None, processed=now(), error=error, log=tail)
    shutil.rmtree(config.WORK / submission_id, ignore_errors=True)
    try:
        notify(read_record(directory))
    except Exception:
        log.exception("Emails of the compilation of %s not sent", submission_id)


def pending() -> list[str]:
    """Submissions to compile, oldest first. Those interrupted while compiling are compiled
    again, MAX_ATTEMPTS times at most."""
    ids = []
    for path in sorted(config.SUBMISSIONS.glob("*/submission.json")):
        try:
            record = read_record(path.parent)
        except (OSError, ValueError):
            continue
        if record.get("status") == "received":
            ids.append(path.parent.name)
        elif record.get("status") == "processing":
            if record.get("attempts", 0) < MAX_ATTEMPTS:
                ids.append(path.parent.name)
            else:
                update(path.parent, status="failed", step=None, error="Interrupted twice while compiling")
                try:
                    notify(read_record(path.parent))
                except Exception:
                    log.exception("Emails of %s not sent", path.parent.name)
    return ids


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config.SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    log.info("Worker started")
    # Interrupted compilations are only found once, at the start: later, "processing" is the
    # one running here
    queue = pending()
    while True:
        for submission_id in queue:
            process(submission_id)
        time.sleep(POLL_SECONDS)
        queue = [i for i in pending() if read_record(config.SUBMISSIONS / i).get("status") == "received"]


if __name__ == "__main__":
    main()
