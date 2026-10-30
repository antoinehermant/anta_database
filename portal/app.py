"""Submission portal of the AntADatabase: receives the datasets sent from antadatabase.com/submit/
and the messages of the contact form.

Each submission is stored as a dataset directory ready for CompileDatabase, with what the
contributor filled in the form:

    <PORTAL_DATA>/submissions/<id>/
        submission.json                  form, contributor, status
        <Dataset>/raw/*.csv|*.tab        the files, as uploaded
        <Dataset>/raw_md.json            metadata, as CompileDatabase reads it
        <Dataset>/original_new_column_names.csv

Contributors first verify their email address with a code sent to it; submissions are only
accepted from a verified address. Nothing is checked beyond the form and the file names here:
the worker (worker.py, pipeline.py) then checks and compiles each submission, and builds its
private 3D preview, served here to whoever has its key (viewer/?preview=<id>&key=<key>).

The portal is closed (no codes, submissions or messages) while <PORTAL_DATA>/CLOSED exists:

    touch portal/data/CLOSED      # close; the text of the file, if any, is shown on the page
    rm portal/data/CLOSED         # open again

    uvicorn portal.app:app --reload        (from the repository, see portal/README.md)

Settings: see config.py and .env.example.
"""

import csv
import json
import logging
import os
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import config, mail, storage, store

FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(csv|tab)$")
DATASET_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_\d{4}[a-z]?$")
SUBMISSION_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")
PREVIEW_FILE = re.compile(r"^[A-Za-z0-9_.-]+\.(json|bin\.gz)$")
GB = 1024**3
REDUCE = "Try to reduce the size of your files, or contact us."
EMAIL = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
VARIABLES = ("ICE_THK", "SURF_ELEV", "BED_ELEV", "BASAL_UNIT")

log = logging.getLogger("portal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


# --- The form, as sent by web/src/submit/main.ts


class Publication(BaseModel):
    first_author: str = Field(min_length=1, max_length=100)
    et_al: bool
    year: str = Field(pattern=r"^\d{4}$")
    doi_publication: str = Field(default="", max_length=300)
    doi_dataset: str = Field(default="", max_length=300)


class Survey(BaseModel):
    institutes: list[str] = Field(default=[], max_length=20)
    projects: list[str] = Field(default=[], max_length=20)
    radar_instruments: list[str] = Field(default=[], max_length=20)
    acquisition_year: str = Field(default="", pattern=r"^(\d{4})?$")


class Contributor(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str = Field(pattern=EMAIL, max_length=200)
    affiliation: str = Field(default="", max_length=300)
    orcid: str = Field(default="", max_length=60)


class SubmittedFile(BaseModel):
    name: str
    kind: Literal["irh", "variable", "line"]
    irh: Optional[int] = Field(default=None, ge=1)
    variable: Optional[Literal[VARIABLES]] = None
    irh_columns: Optional[list[int]] = None
    rows: int = 0
    line_ids: int = 0
    acquisition_year: Optional[str] = Field(default=None, pattern=r"^\d{4}$")
    project: Optional[str] = Field(default=None, max_length=200)

    @field_validator("name")
    @classmethod
    def plain_name(cls, name: str) -> str:
        if not FILE_NAME.match(name):
            raise ValueError(f"{name!r}: only .csv or .tab files, named with letters, digits, '.', '_' and '-'")
        return name


class Layer(BaseModel):
    irh: int = Field(ge=1)
    age: Optional[float] = Field(default=None, gt=0)
    age_unc: Optional[float] = Field(default=None, ge=0)


class Submission(BaseModel):
    structure: Literal["layer", "line"]
    dataset: str = Field(pattern=DATASET_NAME.pattern)
    citation: str = Field(min_length=1, max_length=200)
    publication: Publication
    survey: Survey
    contributor: Contributor
    notes: str = Field(default="", max_length=10000)
    # "redistribute": added to the database after review; "review": only checked, kept private
    licence: Literal["redistribute", "review"]
    files: list[SubmittedFile] = Field(min_length=1, max_length=config.MAX_FILES)
    layers: list[Layer] = Field(default=[], max_length=1000)
    warnings: list[str] = Field(default=[], max_length=10000)
    # Sent from the development version of the page (/submit/?dev)
    test: bool = False


# --- Inputs of CompileDatabase


def one_or_list(values: list[str]):
    """A name, a list of names, or None, as in the raw_md.json of the existing datasets."""
    values = [v.strip() for v in values if v.strip()]
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def raw_metadata(s: Submission) -> list[dict]:
    """raw_md.json: one entry per file (one file per layer) or per layer (one file per line)."""
    ages = {layer.irh: layer for layer in s.layers}
    common = {
        "dataset": s.dataset,
        "institute": one_or_list(s.survey.institutes),
        "project": one_or_list(s.survey.projects),
        "radar instrument": one_or_list(s.survey.radar_instruments),
        "acquisition year": int(s.survey.acquisition_year) if s.survey.acquisition_year else None,
        "citation": s.citation,
        "DOI dataset": s.publication.doi_dataset,
        "DOI publication": s.publication.doi_publication,
    }
    entries = []
    if s.structure == "layer":
        for f in s.files:
            layer = ages.get(f.irh) if f.kind == "irh" else None
            entry = {
                "raw file": f.name,
                "file structure": "IRH",
                "dataset": s.dataset,
                "IRH": f.irh if f.kind == "irh" else f.variable,
                "age": layer.age if layer else None,
                "age uncertainty": layer.age_unc if layer else None,
                **{k: v for k, v in common.items() if k != "dataset"},
            }
            if f.project:
                entry["project"] = f.project
            if f.acquisition_year:
                entry["acquisition year"] = int(f.acquisition_year)
            entries.append(entry)
    else:
        for irh in sorted({n for f in s.files for n in f.irh_columns or []}):
            layer = ages.get(irh)
            entries.append({
                "file structure": "line",
                "dataset": s.dataset,
                "IRH": irh,
                "age": layer.age if layer else None,
                "age uncertainty": layer.age_unc if layer else None,
                **{k: v for k, v in common.items() if k != "dataset"},
            })
    return entries


def column_names(s: Submission) -> list[tuple[str, str]]:
    """original_new_column_names.csv: the columns already have the standard names, apart from
    the layer columns of the line files (IRH1 -> 1), and Flight_ID, the old name of Line_ID."""
    columns = {c: c for c in ("PSX", "PSY", "lon", "lat")}
    if s.structure == "layer":
        columns["Line_ID"] = "Line_ID"
        columns["Flight_ID"] = "Line_ID"
        if any(f.kind == "irh" for f in s.files):
            columns["IRH_DEPTH"] = "IRH_DEPTH"
        for f in s.files:
            if f.kind == "variable":
                columns[f.variable] = f.variable
    else:
        columns.update({v: v for v in VARIABLES})
        for irh in sorted({n for f in s.files for n in f.irh_columns or []}):
            columns[f"IRH{irh}"] = str(irh)
    return list(columns.items())


def write_submission(directory: Path, s: Submission, record: dict) -> None:
    dataset = directory / s.dataset
    with open(dataset / "raw_md.json", "w") as f:
        json.dump(raw_metadata(s), f, indent=4)
    with open(dataset / "original_new_column_names.csv", "w", newline="") as f:
        writer = csv.writer(f)
        columns = column_names(s)
        writer.writerow([c for c, _ in columns])
        writer.writerow([n for _, n in columns])
    with open(directory / "submission.json", "w") as f:
        json.dump(record, f, indent=2)


# --- Texts of the emails


def submission_summary(s: Submission, submission_id: str, total: int) -> str:
    layers = ", ".join(
        f"IRH {l.irh}: {l.age:g} ± {l.age_unc:g} a" if l.age and l.age_unc is not None
        else f"IRH {l.irh}: {f'{l.age:g} a' if l.age else 'undated'}"
        for l in s.layers
    )
    structure = "one file per layer" if s.structure == "layer" else "one file per line"
    purpose = "added to the database after review" if s.licence == "redistribute" else "REVIEW ONLY, not to be redistributed"
    return "\n".join([
        f"Submission: {submission_id}",
        f"Purpose: {purpose}",
        f"Dataset: {s.dataset}, {s.citation}",
        f"Publication: {s.publication.doi_publication or '(no DOI)'}",
        f"Data: {s.publication.doi_dataset or '(no DOI)'}",
        f"Files: {len(s.files)} ({structure}, {total / 1e6:.1f} MB)",
        f"Layers: {layers or 'none'}",
        f"Institutes: {', '.join(s.survey.institutes) or '-'}; projects: {', '.join(s.survey.projects) or '-'}; "
        f"radar: {', '.join(s.survey.radar_instruments) or '-'}; acquisition year: {s.survey.acquisition_year or '-'}",
    ])


# --- API


async def save_upload(upload: UploadFile, path: Path, max_bytes: int) -> int:
    """Write an uploaded file, refusing it above max_bytes; its size."""
    size = 0
    with open(path, "wb") as out:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"Limit exceeded: {path.name} is larger than {max_bytes / GB:g} GB. {REDUCE}")
            out.write(chunk)
    return size


app = FastAPI(title="AntADatabase submission portal")
app.add_middleware(CORSMiddleware, allow_origins=config.ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


def closed_message() -> str | None:
    """The message of a closed portal, None while open."""
    if not config.CLOSED_FILE.exists():
        return None
    return config.CLOSED_FILE.read_text().strip() or "The portal is closed for the moment."


def ensure_open() -> None:
    message = closed_message()
    if message:
        raise HTTPException(503, message)


def client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def normal_email(email: str) -> str:
    email = email.strip().lower()
    if not re.match(EMAIL, email) or len(email) > 200:
        raise HTTPException(422, "Not a valid email address")
    return email


@app.get("/api/status")
def portal_status():
    message = closed_message()
    return {"open": message is None, "message": message}


class CodeRequest(BaseModel):
    email: str
    website: str = ""  # hidden field, only filled by bots


@app.post("/api/verify")
def send_code(body: CodeRequest, request: Request):
    """Send a verification code to an email address."""
    ensure_open()
    email = normal_email(body.email)
    if body.website:
        return {"sent": True}
    if not store.allow("code-ip", client(request), 20) or not store.allow("code", email, 5):
        raise HTTPException(429, "Too many codes asked for: please try again in an hour.")
    code = store.new_code(email)
    try:
        mail.send(
            email,
            f"AntADatabase verification code: {code}",
            f"Your code to verify your email address for the AntADatabase is\n\n    {code}\n\n"
            f"It is valid for {store.CODE_MINUTES} minutes. If you did not ask for it, you can ignore this email.\n\n"
            f"AntADatabase, {config.SITE_URL}\n",
        )
    except Exception:
        log.exception("Verification code for %s not sent", email)
        raise HTTPException(502, "The email could not be sent: please try again later, or contact us.")
    return {"sent": True}


class CodeCheck(BaseModel):
    email: str
    code: str = Field(pattern=r"^\s*\d{6}\s*$")


@app.post("/api/verify/check")
def check_code(body: CodeCheck):
    """Exchange a verification code for a session token."""
    ensure_open()
    token = store.check_code(normal_email(body.email), body.code.strip())
    if not token:
        raise HTTPException(403, "Wrong or expired code: check it, or ask for a new one.")
    return {"token": token, "hours": store.TOKEN_HOURS}


@app.post("/api/submissions")
async def submit(
    request: Request,
    submission: str = Form(...),
    files: list[UploadFile] = File(...),
    website: str = Form(""),
    authorization: str = Header(""),
):
    ensure_open()
    if website:
        return {"id": "0", "dataset": ""}
    try:
        s = Submission.model_validate_json(submission)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:5])
        raise HTTPException(422, f"Invalid submission: {problems}")

    verified = store.verified_email(authorization.removeprefix("Bearer ").strip())
    if not verified or verified != normal_email(s.contributor.email):
        raise HTTPException(401, "Please verify your email address first (it may have expired after a day).")
    if not store.allow("submission", verified, 10, hours=24):
        raise HTTPException(429, "Too many submissions today: please contact us.")

    names = [f.name for f in s.files]
    uploaded = [os.path.basename(f.filename or "") for f in files]
    if len(set(names)) != len(names):
        raise HTTPException(422, "Duplicate file names")
    if sorted(names) != sorted(uploaded):
        raise HTTPException(422, "The files sent do not match the form")
    if sum(f.size or 0 for f in files) > config.MAX_BYTES:
        raise HTTPException(413, f"Limit exceeded: at most {config.MAX_BYTES // GB} GB per submission. {REDUCE}")

    now = datetime.now(timezone.utc)
    submission_id = f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"
    final = config.SUBMISSIONS / submission_id
    partial = config.SUBMISSIONS / f".{submission_id}.partial"
    raw = partial / s.dataset / "raw"
    raw.mkdir(parents=True)
    try:
        total = 0
        for upload, name in zip(files, uploaded):
            total += await save_upload(upload, raw / name, config.MAX_FILE_BYTES)
            if total > config.MAX_BYTES:
                raise HTTPException(413, f"Limit exceeded: at most {config.MAX_BYTES // GB} GB per submission. {REDUCE}")
        record = {
            "id": submission_id,
            "status": "received",
            # Key of the private 3D preview
            "preview_key": secrets.token_urlsafe(16),
            "received": now.isoformat(timespec="seconds"),
            "bytes": total,
            "ip": client(request),
            **s.model_dump(),
        }
        write_submission(partial, s, record)
        partial.rename(final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    try:
        # Kept on the server too, for the worker
        location = storage.store(final, config.SUBMISSIONS_BUCKET, keep=True)
    except Exception:
        # Kept on the server: not lost, and the maintainer is told
        log.exception("Submission %s not uploaded to the bucket", submission_id)
        location = f"{final} (NOT uploaded to the bucket, see the log)"

    log.info("Submission %s: %s (%d files, %.1f MB) from %s", submission_id, s.citation, len(files), total / 1e6, verified)
    summary = submission_summary(s, submission_id, total)
    contributor = s.contributor
    preview = f"{config.SITE_URL}/viewer/?preview={submission_id}&key={record['preview_key']}"

    try:
        mail.to_maintainer(
            f"{'[test] ' if s.test else ''}New submission{' (review only)' if s.licence == 'review' else ''}: "
            f"{s.citation} ({s.dataset})",
            f"{summary}\n\nFrom: {contributor.name} <{contributor.email}>, {contributor.affiliation or '-'}"
            f"{f', ORCID {contributor.orcid}' if contributor.orcid else ''}\n\n"
            f"Notes:\n{s.notes or '-'}\n\nWarnings of the form:\n"
            + ("\n".join(f"- {w}" for w in s.warnings) or "-")
            + f"\n\nStored in {location}\nPreview: {preview}\n",
            reply_to=contributor.email,
        )
        mail.send(
            contributor.email,
            f"{'[test] ' if s.test else ''}Your AntADatabase submission: {s.citation}",
            f"Dear {contributor.name},\n\nThank you for your submission to the AntADatabase. It was received and "
            f"is now checked and converted to the format of the database, usually in a few minutes. Follow the "
            f"processing, then see your dataset in 3D with the results of the automatic checks, at\n\n"
            f"    {preview}\n\nThis preview is private: only people with the link can see it. "
            + (
                "We will contact you once it is reviewed, before it is added to the next release."
                if s.licence == "redistribute"
                else "As asked, it is only reviewed: it stays private, is not added to the database, and is deleted "
                "after the review. We will send you the result."
            )
            + f"\n\n{summary}\n\n"
            f"For any question, reply to this email or use {config.SITE_URL}/contact/.\n\nAntADatabase\n",
            reply_to=config.MAINTAINER_EMAIL,
        )
    except Exception:
        # The submission is stored: an email problem must not lose it
        log.exception("Emails of submission %s not sent", submission_id)
    return {"id": submission_id, "dataset": s.dataset, "preview_key": record["preview_key"]}


@app.get("/api/submissions/{submission_id}")
def submission_status(submission_id: str):
    if not SUBMISSION_ID.match(submission_id):
        raise HTTPException(404, "No such submission")
    path = config.SUBMISSIONS / submission_id / "submission.json"
    if not path.exists():
        raise HTTPException(404, "No such submission")
    record = json.loads(path.read_text())
    return {k: record[k] for k in ("id", "status", "received", "dataset", "citation")}


def preview_record(submission_id: str, key: str) -> dict:
    """The record of a submission, for whoever has the key of its preview."""
    path = config.SUBMISSIONS / submission_id / "submission.json"
    if not SUBMISSION_ID.match(submission_id) or not path.exists():
        raise HTTPException(404, "No such preview")
    record = json.loads(path.read_text())
    if not secrets.compare_digest(record.get("preview_key", ""), key):
        raise HTTPException(404, "No such preview")
    return record


@app.get("/api/previews/{submission_id}/{key}/status")
def preview_status(submission_id: str, key: str):
    """Progress of the compilation of a submission, then what the preview page shows."""
    record = preview_record(submission_id, key)
    status = record.get("status")
    out = {
        "id": submission_id,
        "status": status,
        "dataset": record["dataset"],
        "citation": record["citation"],
        "licence": record["licence"],
        "steps": record.get("steps") or [],
        "step": record.get("step"),
    }
    if status == "received":
        # Submissions compiled before this one
        out["waiting"] = sum(
            1 for p in config.SUBMISSIONS.glob("*/submission.json")
            if p.parent.name < submission_id and json.loads(p.read_text()).get("status") in ("received", "processing")
        )
    report = record.get("report") or {}
    if status in ("ready", "failed"):
        out["warnings"] = report.get("warnings", [])
        out["errors"] = report.get("errors", [])
        out["bbox"] = report.get("summary", {}).get("bbox")
    if status == "ready" and not (config.PREVIEWS / submission_id / "irh.json").exists():
        out["status"] = "closed"
    return out


@app.get("/api/previews/{submission_id}/{key}/{name}")
def preview_file(submission_id: str, key: str, name: str):
    """Files of the 3D preview of a submission (anta_database.web.irh)."""
    preview_record(submission_id, key)
    path = config.PREVIEWS / submission_id / name
    if not PREVIEW_FILE.match(name) or not path.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(path, headers={"Cache-Control": "private, no-cache"})


TOPICS = {
    "submission": "Submitting data",
    "data": "The data of the database",
    "viewer": "The 3D viewer or the website",
    "python": "The Python package",
    "other": "Other",
}


@app.post("/api/contact")
async def contact(
    request: Request,
    name: str = Form(..., min_length=1, max_length=200),
    email: str = Form(...),
    topic: Literal["submission", "data", "viewer", "python", "other"] = Form(...),
    message: str = Form(..., min_length=10, max_length=10000),
    website: str = Form(""),  # hidden field, only filled by bots
    files: list[UploadFile] = File(default=[]),
    authorization: str = Header(""),
):
    """Message of the contact form, sent to the maintainer. Its files, if any, are stored with
    it (<PORTAL_DATA>/contact/<id>/), from a verified email address only."""
    ensure_open()
    email = normal_email(email)
    if website:
        return {"sent": True}
    if files:
        if store.verified_email(authorization.removeprefix("Bearer ").strip()) != email:
            raise HTTPException(401, "Please verify your email address to send files.")
        if len(files) > config.CONTACT_MAX_FILES:
            raise HTTPException(413, f"Limit exceeded: at most {config.CONTACT_MAX_FILES} files.")
        for upload in files:
            if Path(upload.filename or "").suffix.lower() not in config.CONTACT_EXTENSIONS:
                raise HTTPException(422, f"{upload.filename}: only {', '.join(config.CONTACT_EXTENSIONS)} files.")
    if not store.allow("contact", client(request), 5) or not store.allow("contact-email", email, 5):
        raise HTTPException(429, "Too many messages: please try again in an hour.")

    now = datetime.now(timezone.utc)
    message_id = f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"
    stored = []
    if files:
        directory = config.CONTACT / message_id
        partial = config.CONTACT / f".{message_id}.partial"
        partial.mkdir(parents=True)
        try:
            total = 0
            for i, upload in enumerate(files):
                # Plain names, numbered so that they cannot collide
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(upload.filename or "file"))[-100:]
                path = partial / f"{i + 1}_{safe}"
                total += await save_upload(upload, path, config.CONTACT_MAX_BYTES)
                if total > config.CONTACT_MAX_BYTES:
                    raise HTTPException(413, f"Limit exceeded: at most {config.CONTACT_MAX_BYTES // 1024**2} MB of files. {REDUCE}")
                stored.append(f"{path.name} ({path.stat().st_size / 1e6:.1f} MB)")
            (partial / "message.json").write_text(json.dumps(
                {"id": message_id, "received": now.isoformat(timespec="seconds"), "name": name, "email": email,
                 "topic": topic, "message": message, "ip": client(request)}, indent=2))
            partial.rename(directory)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        try:
            location = storage.store(directory, config.CONTACT_BUCKET)
        except Exception:
            log.exception("Files of contact message %s not uploaded to the bucket", message_id)
            location = f"{directory} (NOT uploaded to the bucket, see the log)"
    files_text = f"\n\nFiles ({len(stored)}), stored in {location}:\n" + "\n".join(f"- {f}" for f in stored) if stored else ""
    try:
        mail.to_maintainer(
            f"Contact: {TOPICS[topic]}, from {name}",
            f"From: {name} <{email}>\nTopic: {TOPICS[topic]}\n\n{message}\n{files_text}",
            reply_to=email,
        )
    except Exception:
        log.exception("Contact message from %s not sent", email)
        raise HTTPException(502, "The message could not be sent: please email us instead.")
    return {"sent": True}
