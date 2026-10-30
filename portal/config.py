"""Settings of the portal, from the environment or from portal/.env (not in git).

See portal/.env.example for the settings. The private ones (maintainer address, SMTP
password) only live in portal/.env on the machine running the portal.
"""

import os
from pathlib import Path

HERE = Path(__file__).parent


def _load_env_file(path: Path) -> None:
    """KEY=VALUE lines of a .env file, without overriding the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env_file(Path(os.environ.get("PORTAL_ENV_FILE", HERE / ".env")))

DATA = Path(os.environ.get("PORTAL_DATA", HERE / "data")).resolve()
SUBMISSIONS = DATA / "submissions"
# The portal is closed while this file exists (its text, if any, is shown on the page)
CLOSED_FILE = DATA / "CLOSED"

ORIGINS = os.environ.get("PORTAL_ORIGINS", "https://antadatabase.com,http://localhost:5173").split(",")
# Limits of a submission, as in the form (web/src/submit/files.ts)
MAX_FILES = 100
MAX_FILE_BYTES = 1024**3
MAX_BYTES = int(os.environ.get("PORTAL_MAX_GB", "5")) * 1024**3
# Files of contact messages: stored with the message (<PORTAL_DATA>/contact/<id>/), not emailed.
# Only from a verified email address
CONTACT = DATA / "contact"
CONTACT_MAX_FILES = 10
CONTACT_MAX_BYTES = 500 * 1024**2
CONTACT_EXTENSIONS = (".csv", ".tab", ".txt", ".zip", ".pdf", ".png", ".jpg", ".jpeg")
SITE_URL = os.environ.get("PORTAL_SITE_URL", "https://antadatabase.com")

# Private buckets for the submissions and the files of contact messages (S3 API of the
# Infomaniak Public Cloud). Without them, the files stay in PORTAL_DATA on the server
S3_ENDPOINT = os.environ.get("PORTAL_S3_ENDPOINT", "https://s3.pub1.infomaniak.cloud")
S3_ACCESS_KEY_ID = os.environ.get("PORTAL_S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.environ.get("PORTAL_S3_SECRET_ACCESS_KEY", "")
SUBMISSIONS_BUCKET = os.environ.get("PORTAL_SUBMISSIONS_BUCKET", "")
CONTACT_BUCKET = os.environ.get("PORTAL_CONTACT_BUCKET", "")
# The compiled submissions (worker.py): those to add to the database after review, and those
# only reviewed (kept private, never redistributed), apart
COMPILED_BUCKET = os.environ.get("PORTAL_COMPILED_BUCKET", "")
REVIEW_BUCKET = os.environ.get("PORTAL_REVIEW_BUCKET", "")

# Compilation of the submissions (worker.py): working directories, and the private 3D previews
# served to the contributor (viewer/?preview=<id>&key=<key>) until the review is done
WORK = DATA / "work"
PREVIEWS = DATA / "previews"
# usurf and thk of Bedmap3, for the 3D preview and the checks (see README.md)
BEDMAP = Path(os.environ.get("PORTAL_BEDMAP", DATA / "bedmap3_surface.nc"))
VAR_ATTRS = HERE / "var_attrs.json"
# Longest compilation of a submission (s)
COMPILE_TIMEOUT = int(os.environ.get("PORTAL_COMPILE_TIMEOUT", str(3 * 3600)))

# Where new submissions and contact messages are sent
MAINTAINER_EMAIL = os.environ.get("PORTAL_MAINTAINER_EMAIL", "")

# Outgoing mail. Without SMTP_HOST, emails are only written to the log (development)
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", SMTP_USER)
# "ssl" (port 465), "starttls" (port 587) or "none" (local test server)
SMTP_SECURITY = os.environ.get("SMTP_SECURITY", "ssl" if SMTP_PORT == 465 else "starttls")
