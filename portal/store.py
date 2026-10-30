"""State of the portal in a small SQLite database (<PORTAL_DATA>/portal.db): email
verification codes, verified email sessions and the request log for the rate limits."""

import hashlib
import secrets
import sqlite3
import time

from . import config

CODE_MINUTES = 30  # a verification code expires after
CODE_ATTEMPTS = 5  # wrong codes before a new one is needed
TOKEN_HOURS = 24  # a verified email stays verified for


def _db() -> sqlite3.Connection:
    config.DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.DATA / "portal.db")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS codes (email TEXT PRIMARY KEY, code_hash TEXT, expires REAL, attempts INTEGER);
        CREATE TABLE IF NOT EXISTS tokens (token_hash TEXT PRIMARY KEY, email TEXT, expires REAL);
        CREATE TABLE IF NOT EXISTS requests (kind TEXT, key TEXT, time REAL);
        CREATE INDEX IF NOT EXISTS requests_key ON requests (kind, key, time);
        """
    )
    return db


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def allow(kind: str, key: str, limit: int, hours: float = 1) -> bool:
    """Count a request of a kind (e.g. "code") by a key (IP or email): False above limit per period."""
    now = time.time()
    with _db() as db:
        db.execute("DELETE FROM requests WHERE time < ?", (now - 24 * 3600,))
        (count,) = db.execute(
            "SELECT COUNT(*) FROM requests WHERE kind = ? AND key = ? AND time > ?", (kind, key, now - hours * 3600)
        ).fetchone()
        if count >= limit:
            return False
        db.execute("INSERT INTO requests VALUES (?, ?, ?)", (kind, key, now))
    return True


def new_code(email: str) -> str:
    """A 6-digit code for an email, replacing any previous one."""
    code = f"{secrets.randbelow(10**6):06d}"
    with _db() as db:
        db.execute(
            "INSERT OR REPLACE INTO codes VALUES (?, ?, ?, 0)", (email, _hash(code), time.time() + CODE_MINUTES * 60)
        )
    return code


def check_code(email: str, code: str) -> str | None:
    """A session token if the code is right, None otherwise."""
    with _db() as db:
        row = db.execute("SELECT code_hash, expires, attempts FROM codes WHERE email = ?", (email,)).fetchone()
        if not row or row[1] < time.time() or row[2] >= CODE_ATTEMPTS:
            return None
        if not secrets.compare_digest(row[0], _hash(code)):
            db.execute("UPDATE codes SET attempts = attempts + 1 WHERE email = ?", (email,))
            return None
        db.execute("DELETE FROM codes WHERE email = ?", (email,))
        token = secrets.token_urlsafe(32)
        db.execute("DELETE FROM tokens WHERE expires < ?", (time.time(),))
        db.execute("INSERT INTO tokens VALUES (?, ?, ?)", (_hash(token), email, time.time() + TOKEN_HOURS * 3600))
    return token


def verified_email(token: str) -> str | None:
    """The email a session token was given for, while valid."""
    if not token:
        return None
    with _db() as db:
        row = db.execute("SELECT email, expires FROM tokens WHERE token_hash = ?", (_hash(token),)).fetchone()
    return row[0] if row and row[1] > time.time() else None
