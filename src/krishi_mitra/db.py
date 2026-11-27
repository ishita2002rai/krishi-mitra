"""SQLite scan history for Increment 4, extended in Increment 5 with plot
tagging and farmer-reported treatment outcomes. One row per completed
diagnosis. Farmer accounts (login) were added afterwards: each scan belongs to
one farmer via scans.farmer_id -> users.id."""

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import uuid

from krishi_mitra.config import DB_DIR, DB_PATH, UPLOADS_DIR
from krishi_mitra.weather import format_place

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    image_path TEXT,
    location TEXT,
    plot_name TEXT,
    diagnosis TEXT NOT NULL,
    confidence REAL,
    reasoning TEXT,
    weather_json TEXT,
    risk_assessment TEXT,
    treatment_plan TEXT,
    outcome_rating INTEGER,
    outcome_note TEXT,
    farmer_id TEXT REFERENCES users(id)
)
"""

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

# Columns added after the original Increment 4 schema. init_db() adds any of
# these missing from an existing scans.db without touching existing rows.
_MIGRATIONS = [
    "ALTER TABLE scans ADD COLUMN plot_name TEXT",
    "ALTER TABLE scans ADD COLUMN outcome_rating INTEGER",
    "ALTER TABLE scans ADD COLUMN outcome_note TEXT",
    "ALTER TABLE scans ADD COLUMN farmer_id TEXT",
]

# Password hashing: PBKDF2-HMAC-SHA256 from Python's standard library, with a
# random 16-byte salt per user. Only "salt$hash" is stored, never the password.
_PBKDF2_ITERATIONS = 200_000
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,30}$")
MIN_PASSWORD_LENGTH = 6


def init_db() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(USERS_SCHEMA)
        conn.execute(SCHEMA)
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(scans)")}
        for migration in _MIGRATIONS:
            column = migration.split("ADD COLUMN")[1].split()[0]
            if column not in existing_columns:
                conn.execute(migration)


def _hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    salt_hex, _ = stored.split("$", 1)
    # compare_digest takes the same time whether or not the hashes match, so
    # response timing can't leak how much of a guessed password was right.
    return hmac.compare_digest(_hash_password(password, bytes.fromhex(salt_hex)), stored)


def create_user(username: str, password: str) -> dict:
    """Register a farmer. Raises ValueError with a plain-language message if the
    username is invalid/taken or the password is too short. The first account
    created also takes ownership of any scans saved before logins existed."""
    username = username.strip()
    if not USERNAME_PATTERN.match(username):
        raise ValueError("Username must be 3-30 characters: letters, numbers or underscore only.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    init_db()
    user_id = str(uuid.uuid4())
    with sqlite3.connect(DB_PATH) as conn:
        is_first_user = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        try:
            conn.execute(
                "INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
                (user_id, username, _hash_password(password)),
            )
        except sqlite3.IntegrityError:
            raise ValueError("That username is already taken. Please choose another.")
        if is_first_user:
            conn.execute("UPDATE scans SET farmer_id = ? WHERE farmer_id IS NULL", (user_id,))
    return {"id": user_id, "username": username}


def authenticate(username: str, password: str):
    """Return {"id", "username"} if the credentials are correct, else None."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    if row and _verify_password(password, row[2]):
        return {"id": row[0], "username": row[1]}
    return None


def save_image(image_bytes: bytes, suffix: str = ".jpg") -> str:
    """Save uploaded image bytes under UPLOADS_DIR and return the saved path."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOADS_DIR / f"{uuid.uuid4()}{suffix}"
    path.write_bytes(image_bytes)
    return str(path)


def save_scan(state: dict, image_path: str = None, plot_name: str = None, farmer_id: str = None) -> str:
    """Persist a completed agent run for a farmer. Returns the new scan's id."""
    init_db()
    scan_id = str(uuid.uuid4())
    location = state.get("location")
    location_str = format_place(location) if isinstance(location, dict) else location
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO scans
               (id, farmer_id, image_path, location, plot_name, diagnosis, confidence, reasoning,
                weather_json, risk_assessment, treatment_plan)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scan_id,
                farmer_id,
                image_path or state.get("image_path"),
                location_str,
                plot_name or None,
                state.get("diagnosis"),
                state.get("confidence"),
                state.get("reasoning"),
                json.dumps(state["weather"]) if state.get("weather") else None,
                state.get("risk_assessment"),
                state.get("treatment_plan"),
            ),
        )
    return scan_id


def update_outcome(scan_id: str, rating: int, note: str = None, farmer_id: str = None) -> None:
    """Record how well a scan's treatment plan worked (1-5), closing the agentic loop.
    Only updates the scan if it belongs to farmer_id."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE scans SET outcome_rating = ?, outcome_note = ? WHERE id = ? AND farmer_id IS ?",
            (rating, note or None, scan_id, farmer_id),
        )


def get_history(farmer_id: str = None, limit: int = 500) -> list:
    """One farmer's scans, most recent first, as a list of dicts."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scans WHERE farmer_id IS ? ORDER BY created_at DESC LIMIT ?", (farmer_id, limit)
        ).fetchall()
    return [dict(row) for row in rows]
