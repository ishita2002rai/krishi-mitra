"""SQLite scan history for Increment 4, extended in Increment 5 with plot
tagging and farmer-reported treatment outcomes. One row per completed
diagnosis."""

import json
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
    outcome_note TEXT
)
"""

# Columns added after the original Increment 4 schema. init_db() adds any of
# these missing from an existing scans.db without touching existing rows.
_MIGRATIONS = [
    "ALTER TABLE scans ADD COLUMN plot_name TEXT",
    "ALTER TABLE scans ADD COLUMN outcome_rating INTEGER",
    "ALTER TABLE scans ADD COLUMN outcome_note TEXT",
]


def init_db() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(SCHEMA)
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(scans)")}
        for migration in _MIGRATIONS:
            column = migration.split("ADD COLUMN")[1].split()[0]
            if column not in existing_columns:
                conn.execute(migration)


def save_image(image_bytes: bytes, suffix: str = ".jpg") -> str:
    """Save uploaded image bytes under UPLOADS_DIR and return the saved path."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOADS_DIR / f"{uuid.uuid4()}{suffix}"
    path.write_bytes(image_bytes)
    return str(path)


def save_scan(state: dict, image_path: str = None, plot_name: str = None) -> str:
    """Persist a completed agent run. Returns the new scan's id."""
    init_db()
    scan_id = str(uuid.uuid4())
    location = state.get("location")
    location_str = format_place(location) if isinstance(location, dict) else location
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO scans
               (id, image_path, location, plot_name, diagnosis, confidence, reasoning,
                weather_json, risk_assessment, treatment_plan)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scan_id,
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


def update_outcome(scan_id: str, rating: int, note: str = None) -> None:
    """Record how well a scan's treatment plan worked (1-5), closing the agentic loop."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE scans SET outcome_rating = ?, outcome_note = ? WHERE id = ?",
            (rating, note or None, scan_id),
        )


def get_history(limit: int = 500) -> list:
    """Most recent scans first, as a list of dicts."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM scans ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
