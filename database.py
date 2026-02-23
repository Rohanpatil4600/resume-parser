"""SQLite persistence for application threads and run results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools import hash_text

DB_PATH = Path(__file__).resolve().parent / "resume_engine.db"


def get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT UNIQUE NOT NULL,
                company TEXT,
                role TEXT,
                jd_url TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                resume_hash TEXT,
                jd_hash TEXT,
                match_score REAL,
                report_json TEXT,
                rewrites_json TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (thread_id) REFERENCES applications(thread_id)
            )
            """
        )
        conn.commit()


def save_application(
    thread_id: str,
    company: Optional[str] = None,
    role: Optional[str] = None,
    jd_url: Optional[str] = None,
) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO applications (thread_id, company, role, jd_url)
            VALUES (?, ?, ?, ?)
            """,
            (thread_id, company or "", role or "", jd_url or ""),
        )
        conn.commit()


def save_run(
    thread_id: str,
    resume_hash: str,
    jd_hash: str,
    match_score: float,
    report_json: Dict[str, Any],
    rewrites_json: Optional[Dict[str, Any]] = None,
) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO runs (thread_id, resume_hash, jd_hash, match_score, report_json, rewrites_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                resume_hash,
                jd_hash,
                match_score,
                json.dumps(report_json),
                json.dumps(rewrites_json) if rewrites_json else None,
            ),
        )
        conn.commit()


def list_applications() -> List[Dict[str, Any]]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, company, role, jd_url, created_at
            FROM applications
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]
