"""SQLite connection helpers and schema creation.

Each operation opens its own short-lived connection. That keeps concurrent
requests and workers from sharing one connection, and SQLite is fast enough
at this scale that the cost is negligible. WAL mode lets readers proceed
while a write is in flight, and busy_timeout makes writers wait briefly
instead of failing immediately with "database is locked".

The database path comes from the TASKS_DB_PATH environment variable
(default: tasks.db) and is read on every call, so tests can point each test
at its own temporary file.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

DEFAULT_DB_PATH = "tasks.db"
BUSY_TIMEOUT_MS = 30_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id             TEXT PRIMARY KEY,
    username       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash  TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users (id),
    idempotency_key  TEXT,
    request_hash     TEXT NOT NULL,
    type             TEXT NOT NULL
        CHECK (type IN ('text_analysis', 'transaction_analysis')),
    status           TEXT NOT NULL
        CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    payload          TEXT NOT NULL,
    result           TEXT,
    error            TEXT,
    retry_count      INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    completed_at     TEXT,
    -- Idempotency keys are per user. Tasks without a key (NULL) never conflict.
    UNIQUE (user_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_user_created ON tasks (user_id, created_at);
"""


def get_db_path() -> str:
    return os.environ.get("TASKS_DB_PATH", DEFAULT_DB_PATH)


@asynccontextmanager
async def get_connection(db_path: str | None = None) -> AsyncIterator[aiosqlite.Connection]:
    """Open a connection configured for concurrent use."""
    async with aiosqlite.connect(db_path or get_db_path(), timeout=BUSY_TIMEOUT_MS / 1000) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        await conn.execute("PRAGMA foreign_keys = ON")
        yield conn


async def _check_not_outdated(conn: aiosqlite.Connection, path: str) -> None:
    """Fail with a clear message if the file was created by the pre-login version."""
    cursor = await conn.execute("PRAGMA table_info(tasks)")
    columns = {row["name"] for row in await cursor.fetchall()}
    if columns and "user_id" not in columns:
        raise RuntimeError(
            f"The database {path!r} uses the old schema (tasks without owners). "
            "Stop the server, delete tasks.db, tasks.db-wal and tasks.db-shm, "
            "and start again to create a fresh database."
        )


async def init_db(db_path: str | None = None) -> None:
    """Create the database file, enable WAL and create tables if missing."""
    path = db_path or get_db_path()
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)

    async with get_connection(path) as conn:
        await _check_not_outdated(conn, path)
        # WAL is a persistent property of the database file.
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA synchronous = NORMAL")
        await conn.executescript(SCHEMA)
        await conn.commit()