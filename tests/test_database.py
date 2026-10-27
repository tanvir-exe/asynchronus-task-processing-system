import sqlite3

from task_processing_system.database import get_connection, init_db

OLD_SCHEMA = """
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE,
    request_hash TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    result TEXT, error TEXT,
    created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT
);
"""


async def test_init_db_is_safe_to_run_twice(db):
    await init_db()
    await init_db()


async def test_old_schema_database_gets_a_clear_error(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.commit()
    conn.close()
    monkeypatch.setenv("TASKS_DB_PATH", str(path))

    try:
        await init_db()
    except RuntimeError as exc:
        assert "delete tasks.db" in str(exc)
        return
    raise AssertionError("expected RuntimeError for the old schema")


async def test_foreign_keys_are_enforced(db):
    async with get_connection() as conn:
        cursor = await conn.execute("PRAGMA foreign_keys")
        assert (await cursor.fetchone())[0] == 1