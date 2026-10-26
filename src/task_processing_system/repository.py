"""All database access for tasks lives here.

Concurrency notes:
  * Idempotency is enforced by the UNIQUE constraint on idempotency_key.
    We insert first and handle IntegrityError, instead of "check then insert",
    which would be racy when two identical requests arrive together.
  * State changes are single conditional UPDATEs
    (``... WHERE id = ? AND status = 'pending'``) and we look at rowcount to
    see whether we won. That makes claim-vs-cancel race-safe: exactly one of
    the worker or the cancel request can move a pending task.

SQL fragments built with f-strings only ever contain fixed text; every value
goes through ``?`` placeholders.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from .database import get_connection
from .schemas import TaskStatus

PENDING = TaskStatus.PENDING.value
PROCESSING = TaskStatus.PROCESSING.value
COMPLETED = TaskStatus.COMPLETED.value
FAILED = TaskStatus.FAILED.value
CANCELLED = TaskStatus.CANCELLED.value

CancelOutcome = Literal["cancelled", "not_found", "not_cancellable"]

# Columns the API returns for status/list/cancel. payload and result are
# deliberately left out: they can be ~1 MB each, and parsing them on the event
# loop for a response that never includes them freezes the server.
SUMMARY_COLUMNS = (
    "id, idempotency_key, type, status, error, created_at, started_at, completed_at"
)


class IdempotencyConflictError(Exception):
    """The idempotency key was already used with a different request body."""

    def __init__(self, task_id: str, key: str) -> None:
        super().__init__(f"Idempotency key {key!r} was already used with a different request")
        self.task_id = task_id
        self.key = key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _value(item: Any) -> Any:
    return item.value if isinstance(item, Enum) else item


def compute_request_hash(task_type: Any, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"type": _value(task_type), "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _row_to_task(row: sqlite3.Row) -> dict[str, Any]:
    task = dict(row)
    # Only parse the JSON columns that were actually selected.
    if task.get("payload") is not None:
        task["payload"] = json.loads(task["payload"])
    if task.get("result") is not None:
        task["result"] = json.loads(task["result"])
    return task


async def _fetch(
    conn: Any, task_id: str, columns: str = SUMMARY_COLUMNS
) -> dict[str, Any] | None:
    # `columns` only ever comes from constants in this file, never user input.
    cursor = await conn.execute(f"SELECT {columns} FROM tasks WHERE id = ?", (task_id,))
    row = await cursor.fetchone()
    return _row_to_task(row) if row else None


# ---------------------------------------------------------------------------
# Create / read
# ---------------------------------------------------------------------------


async def create_task(
    task_type: Any,
    payload: dict[str, Any],
    idempotency_key: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Insert a pending task.

    Returns ``(task, created)``. ``created`` is False when an earlier task with
    the same idempotency key and the same request body already exists, in
    which case that existing task is returned. Raises IdempotencyConflictError
    if the key exists but the request body differs.
    """
    task_type = _value(task_type)
    request_hash = compute_request_hash(task_type, payload)
    task_id = str(uuid.uuid4())

    async with get_connection() as conn:
        try:
            await conn.execute(
                """
                INSERT INTO tasks
                    (id, idempotency_key, request_hash, type, status, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    idempotency_key,
                    request_hash,
                    task_type,
                    PENDING,
                    json.dumps(payload),
                    utc_now(),
                ),
            )
            await conn.commit()
        except sqlite3.IntegrityError:
            await conn.rollback()
            if idempotency_key is None:
                raise
            cursor = await conn.execute(
                f"SELECT request_hash, {SUMMARY_COLUMNS} FROM tasks WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise
            if row["request_hash"] != request_hash:
                raise IdempotencyConflictError(row["id"], idempotency_key)
            return _row_to_task(row), False

        task = await _fetch(conn, task_id)
        assert task is not None
        return task, True


async def get_task(task_id: str) -> dict[str, Any] | None:
    """Task summary (no payload or result)."""
    async with get_connection() as conn:
        return await _fetch(conn, task_id)


async def get_task_result(task_id: str) -> dict[str, Any] | None:
    """Task summary plus the parsed result. Used only by /tasks/{id}/result."""
    async with get_connection() as conn:
        return await _fetch(conn, task_id, f"{SUMMARY_COLUMNS}, result")


async def list_tasks(
    status: Any = None,
    task_type: Any = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return ``(items, total)``, newest first. ``total`` ignores limit/offset."""
    conditions: list[str] = []
    params: list[Any] = []
    if status is not None:
        conditions.append("status = ?")
        params.append(_value(status))
    if task_type is not None:
        conditions.append("type = ?")
        params.append(_value(task_type))
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    async with get_connection() as conn:
        cursor = await conn.execute(f"SELECT COUNT(*) FROM tasks {where}", params)
        total = (await cursor.fetchone())[0]
        cursor = await conn.execute(
            f"SELECT {SUMMARY_COLUMNS} FROM tasks {where} "
            "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        rows = await cursor.fetchall()
    return [_row_to_task(r) for r in rows], total


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


async def claim_task(task_id: str) -> dict[str, Any] | None:
    """Atomically move pending -> processing.

    Returns the task if this caller won the claim, otherwise None (the task
    was cancelled, already claimed, or does not exist).
    """
    async with get_connection() as conn:
        cursor = await conn.execute(
            "UPDATE tasks SET status = ?, started_at = ? WHERE id = ? AND status = ?",
            (PROCESSING, utc_now(), task_id, PENDING),
        )
        await conn.commit()
        if cursor.rowcount != 1:
            return None
        # The worker is the one caller that needs the payload.
        return await _fetch(conn, task_id, f"{SUMMARY_COLUMNS}, payload")


async def cancel_task(task_id: str) -> tuple[CancelOutcome, dict[str, Any] | None]:
    """Cancel a task only if it is still pending."""
    async with get_connection() as conn:
        cursor = await conn.execute(
            "UPDATE tasks SET status = ?, completed_at = ? WHERE id = ? AND status = ?",
            (CANCELLED, utc_now(), task_id, PENDING),
        )
        await conn.commit()
        task = await _fetch(conn, task_id)
        if cursor.rowcount == 1:
            return "cancelled", task
        if task is None:
            return "not_found", None
        return "not_cancellable", task


async def complete_task(task_id: str, result: dict[str, Any]) -> bool:
    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            UPDATE tasks
               SET status = ?, result = ?, error = NULL, completed_at = ?
             WHERE id = ? AND status = ?
            """,
            (COMPLETED, json.dumps(result), utc_now(), task_id, PROCESSING),
        )
        await conn.commit()
        return cursor.rowcount == 1


async def fail_task(task_id: str, error: str) -> bool:
    async with get_connection() as conn:
        cursor = await conn.execute(
            "UPDATE tasks SET status = ?, error = ?, completed_at = ? WHERE id = ? AND status = ?",
            (FAILED, error, utc_now(), task_id, PROCESSING),
        )
        await conn.commit()
        return cursor.rowcount == 1


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------


async def reset_stale_processing() -> int:
    """Put tasks left in 'processing' by a crash back to 'pending'.

    Processing is deterministic and side-effect free, so re-running is safe.
    Returns how many tasks were reset.
    """
    async with get_connection() as conn:
        cursor = await conn.execute(
            "UPDATE tasks SET status = ?, started_at = NULL WHERE status = ?",
            (PENDING, PROCESSING),
        )
        await conn.commit()
        return cursor.rowcount


async def get_pending_ids() -> list[str]:
    """IDs of all pending tasks, oldest first (used to refill the queue)."""
    async with get_connection() as conn:
        cursor = await conn.execute(
            "SELECT id FROM tasks WHERE status = ? ORDER BY created_at, rowid", (PENDING,)
        )
        return [row["id"] for row in await cursor.fetchall()]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


async def get_stats() -> dict[str, Any]:
    async with get_connection() as conn:
        cursor = await conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status")
        counts = {row["status"]: row["n"] for row in await cursor.fetchall()}

        cursor = await conn.execute(
            """
            SELECT AVG((julianday(completed_at) - julianday(started_at)) * 86400.0)
              FROM tasks
             WHERE status = ? AND started_at IS NOT NULL AND completed_at IS NOT NULL
            """,
            (COMPLETED,),
        )
        average = (await cursor.fetchone())[0]

    by_status = {s.value: counts.get(s.value, 0) for s in TaskStatus}
    return {
        "total_tasks": sum(by_status.values()),
        "by_status": by_status,
        "average_processing_seconds": round(average, 3) if average is not None else None,
    }