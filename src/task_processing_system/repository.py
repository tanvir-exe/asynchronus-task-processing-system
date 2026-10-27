"""All database access for tasks lives here.

Concurrency notes:
  * Idempotency is enforced by the UNIQUE (user_id, idempotency_key) constraint (keys are per user).
    We insert first and handle IntegrityError, instead of "check then insert",
    which would be racy when two identical requests arrive together.
  * State changes are single conditional UPDATEs
    (``... WHERE id = ? AND status = 'pending'``) and we look at rowcount to
    see whether we won. That makes claim-vs-cancel race-safe: exactly one of
    the worker or the cancel request can move a pending task.

Ownership:
  Every task belongs to a user. Functions that serve API requests (create,
  get, list, cancel, stats) REQUIRE a user_id and only ever touch that user's
  rows; a task owned by someone else looks exactly like a task that does not
  exist. Only the background-worker functions (claim, complete, fail, startup
  recovery) work across all users, because workers serve everyone.

SQL fragments built with f-strings only ever contain fixed text; every value
goes through ``?`` placeholders.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal
from .schemas import MAX_RETRIES, TaskStatus
from .database import get_connection
from .schemas import TaskStatus

PENDING = TaskStatus.PENDING.value
PROCESSING = TaskStatus.PROCESSING.value
COMPLETED = TaskStatus.COMPLETED.value
FAILED = TaskStatus.FAILED.value
CANCELLED = TaskStatus.CANCELLED.value

CancelOutcome = Literal["cancelled", "not_found", "not_cancellable"]
RetryOutcome = Literal["retried", "not_found", "not_failed", "limit_reached"]
# Columns the API returns for status/list/cancel. payload and result are
# deliberately left out: they can be ~1 MB each, and parsing them on the event
# loop for a response that never includes them freezes the server.
SUMMARY_COLUMNS = (
    "id, idempotency_key, type, status, error, retry_count, "
    "created_at, started_at, completed_at"
)


class IdempotencyConflictError(Exception):
    """The idempotency key was already used with a different request body."""

    def __init__(self, task_id: str, key: str) -> None:
        super().__init__(f"Idempotency key {key!r} was already used with a different request")
        self.task_id = task_id
        self.key = key


class UsernameTakenError(Exception):
    """A user with this username (case-insensitive) already exists."""


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
    conn: Any,
    task_id: str,
    columns: str = SUMMARY_COLUMNS,
    user_id: str | None = None,
) -> dict[str, Any] | None:
    """Fetch one task. With ``user_id`` only that user's task can be found."""
    # `columns` only ever comes from constants in this file, never user input.
    sql = f"SELECT {columns} FROM tasks WHERE id = ?"
    params: list[Any] = [task_id]
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)
    cursor = await conn.execute(sql, params)
    row = await cursor.fetchone()
    return _row_to_task(row) if row else None


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


async def create_user(username: str, password_hash: str) -> dict[str, Any]:
    """Create a user. Raises UsernameTakenError if the name exists (any case)."""
    user_id = str(uuid.uuid4())
    created_at = utc_now()
    async with get_connection() as conn:
        try:
            await conn.execute(
                "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, username, password_hash, created_at),
            )
            await conn.commit()
        except sqlite3.IntegrityError as exc:
            await conn.rollback()
            raise UsernameTakenError(username) from exc
    return {"id": user_id, "username": username, "created_at": created_at}


async def get_user_by_username(username: str) -> dict[str, Any] | None:
    """Includes password_hash: only for the login check, never return it to clients."""
    async with get_connection() as conn:
        cursor = await conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),  # column is COLLATE NOCASE, so this match ignores case
        )
        row = await cursor.fetchone()
    return dict(row) if row else None


async def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    async with get_connection() as conn:
        cursor = await conn.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?", (user_id,)
        )
        row = await cursor.fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Create / read
# ---------------------------------------------------------------------------


async def create_task(
    user_id: str,
    task_type: Any,
    payload: dict[str, Any],
    idempotency_key: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Insert a pending task.

    Returns ``(task, created)``. ``created`` is False when an earlier task of the
    same user with the same idempotency key and request body already exists, in
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
                    (id, user_id, idempotency_key, request_hash, type, status, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    user_id,
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
                f"SELECT request_hash, {SUMMARY_COLUMNS} FROM tasks "
                "WHERE user_id = ? AND idempotency_key = ?",
                (user_id, idempotency_key),
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


async def retry_task(
    task_id: str, user_id: str, max_retries: int = MAX_RETRIES
) -> tuple[RetryOutcome, dict[str, Any] | None]:
    """Move one of the user's failed tasks back to pending, up to ``max_retries`` times.

    A single conditional UPDATE, so two simultaneous retry requests cannot both win.
    """
    async with get_connection() as conn:
        cursor = await conn.execute(
            "UPDATE tasks SET status = ?, error = NULL, started_at = NULL, "
            "completed_at = NULL, retry_count = retry_count + 1 "
            "WHERE id = ? AND user_id = ? AND status = ? AND retry_count < ?",
            (PENDING, task_id, user_id, FAILED, max_retries),
        )
        await conn.commit()
        task = await _fetch(conn, task_id, user_id=user_id)
        if cursor.rowcount == 1:
            return "retried", task
        if task is None:
            return "not_found", None
        if task["status"] != FAILED:
            return "not_failed", task
        return "limit_reached", task

async def get_task(task_id: str, user_id: str) -> dict[str, Any] | None:
    """Task summary (no payload or result), only if it belongs to ``user_id``."""
    async with get_connection() as conn:
        return await _fetch(conn, task_id, user_id=user_id)


async def get_task_result(task_id: str, user_id: str) -> dict[str, Any] | None:
    """Task summary plus the parsed result. Used only by /tasks/{id}/result."""
    async with get_connection() as conn:
        return await _fetch(conn, task_id, f"{SUMMARY_COLUMNS}, result", user_id=user_id)


async def list_tasks(
    user_id: str,
    status: Any = None,
    task_type: Any = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return ``(items, total)`` for one user, newest first. ``total`` ignores limit/offset."""
    conditions: list[str] = ["user_id = ?"]
    params: list[Any] = [user_id]
    if status is not None:
        conditions.append("status = ?")
        params.append(_value(status))
    if task_type is not None:
        conditions.append("type = ?")
        params.append(_value(task_type))
    where = f"WHERE {' AND '.join(conditions)}"

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


async def cancel_task(
    task_id: str, user_id: str
) -> tuple[CancelOutcome, dict[str, Any] | None]:
    """Cancel one of the user's tasks, only if it is still pending."""
    async with get_connection() as conn:
        cursor = await conn.execute(
            "UPDATE tasks SET status = ?, completed_at = ? "
            "WHERE id = ? AND user_id = ? AND status = ?",
            (CANCELLED, utc_now(), task_id, user_id, PENDING),
        )
        await conn.commit()
        task = await _fetch(conn, task_id, user_id=user_id)
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


async def reset_stale_processing(older_than_seconds: float = 0) -> int:
    """Put abandoned 'processing' tasks back to 'pending'.

    Only tasks that have been processing for at least ``older_than_seconds``
    are reset. With several server processes sharing one database, resetting
    every 'processing' row at startup would steal tasks that another live
    process is running right now (they would then run twice). The age
    threshold leaves those alone; it must be longer than any task normally
    takes. A threshold of 0 resets everything (single-process behaviour).

    Processing is deterministic and side-effect free, so re-running is safe.
    Returns how many tasks were reset.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
    ).isoformat(timespec="milliseconds")
    async with get_connection() as conn:
        cursor = await conn.execute(
            "UPDATE tasks SET status = ?, started_at = NULL "
            "WHERE status = ? AND started_at <= ?",
            (PENDING, PROCESSING, cutoff),
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


async def get_stats(user_id: str) -> dict[str, Any]:
    """Statistics for one user's tasks."""
    async with get_connection() as conn:
        cursor = await conn.execute(
            "SELECT status, COUNT(*) AS n FROM tasks WHERE user_id = ? GROUP BY status",
            (user_id,),
        )
        counts = {row["status"]: row["n"] for row in await cursor.fetchall()}

        cursor = await conn.execute(
            """
            SELECT AVG((julianday(completed_at) - julianday(started_at)) * 86400.0)
              FROM tasks
             WHERE user_id = ? AND status = ?
               AND started_at IS NOT NULL AND completed_at IS NOT NULL
            """,
            (user_id, COMPLETED),
        )
        average = (await cursor.fetchone())[0]

    by_status = {s.value: counts.get(s.value, 0) for s in TaskStatus}
    return {
        "total_tasks": sum(by_status.values()),
        "by_status": by_status,
        "average_processing_seconds": round(average, 3) if average is not None else None,
    }