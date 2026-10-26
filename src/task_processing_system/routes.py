"""HTTP API for submitting and managing tasks."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response

from . import repository as repo
from .schemas import (
    MAX_IDEMPOTENCY_KEY_LENGTH,
    StatsResponse,
    TaskCreate,
    TaskListResponse,
    TaskResponse,
    TaskResultResponse,
    TaskStatus,
    TaskType,
)

router = APIRouter()


def _not_found(task_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Task {task_id} not found")


def _clean_idempotency_key(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = raw.strip()
    if not key or len(key) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Idempotency-Key must be 1-{MAX_IDEMPOTENCY_KEY_LENGTH} characters",
        )
    return key


@router.post("/tasks", response_model=TaskResponse, status_code=202, tags=["tasks"])
async def submit_task(
    body: TaskCreate,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str | None, Header()] = None,
) -> Any:
    """Submit a task for background processing.

    Returns 202 immediately with the task ID. Send an `Idempotency-Key` header
    to make retries safe: repeating the same request with the same key returns
    the original task (200) instead of creating a new one. Reusing a key with a
    different body returns 409.
    """
    key = _clean_idempotency_key(idempotency_key)
    payload = body.payload.model_dump()

    try:
        task, created = await repo.create_task(body.type, payload, key)
    except repo.IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if created:
        request.app.state.worker_pool.enqueue(task["id"])
    else:
        response.status_code = 200
        response.headers["Idempotent-Replayed"] = "true"
    return task


@router.get("/tasks", response_model=TaskListResponse, tags=["tasks"])
async def list_tasks(
    status: TaskStatus | None = None,
    task_type: Annotated[TaskType | None, Query(alias="type")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Any:
    """List tasks, newest first. Filter with `status` and `type`; page with `limit`/`offset`."""
    items, total = await repo.list_tasks(
        status=status, task_type=task_type, limit=limit, offset=offset
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task_status(task_id: str) -> Any:
    """Check a task's current status."""
    task = await repo.get_task(task_id)
    if task is None:
        raise _not_found(task_id)
    return task


@router.get("/tasks/{task_id}/result", response_model=TaskResultResponse, tags=["tasks"])
async def get_task_result(task_id: str) -> Any:
    """Fetch the result of a completed task (409 if it has no result yet)."""
    task = await repo.get_task(task_id)
    if task is None:
        raise _not_found(task_id)

    if task["status"] != TaskStatus.COMPLETED.value:
        detail = f"Task has no result (status: {task['status']})"
        if task["status"] == TaskStatus.FAILED.value and task["error"]:
            detail += f"; error: {task['error']}"
        raise HTTPException(status_code=409, detail=detail)
    return task


@router.post("/tasks/{task_id}/cancel", response_model=TaskResponse, tags=["tasks"])
async def cancel_task(task_id: str) -> Any:
    """Cancel a task. Only tasks that are still `pending` can be cancelled."""
    outcome, task = await repo.cancel_task(task_id)
    if outcome == "not_found":
        raise _not_found(task_id)
    if outcome == "not_cancellable":
        raise HTTPException(
            status_code=409,
            detail=f"Only pending tasks can be cancelled (current status: {task['status']})",
        )
    return task


@router.get("/stats", response_model=StatsResponse, tags=["stats"])
async def get_stats() -> Any:
    """Task counts by status and average processing time of completed tasks."""
    return await repo.get_stats()