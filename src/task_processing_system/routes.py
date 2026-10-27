"""HTTP API for submitting and managing tasks.

Every route here requires a valid Bearer token (the whole router is guarded) and
only ever sees the logged-in user's own tasks.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response

from . import repository as repo
from .auth import CurrentUser, get_current_user
from .schemas import (
    MAX_RETRIES,
    MAX_IDEMPOTENCY_KEY_LENGTH,
    StatsResponse,
    TaskCreate,
    TaskListResponse,
    TaskResponse,
    TaskResultResponse,
    TaskStatus,
    TaskType,
)

# The router-level dependency means a route added here later cannot accidentally
# be public; the per-route `user` parameter is how handlers get the user itself.
router = APIRouter(dependencies=[Depends(get_current_user)])


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
    user: CurrentUser,
    idempotency_key: Annotated[str | None, Header()] = None,
) -> Any:
    """Submit a task for background processing.

    Returns 202 immediately with the task ID. Send an `Idempotency-Key` header
    to make retries safe: repeating the same request with the same key (per user) returns
    the original task (200) instead of creating a new one. Reusing a key with a
    different body returns 409.
    """
    key = _clean_idempotency_key(idempotency_key)
    payload = body.payload.model_dump()

    try:
        task, created = await repo.create_task(user["id"], body.type, payload, key)
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
    user: CurrentUser,
    status: TaskStatus | None = None,
    task_type: Annotated[TaskType | None, Query(alias="type")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Any:
    """List your tasks, newest first. Filter with `status` and `type`; page with `limit`/`offset`."""
    items, total = await repo.list_tasks(
        user["id"], status=status, task_type=task_type, limit=limit, offset=offset
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task_status(task_id: str, user: CurrentUser) -> Any:
    """Check the status of one of your tasks."""
    task = await repo.get_task(task_id, user["id"])
    if task is None:
        raise _not_found(task_id)
    return task


@router.get("/tasks/{task_id}/result", response_model=TaskResultResponse, tags=["tasks"])
async def get_task_result(task_id: str, user: CurrentUser) -> Any:
    """Fetch the result of a completed task (409 if it has no result yet)."""
    task = await repo.get_task_result(task_id, user["id"])
    if task is None:
        raise _not_found(task_id)

    if task["status"] != TaskStatus.COMPLETED.value:
        detail = f"Task has no result (status: {task['status']})"
        if task["status"] == TaskStatus.FAILED.value and task["error"]:
            detail += f"; error: {task['error']}"
        raise HTTPException(status_code=409, detail=detail)
    return task


@router.post("/tasks/{task_id}/cancel", response_model=TaskResponse, tags=["tasks"])
async def cancel_task(task_id: str, user: CurrentUser) -> Any:
    """Cancel one of your tasks. Only tasks that are still `pending` can be cancelled."""
    outcome, task = await repo.cancel_task(task_id, user["id"])
    if outcome == "not_found":
        raise _not_found(task_id)
    if outcome == "not_cancellable":
        raise HTTPException(
            status_code=409,
            detail=f"Only pending tasks can be cancelled (current status: {task['status']})",
        )
    return task


@router.get("/stats", response_model=StatsResponse, tags=["stats"])
async def get_stats(user: CurrentUser) -> Any:
    """Your task counts by status and average processing time of completed tasks."""
    return await repo.get_stats(user["id"])


@router.post("/tasks/{task_id}/retry", response_model=TaskResponse, status_code=202, tags=["tasks"])
async def retry_task(task_id: str, request: Request, user: CurrentUser) -> Any:
    """Retry one of your failed tasks (at most 3 times)."""
    outcome, task = await repo.retry_task(task_id, user["id"])
    if outcome == "not_found":
        raise _not_found(task_id)
    if outcome == "not_failed":
        raise HTTPException(
            status_code=409,
            detail=f"Only failed tasks can be retried (current status: {task['status']})",
        )
    if outcome == "limit_reached":
        raise HTTPException(
            status_code=409,
            detail=f"Retry limit reached ({MAX_RETRIES} retries)",
        )
    request.app.state.worker_pool.enqueue(task["id"])
    return task