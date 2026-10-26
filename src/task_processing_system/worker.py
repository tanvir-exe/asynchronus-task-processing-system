"""Background workers that execute queued tasks.

Design:
  * The API saves a task as 'pending' in SQLite, then puts its ID on an
    in-memory asyncio.Queue. SQLite is the source of truth; the queue is
    only a wake-up signal. If the server restarts, pending tasks are re-queued
    from the database (see TaskWorkerPool.start).
  * N worker coroutines pull IDs from the queue. Each one first *claims* the
    task with an atomic UPDATE (pending -> processing). If the claim fails the
    task was cancelled in the meantime and is skipped.
  * The analysis functions are CPU-bound, so they run in a thread via
    asyncio.to_thread and never block the event loop (other requests keep
    being served).
  * Any exception inside a task marks that task 'failed' and the worker moves
    on. Workers only stop when cancelled at shutdown.

Environment variables (both optional):
  TASK_WORKERS                    number of workers (default 3)
  TASK_PROCESSING_DELAY_SECONDS   artificial delay per task, handy for demoing
                                  cancellation and concurrency (default 0)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable

from . import repository as repo
from .processors import analyze_text, analyze_transactions

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 3


def _run_text_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    return analyze_text(payload["text"])


def _run_transaction_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    return analyze_transactions(payload["transactions"])


PROCESSORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "text_analysis": _run_text_analysis,
    "transaction_analysis": _run_transaction_analysis,
}


def get_worker_count() -> int:
    try:
        return max(1, int(os.environ.get("TASK_WORKERS", DEFAULT_WORKERS)))
    except ValueError:
        return DEFAULT_WORKERS


def _processing_delay() -> float:
    try:
        return max(0.0, float(os.environ.get("TASK_PROCESSING_DELAY_SECONDS", "0")))
    except ValueError:
        return 0.0


async def _safe_fail(task_id: str, exc: Exception) -> None:
    try:
        await repo.fail_task(task_id, f"{type(exc).__name__}: {exc}")
    except Exception:
        logger.exception("Could not record failure for task %s", task_id)


async def process_task(task_id: str) -> None:
    """Run one task end to end. Never raises for task-level errors."""
    task = await repo.claim_task(task_id)
    if task is None:
        logger.info("Task %s was not claimable (cancelled or already taken)", task_id)
        return

    try:
        delay = _processing_delay()
        if delay:
            await asyncio.sleep(delay)

        processor = PROCESSORS.get(task["type"])
        if processor is None:
            raise ValueError(f"Unknown task type: {task['type']}")

        result = await asyncio.to_thread(processor, task["payload"])
        await repo.complete_task(task_id, result)
    except Exception as exc:  # CancelledError is not an Exception subclass
        logger.exception("Task %s failed", task_id)
        await _safe_fail(task_id, exc)


class TaskWorkerPool:
    def __init__(self, num_workers: int | None = None) -> None:
        self.num_workers = num_workers or get_worker_count()
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        """Recover unfinished work from the database, then start the workers."""
        reset = await repo.reset_stale_processing()
        if reset:
            logger.warning("Re-queued %d task(s) interrupted by a previous shutdown", reset)
        for task_id in await repo.get_pending_ids():
            self.queue.put_nowait(task_id)

        self._workers = [
            asyncio.create_task(self._worker_loop(i), name=f"task-worker-{i}")
            for i in range(self.num_workers)
        ]
        logger.info("Started %d worker(s)", self.num_workers)

    async def stop(self) -> None:
        """Cancel the workers. A task caught mid-run stays 'processing' in the
        database and is re-queued by start() on the next boot."""
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    def enqueue(self, task_id: str) -> None:
        self.queue.put_nowait(task_id)

    async def _worker_loop(self, index: int) -> None:
        while True:
            task_id = await self.queue.get()
            try:
                await process_task(task_id)
            except Exception:
                # Last line of defence (e.g. the database is unreachable).
                logger.exception("Worker %d hit an unexpected error on task %s", index, task_id)
            finally:
                self.queue.task_done()