import asyncio

from task_processing_system import repository as repo
from task_processing_system.database import get_connection
from task_processing_system.worker import TaskWorkerPool, process_task

GOOD_TEXT = {"text": "The cat sat. The cat ran away fast!"}


async def wait_for_status(user_id, task_id, status, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        task = await repo.get_task(task_id, user_id)
        if task["status"] == status:
            return task
        if loop.time() > deadline:
            raise AssertionError(f"task still {task['status']!r}, expected {status!r}")
        await asyncio.sleep(0.02)


async def test_process_task_completes_text_analysis(db, user_id):
    task, _ = await repo.create_task(user_id, "text_analysis", GOOD_TEXT)
    await process_task(task["id"])
    done = await repo.get_task_result(task["id"], user_id)
    assert done["status"] == "completed"
    assert done["result"]["total_words"] == 8
    assert done["started_at"] and done["completed_at"]


async def test_process_task_completes_transaction_analysis(db, user_id):
    payload = {
        "transactions": [
            {"transaction_id": "t1", "customer_id": "a", "amount": 10, "timestamp": "2024-01-01"},
            {"transaction_id": "t1", "customer_id": "a", "amount": 10, "timestamp": "2024-01-01"},
            {"bad": "record"},
        ]
    }
    task, _ = await repo.create_task(user_id, "transaction_analysis", payload)
    await process_task(task["id"])
    done = await repo.get_task_result(task["id"], user_id)
    assert done["status"] == "completed"
    assert done["result"]["total_revenue"] == 10.0
    assert done["result"]["duplicate_count"] == 1
    assert done["result"]["invalid_count"] == 1


async def test_processor_error_marks_task_failed(db, user_id):
    # "transactions" is not a list, so the processor raises TypeError.
    task, _ = await repo.create_task(user_id, "transaction_analysis", {"transactions": "nope"})
    await process_task(task["id"])
    failed = await repo.get_task_result(task["id"], user_id)
    assert failed["status"] == "failed"
    assert "TypeError" in failed["error"]
    assert failed["result"] is None  # nothing was stored for a failed task


async def test_process_task_skips_cancelled_task(db, user_id):
    task, _ = await repo.create_task(user_id, "text_analysis", GOOD_TEXT)
    await repo.cancel_task(task["id"], user_id)
    await process_task(task["id"])
    assert (await repo.get_task(task["id"], user_id))["status"] == "cancelled"


async def test_process_task_ignores_unknown_id(db, user_id):
    await process_task("does-not-exist")  # must not raise


async def test_pool_processes_many_tasks_concurrently(db, user_id):
    pool = TaskWorkerPool(num_workers=3)
    await pool.start()
    try:
        ids = []
        for i in range(15):
            task, _ = await repo.create_task(user_id, "text_analysis", {"text": f"word{i} word"})
            pool.enqueue(task["id"])
            ids.append(task["id"])
        for task_id in ids:
            await wait_for_status(user_id, task_id, "completed")
    finally:
        await pool.stop()
    stats = await repo.get_stats(user_id)
    assert stats["by_status"]["completed"] == 15


async def test_worker_survives_a_failing_task(db, user_id):
    pool = TaskWorkerPool(num_workers=1)
    await pool.start()
    try:
        bad, _ = await repo.create_task(user_id, "transaction_analysis", {"transactions": "nope"})
        good, _ = await repo.create_task(user_id, "text_analysis", GOOD_TEXT)
        pool.enqueue(bad["id"])
        pool.enqueue(good["id"])
        await wait_for_status(user_id, bad["id"], "failed")
        await wait_for_status(user_id, good["id"], "completed")
    finally:
        await pool.stop()


async def test_pool_recovers_pending_and_interrupted_tasks_on_start(db, user_id, monkeypatch):
    monkeypatch.setenv("TASK_STALE_AFTER_SECONDS", "0")  # treat all processing as abandoned
    pending, _ = await repo.create_task(user_id, "text_analysis", {"text": "pending one"})
    stale, _ = await repo.create_task(user_id, "text_analysis", {"text": "stale one"})
    await repo.claim_task(stale["id"])  # simulates a crash mid-processing

    pool = TaskWorkerPool(num_workers=2)
    await pool.start()
    try:
        await wait_for_status(user_id, pending["id"], "completed")
        await wait_for_status(user_id, stale["id"], "completed")
    finally:
        await pool.stop()


async def test_cancel_pending_task_while_worker_is_busy(db, user_id, monkeypatch):
    monkeypatch.setenv("TASK_PROCESSING_DELAY_SECONDS", "0.3")
    pool = TaskWorkerPool(num_workers=1)
    await pool.start()
    try:
        first, _ = await repo.create_task(user_id, "text_analysis", {"text": "first"})
        second, _ = await repo.create_task(user_id, "text_analysis", {"text": "second"})
        pool.enqueue(first["id"])
        pool.enqueue(second["id"])

        await wait_for_status(user_id, first["id"], "processing")
        outcome, _ = await repo.cancel_task(second["id"], user_id)
        assert outcome == "cancelled"

        await wait_for_status(user_id, first["id"], "completed")
        await asyncio.sleep(0.5)  # give the worker time to (not) pick up the second
        assert (await repo.get_task(second["id"], user_id))["status"] == "cancelled"
    finally:
        await pool.stop()


async def test_stop_is_clean_and_idempotent(db, user_id):
    pool = TaskWorkerPool(num_workers=2)
    await pool.start()
    await pool.stop()
    await pool.stop()


async def test_start_leaves_recently_started_tasks_alone_but_recovers_old_ones(db, user_id, monkeypatch):
    """Simulates two server processes: a task another live process just started
    must not be stolen, while one abandoned long ago must be recovered."""
    monkeypatch.setenv("TASK_STALE_AFTER_SECONDS", "60")

    live, _ = await repo.create_task(user_id, "text_analysis", {"text": "running elsewhere"})
    abandoned, _ = await repo.create_task(user_id, "text_analysis", {"text": "crashed long ago"})
    await repo.claim_task(live["id"])
    await repo.claim_task(abandoned["id"])
    async with get_connection() as conn:
        await conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?",
            ("2020-01-01T00:00:00.000+00:00", abandoned["id"]),
        )
        await conn.commit()

    pool = TaskWorkerPool(num_workers=1)
    await pool.start()
    try:
        await wait_for_status(user_id, abandoned["id"], "completed")
        await asyncio.sleep(0.3)
        still_live = await repo.get_task(live["id"], user_id)
        assert still_live["status"] == "processing"  # not reset, not run twice
    finally:
        await pool.stop()