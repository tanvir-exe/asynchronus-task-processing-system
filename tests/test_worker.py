import asyncio

from task_processing_system import repository as repo
from task_processing_system.worker import TaskWorkerPool, process_task

GOOD_TEXT = {"text": "The cat sat. The cat ran away fast!"}


async def wait_for_status(task_id, status, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        task = await repo.get_task(task_id)
        if task["status"] == status:
            return task
        if loop.time() > deadline:
            raise AssertionError(f"task still {task['status']!r}, expected {status!r}")
        await asyncio.sleep(0.02)


async def test_process_task_completes_text_analysis(db):
    task, _ = await repo.create_task("text_analysis", GOOD_TEXT)
    await process_task(task["id"])
    done = await repo.get_task(task["id"])
    assert done["status"] == "completed"
    assert done["result"]["total_words"] == 8
    assert done["started_at"] and done["completed_at"]


async def test_process_task_completes_transaction_analysis(db):
    payload = {
        "transactions": [
            {"transaction_id": "t1", "customer_id": "a", "amount": 10, "timestamp": "2024-01-01"},
            {"transaction_id": "t1", "customer_id": "a", "amount": 10, "timestamp": "2024-01-01"},
            {"bad": "record"},
        ]
    }
    task, _ = await repo.create_task("transaction_analysis", payload)
    await process_task(task["id"])
    done = await repo.get_task(task["id"])
    assert done["status"] == "completed"
    assert done["result"]["total_revenue"] == 10.0
    assert done["result"]["duplicate_count"] == 1
    assert done["result"]["invalid_count"] == 1


async def test_processor_error_marks_task_failed(db):
    # "transactions" is not a list, so the processor raises TypeError.
    task, _ = await repo.create_task("transaction_analysis", {"transactions": "nope"})
    await process_task(task["id"])
    failed = await repo.get_task(task["id"])
    assert failed["status"] == "failed"
    assert "TypeError" in failed["error"]
    assert failed["result"] is None


async def test_process_task_skips_cancelled_task(db):
    task, _ = await repo.create_task("text_analysis", GOOD_TEXT)
    await repo.cancel_task(task["id"])
    await process_task(task["id"])
    assert (await repo.get_task(task["id"]))["status"] == "cancelled"


async def test_process_task_ignores_unknown_id(db):
    await process_task("does-not-exist")  # must not raise


async def test_pool_processes_many_tasks_concurrently(db):
    pool = TaskWorkerPool(num_workers=3)
    await pool.start()
    try:
        ids = []
        for i in range(15):
            task, _ = await repo.create_task("text_analysis", {"text": f"word{i} word"})
            pool.enqueue(task["id"])
            ids.append(task["id"])
        for task_id in ids:
            await wait_for_status(task_id, "completed")
    finally:
        await pool.stop()
    stats = await repo.get_stats()
    assert stats["by_status"]["completed"] == 15


async def test_worker_survives_a_failing_task(db):
    pool = TaskWorkerPool(num_workers=1)
    await pool.start()
    try:
        bad, _ = await repo.create_task("transaction_analysis", {"transactions": "nope"})
        good, _ = await repo.create_task("text_analysis", GOOD_TEXT)
        pool.enqueue(bad["id"])
        pool.enqueue(good["id"])
        await wait_for_status(bad["id"], "failed")
        await wait_for_status(good["id"], "completed")
    finally:
        await pool.stop()


async def test_pool_recovers_pending_and_interrupted_tasks_on_start(db):
    pending, _ = await repo.create_task("text_analysis", {"text": "pending one"})
    stale, _ = await repo.create_task("text_analysis", {"text": "stale one"})
    await repo.claim_task(stale["id"])  # simulates a crash mid-processing

    pool = TaskWorkerPool(num_workers=2)
    await pool.start()
    try:
        await wait_for_status(pending["id"], "completed")
        await wait_for_status(stale["id"], "completed")
    finally:
        await pool.stop()


async def test_cancel_pending_task_while_worker_is_busy(db, monkeypatch):
    monkeypatch.setenv("TASK_PROCESSING_DELAY_SECONDS", "0.3")
    pool = TaskWorkerPool(num_workers=1)
    await pool.start()
    try:
        first, _ = await repo.create_task("text_analysis", {"text": "first"})
        second, _ = await repo.create_task("text_analysis", {"text": "second"})
        pool.enqueue(first["id"])
        pool.enqueue(second["id"])

        await wait_for_status(first["id"], "processing")
        outcome, _ = await repo.cancel_task(second["id"])
        assert outcome == "cancelled"

        await wait_for_status(first["id"], "completed")
        await asyncio.sleep(0.5)  # give the worker time to (not) pick up the second
        assert (await repo.get_task(second["id"]))["status"] == "cancelled"
    finally:
        await pool.stop()


async def test_stop_is_clean_and_idempotent(db):
    pool = TaskWorkerPool(num_workers=2)
    await pool.start()
    await pool.stop()
    await pool.stop()