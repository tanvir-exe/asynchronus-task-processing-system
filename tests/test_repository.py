import asyncio

from task_processing_system import repository as repo

TEXT = "text_analysis"
TX = "transaction_analysis"


async def make(payload=None, key=None, task_type=TEXT):
    return await repo.create_task(task_type, payload or {"text": "hello"}, key)


# ---------------------------------------------------------------------------
# Create / idempotency
# ---------------------------------------------------------------------------


async def test_create_task_defaults(db):
    task, created = await make()
    assert created is True
    assert task["status"] == "pending"
    assert task["type"] == TEXT
    assert task["started_at"] is None
    # Large columns are not loaded for summaries.
    assert "payload" not in task and "result" not in task
    assert len(task["id"]) == 36


async def test_same_key_same_body_returns_existing_task(db):
    first, created1 = await make(key="abc")
    second, created2 = await make(key="abc")
    assert created1 is True
    assert created2 is False
    assert first["id"] == second["id"]
    _, total = await repo.list_tasks()
    assert total == 1


async def test_same_key_different_body_raises_conflict(db):
    first, _ = await make({"text": "one"}, key="abc")
    try:
        await make({"text": "two"}, key="abc")
    except repo.IdempotencyConflictError as exc:
        assert exc.task_id == first["id"]
        return
    raise AssertionError("expected IdempotencyConflictError")


async def test_same_key_different_type_raises_conflict(db):
    await make({"text": "x"}, key="abc")
    try:
        await make({"transactions": []}, key="abc", task_type=TX)
    except repo.IdempotencyConflictError:
        return
    raise AssertionError("expected IdempotencyConflictError")


async def test_no_key_creates_separate_tasks(db):
    a, _ = await make()
    b, _ = await make()
    assert a["id"] != b["id"]
    _, total = await repo.list_tasks()
    assert total == 2


async def test_concurrent_requests_with_same_key_create_one_task(db):
    results = await asyncio.gather(*(make(key="race") for _ in range(20)))
    assert len({task["id"] for task, _ in results}) == 1
    assert sum(1 for _, created in results if created) == 1
    _, total = await repo.list_tasks()
    assert total == 1


async def test_get_task(db):
    task, _ = await make({"text": "abc"})
    fetched = await repo.get_task(task["id"])
    assert fetched["id"] == task["id"]
    assert fetched["status"] == "pending"
    assert await repo.get_task("does-not-exist") is None


async def test_summary_queries_do_not_load_payload_or_result(db):
    task, _ = await make({"text": "abc"})
    await repo.claim_task(task["id"])
    await repo.complete_task(task["id"], {"total_words": 1})

    created, _ = await make({"text": "other"}, key="k")
    replayed, _ = await make({"text": "other"}, key="k")
    items, _ = await repo.list_tasks()
    outcome, cancelled = await repo.cancel_task(created["id"])

    summaries = [await repo.get_task(task["id"]), created, replayed, cancelled, *items]
    for summary in summaries:
        assert "payload" not in summary
        assert "result" not in summary


async def test_get_task_result(db):
    task, _ = await make()
    before = await repo.get_task_result(task["id"])
    assert before["status"] == "pending" and before["result"] is None

    await repo.claim_task(task["id"])
    await repo.complete_task(task["id"], {"total_words": 3})
    after = await repo.get_task_result(task["id"])
    assert after["status"] == "completed"
    assert after["result"] == {"total_words": 3}
    assert "payload" not in after
    assert await repo.get_task_result("does-not-exist") is None


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


async def test_claim_task_only_succeeds_once(db):
    task, _ = await make()
    claimed = await repo.claim_task(task["id"])
    assert claimed["status"] == "processing"
    assert claimed["started_at"] is not None
    assert claimed["payload"] == {"text": "hello"}  # the worker needs this
    assert await repo.claim_task(task["id"]) is None


async def test_concurrent_claims_have_one_winner(db):
    task, _ = await make()
    results = await asyncio.gather(*(repo.claim_task(task["id"]) for _ in range(10)))
    assert sum(1 for r in results if r is not None) == 1


async def test_cancel_pending_task(db):
    task, _ = await make()
    outcome, cancelled = await repo.cancel_task(task["id"])
    assert outcome == "cancelled"
    assert cancelled["status"] == "cancelled"
    assert cancelled["completed_at"] is not None


async def test_cancelled_task_cannot_be_claimed(db):
    task, _ = await make()
    await repo.cancel_task(task["id"])
    assert await repo.claim_task(task["id"]) is None
    assert (await repo.get_task(task["id"]))["status"] == "cancelled"


async def test_cancel_after_claim_is_rejected(db):
    task, _ = await make()
    await repo.claim_task(task["id"])
    outcome, current = await repo.cancel_task(task["id"])
    assert outcome == "not_cancellable"
    assert current["status"] == "processing"


async def test_cancel_completed_and_missing(db):
    task, _ = await make()
    await repo.claim_task(task["id"])
    await repo.complete_task(task["id"], {"ok": True})
    outcome, _ = await repo.cancel_task(task["id"])
    assert outcome == "not_cancellable"
    outcome, none_task = await repo.cancel_task("missing")
    assert outcome == "not_found"
    assert none_task is None


async def test_complete_task_stores_result(db):
    task, _ = await make()
    await repo.claim_task(task["id"])
    assert await repo.complete_task(task["id"], {"total_words": 3}) is True
    done = await repo.get_task_result(task["id"])
    assert done["status"] == "completed"
    assert done["result"] == {"total_words": 3}
    assert done["completed_at"] is not None
    assert done["error"] is None


async def test_complete_requires_processing_state(db):
    task, _ = await make()
    assert await repo.complete_task(task["id"], {}) is False  # still pending
    await repo.cancel_task(task["id"])
    assert await repo.complete_task(task["id"], {}) is False  # cancelled


async def test_fail_task_stores_error(db):
    task, _ = await make()
    await repo.claim_task(task["id"])
    assert await repo.fail_task(task["id"], "boom") is True
    failed = await repo.get_task(task["id"])
    assert failed["status"] == "failed"
    assert failed["error"] == "boom"


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


async def test_list_filters_and_pagination(db):
    ids = []
    for i in range(5):
        task, _ = await make({"text": f"t{i}"})
        ids.append(task["id"])
    await repo.claim_task(ids[0])
    await repo.complete_task(ids[0], {})
    await make({"transactions": []}, task_type=TX)

    items, total = await repo.list_tasks()
    assert total == 6
    assert items[0]["type"] == TX  # newest first

    items, total = await repo.list_tasks(status="completed")
    assert total == 1 and items[0]["id"] == ids[0]

    items, total = await repo.list_tasks(status="pending", task_type=TEXT)
    assert total == 4

    page1, total = await repo.list_tasks(limit=2, offset=0)
    page2, _ = await repo.list_tasks(limit=2, offset=2)
    assert total == 6
    assert len(page1) == 2 and len(page2) == 2
    assert not {t["id"] for t in page1} & {t["id"] for t in page2}

    items, total = await repo.list_tasks(limit=10, offset=100)
    assert items == [] and total == 6


async def test_list_empty(db):
    items, total = await repo.list_tasks()
    assert items == [] and total == 0


# ---------------------------------------------------------------------------
# Recovery and stats
# ---------------------------------------------------------------------------


async def test_reset_stale_processing_and_pending_ids(db):
    a, _ = await make({"text": "a"})
    b, _ = await make({"text": "b"})
    c, _ = await make({"text": "c"})
    await repo.claim_task(b["id"])  # simulate crash mid-processing
    await repo.claim_task(c["id"])
    await repo.complete_task(c["id"], {})

    assert await repo.reset_stale_processing() == 1
    reset = await repo.get_task(b["id"])
    assert reset["status"] == "pending" and reset["started_at"] is None
    assert await repo.get_pending_ids() == [a["id"], b["id"]]


async def test_stats_empty(db):
    stats = await repo.get_stats()
    assert stats["total_tasks"] == 0
    assert stats["by_status"] == {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }
    assert stats["average_processing_seconds"] is None


async def test_stats_counts_and_average(db):
    done, _ = await make({"text": "1"})
    failed, _ = await make({"text": "2"})
    cancelled, _ = await make({"text": "3"})
    await make({"text": "4"})

    await repo.claim_task(done["id"])
    await asyncio.sleep(0.05)
    await repo.complete_task(done["id"], {})
    await repo.claim_task(failed["id"])
    await repo.fail_task(failed["id"], "x")
    await repo.cancel_task(cancelled["id"])

    stats = await repo.get_stats()
    assert stats["total_tasks"] == 4
    assert stats["by_status"]["completed"] == 1
    assert stats["by_status"]["failed"] == 1
    assert stats["by_status"]["cancelled"] == 1
    assert stats["by_status"]["pending"] == 1
    assert stats["average_processing_seconds"] >= 0.04