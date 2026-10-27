import asyncio
import sqlite3

from task_processing_system import repository as repo

TEXT = "text_analysis"
TX = "transaction_analysis"


async def make(uid, payload=None, key=None, task_type=TEXT):
    return await repo.create_task(uid, task_type, payload or {"text": "hello"}, key)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


async def test_create_and_fetch_user(db):
    user = await repo.create_user("Carol", "hash-1")
    assert set(user) == {"id", "username", "created_at"}
    assert len(user["id"]) == 36

    by_id = await repo.get_user_by_id(user["id"])
    assert by_id["username"] == "Carol"
    assert "password_hash" not in by_id  # never exposed by the id lookup

    by_name = await repo.get_user_by_username("Carol")
    assert by_name["id"] == user["id"]
    assert by_name["password_hash"] == "hash-1"  # needed for the login check


async def test_username_lookup_ignores_case(db):
    user = await repo.create_user("Carol", "h")
    assert (await repo.get_user_by_username("carol"))["id"] == user["id"]
    assert (await repo.get_user_by_username("CAROL"))["id"] == user["id"]


async def test_duplicate_username_is_rejected_ignoring_case(db):
    await repo.create_user("Carol", "h")
    for name in ("Carol", "carol", "CAROL"):
        try:
            await repo.create_user(name, "h2")
        except repo.UsernameTakenError:
            continue
        raise AssertionError(f"{name!r} should have been rejected")


async def test_unknown_users_return_none(db):
    assert await repo.get_user_by_username("nobody") is None
    assert await repo.get_user_by_id("nope") is None


# ---------------------------------------------------------------------------
# Create / idempotency
# ---------------------------------------------------------------------------


async def test_create_task_defaults(user_id):
    task, created = await make(user_id)
    assert created is True
    assert task["status"] == "pending"
    assert task["type"] == TEXT
    assert task["retry_count"] == 0
    assert task["started_at"] is None
    assert len(task["id"]) == 36
    # Large columns are not part of the summary.
    assert "payload" not in task and "result" not in task


async def test_task_requires_an_existing_user(db):
    try:
        await make("no-such-user")
    except sqlite3.IntegrityError:
        return
    raise AssertionError("expected a foreign-key error")


async def test_same_key_same_body_returns_existing_task(user_id):
    first, created1 = await make(user_id, key="abc")
    second, created2 = await make(user_id, key="abc")
    assert created1 is True
    assert created2 is False
    assert first["id"] == second["id"]
    _, total = await repo.list_tasks(user_id)
    assert total == 1


async def test_same_key_different_body_raises_conflict(user_id):
    first, _ = await make(user_id, {"text": "one"}, key="abc")
    try:
        await make(user_id, {"text": "two"}, key="abc")
    except repo.IdempotencyConflictError as exc:
        assert exc.task_id == first["id"]
        return
    raise AssertionError("expected IdempotencyConflictError")


async def test_same_key_different_type_raises_conflict(user_id):
    await make(user_id, {"text": "x"}, key="abc")
    try:
        await make(user_id, {"transactions": []}, key="abc", task_type=TX)
    except repo.IdempotencyConflictError:
        return
    raise AssertionError("expected IdempotencyConflictError")


async def test_no_key_creates_separate_tasks(user_id):
    a, _ = await make(user_id)
    b, _ = await make(user_id)
    assert a["id"] != b["id"]
    _, total = await repo.list_tasks(user_id)
    assert total == 2


async def test_concurrent_requests_with_same_key_create_one_task(user_id):
    results = await asyncio.gather(*(make(user_id, key="race") for _ in range(20)))
    assert len({task["id"] for task, _ in results}) == 1
    assert sum(1 for _, created in results if created) == 1
    _, total = await repo.list_tasks(user_id)
    assert total == 1


async def test_get_task(user_id):
    task, _ = await make(user_id, {"text": "abc"})
    fetched = await repo.get_task(task["id"], user_id)
    assert fetched["id"] == task["id"]
    assert fetched["status"] == "pending"
    assert await repo.get_task("does-not-exist", user_id) is None


async def test_summary_queries_do_not_load_payload_or_result(user_id):
    task, _ = await make(user_id, {"text": "abc"})
    await repo.claim_task(task["id"])
    await repo.complete_task(task["id"], {"total_words": 1})

    created, _ = await make(user_id, {"text": "other"}, key="k")
    replayed, _ = await make(user_id, {"text": "other"}, key="k")
    items, _ = await repo.list_tasks(user_id)
    outcome, cancelled = await repo.cancel_task(created["id"], user_id)

    summaries = [await repo.get_task(task["id"], user_id), created, replayed, cancelled, *items]
    for summary in summaries:
        assert "payload" not in summary
        assert "result" not in summary


async def test_get_task_result(user_id):
    task, _ = await make(user_id)
    before = await repo.get_task_result(task["id"], user_id)
    assert before["status"] == "pending" and before["result"] is None

    await repo.claim_task(task["id"])
    await repo.complete_task(task["id"], {"total_words": 3})
    after = await repo.get_task_result(task["id"], user_id)
    assert after["status"] == "completed"
    assert after["result"] == {"total_words": 3}
    assert "payload" not in after
    assert await repo.get_task_result("does-not-exist", user_id) is None


# ---------------------------------------------------------------------------
# Ownership: users only ever see and change their own tasks
# ---------------------------------------------------------------------------


async def test_other_users_cannot_read_a_task(user_id, other_user_id):
    task, _ = await make(user_id)
    await repo.claim_task(task["id"])
    await repo.complete_task(task["id"], {"secret": 1})

    assert await repo.get_task(task["id"], other_user_id) is None
    assert await repo.get_task_result(task["id"], other_user_id) is None
    assert (await repo.get_task(task["id"], user_id))["status"] == "completed"


async def test_other_users_cannot_cancel_a_task(user_id, other_user_id):
    task, _ = await make(user_id)
    outcome, found = await repo.cancel_task(task["id"], other_user_id)
    assert outcome == "not_found"  # indistinguishable from a task that does not exist
    assert found is None
    assert (await repo.get_task(task["id"], user_id))["status"] == "pending"  # untouched


async def test_list_only_returns_own_tasks(user_id, other_user_id):
    mine = [(await make(user_id, {"text": f"mine {i}"}))[0]["id"] for i in range(3)]
    theirs = [(await make(other_user_id, {"text": f"theirs {i}"}))[0]["id"] for i in range(2)]

    items, total = await repo.list_tasks(user_id)
    assert total == 3 and {t["id"] for t in items} == set(mine)
    items, total = await repo.list_tasks(other_user_id)
    assert total == 2 and {t["id"] for t in items} == set(theirs)


async def test_stats_only_count_own_tasks(user_id, other_user_id):
    task, _ = await make(user_id)
    await repo.claim_task(task["id"])
    await repo.complete_task(task["id"], {})
    await make(other_user_id)
    await make(other_user_id)

    assert (await repo.get_stats(user_id))["total_tasks"] == 1
    other = await repo.get_stats(other_user_id)
    assert other["total_tasks"] == 2
    assert other["by_status"]["completed"] == 0
    assert other["average_processing_seconds"] is None


async def test_idempotency_keys_are_per_user(user_id, other_user_id):
    mine, created_mine = await make(user_id, {"text": "same"}, key="shared-key")
    theirs, created_theirs = await make(other_user_id, {"text": "different"}, key="shared-key")
    assert created_mine and created_theirs  # no clash, and no leak of the other user's task
    assert mine["id"] != theirs["id"]

    again, created = await make(other_user_id, {"text": "different"}, key="shared-key")
    assert created is False and again["id"] == theirs["id"]


async def test_workers_can_claim_any_users_task(user_id, other_user_id):
    mine, _ = await make(user_id)
    theirs, _ = await make(other_user_id)
    assert (await repo.claim_task(mine["id"]))["status"] == "processing"
    assert (await repo.claim_task(theirs["id"]))["status"] == "processing"


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


async def test_claim_task_only_succeeds_once(user_id):
    task, _ = await make(user_id)
    claimed = await repo.claim_task(task["id"])
    assert claimed["status"] == "processing"
    assert claimed["started_at"] is not None
    assert claimed["payload"] == {"text": "hello"}  # the worker needs this
    assert await repo.claim_task(task["id"]) is None


async def test_concurrent_claims_have_one_winner(user_id):
    task, _ = await make(user_id)
    results = await asyncio.gather(*(repo.claim_task(task["id"]) for _ in range(10)))
    assert sum(1 for r in results if r is not None) == 1


async def test_cancel_pending_task(user_id):
    task, _ = await make(user_id)
    outcome, cancelled = await repo.cancel_task(task["id"], user_id)
    assert outcome == "cancelled"
    assert cancelled["status"] == "cancelled"
    assert cancelled["completed_at"] is not None


async def test_cancelled_task_cannot_be_claimed(user_id):
    task, _ = await make(user_id)
    await repo.cancel_task(task["id"], user_id)
    assert await repo.claim_task(task["id"]) is None
    assert (await repo.get_task(task["id"], user_id))["status"] == "cancelled"


async def test_cancel_after_claim_is_rejected(user_id):
    task, _ = await make(user_id)
    await repo.claim_task(task["id"])
    outcome, current = await repo.cancel_task(task["id"], user_id)
    assert outcome == "not_cancellable"
    assert current["status"] == "processing"


async def test_cancel_completed_and_missing(user_id):
    task, _ = await make(user_id)
    await repo.claim_task(task["id"])
    await repo.complete_task(task["id"], {"ok": True})
    outcome, _ = await repo.cancel_task(task["id"], user_id)
    assert outcome == "not_cancellable"
    outcome, none_task = await repo.cancel_task("missing", user_id)
    assert outcome == "not_found"
    assert none_task is None


async def test_complete_task_stores_result(user_id):
    task, _ = await make(user_id)
    await repo.claim_task(task["id"])
    assert await repo.complete_task(task["id"], {"total_words": 3}) is True
    done = await repo.get_task_result(task["id"], user_id)
    assert done["status"] == "completed"
    assert done["result"] == {"total_words": 3}
    assert done["completed_at"] is not None
    assert done["error"] is None


async def test_complete_requires_processing_state(user_id):
    task, _ = await make(user_id)
    assert await repo.complete_task(task["id"], {}) is False  # still pending
    await repo.cancel_task(task["id"], user_id)
    assert await repo.complete_task(task["id"], {}) is False  # cancelled


async def test_fail_task_stores_error(user_id):
    task, _ = await make(user_id)
    await repo.claim_task(task["id"])
    assert await repo.fail_task(task["id"], "boom") is True
    failed = await repo.get_task(task["id"], user_id)
    assert failed["status"] == "failed"
    assert failed["error"] == "boom"


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


async def test_list_filters_and_pagination(user_id):
    ids = []
    for i in range(5):
        task, _ = await make(user_id, {"text": f"t{i}"})
        ids.append(task["id"])
    await repo.claim_task(ids[0])
    await repo.complete_task(ids[0], {})
    await make(user_id, {"transactions": []}, task_type=TX)

    items, total = await repo.list_tasks(user_id)
    assert total == 6
    assert items[0]["type"] == TX  # newest first

    items, total = await repo.list_tasks(user_id, status="completed")
    assert total == 1 and items[0]["id"] == ids[0]

    items, total = await repo.list_tasks(user_id, status="pending", task_type=TEXT)
    assert total == 4

    page1, total = await repo.list_tasks(user_id, limit=2, offset=0)
    page2, _ = await repo.list_tasks(user_id, limit=2, offset=2)
    assert total == 6
    assert len(page1) == 2 and len(page2) == 2
    assert not {t["id"] for t in page1} & {t["id"] for t in page2}

    items, total = await repo.list_tasks(user_id, limit=10, offset=100)
    assert items == [] and total == 6


async def test_list_empty(user_id):
    items, total = await repo.list_tasks(user_id)
    assert items == [] and total == 0


# ---------------------------------------------------------------------------
# Recovery and stats
# ---------------------------------------------------------------------------


async def test_reset_stale_processing_and_pending_ids(user_id):
    a, _ = await make(user_id, {"text": "a"})
    b, _ = await make(user_id, {"text": "b"})
    c, _ = await make(user_id, {"text": "c"})
    await repo.claim_task(b["id"])  # simulate crash mid-processing
    await repo.claim_task(c["id"])
    await repo.complete_task(c["id"], {})

    assert await repo.reset_stale_processing() == 1
    reset = await repo.get_task(b["id"], user_id)
    assert reset["status"] == "pending" and reset["started_at"] is None
    assert await repo.get_pending_ids() == [a["id"], b["id"]]


async def test_reset_stale_processing_respects_age_threshold(user_id):
    task, _ = await make(user_id)
    await repo.claim_task(task["id"])

    assert await repo.reset_stale_processing(older_than_seconds=60) == 0  # too recent
    assert (await repo.get_task(task["id"], user_id))["status"] == "processing"

    assert await repo.reset_stale_processing(older_than_seconds=0) == 1
    assert (await repo.get_task(task["id"], user_id))["status"] == "pending"


async def test_stats_empty(user_id):
    stats = await repo.get_stats(user_id)
    assert stats["total_tasks"] == 0
    assert stats["by_status"] == {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }
    assert stats["average_processing_seconds"] is None


async def test_stats_counts_and_average(user_id):
    done, _ = await make(user_id, {"text": "1"})
    failed, _ = await make(user_id, {"text": "2"})
    cancelled, _ = await make(user_id, {"text": "3"})
    await make(user_id, {"text": "4"})

    await repo.claim_task(done["id"])
    await asyncio.sleep(0.05)
    await repo.complete_task(done["id"], {})
    await repo.claim_task(failed["id"])
    await repo.fail_task(failed["id"], "x")
    await repo.cancel_task(cancelled["id"], user_id)

    stats = await repo.get_stats(user_id)
    assert stats["total_tasks"] == 4
    assert stats["by_status"]["completed"] == 1
    assert stats["by_status"]["failed"] == 1
    assert stats["by_status"]["cancelled"] == 1
    assert stats["by_status"]["pending"] == 1
    assert stats["average_processing_seconds"] >= 0.04

# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


async def fail(task_id):
    await repo.claim_task(task_id)
    await repo.fail_task(task_id, "boom")


async def test_retry_failed_task_goes_back_to_pending(user_id):
    task, _ = await make(user_id)
    await fail(task["id"])

    outcome, retried = await repo.retry_task(task["id"], user_id)
    assert outcome == "retried"
    assert retried["status"] == "pending"
    assert retried["retry_count"] == 1
    assert retried["error"] is None
    assert retried["started_at"] is None and retried["completed_at"] is None
    assert await repo.get_pending_ids() == [task["id"]]


async def test_retry_is_limited_to_three_times(user_id):
    task, _ = await make(user_id)
    for expected in (1, 2, 3):
        await fail(task["id"])
        outcome, retried = await repo.retry_task(task["id"], user_id)
        assert outcome == "retried" and retried["retry_count"] == expected

    await fail(task["id"])
    outcome, current = await repo.retry_task(task["id"], user_id)
    assert outcome == "limit_reached"
    assert current["status"] == "failed" and current["retry_count"] == 3


async def test_only_failed_tasks_can_be_retried(user_id):
    pending, _ = await make(user_id, {"text": "a"})
    processing, _ = await make(user_id, {"text": "b"})
    completed, _ = await make(user_id, {"text": "c"})
    cancelled, _ = await make(user_id, {"text": "d"})
    await repo.claim_task(processing["id"])
    await repo.claim_task(completed["id"])
    await repo.complete_task(completed["id"], {})
    await repo.cancel_task(cancelled["id"], user_id)

    for task in (pending, processing, completed, cancelled):
        outcome, current = await repo.retry_task(task["id"], user_id)
        assert outcome == "not_failed"
        assert current["retry_count"] == 0


async def test_retry_missing_task(user_id):
    outcome, task = await repo.retry_task("missing", user_id)
    assert outcome == "not_found" and task is None


async def test_other_users_cannot_retry_a_task(user_id, other_user_id):
    task, _ = await make(user_id)
    await fail(task["id"])
    outcome, found = await repo.retry_task(task["id"], other_user_id)
    assert outcome == "not_found" and found is None
    assert (await repo.get_task(task["id"], user_id))["status"] == "failed"


async def test_concurrent_retries_have_one_winner(user_id):
    task, _ = await make(user_id)
    await fail(task["id"])
    results = await asyncio.gather(*(repo.retry_task(task["id"], user_id) for _ in range(10)))
    assert sum(1 for outcome, _ in results if outcome == "retried") == 1
    assert (await repo.get_task(task["id"], user_id))["retry_count"] == 1