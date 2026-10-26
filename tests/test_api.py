"""End-to-end API tests.

Each test gets its own app instance (lifespan + workers) and its own SQLite
file. Requests go through httpx's ASGI transport, so no real server or port
is needed, and asyncio.gather lets us fire truly concurrent requests.

Two fixtures:
  client       fast workers (3), no delay  -> tasks finish almost instantly
  slow_client  1 worker, 0.5s per task     -> tasks stay pending/processing
                                              long enough to test cancel etc.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager

import httpx
import pytest

from task_processing_system import worker as worker_module
from task_processing_system.main import app

TEXT_BODY = {
    "type": "text_analysis",
    "payload": {"text": "The cat sat. The cat ran away fast!"},
}

TRANSACTION_BODY = {
    "type": "transaction_analysis",
    "payload": {
        "transactions": [
            {"transaction_id": "t1", "customer_id": "alice", "amount": 100, "timestamp": "2024-01-15T10:30:00Z"},
            {"transaction_id": "t1", "customer_id": "alice", "amount": 100, "timestamp": "2024-01-15T10:30:00Z"},
            {"transaction_id": "t2", "customer_id": "bob", "amount": 50.5, "timestamp": "2024-01-16T09:00:00Z"},
            {"transaction_id": "t3", "customer_id": "carol", "amount": -5, "timestamp": "2024-01-17T09:00:00Z"},
        ]
    },
}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def running_app(tmp_path, monkeypatch, workers, delay):
    monkeypatch.setenv("TASKS_DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("TASK_WORKERS", str(workers))
    monkeypatch.setenv("TASK_PROCESSING_DELAY_SECONDS", str(delay))
    # ASGITransport does not run lifespan events, so we run them ourselves.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


@pytest.fixture
async def client(tmp_path, monkeypatch):
    async with running_app(tmp_path, monkeypatch, workers=3, delay=0) as http:
        yield http


@pytest.fixture
async def slow_client(tmp_path, monkeypatch):
    async with running_app(tmp_path, monkeypatch, workers=1, delay=0.5) as http:
        yield http


async def submit(http, body=None, key=None):
    headers = {"Idempotency-Key": key} if key is not None else {}
    return await http.post("/tasks", json=body or TEXT_BODY, headers=headers)


async def wait_for_status(http, task_id, status, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        body = (await http.get(f"/tasks/{task_id}")).json()
        if body["status"] == status:
            return body
        if loop.time() > deadline:
            raise AssertionError(f"task still {body['status']!r}, expected {status!r}")
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Submit -> status -> result
# ---------------------------------------------------------------------------


async def test_submit_text_task_end_to_end(client):
    response = await submit(client)
    assert response.status_code == 202
    task = response.json()
    uuid.UUID(task["id"])  # valid UUID
    assert task["type"] == "text_analysis"
    assert task["status"] == "pending"
    assert task["idempotency_key"] is None

    done = await wait_for_status(client, task["id"], "completed")
    assert done["started_at"] and done["completed_at"]

    result = await client.get(f"/tasks/{task['id']}/result")
    assert result.status_code == 200
    body = result.json()
    assert body["status"] == "completed"
    assert body["result"]["total_words"] == 8
    assert body["result"]["unique_words"] == 6
    assert body["result"]["average_word_length"] == 3.25
    assert body["result"]["top_five_words"][0] == {"word": "the", "count": 2}
    assert body["result"]["longest_sentence"]["text"] == "The cat ran away fast!"


async def test_submit_transaction_task_end_to_end(client):
    task = (await submit(client, TRANSACTION_BODY)).json()
    await wait_for_status(client, task["id"], "completed")

    result = (await client.get(f"/tasks/{task['id']}/result")).json()["result"]
    assert result["total_records"] == 4
    assert result["valid_records"] == 3
    assert result["duplicate_count"] == 1
    assert result["invalid_count"] == 1
    assert result["invalid_records"][0]["index"] == 3
    assert result["total_revenue"] == 150.5
    assert [c["customer_id"] for c in result["top_customers"]] == ["alice", "bob"]


async def test_invalid_transaction_records_do_not_fail_the_task(client):
    body = {
        "type": "transaction_analysis",
        "payload": {"transactions": [{"nonsense": True}, "text", 42, None]},
    }
    task = (await submit(client, body)).json()
    await wait_for_status(client, task["id"], "completed")
    result = (await client.get(f"/tasks/{task['id']}/result")).json()["result"]
    assert result["invalid_count"] == 4
    assert result["total_revenue"] == 0.0


async def test_unknown_task_returns_404(client):
    for path in ("/tasks/nope", "/tasks/nope/result"):
        assert (await client.get(path)).status_code == 404
    assert (await client.post("/tasks/nope/cancel")).status_code == 404


# ---------------------------------------------------------------------------
# Not blocking, pending results
# ---------------------------------------------------------------------------


async def test_submit_returns_immediately_and_server_stays_responsive(slow_client):
    task = (await submit(slow_client)).json()
    assert task["status"] == "pending"

    # The task is still being worked on, yet other endpoints answer normally.
    assert (await slow_client.get("/health")).status_code == 200
    assert (await slow_client.get("/stats")).status_code == 200
    current = (await slow_client.get(f"/tasks/{task['id']}")).json()
    assert current["status"] in ("pending", "processing")


async def test_result_of_unfinished_task_returns_409(slow_client):
    first = (await submit(slow_client)).json()
    second = (await submit(slow_client)).json()

    await wait_for_status(slow_client, first["id"], "processing")
    running = await slow_client.get(f"/tasks/{first['id']}/result")
    queued = await slow_client.get(f"/tasks/{second['id']}/result")
    assert running.status_code == 409
    assert "processing" in running.json()["detail"]
    assert queued.status_code == 409
    assert "pending" in queued.json()["detail"]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_same_idempotency_key_returns_same_task(client):
    first = await submit(client, key="order-42")
    second = await submit(client, key="order-42")
    assert first.status_code == 202
    assert second.status_code == 200
    assert second.headers["idempotent-replayed"] == "true"
    assert first.json()["id"] == second.json()["id"]
    assert first.json()["idempotency_key"] == "order-42"

    listing = (await client.get("/tasks")).json()
    assert listing["total"] == 1


async def test_idempotency_key_reused_with_different_body_returns_409(client):
    await submit(client, key="order-42")
    other = {"type": "text_analysis", "payload": {"text": "something else"}}
    response = await submit(client, other, key="order-42")
    assert response.status_code == 409
    assert "order-42" in response.json()["detail"]


async def test_different_keys_create_different_tasks(client):
    a = (await submit(client, key="a")).json()
    b = (await submit(client, key="b")).json()
    assert a["id"] != b["id"]


async def test_requests_without_key_are_never_deduplicated(client):
    a = (await submit(client)).json()
    b = (await submit(client)).json()
    assert a["id"] != b["id"]


async def test_concurrent_requests_with_same_key_create_one_task(client):
    responses = await asyncio.gather(*(submit(client, key="race") for _ in range(10)))
    codes = sorted(r.status_code for r in responses)
    assert codes == [200] * 9 + [202]
    assert len({r.json()["id"] for r in responses}) == 1
    assert (await client.get("/tasks")).json()["total"] == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

INVALID_BODIES = {
    "blank text": {"type": "text_analysis", "payload": {"text": "   "}},
    "empty text": {"type": "text_analysis", "payload": {"text": ""}},
    "text too long": {"type": "text_analysis", "payload": {"text": "a" * 100_001}},
    "missing payload": {"type": "text_analysis"},
    "missing type": {"payload": {"text": "hi"}},
    "unknown type": {"type": "bogus", "payload": {"text": "hi"}},
    "extra top-level field": {**TEXT_BODY, "surprise": 1},
    "extra payload field": {"type": "text_analysis", "payload": {"text": "hi", "x": 1}},
    "payload of wrong type": {"type": "text_analysis", "payload": {"transactions": [{}]}},
    "text is not a string": {"type": "text_analysis", "payload": {"text": 123}},
    "empty transactions": {"type": "transaction_analysis", "payload": {"transactions": []}},
    "transactions not a list": {"type": "transaction_analysis", "payload": {"transactions": "nope"}},
    "too many transactions": {
        "type": "transaction_analysis",
        "payload": {"transactions": [{}] * 10_001},
    },
}


@pytest.mark.parametrize("body", INVALID_BODIES.values(), ids=INVALID_BODIES.keys())
async def test_invalid_input_returns_422(client, body):
    response = await client.post("/tasks", json=body)
    assert response.status_code == 422
    assert (await client.get("/tasks")).json()["total"] == 0  # nothing was saved


async def test_malformed_json_returns_422(client):
    response = await client.post(
        "/tasks", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


async def test_overlong_idempotency_key_returns_422(client):
    response = await submit(client, key="k" * 256)
    assert response.status_code == 422
    assert (await client.get("/tasks")).json()["total"] == 0


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


async def test_cancel_pending_task(slow_client):
    first = (await submit(slow_client)).json()
    second = (await submit(slow_client)).json()
    await wait_for_status(slow_client, first["id"], "processing")

    cancelled = await slow_client.post(f"/tasks/{second['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["completed_at"] is not None

    # The running task is unaffected and the cancelled one is never picked up.
    await wait_for_status(slow_client, first["id"], "completed")
    await asyncio.sleep(0.8)
    assert (await slow_client.get(f"/tasks/{second['id']}")).json()["status"] == "cancelled"
    assert (await slow_client.get(f"/tasks/{second['id']}/result")).status_code == 409

    stats = (await slow_client.get("/stats")).json()["by_status"]
    assert stats["completed"] == 1
    assert stats["cancelled"] == 1


async def test_cannot_cancel_processing_cancelled_or_completed_tasks(slow_client):
    first = (await submit(slow_client)).json()
    second = (await submit(slow_client)).json()
    await wait_for_status(slow_client, first["id"], "processing")

    running = await slow_client.post(f"/tasks/{first['id']}/cancel")
    assert running.status_code == 409
    assert "processing" in running.json()["detail"]

    assert (await slow_client.post(f"/tasks/{second['id']}/cancel")).status_code == 200
    again = await slow_client.post(f"/tasks/{second['id']}/cancel")
    assert again.status_code == 409
    assert "cancelled" in again.json()["detail"]

    await wait_for_status(slow_client, first["id"], "completed")
    done = await slow_client.post(f"/tasks/{first['id']}/cancel")
    assert done.status_code == 409
    assert "completed" in done.json()["detail"]


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


async def test_failed_task_reports_error_and_workers_keep_going(client, monkeypatch):
    def boom(payload):
        raise RuntimeError("boom")

    monkeypatch.setitem(worker_module.PROCESSORS, "text_analysis", boom)

    bad = (await submit(client)).json()
    failed = await wait_for_status(client, bad["id"], "failed")
    assert "RuntimeError: boom" in failed["error"]

    result = await client.get(f"/tasks/{bad['id']}/result")
    assert result.status_code == 409
    assert "boom" in result.json()["detail"]

    # A different task type still works, so the workers did not die.
    good = (await submit(client, TRANSACTION_BODY)).json()
    await wait_for_status(client, good["id"], "completed")

    stats = (await client.get("/stats")).json()["by_status"]
    assert stats["failed"] == 1
    assert stats["completed"] == 1


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_many_concurrent_submissions_all_complete(client):
    bodies = [
        {"type": "text_analysis", "payload": {"text": f"task number {i} here"}}
        for i in range(30)
    ]
    responses = await asyncio.gather(*(submit(client, body) for body in bodies))
    assert all(r.status_code == 202 for r in responses)
    ids = [r.json()["id"] for r in responses]
    assert len(set(ids)) == 30

    for task_id in ids:
        await wait_for_status(client, task_id, "completed")

    stats = (await client.get("/stats")).json()
    assert stats["total_tasks"] == 30
    assert stats["by_status"]["completed"] == 30


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


async def test_list_tasks_with_filters_and_pagination(client):
    ids = []
    for i in range(3):
        body = {"type": "text_analysis", "payload": {"text": f"text {i}"}}
        ids.append((await submit(client, body)).json()["id"])
    tx_id = (await submit(client, TRANSACTION_BODY)).json()["id"]
    for task_id in [*ids, tx_id]:
        await wait_for_status(client, task_id, "completed")

    everything = (await client.get("/tasks")).json()
    assert everything["total"] == 4
    assert everything["limit"] == 20 and everything["offset"] == 0
    assert everything["items"][0]["id"] == tx_id  # newest first

    only_tx = (await client.get("/tasks", params={"type": "transaction_analysis"})).json()
    assert only_tx["total"] == 1 and only_tx["items"][0]["id"] == tx_id

    completed = (await client.get("/tasks", params={"status": "completed"})).json()
    assert completed["total"] == 4
    assert (await client.get("/tasks", params={"status": "failed"})).json()["total"] == 0

    page1 = (await client.get("/tasks", params={"limit": 2, "offset": 0})).json()
    page2 = (await client.get("/tasks", params={"limit": 2, "offset": 2})).json()
    assert page1["total"] == 4 and len(page1["items"]) == 2 and len(page2["items"]) == 2
    assert not {t["id"] for t in page1["items"]} & {t["id"] for t in page2["items"]}


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"status": "bogus"}, {"type": "bogus"}],
)
async def test_list_tasks_rejects_bad_query_params(client, params):
    assert (await client.get("/tasks", params=params)).status_code == 422


async def test_list_empty(client):
    body = (await client.get("/tasks")).json()
    assert body["items"] == [] and body["total"] == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


async def test_stats_when_empty(client):
    stats = (await client.get("/stats")).json()
    assert stats["total_tasks"] == 0
    assert stats["by_status"] == {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }
    assert stats["average_processing_seconds"] is None


async def test_stats_after_processing(client):
    ids = [(await submit(client)).json()["id"] for _ in range(3)]
    for task_id in ids:
        await wait_for_status(client, task_id, "completed")

    stats = (await client.get("/stats")).json()
    assert stats["total_tasks"] == 3
    assert stats["by_status"]["completed"] == 3
    assert stats["average_processing_seconds"] is not None
    assert stats["average_processing_seconds"] >= 0