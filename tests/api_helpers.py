"""Constants and small helpers shared by the API test files."""

import asyncio

TEST_SECRET = "a-test-secret-that-is-at-least-32-characters-long"
PASSWORD = "correct-horse-battery"

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