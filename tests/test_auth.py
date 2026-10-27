"""Authentication and ownership tests, through the real HTTP API.

Fixtures come from conftest.py:
  client / slow_client   logged in as "alice"
  bob / slow_bob         a second logged-in user on the same app
  anon                   not logged in
"""

import asyncio
import time
import uuid

import jwt
import pytest

from task_processing_system import security
from api_helpers import PASSWORD, TEST_SECRET, TEXT_BODY, submit, wait_for_status


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


async def test_register_creates_an_account(anon):
    response = await anon.post("/auth/register", json={"username": "carol", "password": PASSWORD})
    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"id", "username", "created_at"}
    assert body["username"] == "carol"
    uuid.UUID(body["id"])
    assert "password" not in response.text.lower()  # no password or hash in the response


INVALID_REGISTRATIONS = {
    "password too short": {"username": "carol", "password": "short"},
    "password too long": {"username": "carol", "password": "x" * 129},
    "password not a string": {"username": "carol", "password": 12345678},
    "username too short": {"username": "ab", "password": PASSWORD},
    "username too long": {"username": "u" * 33, "password": PASSWORD},
    "username with space": {"username": "bad name", "password": PASSWORD},
    "username with symbol": {"username": "bad@name", "password": PASSWORD},
    "missing password": {"username": "carol"},
    "missing username": {"password": PASSWORD},
    "unknown field": {"username": "carol", "password": PASSWORD, "role": "admin"},
}


@pytest.mark.parametrize("body", INVALID_REGISTRATIONS.values(), ids=INVALID_REGISTRATIONS.keys())
async def test_invalid_registration_returns_422(anon, body):
    response = await anon.post("/auth/register", json=body)
    assert response.status_code == 422


async def test_duplicate_username_returns_409_ignoring_case(anon):
    await anon.post("/auth/register", json={"username": "Dave", "password": PASSWORD})
    for name in ("Dave", "dave", "DAVE"):
        response = await anon.post("/auth/register", json={"username": name, "password": PASSWORD})
        assert response.status_code == 409


async def test_concurrent_registrations_of_one_name_create_one_account(anon):
    responses = await asyncio.gather(
        *(
            anon.post("/auth/register", json={"username": "racer", "password": PASSWORD})
            for _ in range(5)
        )
    )
    assert sorted(r.status_code for r in responses) == [201, 409, 409, 409, 409]


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


async def test_login_returns_a_working_token(anon):
    await anon.post("/auth/register", json={"username": "carol", "password": PASSWORD})
    response = await anon.post("/auth/login", data={"username": "carol", "password": PASSWORD})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 30 * 60
    assert body["access_token"].count(".") == 2  # header.payload.signature

    me = await anon.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["username"] == "carol"


async def test_token_contains_only_the_user_id_and_times(anon):
    user = (await anon.post("/auth/register", json={"username": "carol", "password": PASSWORD})).json()
    token = (await anon.post("/auth/login", data={"username": "carol", "password": PASSWORD})).json()
    claims = jwt.decode(token["access_token"], TEST_SECRET, algorithms=["HS256"])
    assert set(claims) == {"sub", "iat", "exp"}
    assert claims["sub"] == user["id"]


async def test_login_username_ignores_case(anon):
    await anon.post("/auth/register", json={"username": "Carol", "password": PASSWORD})
    response = await anon.post("/auth/login", data={"username": "CAROL", "password": PASSWORD})
    assert response.status_code == 200


async def test_token_lifetime_follows_configuration(anon, monkeypatch):
    await anon.post("/auth/register", json={"username": "carol", "password": PASSWORD})
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "5")
    response = await anon.post("/auth/login", data={"username": "carol", "password": PASSWORD})
    assert response.json()["expires_in"] == 300


async def test_wrong_password_and_unknown_user_get_the_same_401(anon):
    await anon.post("/auth/register", json={"username": "carol", "password": PASSWORD})
    wrong = await anon.post("/auth/login", data={"username": "carol", "password": "wrong-password"})
    unknown = await anon.post("/auth/login", data={"username": "nobody", "password": PASSWORD})
    for response in (wrong, unknown):
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert "access_token" not in response.text
    assert wrong.json() == unknown.json()  # does not reveal which usernames exist


async def test_login_with_oversized_password_is_rejected(anon):
    await anon.post("/auth/register", json={"username": "carol", "password": PASSWORD})
    response = await anon.post("/auth/login", data={"username": "carol", "password": "x" * 5000})
    assert response.status_code == 401


async def test_login_requires_both_fields_as_a_form(anon):
    assert (await anon.post("/auth/login", data={"username": "carol"})).status_code == 422
    assert (await anon.post("/auth/login", data={"password": PASSWORD})).status_code == 422
    # JSON is not accepted: the endpoint takes an OAuth2 form so /docs "Authorize" works
    as_json = await anon.post("/auth/login", json={"username": "carol", "password": PASSWORD})
    assert as_json.status_code == 422


# ---------------------------------------------------------------------------
# Protected endpoints
# ---------------------------------------------------------------------------

PROTECTED = [
    ("GET", "/auth/me"),
    ("POST", "/tasks"),
    ("GET", "/tasks"),
    ("GET", "/tasks/some-id"),
    ("GET", "/tasks/some-id/result"),
    ("POST", "/tasks/some-id/cancel"),
    ("GET", "/stats"),
    ("POST", "/tasks/some-id/retry"),
]


@pytest.mark.parametrize("method,path", PROTECTED)
async def test_endpoints_require_a_token(anon, method, path):
    kwargs = {"json": TEXT_BODY} if method == "POST" and path == "/tasks" else {}
    response = await anon.request(method, path, **kwargs)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


async def test_public_endpoints_need_no_token(anon):
    assert (await anon.get("/health")).status_code == 200
    assert (await anon.get("/docs")).status_code == 200
    assert (await anon.get("/openapi.json")).status_code == 200


async def test_openapi_marks_task_routes_as_secured(anon):
    spec = (await anon.get("/openapi.json")).json()
    assert spec["components"]["securitySchemes"]  # the Authorize button in /docs
    assert spec["paths"]["/tasks"]["get"]["security"]
    assert spec["paths"]["/stats"]["get"]["security"]
    assert "security" not in spec["paths"]["/health"]["get"]
    assert "security" not in spec["paths"]["/auth/login"]["post"]


async def test_bad_tokens_are_rejected(anon, client):
    user = (await client.get("/auth/me")).json()
    now = int(time.time())

    bad_tokens = {
        "garbage": "not-a-token",
        "empty": "",
        "expired": security.create_access_token(user["id"], expires_minutes=-1),
        "signed with another secret": jwt.encode(
            {"sub": user["id"], "exp": now + 600}, "some-other-secret-value-that-is-long-enough", algorithm="HS256"
        ),
        "unsigned (alg none)": jwt.encode({"sub": user["id"], "exp": now + 600}, key=None, algorithm="none"),
        "no expiry": jwt.encode({"sub": user["id"]}, TEST_SECRET, algorithm="HS256"),
        "no subject": jwt.encode({"exp": now + 600}, TEST_SECRET, algorithm="HS256"),
        "user does not exist": security.create_access_token(str(uuid.uuid4())),
    }
    valid = security.create_access_token(user["id"])
    header, payload, signature = valid.split(".")
    bad_tokens["tampered signature"] = f"{header}.{payload}.{'A' if signature[0] != 'A' else 'B'}{signature[1:]}"

    for name, token in bad_tokens.items():
        for path in ("/auth/me", "/tasks", "/stats"):
            response = await anon.get(path, headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 401, f"{name} on {path} -> {response.status_code}"

    # sanity check: the well-formed token for a real user is accepted
    ok = await anon.get("/auth/me", headers={"Authorization": f"Bearer {valid}"})
    assert ok.status_code == 200


async def test_wrong_authorization_scheme_is_rejected(anon, client):
    token = security.create_access_token((await client.get("/auth/me")).json()["id"])
    for header in (f"Basic {token}", f"Token {token}", token):
        response = await anon.get("/tasks", headers={"Authorization": header})
        assert response.status_code == 401


async def test_expired_token_stops_working(anon, monkeypatch):
    await anon.post("/auth/register", json={"username": "carol", "password": PASSWORD})
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "0.02")  # about 1.2 seconds
    token = (await anon.post("/auth/login", data={"username": "carol", "password": PASSWORD})).json()
    headers = {"Authorization": f"Bearer {token['access_token']}"}

    assert (await anon.get("/tasks", headers=headers)).status_code == 200
    await asyncio.sleep(2.2)
    assert (await anon.get("/tasks", headers=headers)).status_code == 401


# ---------------------------------------------------------------------------
# Ownership: users only see and manage their own tasks
# ---------------------------------------------------------------------------


async def test_users_cannot_see_each_others_tasks(client, bob):
    task = (await submit(client)).json()
    await wait_for_status(client, task["id"], "completed")

    # Bob gets 404, exactly what he'd get for an ID that never existed.
    for path in (f"/tasks/{task['id']}", f"/tasks/{task['id']}/result"):
        assert (await bob.get(path)).status_code == 404
    assert (await bob.post(f"/tasks/{task['id']}/cancel")).status_code == 404
    real = await bob.get(f"/tasks/{task['id']}")
    fake_id = str(uuid.uuid4())
    fake = await bob.get(f"/tasks/{fake_id}")
    assert real.status_code == fake.status_code == 404
    assert real.json()["detail"] == f"Task {task['id']} not found"
    assert fake.json()["detail"] == f"Task {fake_id} not found"  # same shape: nothing leaks

    # Alice still sees her own task, untouched.
    assert (await client.get(f"/tasks/{task['id']}")).json()["status"] == "completed"
    assert (await client.get(f"/tasks/{task['id']}/result")).status_code == 200


async def test_listing_and_stats_are_per_user(client, bob):
    mine = [(await submit(client, {"type": "text_analysis", "payload": {"text": f"a{i}"}})).json()["id"] for i in range(3)]
    theirs = [(await submit(bob, {"type": "text_analysis", "payload": {"text": f"b{i}"}})).json()["id"] for i in range(2)]
    for task_id in mine:
        await wait_for_status(client, task_id, "completed")
    for task_id in theirs:
        await wait_for_status(bob, task_id, "completed")

    alice_list = (await client.get("/tasks")).json()
    bob_list = (await bob.get("/tasks")).json()
    assert alice_list["total"] == 3 and {t["id"] for t in alice_list["items"]} == set(mine)
    assert bob_list["total"] == 2 and {t["id"] for t in bob_list["items"]} == set(theirs)

    assert (await client.get("/stats")).json()["total_tasks"] == 3
    assert (await bob.get("/stats")).json()["total_tasks"] == 2


async def test_a_new_user_starts_with_an_empty_history(client, bob):
    await submit(client)
    assert (await bob.get("/tasks")).json()["total"] == 0
    assert (await bob.get("/stats")).json()["total_tasks"] == 0


async def test_another_user_cannot_cancel_my_pending_task(slow_client, slow_bob):
    first = (await submit(slow_client)).json()
    second = (await submit(slow_client)).json()
    await wait_for_status(slow_client, first["id"], "processing")

    assert (await slow_bob.post(f"/tasks/{second['id']}/cancel")).status_code == 404
    assert (await slow_client.get(f"/tasks/{second['id']}")).json()["status"] == "pending"

    # The owner can still cancel it.
    assert (await slow_client.post(f"/tasks/{second['id']}/cancel")).status_code == 200


async def test_idempotency_keys_do_not_clash_or_leak_between_users(client, bob):
    mine = await submit(client, key="shared-key")
    other_body = {"type": "text_analysis", "payload": {"text": "a different request"}}
    theirs = await submit(bob, other_body, key="shared-key")
    same_body = await submit(bob, TEXT_BODY, key="shared-key-2")

    assert mine.status_code == 202
    assert theirs.status_code == 202  # no 409, which would reveal Alice used this key
    assert mine.json()["id"] != theirs.json()["id"] != same_body.json()["id"]

    replay = await submit(bob, other_body, key="shared-key")
    assert replay.status_code == 200 and replay.json()["id"] == theirs.json()["id"]


async def test_the_task_owner_is_never_exposed(client):
    task = (await submit(client)).json()
    await wait_for_status(client, task["id"], "completed")
    for response in (
        await client.get(f"/tasks/{task['id']}"),
        await client.get(f"/tasks/{task['id']}/result"),
        await client.get("/tasks"),
    ):
        assert "user_id" not in response.text
        assert "password" not in response.text.lower()

async def test_another_user_cannot_retry_my_failed_task(client, bob, monkeypatch):
    from task_processing_system import worker as worker_module

    def boom(payload):
        raise RuntimeError("boom")

    monkeypatch.setitem(worker_module.PROCESSORS, "text_analysis", boom)
    task = (await submit(client)).json()
    await wait_for_status(client, task["id"], "failed")

    assert (await bob.post(f"/tasks/{task['id']}/retry")).status_code == 404
    assert (await client.get(f"/tasks/{task['id']}")).json()["retry_count"] == 0