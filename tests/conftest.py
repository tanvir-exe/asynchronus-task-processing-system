from contextlib import asynccontextmanager

import httpx
import pytest
from api_helpers import PASSWORD, TEST_SECRET

from task_processing_system import repository as repo
from task_processing_system.database import init_db
from task_processing_system.main import app

# ---------------------------------------------------------------------------
# Repository-level fixtures: a fresh database and ready-made users
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """Point the app at a fresh SQLite file for each test."""
    monkeypatch.setenv("TASKS_DB_PATH", str(tmp_path / "test.db"))
    await init_db()


@pytest.fixture
async def user_id(db):
    """ID of a ready-made user. Every task belongs to a user."""
    return (await repo.create_user("alice", "not-a-real-hash"))["id"]


@pytest.fixture
async def other_user_id(db):
    """ID of a second user, for ownership tests."""
    return (await repo.create_user("bob", "not-a-real-hash"))["id"]


# ---------------------------------------------------------------------------
# API fixtures: a running app (lifespan + workers) and logged-in HTTP clients
# ---------------------------------------------------------------------------


def new_http_client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def register_and_login(http, username, password=PASSWORD):
    """Create an account, log in, and make `http` send the token from now on."""
    response = await http.post("/auth/register", json={"username": username, "password": password})
    assert response.status_code == 201, response.text
    response = await http.post("/auth/login", data={"username": username, "password": password})
    assert response.status_code == 200, response.text
    http.headers["Authorization"] = f"Bearer {response.json()['access_token']}"


@asynccontextmanager
async def running_app(tmp_path, monkeypatch, workers, delay):
    monkeypatch.setenv("TASKS_DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("TASK_WORKERS", str(workers))
    monkeypatch.setenv("TASK_PROCESSING_DELAY_SECONDS", str(delay))
    monkeypatch.setenv("JWT_SECRET", TEST_SECRET)
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
    # ASGITransport does not run lifespan events, so we run them ourselves.
    async with app.router.lifespan_context(app):
        async with new_http_client() as http:
            await register_and_login(http, "alice")
            yield http


@pytest.fixture
async def client(tmp_path, monkeypatch):
    """Logged in as 'alice'. 3 fast workers: tasks finish almost instantly."""
    async with running_app(tmp_path, monkeypatch, workers=3, delay=0) as http:
        yield http


@pytest.fixture
async def slow_client(tmp_path, monkeypatch):
    """Logged in as 'alice'. 1 worker, 0.5s per task: tasks stay pending/processing."""
    async with running_app(tmp_path, monkeypatch, workers=1, delay=0.5) as http:
        yield http


@pytest.fixture
async def anon(client):
    """Same running app, but not logged in."""
    async with new_http_client() as http:
        yield http


@pytest.fixture
async def bob(client):
    """A second user, logged in, talking to the same app as `client`."""
    async with new_http_client() as http:
        await register_and_login(http, "bob")
        yield http


@pytest.fixture
async def slow_bob(slow_client):
    """A second user for the slow app."""
    async with new_http_client() as http:
        await register_and_login(http, "bob")
        yield http