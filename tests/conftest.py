import pytest

from task_processing_system.database import init_db


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """Point the app at a fresh SQLite file for each test."""
    monkeypatch.setenv("TASKS_DB_PATH", str(tmp_path / "test.db"))
    await init_db()