"""Application entry point: wires up the database, workers and routes."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import security
from .auth import router as auth_router
from .database import init_db
from .routes import router
from .worker import TaskWorkerPool

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    security.check_security_config()
    await init_db()
    pool = TaskWorkerPool()
    await pool.start()
    app.state.worker_pool = pool
    try:
        yield
    finally:
        await pool.stop()


app = FastAPI(
    title="Asynchronous Task Processing System",
    description=(
        "Submit text and transaction analysis jobs, then poll for results. "
        "Register and log in first (use the Authorize button), then call the task endpoints."
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.include_router(auth_router)
app.include_router(router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(sqlite3.OperationalError)
async def database_unavailable(request: Request, exc: sqlite3.OperationalError) -> JSONResponse:
    # e.g. "database is locked" after the busy timeout. Ask the client to retry
    # rather than returning an opaque 500.
    logger.error("Database error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Database temporarily unavailable, please retry"},
        headers={"Retry-After": "1"},
    )