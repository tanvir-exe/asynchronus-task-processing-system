"""Pydantic models for API requests/responses, plus shared enums."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_TEXT_LENGTH = 100_000
MAX_TRANSACTIONS = 10_000
MAX_IDEMPOTENCY_KEY_LENGTH = 255


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    TEXT_ANALYSIS = "text_analysis"
    TRANSACTION_ANALYSIS = "transaction_analysis"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class TextPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class TransactionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Items are deliberately typed as Any: malformed individual records must
    # not reject the whole request. The processor reports them as invalid.
    transactions: list[Any] = Field(min_length=1, max_length=MAX_TRANSACTIONS)


class TextTaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text_analysis"]
    payload: TextPayload


class TransactionTaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["transaction_analysis"]
    payload: TransactionPayload


# Pydantic picks the right model based on the "type" field.
TaskCreate = Annotated[
    Union[TextTaskCreate, TransactionTaskCreate],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class TaskResponse(BaseModel):
    id: str
    type: TaskType
    status: TaskStatus
    idempotency_key: str | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class TaskResultResponse(BaseModel):
    id: str
    type: TaskType
    status: TaskStatus
    result: dict[str, Any]


class TaskListResponse(BaseModel):
    items: list[TaskResponse]
    total: int
    limit: int
    offset: int


class StatsResponse(BaseModel):
    total_tasks: int
    by_status: dict[str, int]
    average_processing_seconds: float | None = None