from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """HTTP request for one grounded academic question."""

    query: str = Field(min_length=1, max_length=1000)
    user_faculty: str = Field(default="GLOBAL", min_length=1, max_length=100)
    user_level: int = Field(default=1, ge=1, le=10)


class Citation(BaseModel):
    """Source reference attached to a generated answer."""

    title: str
    chunk_id: str
    source: str | None = None


class ChatResponse(BaseModel):
    """Stable response contract for the chat endpoint."""

    trace_id: str
    query: str
    response: str
    intent: str | None
    citations: list[Citation]
    suggestions: list[str]


class ErrorResponse(BaseModel):
    """Shared error response contract."""

    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
