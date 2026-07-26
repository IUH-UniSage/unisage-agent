from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval before generation."""

    chunk_id: str
    content: str
    source: str
    faculty: str = "GLOBAL"
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    """Internal retrieval input with access metadata."""

    query: str
    user_faculty: str = "GLOBAL"
    user_level: int = Field(default=1, ge=1)
    limit: int = Field(default=5, ge=1, le=20)
