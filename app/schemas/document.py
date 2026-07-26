from typing import Any

from pydantic import BaseModel, Field


class DocumentIngestionRequest(BaseModel):
    """Minimal text ingestion contract for the base service."""

    source: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=1_000_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestionResponse(BaseModel):
    """Result returned after parsing and chunking a document."""

    source: str
    chunk_count: int
    chunks: list[str]
