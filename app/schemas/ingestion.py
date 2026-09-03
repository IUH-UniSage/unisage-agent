from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChunkingStrategyName(StrEnum):
    """The five chunking strategies selectable by clients."""

    RECURSIVE = "recursive"
    TOKEN_BASED = "token_based"
    SEMANTIC = "semantic"
    MARKDOWN_AWARE = "markdown_aware"
    EXCEL_ROW = "excel_row"


class RegionType(StrEnum):
    """The kind of content a parsed region or chunk originated from."""

    TEXT = "text"
    TABLE = "table"
    EXCEL_ROW = "excel_row"


class PreviewRequest(BaseModel):
    """Request to fetch and preview the raw text of a stored object."""

    department_id: str = Field(min_length=1, max_length=100)
    object_key: str = Field(min_length=1, max_length=1024)


class PreviewResponse(BaseModel):
    """Raw text extracted from the requested object."""

    raw_text: str


class ChunkingRequest(BaseModel):
    """Request to chunk a stored object using the given strategy."""

    document_id: str = Field(min_length=1, max_length=100)
    department_id: str = Field(min_length=1, max_length=100)
    object_key: str = Field(min_length=1, max_length=1024)
    strategy: ChunkingStrategyName
    params: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """One chunk produced by a chunking strategy."""

    chunk_index: int = Field(ge=0)
    content: str
    region_type: RegionType


class ChunkingResponse(BaseModel):
    """Chunks produced for a single chunking request."""

    chunks: list[Chunk]


class EmbeddingRequest(BaseModel):
    """Request to enrich, embed, and upsert a client-approved chunk list."""

    document_id: str = Field(min_length=1, max_length=100)
    department_id: str = Field(min_length=1, max_length=100)
    access_level: int = Field(ge=0)
    object_key: str = Field(min_length=1, max_length=1024)
    chunks: list[Chunk] = Field(min_length=1)


class EmbeddingAcceptedResponse(BaseModel):
    """Returned immediately after an embedding task is dispatched."""

    task_id: str


class EmbeddingStatusResponse(BaseModel):
    """One embedding task's current progress - the HTTP-pollable equivalent
    of a single `/ingestion/embedding/{task_id}/progress` WebSocket frame."""

    percent: int
    state: str


class IngestionJobResponse(BaseModel):
    """A resumable chunking draft, or in-flight embedding job, for one document."""

    object_key: str
    current_step: str
    chunking_strategy: str
    chunking_params: dict[str, Any]
    chunks: list[Chunk]
    task_id: str | None = None
