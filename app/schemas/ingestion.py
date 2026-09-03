from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.database.models import DocumentProcessStep

if TYPE_CHECKING:
    from app.database.repositories.ingestion_job import DraftDTO


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


class TaskProgress(BaseModel):
    """One embedding task's current progress, read from the Celery result backend."""

    percent: int
    state: str


class IngestionJobResponse(BaseModel):
    """A resumable chunking draft, or in-flight embedding job, for one document.

    When `current_step` is `embedding`, `task_state` / `task_percent` carry
    the live Celery task progress read server-side, so the client's
    reconciliation sweep needs a single authorized call.
    """

    object_key: str
    current_step: DocumentProcessStep
    chunking_strategy: ChunkingStrategyName
    chunking_params: dict[str, Any]
    chunks: list[Chunk]
    task_id: str | None = None
    task_state: str | None = None
    task_percent: int | None = None

    @classmethod
    def from_draft(
        cls, draft: "DraftDTO", task_progress: TaskProgress | None = None
    ) -> "IngestionJobResponse":
        return cls(
            object_key=draft.object_key,
            current_step=draft.current_step,
            chunking_strategy=ChunkingStrategyName(draft.chunking_strategy),
            chunking_params=draft.chunking_params,
            chunks=draft.chunks,
            task_id=draft.celery_task_id,
            task_state=task_progress.state if task_progress else None,
            task_percent=task_progress.percent if task_progress else None,
        )
