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


class HeaderSource(StrEnum):
    """Where a table's header row information came from."""

    EXPLICIT = "explicit"  # real structural header signal from the source: HTML <th>,
    # DOCX <w:tblHeader>, XLSX header row (product contract, see table_aware_parser)
    INFERRED = "inferred"  # no structural signal - the first row is assumed to be the
    # header (PDF always lands here: pymupdf4llm gives no such signal)
    MISSING = "missing"  # could not be determined -> column_names = None


class SourceType(StrEnum):
    """The file format a chunk's originating region was parsed from."""

    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    TXT = "txt"
    XLSX = "xlsx"


class SourceLocator(BaseModel):
    """Fine-grained location of a chunk within its source document.

    All fields are optional/`None` by default because most fields only make
    sense for a subset of `region_type`/`source_type` combinations (e.g.
    `sheet_name` only for XLSX, `row_start`/`row_end`/`row_count` only for
    TABLE/EXCEL_ROW chunks) - see `validate_chunks` (Phase 4) for which
    fields become mandatory for which chunk kind.
    """

    section: str | None = None  # DOCX/HTML/TXT: heading_path joined with " > ",
    # None when there is no heading
    sheet_name: str | None = None  # XLSX
    row_start: int | None = None  # XLSX or TABLE chunk: first data row (1-indexed,
    # header not counted)
    row_end: int | None = None  # XLSX or TABLE chunk: last data row
    row_count: int | None = None  # number of real data rows in this chunk - an
    # independent figure the validator (Phase 4) cross-checks against
    # `row_end - row_start + 1` (a round-trip consistency check, NOT a
    # re-verification of the actual cell content - see Task 2.2/4.2)
    table_id: str | None = None  # "table-{block_index}" - distinguishes two tables
    # sharing the same page/heading
    row_part: int | None = None  # 1-indexed, only set when one row had to be split
    # across multiple chunks (oversized-row fallback)
    row_part_count: int | None = None
    is_partial_row: bool = False


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
    """One chunk produced by a chunking strategy.

    `source_type`/`block_index` default to `None` rather than being
    required so that every existing `Chunk(chunk_index=.., content=..,
    region_type=..)` call (tests, older chunkers before Phase 3) keeps
    constructing successfully - the chunkers wired through
    `strategy.dispatch()` (Phase 3) always set them explicitly, and
    `validate_chunks` (Phase 4) is the place that enforces they are set,
    not the Pydantic model itself. `block_index=None` is distinct from
    `block_index=0` (a valid region index) - `None` means "no chunker has
    assigned this yet".
    """

    chunk_index: int = Field(ge=0)
    content: str
    region_type: RegionType
    source_type: SourceType | None = None
    block_index: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    source_locator: SourceLocator | None = None
    column_names: list[str] | None = None
    has_header: bool = False
    header_source: HeaderSource = HeaderSource.MISSING
    header_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # Deliberately NOT `settings.CHUNKING_VERSION` - that default would also
    # apply when deserializing an old row that never had this field at all,
    # mislabeling legacy data as produced by the current chunking logic.
    # Every new chunker (Phase 2/3) sets this explicitly at construction.
    chunking_version: str = "legacy"


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
