from typing import Any

from pydantic import BaseModel, Field

from app.schemas.ingestion import SourceLocator


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval before generation.

    The structural-metadata fields below (Phase 6) carry `Chunk`'s
    page/heading/table info all the way to citation-building
    (`build_prepared_context_section`) - without them, `page_start`/
    `page_end` stop at the Qdrant payload (Phase 5) and the LLM never sees
    what page a chunk came from, defeating the whole point of "citation by
    page". All default to `None`/`[]` so a point upserted before Phase 5
    (missing these payload keys entirely) still parses into a valid
    `RetrievedChunk` - just without page/heading/table info available for
    that older chunk.
    """

    chunk_id: str
    content: str
    source: str
    faculty: str = "GLOBAL"
    score: float = Field(ge=0.0, le=1.0)
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    source_type: str | None = None
    source_locator: SourceLocator | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    """Internal retrieval input with access metadata."""

    query: str
    user_faculty: str = "GLOBAL"
    user_level: int = Field(default=1, ge=1)
    limit: int = Field(default=5, ge=1, le=20)
