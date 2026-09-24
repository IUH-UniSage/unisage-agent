from typing import Any

from pydantic import BaseModel, Field

from app.schemas.ingestion import SourceLocator


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval before generation.

    The structural-metadata fields carry page/heading/table info through to
    citations. They default to `None`/`[]` so an older point without these
    payload keys still parses.
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
