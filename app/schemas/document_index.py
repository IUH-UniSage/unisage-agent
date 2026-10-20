"""Backend-only document index sync - changes/17-10-2026-Embedding-Reindex (unisage-backend)
plan.md §2.4."""

from pydantic import BaseModel, Field


class IndexMetadataUpdateRequest(BaseModel):
    """The access payload every point of a document carries, after a document edit."""

    department_id: str | None
    access_level: int = Field(ge=0)
    is_public: bool


class IndexMetadataUpdateResponse(BaseModel):
    updated_points: int


class IndexDeleteResponse(BaseModel):
    deleted_points: int
