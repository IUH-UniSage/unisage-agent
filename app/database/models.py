import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for SQLAlchemy persistence models."""


class DocumentProcessStep(StrEnum):
    """Which ingestion stage last wrote a document's draft record.

    Ships with exactly one member on purpose: this feature only tracks "a
    chunking draft exists, not yet embedded." Preview isn't tracked (no row
    created for it) and a successful embed deletes the row entirely, so
    there is no state machine to model yet - this enum is a
    forward-compatible extension point for a later feature to add one.
    """

    CHUNKED = "chunked"


class DocumentProcessLog(Base):
    """A resumable chunking draft for one document, keyed by Java's document_id.

    `document_id` is a plain, unconstrained column (no FK into Java's
    `documents` table): this feature lives in Python's own schema/database,
    separate from Java's Hibernate-managed one, by design.
    """

    __tablename__ = "document_process_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    object_key: Mapped[str] = mapped_column(String, nullable=False)
    current_step: Mapped[DocumentProcessStep] = mapped_column(
        SqlEnum(DocumentProcessStep, name="documentprocessstep", native_enum=True),
        nullable=False,
        default=DocumentProcessStep.CHUNKED,
    )
    chunking_strategy: Mapped[str] = mapped_column(String, nullable=False)
    chunking_params: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="process_log",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    """One chunk of a `DocumentProcessLog` draft."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    process_log_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("document_process_logs.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    region_type: Mapped[str] = mapped_column(String, nullable=False)

    process_log: Mapped[DocumentProcessLog] = relationship(back_populates="chunks")
