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
    """Which ingestion stage last wrote a document's process-log record.

    `CHUNKED`: a chunking draft exists, not yet embedded - resuming restores
    the review step. `EMBEDDING`: the client dispatched the Celery embed
    task for this draft's chunks and the row's `celery_task_id` identifies
    it - resuming reconnects the progress view instead of restarting from
    preview.

    There is deliberately no terminal "embedded" value. The row is a
    permanent history record - it is never deleted on completion. Whether an
    embed has finished is answered by the Celery task state (broadcast on
    `WS /ingestion/events` and read back inline on
    `GET /ingestion/jobs/{document_id}`) and, authoritatively, by Java's
    `Document.status`, not by a column here or by row deletion.
    """

    CHUNKED = "chunked"
    EMBEDDING = "embedding"


class DocumentProcessLog(Base):
    """A resumable chunking draft / embedding record for one document, keyed
    by Java's document_id.

    `document_id` is a plain, unconstrained column (no FK into Java's
    `documents` table): this feature lives in Python's own schema/database,
    separate from Java's Hibernate-managed one, by design.

    The row is a permanent history record - it is not deleted when embedding
    finishes (see `DocumentProcessStep`).
    """

    __tablename__ = "document_process_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    object_key: Mapped[str] = mapped_column(String, nullable=False)
    # Owning department, copied from the chunking request. Nullable only for
    # rows written before the column existed; new rows always set it, and
    # `GET /ingestion/jobs/{document_id}` authorizes membership against it.
    department_id: Mapped[str | None] = mapped_column(String, nullable=True)
    current_step: Mapped[DocumentProcessStep] = mapped_column(
        SqlEnum(DocumentProcessStep, name="documentprocessstep", native_enum=True),
        nullable=False,
        default=DocumentProcessStep.CHUNKED,
    )
    chunking_strategy: Mapped[str] = mapped_column(String, nullable=False)
    chunking_params: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=False, default=dict
    )
    celery_task_id: Mapped[str | None] = mapped_column(String, nullable=True)
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
    # Every `Chunk` field other than chunk_index/content/region_type
    # (heading_path, source_type, block_index, source_locator, ...),
    # serialized as-is via `Chunk.model_dump()`. `default={}` (not just
    # server-side) so a row inserted before this column existed, or by code
    # that forgets to set it, reads back as an empty dict rather than NULL -
    # `Chunk(**{})` then falls back to every field's own Pydantic default,
    # including `chunking_version="legacy"`.
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        default=dict,
    )

    process_log: Mapped[DocumentProcessLog] = relationship(back_populates="chunks")


class ConversationClarificationState(Base):
    """Missing-metadata clarification state for one conversation, keyed by
    Java's conversation_id.

    Same pattern as `DocumentProcessLog`: a plain, unconstrained
    `conversation_id` column (no FK into Java's `conversations` table) — this
    is Python's own internal processing state, not chat history, and lives
    in Python's own schema/database by design (see tasks/plan.md).

    `pending_clarification` holds a `PendingClarification`-shaped JSON object
    (see `app/schemas/clarification.py`) or `None` when nothing is pending.
    `confirmed_metadata` accumulates self-declared student attributes across
    the whole conversation and never expires on its own - it is only ever
    read to pick which answer branch to render, never as a Qdrant filter
    (see tasks/plan.md security boundary note).
    """

    __tablename__ = "conversation_clarification_states"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    pending_clarification: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True
    )
    confirmed_metadata: Mapped[dict[str, str]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=False, default=dict
    )
    # Panel v2 state machine (UNISAGE-99): NULL = no round, OPEN, or PROCESSING while
    # one request holds `claim_token` until `claim_expires_at`.
    pending_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    pending_panel_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True, index=True)
    claim_token: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
