from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database.models import DocumentChunk, DocumentProcessLog, DocumentProcessStep
from app.schemas.ingestion import Chunk, RegionType


@dataclass(frozen=True)
class DraftDTO:
    """A resumable chunking or in-flight-embedding draft, read back from
    `document_process_logs`."""

    object_key: str
    current_step: DocumentProcessStep
    chunking_strategy: str
    chunking_params: dict[str, Any]
    chunks: list[Chunk]
    celery_task_id: str | None


async def upsert_chunking_draft(
    session: AsyncSession,
    *,
    document_id: str,
    object_key: str,
    strategy: str,
    params: dict[str, Any],
    chunks: list[Chunk],
) -> None:
    """Persist (or replace) the chunking draft for one document.

    Insert-or-update the log row, then fully replace its chunks (delete
    existing, insert the freshly computed list) - "last write wins", no
    locking, per SPEC-ingestion-resume.md.
    """

    result = await session.execute(
        select(DocumentProcessLog).where(DocumentProcessLog.document_id == document_id)
    )
    log = result.scalar_one_or_none()

    if log is None:
        log = DocumentProcessLog(
            id=uuid4(),
            document_id=document_id,
            object_key=object_key,
            current_step=DocumentProcessStep.CHUNKED,
            chunking_strategy=strategy,
            chunking_params=params,
        )
        session.add(log)
    else:
        log.object_key = object_key
        log.current_step = DocumentProcessStep.CHUNKED
        log.chunking_strategy = strategy
        log.chunking_params = params
        log.celery_task_id = None
        await session.execute(delete(DocumentChunk).where(DocumentChunk.process_log_id == log.id))

    session.add_all(
        DocumentChunk(
            id=uuid4(),
            process_log_id=log.id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            region_type=chunk.region_type.value,
        )
        for chunk in chunks
    )
    await session.commit()


async def mark_embedding(session: AsyncSession, *, document_id: str, celery_task_id: str) -> None:
    """Flip an existing draft to `EMBEDDING` and record the dispatched task's id.

    A no-op if no draft row exists for `document_id` - shouldn't happen in
    practice (embedding always follows a chunking call, which upserts the
    row), but there's nothing to persist a task id onto if it's missing.
    """

    result = await session.execute(
        select(DocumentProcessLog).where(DocumentProcessLog.document_id == document_id)
    )
    log = result.scalar_one_or_none()
    if log is None:
        return

    log.current_step = DocumentProcessStep.EMBEDDING
    log.celery_task_id = celery_task_id
    await session.commit()


async def get_draft(session: AsyncSession, document_id: str) -> DraftDTO | None:
    """Fetch the chunking draft for one document, or `None` if there isn't one."""

    result = await session.execute(
        select(DocumentProcessLog)
        .options(selectinload(DocumentProcessLog.chunks))
        .where(DocumentProcessLog.document_id == document_id)
    )
    log = result.scalar_one_or_none()
    if log is None:
        return None

    return DraftDTO(
        object_key=log.object_key,
        current_step=log.current_step,
        chunking_strategy=log.chunking_strategy,
        chunking_params=log.chunking_params,
        celery_task_id=log.celery_task_id,
        chunks=[
            Chunk(
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                region_type=RegionType(chunk.region_type),
            )
            for chunk in sorted(log.chunks, key=lambda chunk: chunk.chunk_index)
        ],
    )


async def delete_draft(session: AsyncSession, document_id: str) -> None:
    """Delete the draft (and its chunks, via cascade) for one document, if any.

    A no-op, not an error, when there was no draft to begin with.
    """

    result = await session.execute(
        select(DocumentProcessLog).where(DocumentProcessLog.document_id == document_id)
    )
    log = result.scalar_one_or_none()
    if log is not None:
        await session.delete(log)
        await session.commit()
