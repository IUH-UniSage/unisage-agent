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
    department_id: str | None
    current_step: DocumentProcessStep
    chunking_strategy: str
    chunking_params: dict[str, Any]
    chunks: list[Chunk]
    celery_task_id: str | None


async def _load_log(
    session: AsyncSession, document_id: str, *, with_chunks: bool = False
) -> DocumentProcessLog | None:
    """Fetch the single process-log row for a document, or `None`."""

    query = select(DocumentProcessLog).where(DocumentProcessLog.document_id == document_id)
    if with_chunks:
        query = query.options(selectinload(DocumentProcessLog.chunks))
    result = await session.execute(query)
    return result.scalar_one_or_none()


async def upsert_chunking_draft(
    session: AsyncSession,
    *,
    document_id: str,
    object_key: str,
    department_id: str,
    strategy: str,
    params: dict[str, Any],
    chunks: list[Chunk],
) -> None:
    """Persist (or replace) the chunking draft for one document.

    Insert-or-update the log row, then fully replace its chunks (delete
    existing, insert the freshly computed list) - "last write wins", no
    locking, per SPEC-ingestion-resume.md.
    """

    log = await _load_log(session, document_id)

    if log is None:
        log = DocumentProcessLog(
            id=uuid4(),
            document_id=document_id,
            object_key=object_key,
            department_id=department_id,
            current_step=DocumentProcessStep.CHUNKED,
            chunking_strategy=strategy,
            chunking_params=params,
        )
        session.add(log)
    else:
        log.object_key = object_key
        log.department_id = department_id
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
            chunk_metadata=chunk.model_dump(
                mode="json", exclude={"chunk_index", "content", "region_type"}
            ),
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

    log = await _load_log(session, document_id)
    if log is None:
        return

    log.current_step = DocumentProcessStep.EMBEDDING
    log.celery_task_id = celery_task_id
    await session.commit()


async def get_draft(session: AsyncSession, document_id: str) -> DraftDTO | None:
    """Fetch the process-log record for one document, or `None` if there isn't one."""

    log = await _load_log(session, document_id, with_chunks=True)
    if log is None:
        return None

    return DraftDTO(
        object_key=log.object_key,
        department_id=log.department_id,
        current_step=log.current_step,
        chunking_strategy=log.chunking_strategy,
        chunking_params=log.chunking_params,
        celery_task_id=log.celery_task_id,
        chunks=[
            Chunk(
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                region_type=RegionType(chunk.region_type),
                **chunk.chunk_metadata,
            )
            for chunk in sorted(log.chunks, key=lambda chunk: chunk.chunk_index)
        ],
    )
