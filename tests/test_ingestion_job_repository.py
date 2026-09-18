from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DocumentChunk, DocumentProcessLog, DocumentProcessStep
from app.database.repositories.ingestion_job import (
    get_draft,
    mark_embedding,
    upsert_chunking_draft,
)
from app.schemas.ingestion import Chunk, HeaderSource, RegionType, SourceLocator, SourceType


def _chunk(index: int, content: str) -> Chunk:
    return Chunk(chunk_index=index, content=content, region_type=RegionType.TEXT)


@pytest.mark.asyncio
async def test_upsert_creates_a_draft(db_session: AsyncSession) -> None:
    await upsert_chunking_draft(
        db_session,
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        department_id="CNTT",
        strategy="recursive",
        params={"chunk_size": 800, "overlap": 120},
        chunks=[_chunk(0, "first"), _chunk(1, "second")],
    )

    draft = await get_draft(db_session, "doc-1")

    assert draft is not None
    assert draft.object_key == "docs/handbook.pdf"
    assert draft.department_id == "CNTT"
    assert draft.current_step == DocumentProcessStep.CHUNKED
    assert draft.chunking_strategy == "recursive"
    assert draft.chunking_params == {"chunk_size": 800, "overlap": 120}
    assert [c.content for c in draft.chunks] == ["first", "second"]


@pytest.mark.asyncio
async def test_upsert_twice_updates_in_place_and_replaces_chunks(
    db_session: AsyncSession,
) -> None:
    await upsert_chunking_draft(
        db_session,
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        department_id="CNTT",
        strategy="recursive",
        params={"chunk_size": 800},
        chunks=[_chunk(0, "old chunk")],
    )

    await upsert_chunking_draft(
        db_session,
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        department_id="CNTT",
        strategy="token_based",
        params={"chunk_size": 400},
        chunks=[_chunk(0, "new chunk 1"), _chunk(1, "new chunk 2")],
    )

    draft = await get_draft(db_session, "doc-1")

    assert draft is not None
    assert draft.chunking_strategy == "token_based"
    assert [c.content for c in draft.chunks] == ["new chunk 1", "new chunk 2"]

    count_result = await db_session.execute(select(func.count()).select_from(DocumentProcessLog))
    assert count_result.scalar_one() == 1


@pytest.mark.asyncio
async def test_upsert_on_an_embedding_draft_still_replaces_chunks(
    db_session: AsyncSession,
) -> None:
    """Re-chunking after an embed was dispatched: the row is kept (never
    deleted), flipped back to CHUNKED, its task id cleared, and its chunks
    fully replaced."""

    await upsert_chunking_draft(
        db_session,
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        department_id="CNTT",
        strategy="recursive",
        params={},
        chunks=[_chunk(0, "old chunk")],
    )
    await mark_embedding(db_session, document_id="doc-1", celery_task_id="task-abc")

    await upsert_chunking_draft(
        db_session,
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        department_id="CNTT",
        strategy="token_based",
        params={},
        chunks=[_chunk(0, "fresh chunk")],
    )

    draft = await get_draft(db_session, "doc-1")
    assert draft is not None
    assert draft.current_step == DocumentProcessStep.CHUNKED
    assert draft.celery_task_id is None
    assert [c.content for c in draft.chunks] == ["fresh chunk"]

    count_result = await db_session.execute(select(func.count()).select_from(DocumentChunk))
    assert count_result.scalar_one() == 1


@pytest.mark.asyncio
async def test_get_draft_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await get_draft(db_session, "no-such-document") is None


@pytest.mark.asyncio
async def test_upsert_and_get_draft_round_trips_full_structural_metadata(
    db_session: AsyncSession,
) -> None:
    """Task 0.2 acceptance: a `Chunk` with every new field set, persisted and
    read back, must match 100%."""

    chunk = Chunk(
        chunk_index=0,
        content="| Name | Score |\n| --- | --- |\n| Alice | 90 |",
        region_type=RegionType.TABLE,
        source_type=SourceType.HTML,
        block_index=3,
        heading_path=["Hoc phi", "Chinh quy"],
        page_start=None,
        page_end=None,
        source_locator=SourceLocator(
            table_id="table-3", row_start=1, row_end=1, row_count=1, section="Hoc phi > Chinh quy"
        ),
        column_names=["Name", "Score"],
        has_header=True,
        header_source=HeaderSource.EXPLICIT,
        header_confidence=1.0,
        chunking_version="2026-09-structural-v1",
    )

    await upsert_chunking_draft(
        db_session,
        document_id="doc-round-trip",
        object_key="docs/handbook.html",
        department_id="CNTT",
        strategy="recursive",
        params={},
        chunks=[chunk],
    )

    draft = await get_draft(db_session, "doc-round-trip")

    assert draft is not None
    assert len(draft.chunks) == 1
    assert draft.chunks[0] == chunk


@pytest.mark.asyncio
async def test_get_draft_defaults_chunking_version_to_legacy_for_pre_migration_rows(
    db_session: AsyncSession,
) -> None:
    """A row inserted before this feature existed (or by any code that never
    sets `chunk_metadata`) has an empty `metadata` JSON object - reading it
    back must resolve `chunking_version` to "legacy", never to the current
    `settings.CHUNKING_VERSION`."""

    log = DocumentProcessLog(
        id=uuid4(),
        document_id="doc-legacy",
        object_key="docs/handbook.pdf",
        department_id="CNTT",
        current_step=DocumentProcessStep.CHUNKED,
        chunking_strategy="recursive",
        chunking_params={},
    )
    db_session.add(log)
    await db_session.flush()
    db_session.add(
        DocumentChunk(
            id=uuid4(),
            process_log_id=log.id,
            chunk_index=0,
            content="old chunk, no metadata column populated",
            region_type=RegionType.TEXT.value,
            chunk_metadata={},
        )
    )
    await db_session.commit()

    draft = await get_draft(db_session, "doc-legacy")

    assert draft is not None
    assert draft.chunks[0].chunking_version == "legacy"
    assert draft.chunks[0].source_type is None
    assert draft.chunks[0].block_index is None
