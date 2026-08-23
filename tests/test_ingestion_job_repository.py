import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DocumentChunk, DocumentProcessLog, DocumentProcessStep
from app.database.repositories.ingestion_job import delete_draft, get_draft, upsert_chunking_draft
from app.schemas.ingestion import Chunk, RegionType


def _chunk(index: int, content: str) -> Chunk:
    return Chunk(chunk_index=index, content=content, region_type=RegionType.TEXT)


@pytest.mark.asyncio
async def test_upsert_creates_a_draft(db_session: AsyncSession) -> None:
    await upsert_chunking_draft(
        db_session,
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        strategy="recursive",
        params={"chunk_size": 800, "overlap": 120},
        chunks=[_chunk(0, "first"), _chunk(1, "second")],
    )

    draft = await get_draft(db_session, "doc-1")

    assert draft is not None
    assert draft.object_key == "docs/handbook.pdf"
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
        strategy="recursive",
        params={"chunk_size": 800},
        chunks=[_chunk(0, "old chunk")],
    )

    await upsert_chunking_draft(
        db_session,
        document_id="doc-1",
        object_key="docs/handbook.pdf",
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
async def test_delete_draft_removes_log_and_cascades_to_chunks(db_session: AsyncSession) -> None:
    await upsert_chunking_draft(
        db_session,
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        strategy="recursive",
        params={},
        chunks=[_chunk(0, "a chunk")],
    )

    await delete_draft(db_session, "doc-1")

    assert await get_draft(db_session, "doc-1") is None

    count_result = await db_session.execute(select(func.count()).select_from(DocumentChunk))
    assert count_result.scalar_one() == 0


@pytest.mark.asyncio
async def test_delete_draft_is_a_noop_when_nothing_to_delete(db_session: AsyncSession) -> None:
    await delete_draft(db_session, "no-such-document")


@pytest.mark.asyncio
async def test_get_draft_returns_none_when_absent(db_session: AsyncSession) -> None:
    assert await get_draft(db_session, "no-such-document") is None
