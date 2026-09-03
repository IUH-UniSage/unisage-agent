import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DocumentChunk, DocumentProcessLog, DocumentProcessStep


def test_document_process_step_has_exactly_two_members() -> None:
    assert list(DocumentProcessStep) == [
        DocumentProcessStep.CHUNKED,
        DocumentProcessStep.EMBEDDING,
    ]


@pytest.mark.asyncio
async def test_document_id_unique_constraint_is_enforced(db_session: AsyncSession) -> None:
    db_session.add(
        DocumentProcessLog(
            document_id="doc-1",
            object_key="docs/handbook.pdf",
            chunking_strategy="recursive",
            chunking_params={"chunk_size": 800},
        )
    )
    await db_session.commit()

    db_session.add(
        DocumentProcessLog(
            document_id="doc-1",
            object_key="docs/other.pdf",
            chunking_strategy="token_based",
            chunking_params={},
        )
    )
    with pytest.raises(Exception):  # noqa: B017 - dialect-specific IntegrityError
        await db_session.commit()


@pytest.mark.asyncio
async def test_deleting_process_log_cascades_to_its_chunks(db_session: AsyncSession) -> None:
    log = DocumentProcessLog(
        document_id="doc-2",
        object_key="docs/handbook.pdf",
        chunking_strategy="recursive",
        chunking_params={"chunk_size": 800},
    )
    log.chunks = [
        DocumentChunk(id=uuid.uuid4(), chunk_index=0, content="a", region_type="text"),
        DocumentChunk(id=uuid.uuid4(), chunk_index=1, content="b", region_type="text"),
    ]
    db_session.add(log)
    await db_session.commit()

    await db_session.delete(log)
    await db_session.commit()

    remaining = await db_session.execute(select(DocumentChunk))
    assert remaining.scalars().all() == []
