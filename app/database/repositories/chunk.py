from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import DocumentChunk, DocumentProcessLog
from app.schemas.ingestion import Chunk, RegionType


@dataclass(frozen=True)
class ChunkPageDTO:
    """One page of a document's chunks, plus the owning department (for the
    caller to authorize) and the total chunk count (for pagination)."""

    department_id: str | None
    total: int
    chunks: list[Chunk]


class ChunkRepository:
    """Persistence boundary for document chunks."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_page(self, document_id: str, *, limit: int, offset: int) -> ChunkPageDTO | None:
        """Return a page of chunks for one document, ordered by `chunk_index`.

        Returns `None` if the document has no chunking draft at all (distinct
        from a draft with zero chunks, which would return a page with an
        empty `chunks` list and `total=0`).
        """

        log_result = await self._session.execute(
            select(DocumentProcessLog.id, DocumentProcessLog.department_id).where(
                DocumentProcessLog.document_id == document_id
            )
        )
        log_row = log_result.first()
        if log_row is None:
            return None
        process_log_id, department_id = log_row

        count_result = await self._session.execute(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.process_log_id == process_log_id)
        )
        total = count_result.scalar_one()

        chunks_result = await self._session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.process_log_id == process_log_id)
            .order_by(DocumentChunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
        chunks = [
            Chunk(
                chunk_index=row.chunk_index,
                content=row.content,
                region_type=RegionType(row.region_type),
                **row.chunk_metadata,
            )
            for row in chunks_result.scalars()
        ]

        return ChunkPageDTO(department_id=department_id, total=total, chunks=chunks)
