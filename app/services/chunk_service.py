from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.core.errors.exceptions import DocumentChunksNotFoundException
from app.database.repositories.chunk import ChunkPageDTO, ChunkRepository
from app.rag.vectorstore import qdrant_store
from app.schemas.ingestion import IndexedChunk

MAX_LIMIT = 200


def _with_chunk_index(payload: dict[str, object]) -> dict[str, object]:
    """Qdrant payloads store `chunk_id` (`"{document_id}:{chunk_index}"`) but
    not `chunk_index` itself, which `Chunk` (and therefore `IndexedChunk`)
    requires - derive it back out so `model_validate` succeeds."""

    if "chunk_index" in payload:
        return payload
    chunk_id = str(payload.get("chunk_id", ""))
    _, _, index_part = chunk_id.rpartition(":")
    return {**payload, "chunk_index": int(index_part) if index_part.isdigit() else 0}


@dataclass(frozen=True)
class IndexedChunkPageDTO:
    """One page of a document's live-indexed (Qdrant) chunks, plus the owning
    department (for the caller to authorize, read off the first point's
    payload since every point for one document shares the same department)."""

    department_id: str | None
    total: int
    chunks: list[IndexedChunk]


class ChunkService:
    """Business logic for reading a document's chunks - sits between the API
    route and `ChunkRepository`/Qdrant."""

    def __init__(self, repository: ChunkRepository) -> None:
        self._repository = repository

    async def list_document_chunks(
        self, document_id: str, *, page: int, limit: int
    ) -> ChunkPageDTO:
        """Return one page (1-indexed, matching Java's `PageResponse`
        convention) of a document's chunks.

        Raises `DocumentChunksNotFoundException` if the document has no
        chunking draft at all yet.
        """

        clamped_limit = min(limit, MAX_LIMIT)
        offset = (page - 1) * clamped_limit
        chunk_page = await self._repository.get_page(
            document_id, limit=clamped_limit, offset=offset
        )
        if chunk_page is None:
            raise DocumentChunksNotFoundException(document_id)
        return chunk_page

    def list_indexed_chunks(
        self, client: QdrantClient, document_id: str, *, page: int, limit: int
    ) -> IndexedChunkPageDTO:
        """Return one page of `document_id`'s chunks as actually indexed in
        Qdrant (including `summary`/`questions`), sorted by `chunk_id`.

        Empty (not an error) when nothing has been embedded for this document
        yet - a document can legitimately have a chunking draft but no
        indexed chunks (still on the "chunked" ingestion step).
        """

        clamped_limit = min(limit, MAX_LIMIT)
        records = qdrant_store.scroll_chunks_by_document(client, document_id)
        total = len(records)
        offset = (page - 1) * clamped_limit
        page_records = records[offset : offset + clamped_limit]

        department_id = (
            str(records[0].payload["department"]) if records and records[0].payload else None
        )
        chunks = [
            IndexedChunk.model_validate(_with_chunk_index(record.payload))
            for record in page_records
            if record.payload
        ]
        return IndexedChunkPageDTO(department_id=department_id, total=total, chunks=chunks)

    def delete_indexed_chunk(self, client: QdrantClient, document_id: str, chunk_id: str) -> None:
        """Remove one chunk's point from the live Qdrant index.

        Does not touch the Postgres chunking draft - this only affects what
        is actually retrievable by the RAG pipeline.
        """

        qdrant_store.delete_chunk_point(client, document_id, chunk_id)
