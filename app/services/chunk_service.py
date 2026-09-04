from app.core.exceptions import DocumentChunksNotFoundException
from app.database.repositories.chunk import ChunkPageDTO, ChunkRepository

MAX_LIMIT = 200


class ChunkService:
    """Business logic for reading a document's chunks - sits between the API
    route and `ChunkRepository`."""

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
