from sqlalchemy.ext.asyncio import AsyncSession


class ChunkRepository:
    """Persistence boundary for document chunks and embeddings."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
