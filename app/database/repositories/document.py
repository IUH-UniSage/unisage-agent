from sqlalchemy.ext.asyncio import AsyncSession


class DocumentRepository:
    """Persistence boundary for source documents."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
