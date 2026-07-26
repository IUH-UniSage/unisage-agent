from sqlalchemy.ext.asyncio import AsyncSession


class ConversationRepository:
    """Persistence boundary for conversations and messages."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
