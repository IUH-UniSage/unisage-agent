from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.session import get_db_session
from app.graph.deps import ChatDeps


async def get_chat_deps(
    db_session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[ChatDeps, None]:
    """Build graph dependencies from the request-scoped database session."""

    yield ChatDeps(
        db_session=db_session,
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.OPENAI_MODEL,
    )
