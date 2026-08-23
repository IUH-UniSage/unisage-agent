from collections.abc import AsyncGenerator
from dataclasses import dataclass

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import MissingTrustedContextException
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


@dataclass(frozen=True)
class TrustedContext:
    """Caller identity trusted because the API Gateway injected it after JWT verification."""

    department: str
    access_level: str


async def get_trusted_context(
    x_user_department: str | None = Header(default=None),
    x_user_access_level: str | None = Header(default=None),
) -> TrustedContext:
    """Read the gateway-injected trusted headers, failing loudly if either is absent."""

    if not x_user_department:
        raise MissingTrustedContextException("X-User-Department")
    if not x_user_access_level:
        raise MissingTrustedContextException("X-User-Access-Level")
    return TrustedContext(department=x_user_department, access_level=x_user_access_level)
