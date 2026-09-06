from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    # Always off, independent of settings.DEBUG - SQLAlchemy's echo=True
    # forces the "sqlalchemy.engine.Engine" logger to INFO and attaches its
    # own handler that bypasses the level app/main.py sets on it, printing
    # every statement twice. settings.DEBUG now means "dump the generation
    # prompt" (see app/core/graph_trace.py), not "echo SQL".
    echo=False,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield one transactional database session per request."""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
