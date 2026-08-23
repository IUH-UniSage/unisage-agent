from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.database.models import Base
from app.database.session import get_db_session
from app.main import app

settings.INTERNAL_SECRET_KEY = "test-internal-secret"


def _in_memory_sqlite_engine_and_sessions() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Build a fresh in-memory SQLite engine and a session factory bound to it.

    Keeps the default test run free of a live Postgres dependency (same
    principle as the rest of this project's "no live external service in
    the default test run" rule) while still exercising real SQL, cascades,
    and constraints via SQLAlchemy's async engine.
    """

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    return engine, async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client with `get_db_session` backed by an in-memory SQLite DB.

    Sends `X-Internal-Secret` by default on every request so tests don't need
    to know about the API-Gateway-only access gate individually. The session
    is real (not `None`) so endpoints that persist a side effect (e.g. the
    chunking-draft resume feature) work end-to-end within a single test.
    """

    engine, session_factory = _in_memory_sqlite_engine_and_sessions()
    tables_ready = False

    async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        nonlocal tables_ready
        if not tables_ready:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            tables_ready = True
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    with TestClient(
        app, headers={"X-Internal-Secret": settings.INTERNAL_SECRET_KEY}
    ) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """An isolated in-memory SQLite session with all tables created fresh."""

    engine, session_factory = _in_memory_sqlite_engine_and_sessions()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        yield session

    await engine.dispose()
