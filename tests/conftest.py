import asyncio
from collections.abc import AsyncGenerator, Callable, Generator, Sequence

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from pydantic_ai.models.function import FunctionModel
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.api.deps import get_session_factory
from app.core.config import settings
from app.database.models import Base
from app.database.session import get_db_session
from app.main import app
from tests.llm_mocks import (
    make_gated_streaming_llm_model,
    make_sequential_streaming_llm_model,
    make_streaming_llm_model,
    make_sync_llm_model,
)

settings.APP_INTERNAL_SECRET_KEY = "test-internal-secret"


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
    # run_and_persist opens its OWN session via this factory,
    # independent of the request's db_session - point it at the same
    # in-memory SQLite engine so a streaming test can see what it wrote.
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    with TestClient(
        app, headers={"X-Internal-Secret": settings.APP_INTERNAL_SECRET_KEY}
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


@pytest_asyncio.fixture
async def db_session_factory() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """A session factory bound to one fresh in-memory SQLite engine.

    Unlike `db_session`, this hands out the factory itself (not a single
    session) so a test can open multiple independent `AsyncSession`s against
    the same schema/data - needed to simulate two genuinely concurrent
    callers racing the same row (see the clarification-state upsert
    concurrency test).
    """

    engine, session_factory = _in_memory_sqlite_engine_and_sessions()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield session_factory

    await engine.dispose()


@pytest.fixture
def mock_streaming_llm_model() -> Callable[[Sequence[str]], FunctionModel]:
    """Factory fixture: `mock_streaming_llm_model(["Xin ", "chào"])` builds a
    `pydantic_ai` model double whose `Agent.run_stream()` yields those tokens in
    order. For nodes that stream (GenerationSynthesisNode).
    """

    return make_streaming_llm_model


@pytest.fixture
def mock_sequential_streaming_llm_model() -> Callable[[Sequence[Sequence[str]]], FunctionModel]:
    """Factory fixture: `mock_sequential_streaming_llm_model([["a"], ["b"]])` builds
    a model double whose `Agent.run_stream()` yields `["a"]` on the first call
    and `["b"]` on the second - for a node that calls the same agent more than
    once per turn (see GenerationSynthesisNode's JSON-repair follow-up call).
    """

    return make_sequential_streaming_llm_model


@pytest.fixture
def mock_gated_streaming_llm_model() -> Callable[[Sequence[str], asyncio.Event], FunctionModel]:
    """Factory fixture: `mock_gated_streaming_llm_model(["a", "b"], gate)` builds a
    streaming model double that yields the first token, then pauses until `gate`
    is set before yielding the rest - for tests that need to deterministically
    observe a stream "mid-flight" (see tests/llm_mocks.py)."""

    return make_gated_streaming_llm_model


@pytest.fixture
def mock_sync_llm_model() -> Callable[[str], FunctionModel]:
    """Factory fixture: `mock_sync_llm_model("some text")` builds a
    `pydantic_ai` model double whose `Agent.run()`/`run_sync()` returns that text
    as one response. For non-streaming nodes (message classification, query
    transformation, comparison/calculation).
    """

    return make_sync_llm_model
