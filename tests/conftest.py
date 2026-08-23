from collections.abc import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.database.session import get_db_session
from app.main import app

settings.INTERNAL_SECRET_KEY = "test-internal-secret"


async def mock_get_db_session() -> AsyncGenerator[None, None]:
    """Provide a database-free dependency for HTTP tests."""

    yield None


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client with the database dependency overridden.

    Sends `X-Internal-Secret` by default on every request so tests don't need
    to know about the API-Gateway-only access gate individually.
    """

    app.dependency_overrides[get_db_session] = mock_get_db_session
    with TestClient(
        app, headers={"X-Internal-Secret": settings.INTERNAL_SECRET_KEY}
    ) as test_client:
        yield test_client
    app.dependency_overrides.clear()
