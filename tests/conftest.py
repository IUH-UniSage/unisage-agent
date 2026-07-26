from collections.abc import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient

from app.database.session import get_db_session
from app.main import app


async def mock_get_db_session() -> AsyncGenerator[None, None]:
    """Provide a database-free dependency for HTTP tests."""

    yield None


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client with the database dependency overridden."""

    app.dependency_overrides[get_db_session] = mock_get_db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
