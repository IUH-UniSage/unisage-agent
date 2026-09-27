"""`python -m app.tools.register_embedding_index_identity` — the one-off bootstrap CLI, todo.md
Task 13. No live Qdrant, OpenAI, or backend-java anywhere in this file."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from app.integrations.backend_java_client import BackendJavaHTTPError
from app.tools import register_embedding_index_identity as tool

_DIMENSION = 3
_PROBE_VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _mock_openai_cls() -> MagicMock:
    mock_cls = MagicMock()
    mock_cls.return_value.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=vector) for vector in _PROBE_VECTORS]
    )
    return mock_cls


def _mock_qdrant_store(
    *, collection_exists: bool = True, dimension: int | None = _DIMENSION
) -> MagicMock:
    mock_store = MagicMock()
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = collection_exists
    mock_store.get_client.return_value = mock_client
    mock_store.get_collection_dimension.return_value = dimension
    return mock_store


def test_main_fails_without_an_api_key() -> None:
    with patch.object(tool, "qdrant_store", _mock_qdrant_store()):
        exit_code = tool.main(["--api-key", ""])

    assert exit_code == 1


def test_main_fails_when_collection_does_not_exist() -> None:
    with patch.object(tool, "qdrant_store", _mock_qdrant_store(collection_exists=False)):
        exit_code = tool.main(["--api-key", "sk-test"])

    assert exit_code == 1


def test_main_registers_identity_on_first_run() -> None:
    mock_backend_client_cls = MagicMock()
    mock_backend_client_cls.return_value.put_embedding_index_identity = AsyncMock(
        return_value={"established": True}
    )

    with (
        patch.object(tool, "qdrant_store", _mock_qdrant_store()),
        patch.object(tool, "OpenAI", _mock_openai_cls()),
        patch.object(tool, "BackendJavaClient", mock_backend_client_cls),
    ):
        exit_code = tool.main(["--api-key", "sk-test", "--model-name", "text-embedding-3-small"])

    assert exit_code == 0
    mock_backend_client_cls.return_value.put_embedding_index_identity.assert_awaited_once()
    put_kwargs = mock_backend_client_cls.return_value.put_embedding_index_identity.call_args.kwargs
    assert put_kwargs["established_by"] == "bootstrap-cli"
    assert put_kwargs["dimension"] == _DIMENSION
    assert put_kwargs["fingerprint"] == [v for vector in _PROBE_VECTORS for v in vector]


def test_main_refuses_when_measured_dimension_disagrees_with_collection() -> None:
    with (
        patch.object(tool, "qdrant_store", _mock_qdrant_store(dimension=1536)),
        patch.object(tool, "OpenAI", _mock_openai_cls()),
    ):
        exit_code = tool.main(["--api-key", "sk-test"])

    assert exit_code == 1


def test_main_run_twice_second_run_reconciles_without_overwriting() -> None:
    """Second run gets 409 (already registered by the first run) - reads it back, confirms it
    matches, and exits 0 WITHOUT ever calling anything that could overwrite it."""

    mock_backend_client_cls = MagicMock()
    mock_backend_client_cls.return_value.put_embedding_index_identity = AsyncMock(
        side_effect=BackendJavaHTTPError(
            "PUT", "/x", 409, {"error": "EMBEDDING_INDEX_IDENTITY_EXISTS"}
        )
    )
    mock_backend_client_cls.return_value.get_embedding_index_identity = AsyncMock(
        return_value={
            "provider": "openai",
            "modelName": "text-embedding-3-small",
            "modelSourceRef": None,
            "apiBaseUrl": "https://api.openai.com/v1",
            "dimension": _DIMENSION,
            "fingerprint": [v for vector in _PROBE_VECTORS for v in vector],
        }
    )

    with (
        patch.object(tool, "qdrant_store", _mock_qdrant_store()),
        patch.object(tool, "OpenAI", _mock_openai_cls()),
        patch.object(tool, "BackendJavaClient", mock_backend_client_cls),
    ):
        exit_code = tool.main(["--api-key", "sk-test", "--model-name", "text-embedding-3-small"])

    assert exit_code == 0
    mock_backend_client_cls.return_value.get_embedding_index_identity.assert_awaited_once()


def test_main_409_with_mismatched_existing_identity_fails_loudly() -> None:
    mock_backend_client_cls = MagicMock()
    mock_backend_client_cls.return_value.put_embedding_index_identity = AsyncMock(
        side_effect=BackendJavaHTTPError(
            "PUT", "/x", 409, {"error": "EMBEDDING_INDEX_IDENTITY_EXISTS"}
        )
    )
    mock_backend_client_cls.return_value.get_embedding_index_identity = AsyncMock(
        return_value={
            "provider": "openai",
            "modelName": "text-embedding-3-large",
            "modelSourceRef": None,
            "apiBaseUrl": "https://api.openai.com/v1",
            "dimension": _DIMENSION,
            "fingerprint": [v for vector in _PROBE_VECTORS for v in vector],
        }
    )

    with (
        patch.object(tool, "qdrant_store", _mock_qdrant_store()),
        patch.object(tool, "OpenAI", _mock_openai_cls()),
        patch.object(tool, "BackendJavaClient", mock_backend_client_cls),
    ):
        exit_code = tool.main(["--api-key", "sk-test", "--model-name", "text-embedding-3-small"])

    assert exit_code == 1
