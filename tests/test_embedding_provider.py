from unittest.mock import MagicMock, patch

import pytest

import app.core.registry.model_registry as model_registry
from app.core.registry.embedding_identity import (
    EmbeddingIdentityMismatchError,
    reset_verified_cache_for_tests,
)
from app.core.registry.model_registry import parse_snapshot
from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.embeddings.provider import EmbeddingProvider
from app.rag.vectorstore import qdrant_store


class _FakeEmbeddingProvider:
    """Test double proving callers only depend on the `EmbeddingProvider` protocol."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]


def _mock_openai_client(vectors: list[list[float]]) -> MagicMock:
    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=vector) for vector in vectors]
    )
    return client


def test_openai_embedder_returns_vectors_for_each_text() -> None:
    client = _mock_openai_client([[0.1, 0.2], [0.3, 0.4]])
    embedder = OpenAIEmbedder(model="text-embedding-3-small", client=client)

    vectors = embedder.embed(["a", "b"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=["a", "b"]
    )


def test_openai_embedder_returns_empty_list_for_no_texts() -> None:
    client = _mock_openai_client([])
    embedder = OpenAIEmbedder(model="text-embedding-3-small", client=client)

    assert embedder.embed([]) == []
    client.embeddings.create.assert_not_called()


_SNAPSHOT_PAYLOAD = {
    "version": 1,
    "generatedAt": "2026-09-25T03:00:00Z",
    "purposes": {
        "EMBEDDING": [
            {
                "id": "embed-cred",
                "revision": 1,
                "sourceType": "CLOUD_API",
                "provider": "openai",
                "modelName": "text-embedding-3-small",
                "apiBaseUrl": "https://api.openai.com/v1",
                "apiKey": "sk-test",
                "priority": 1,
                "maxRpm": 500,
            }
        ]
    },
    # A collection that already has vectors but no registered identity - the guard's
    # "already has vectors, no identity yet" refusal.
    "embeddingIndexIdentity": None,
}


def test_embedder_never_calls_provider_when_identity_guard_refuses() -> None:
    """Resolving model/client from the registry (no injected `model`/`client`) routes through
    `ensure_embedding_identity` before the provider is ever touched - todo.md Task 13b's
    "provider must NOT be called at all" criterion, proven by a call-count assertion on the
    OpenAI client the embedder itself would have built, not just that an exception was raised."""

    reset_verified_cache_for_tests()
    model_registry._current_snapshot = parse_snapshot(_SNAPSHOT_PAYLOAD)
    try:
        with (
            patch.object(qdrant_store, "collection_has_points", return_value=True),
            patch.object(qdrant_store, "get_collection_dimension", return_value=1536),
            patch("app.rag.embeddings.openai_embedder.OpenAI") as mock_openai_cls,
        ):
            embedder = OpenAIEmbedder()
            with pytest.raises(EmbeddingIdentityMismatchError):
                embedder.embed(["query text"])

            mock_openai_cls.return_value.embeddings.create.assert_not_called()
    finally:
        model_registry._current_snapshot = None
        reset_verified_cache_for_tests()


def test_swapping_provider_requires_no_call_site_changes() -> None:
    provider: EmbeddingProvider = _FakeEmbeddingProvider()

    vectors = provider.embed(["ab", "abc"])

    assert vectors == [[2.0], [3.0]]
