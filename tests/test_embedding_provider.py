from unittest.mock import MagicMock

from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.embeddings.provider import EmbeddingProvider


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


def test_swapping_provider_requires_no_call_site_changes() -> None:
    provider: EmbeddingProvider = _FakeEmbeddingProvider()

    vectors = provider.embed(["ab", "abc"])

    assert vectors == [[2.0], [3.0]]
