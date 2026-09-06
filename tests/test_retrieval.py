"""RetrievalService - embeds the query and searches Qdrant's
`content_vector` for ingested chunks. Mocked Qdrant client + embedder, no
live network anywhere in this file (matches the project's convention for
`BackendJavaClient`/`httpx.MockTransport`)."""

from unittest.mock import MagicMock

from qdrant_client.http.models import QueryResponse, ScoredPoint

from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.retrieval.service import RetrievalService


def _scored_point(chunk_id: str, score: float, content: str = "nội dung") -> ScoredPoint:
    return ScoredPoint(
        id=chunk_id,
        version=0,
        score=score,
        payload={
            "chunk_id": chunk_id,
            "content": content,
            "object_key": "docs/handbook.pdf",
            "document_id": "doc-1",
            "department": "CNTT",
            "access_level": 1,
            "region_type": "text",
        },
    )


def _embedder_returning(vector: list[float]) -> OpenAIEmbedder:
    fake_openai_client = MagicMock()
    fake_openai_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=vector)]
    )
    return OpenAIEmbedder(client=fake_openai_client)


def test_retrieve_embeds_query_and_maps_qdrant_points_to_retrieved_chunks() -> None:
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value = QueryResponse(points=[_scored_point("c1", 0.83)])
    service = RetrievalService(client=fake_qdrant, embedder=_embedder_returning([0.1, 0.2]))

    chunks = service.retrieve("học phí học kỳ này bao nhiêu")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].content == "nội dung"
    assert chunks[0].source == "docs/handbook.pdf"
    assert chunks[0].score == 0.83
    # Fans out across all 3 named vectors (content/summary/questions).
    assert fake_qdrant.query_points.call_count == 3
    used_vectors = {call.kwargs["using"] for call in fake_qdrant.query_points.call_args_list}
    assert used_vectors == {"content_vector", "summary_vector", "questions_vector"}
    for call in fake_qdrant.query_points.call_args_list:
        assert call.kwargs["query"] == [0.1, 0.2]


def test_retrieve_returns_empty_list_when_collection_does_not_exist() -> None:
    """A fresh environment with nothing ingested yet is a normal state, not
    an error - the graph should fall through to ticket fallback, not crash."""

    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = False
    service = RetrievalService(client=fake_qdrant, embedder=_embedder_returning([0.1]))

    chunks = service.retrieve("bất kỳ câu hỏi nào")

    assert chunks == []
    fake_qdrant.query_points.assert_not_called()


def test_retrieve_clamps_score_into_the_0_1_range_the_schema_requires() -> None:
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value = QueryResponse(points=[_scored_point("c1", 1.5)])
    service = RetrievalService(client=fake_qdrant, embedder=_embedder_returning([0.1]))

    chunks = service.retrieve("câu hỏi")

    assert chunks[0].score == 1.0


def test_retrieve_passes_explicit_limit_through_to_qdrant() -> None:
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value = QueryResponse(points=[])
    service = RetrievalService(client=fake_qdrant, embedder=_embedder_returning([0.1]))

    service.retrieve("câu hỏi", limit=3)

    _, kwargs = fake_qdrant.query_points.call_args
    assert kwargs["limit"] == 3
