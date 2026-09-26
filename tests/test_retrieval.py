"""RetrievalService - embeds the query and searches Qdrant's
`content_vector` for ingested chunks. Mocked Qdrant client + embedder, no
live network anywhere in this file (matches the project's convention for
`BackendJavaClient`/`httpx.MockTransport`)."""

from unittest.mock import MagicMock

from qdrant_client.http.models import QueryResponse, ScoredPoint

from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.retrieval.service import RetrievalService
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry


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
    return OpenAIEmbedder(model="text-embedding-3-small", client=fake_openai_client)


def test_retrieve_embeds_query_and_maps_qdrant_points_to_retrieved_chunks() -> None:
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value = QueryResponse(points=[_scored_point("c1", 0.83)])
    service = RetrievalService(client=fake_qdrant, embedder=_embedder_returning([0.1, 0.2]))

    chunks = service.retrieve("học phí học kỳ này bao nhiêu", security=AcademicSecurityContext())

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

    chunks = service.retrieve("bất kỳ câu hỏi nào", security=AcademicSecurityContext())

    assert chunks == []
    fake_qdrant.query_points.assert_not_called()


def test_retrieve_clamps_score_into_the_0_1_range_the_schema_requires() -> None:
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value = QueryResponse(points=[_scored_point("c1", 1.5)])
    service = RetrievalService(client=fake_qdrant, embedder=_embedder_returning([0.1]))

    chunks = service.retrieve("câu hỏi", security=AcademicSecurityContext())

    assert chunks[0].score == 1.0


def test_retrieve_maps_structural_metadata_including_source_locator() -> None:
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    point = ScoredPoint(
        id="c1",
        version=0,
        score=0.9,
        payload={
            "chunk_id": "c1",
            "content": "Điều 5...",
            "object_key": "docs/handbook.pdf",
            "document_id": "doc-1",
            "department": "CNTT",
            "access_level": 1,
            "region_type": "table",
            "source_type": "pdf",
            "heading_path": ["Chương 1", "Điều 5"],
            "page_start": 5,
            "page_end": 6,
            "source_locator": {
                "table_id": "table-2",
                "row_start": 1,
                "row_end": 3,
                "row_count": 3,
            },
        },
    )
    fake_qdrant.query_points.return_value = QueryResponse(points=[point])
    service = RetrievalService(client=fake_qdrant, embedder=_embedder_returning([0.1]))

    chunks = service.retrieve("nội dung điều 5", security=AcademicSecurityContext())

    assert chunks[0].source_type == "pdf"
    assert chunks[0].heading_path == ["Chương 1", "Điều 5"]
    assert chunks[0].page_start == 5
    assert chunks[0].page_end == 6
    assert chunks[0].source_locator is not None
    assert chunks[0].source_locator.table_id == "table-2"
    assert chunks[0].source_locator.row_count == 3


def test_retrieve_defaults_structural_metadata_for_a_legacy_point() -> None:
    """A legacy point without the structural payload keys still parses."""

    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value = QueryResponse(points=[_scored_point("c1", 0.5)])
    service = RetrievalService(client=fake_qdrant, embedder=_embedder_returning([0.1]))

    chunks = service.retrieve("câu hỏi", security=AcademicSecurityContext())

    assert chunks[0].heading_path == []
    assert chunks[0].page_start is None
    assert chunks[0].source_locator is None


def test_retrieve_passes_explicit_limit_through_to_qdrant() -> None:
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value = QueryResponse(points=[])
    service = RetrievalService(client=fake_qdrant, embedder=_embedder_returning([0.1]))

    service.retrieve("câu hỏi", security=AcademicSecurityContext(), limit=3)

    _, kwargs = fake_qdrant.query_points.call_args
    assert kwargs["limit"] == 3


def test_retrieve_applies_the_callers_access_filter_to_every_vector_query() -> None:
    fake_qdrant = MagicMock()
    fake_qdrant.collection_exists.return_value = True
    fake_qdrant.query_points.return_value = QueryResponse(points=[])
    service = RetrievalService(client=fake_qdrant, embedder=_embedder_returning([0.1]))
    security = AcademicSecurityContext(
        department_access=[DepartmentAccessEntry(department_id="KHOA_CNTT", access_level=2)]
    )

    service.retrieve("câu hỏi", security=security)

    assert fake_qdrant.query_points.call_count == 3
    for call in fake_qdrant.query_points.call_args_list:
        query_filter = call.kwargs["query_filter"]
        assert isinstance(query_filter.should, list)
        assert len(query_filter.should) == 2  # public (access_level=0) + KHOA_CNTT grant


def test_retrieve_never_lets_confirmed_metadata_widen_the_access_filter() -> None:
    """Regression guard: `RetrievalService.retrieve` takes no
    `confirmed_metadata` parameter at all - a student's self-declared,
    unverified attribute can never reach the Qdrant permission filter,
    only `security.department_access` (JWT-verified) can."""

    import inspect

    signature = inspect.signature(RetrievalService.retrieve)
    assert "confirmed_metadata" not in signature.parameters
