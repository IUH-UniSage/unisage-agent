from unittest.mock import MagicMock

from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    PayloadSchemaType,
    QueryResponse,
    ScoredPoint,
)

from app.core.config import settings
from app.rag.vectorstore.qdrant_store import (
    ChunkPoint,
    build_access_filter,
    collection_has_points,
    ensure_collection,
    get_collection_dimension,
    search_chunks,
    upsert_chunk,
)
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry


def test_ensure_collection_creates_when_absent() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    ensure_collection(client)

    client.create_collection.assert_called_once()
    _, kwargs = client.create_collection.call_args
    assert kwargs["collection_name"] == settings.QDRANT_COLLECTION
    assert set(kwargs["vectors_config"].keys()) == {
        "content_vector",
        "summary_vector",
        "questions_vector",
    }


def test_ensure_collection_is_idempotent_when_already_present() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True

    ensure_collection(client)

    client.create_collection.assert_not_called()


def test_get_collection_dimension_returns_none_when_collection_missing() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    assert get_collection_dimension(client) is None


def test_get_collection_dimension_reads_content_vector_size() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    content_params = MagicMock(size=1536)
    client.get_collection.return_value.config.params.vectors = {
        "content_vector": content_params,
        "summary_vector": MagicMock(size=1536),
        "questions_vector": MagicMock(size=1536),
    }

    assert get_collection_dimension(client) == 1536


def test_collection_has_points_false_when_collection_missing() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    assert collection_has_points(client) is False


def test_collection_has_points_reflects_points_count() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value.points_count = 0

    assert collection_has_points(client) is False

    client.get_collection.return_value.points_count = 5

    assert collection_has_points(client) is True


def test_upsert_chunk_builds_expected_payload_and_vector_shape() -> None:
    client = MagicMock()
    point = ChunkPoint(
        point_id="doc-1:0",
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        chunk_id="doc-1:0",
        content="chunk text",
        summary="a summary",
        questions=["Q1?", "Q2?"],
        department="CNTT",
        access_level=2,
        category="HOC_VU",
        region_type="text",
        content_vector=[0.1, 0.2],
        summary_vector=[0.3, 0.4],
        questions_vector=[0.5, 0.6],
    )

    upsert_chunk(client, point)

    client.upsert.assert_called_once()
    _, kwargs = client.upsert.call_args
    assert kwargs["collection_name"] == settings.QDRANT_COLLECTION
    (upserted_point,) = kwargs["points"]
    assert upserted_point.id == "doc-1:0"
    assert upserted_point.vector == {
        "content_vector": [0.1, 0.2],
        "summary_vector": [0.3, 0.4],
        "questions_vector": [0.5, 0.6],
    }
    assert upserted_point.payload == {
        "document_id": "doc-1",
        "object_key": "docs/handbook.pdf",
        "chunk_id": "doc-1:0",
        "content": "chunk text",
        "summary": "a summary",
        "questions": ["Q1?", "Q2?"],
        "department": "CNTT",
        "access_level": 2,
        "is_public": False,
        "category": "HOC_VU",
        "region_type": "text",
        "source_type": None,
        "block_index": None,
        "heading_path": [],
        "page_start": None,
        "page_end": None,
        "source_locator": None,
        "column_names": None,
        "has_header": False,
        "header_source": None,
        "chunking_version": "legacy",
        "embedding_identity_key": None,
        "structure_confidence": None,
        "parse_warnings": [],
    }


def test_upsert_chunk_carries_embedding_identity_key_when_set() -> None:
    client = MagicMock()
    point = ChunkPoint(
        point_id="doc-1:0",
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        chunk_id="doc-1:0",
        content="chunk text",
        summary="a summary",
        questions=["Q1?", "Q2?"],
        department="CNTT",
        access_level=2,
        category="HOC_VU",
        region_type="text",
        content_vector=[0.1, 0.2],
        summary_vector=[0.3, 0.4],
        questions_vector=[0.5, 0.6],
        embedding_identity_key="abc123",
    )

    upsert_chunk(client, point)

    _, kwargs = client.upsert.call_args
    (upserted_point,) = kwargs["points"]
    assert upserted_point.payload["embedding_identity_key"] == "abc123"


def test_upsert_chunk_carries_structural_metadata_fields_when_set() -> None:
    client = MagicMock()
    point = ChunkPoint(
        point_id="doc-1:0",
        document_id="doc-1",
        object_key="docs/handbook.pdf",
        chunk_id="doc-1:0",
        content="chunk text",
        summary="a summary",
        questions=["Q1?", "Q2?"],
        department="CNTT",
        access_level=2,
        category="HOC_VU",
        region_type="table",
        content_vector=[0.1],
        summary_vector=[0.3],
        questions_vector=[0.5],
        source_type="pdf",
        block_index=2,
        heading_path=["Chương 1", "Điều 5"],
        page_start=5,
        page_end=6,
        source_locator={"table_id": "table-2", "row_start": 1, "row_end": 3, "row_count": 3},
        column_names=["Tên", "Điểm"],
        has_header=True,
        header_source="inferred",
        chunking_version="2026-09-structural-v1",
        structure_confidence=0.42,
        parse_warnings=["padded_cells"],
    )

    upsert_chunk(client, point)

    (upserted_point,) = client.upsert.call_args.kwargs["points"]
    assert upserted_point.payload["source_type"] == "pdf"
    assert upserted_point.payload["block_index"] == 2
    assert upserted_point.payload["heading_path"] == ["Chương 1", "Điều 5"]
    assert upserted_point.payload["page_start"] == 5
    assert upserted_point.payload["page_end"] == 6
    assert upserted_point.payload["source_locator"]["table_id"] == "table-2"
    assert upserted_point.payload["column_names"] == ["Tên", "Điểm"]
    assert upserted_point.payload["has_header"] is True
    assert upserted_point.payload["header_source"] == "inferred"
    assert upserted_point.payload["chunking_version"] == "2026-09-structural-v1"
    assert upserted_point.payload["structure_confidence"] == 0.42
    assert upserted_point.payload["parse_warnings"] == ["padded_cells"]


def test_search_chunks_returns_empty_list_when_collection_missing() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    points = search_chunks(client, query_vector=[0.1, 0.2], limit=5)

    assert points == []
    client.query_points.assert_not_called()


def test_search_chunks_queries_all_three_named_vectors_with_payload_attached() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    scored_point = ScoredPoint(id="c1", version=0, score=0.9, payload={"chunk_id": "c1"})
    client.query_points.return_value = QueryResponse(points=[scored_point])

    points = search_chunks(client, query_vector=[0.1, 0.2], limit=5)

    assert points == [scored_point]
    assert client.query_points.call_count == 3
    used_vectors = {call.kwargs["using"] for call in client.query_points.call_args_list}
    assert used_vectors == {"content_vector", "summary_vector", "questions_vector"}
    for call in client.query_points.call_args_list:
        assert call.kwargs["collection_name"] == settings.QDRANT_COLLECTION
        assert call.kwargs["query"] == [0.1, 0.2]
        assert call.kwargs["with_payload"] is True


def test_search_chunks_keeps_the_best_score_per_chunk_across_vectors() -> None:
    """A chunk that scores low on content_vector but high on questions_vector
    (its precomputed questions closely match the user's query) must surface
    with the HIGHER score, not be lost or duplicated."""

    client = MagicMock()
    client.collection_exists.return_value = True
    low_score = ScoredPoint(id="c1", version=0, score=0.4, payload={"chunk_id": "c1"})
    high_score = ScoredPoint(id="c1", version=0, score=0.85, payload={"chunk_id": "c1"})
    other_chunk = ScoredPoint(id="c2", version=0, score=0.5, payload={"chunk_id": "c2"})

    client.query_points.side_effect = [
        QueryResponse(points=[low_score]),  # content_vector
        QueryResponse(points=[other_chunk]),  # summary_vector
        QueryResponse(points=[high_score]),  # questions_vector
    ]

    points = search_chunks(client, query_vector=[0.1, 0.2], limit=5)

    assert len(points) == 2
    assert points[0].id == "c1"
    assert points[0].score == 0.85  # kept the higher of the two c1 scores
    assert points[1].id == "c2"


def test_search_chunks_respects_limit_after_merging() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.query_points.side_effect = [
        QueryResponse(
            points=[
                ScoredPoint(id=f"c{i}", version=0, score=1.0 - i * 0.1, payload={})
                for i in range(3)
            ]
        ),
        QueryResponse(points=[]),
        QueryResponse(points=[]),
    ]

    points = search_chunks(client, query_vector=[0.1], limit=2)

    assert len(points) == 2
    assert [p.id for p in points] == ["c0", "c1"]


def test_ensure_collection_creates_department_access_level_and_is_public_payload_index() -> None:
    client = MagicMock()
    client.collection_exists.return_value = False

    ensure_collection(client)

    assert client.create_payload_index.call_count == 3
    calls = {
        call.kwargs["field_name"]: call.kwargs["field_schema"]
        for call in client.create_payload_index.call_args_list
    }
    assert calls["department"] == PayloadSchemaType.KEYWORD
    assert calls["access_level"] == PayloadSchemaType.INTEGER
    assert calls["is_public"] == PayloadSchemaType.BOOL
    for call in client.create_payload_index.call_args_list:
        assert call.kwargs["collection_name"] == settings.QDRANT_COLLECTION


def test_ensure_collection_skips_payload_index_when_already_present() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True

    ensure_collection(client)

    client.create_payload_index.assert_not_called()


def _condition_dump(condition: object) -> dict[str, object]:
    """Recursively serialize a qdrant-client condition/filter model to a
    plain dict, so tests can assert on structure without depending on the
    library's exact model classes."""

    if isinstance(condition, Filter | FieldCondition):
        return condition.model_dump(exclude_none=True)
    raise TypeError(f"unexpected condition type: {type(condition)!r}")


def test_build_access_filter_for_a_guest_only_allows_public_chunks() -> None:
    guest = AcademicSecurityContext()

    access_filter = build_access_filter(guest)

    assert isinstance(access_filter.should, list)
    assert len(access_filter.should) == 1
    condition = _condition_dump(access_filter.should[0])
    assert condition == {"key": "is_public", "match": {"value": True}}


def test_build_access_filter_for_one_department_allows_public_and_that_department() -> None:
    security = AcademicSecurityContext(
        department_access=[DepartmentAccessEntry(department_id="KHOA_CNTT", access_level=2)]
    )

    access_filter = build_access_filter(security)

    assert isinstance(access_filter.should, list)
    assert len(access_filter.should) == 2
    department_clause = _condition_dump(access_filter.should[1])
    assert department_clause == {
        "must": [
            {"key": "department", "match": {"value": "KHOA_CNTT"}},
            {"key": "access_level", "range": {"lte": 2}},
        ]
    }


def test_build_access_filter_for_multiple_departments_uses_each_departments_own_level() -> None:
    security = AcademicSecurityContext(
        department_access=[
            DepartmentAccessEntry(department_id="KHOA_CNTT", access_level=2),
            DepartmentAccessEntry(department_id="PHONG_DAOTAO", access_level=1),
        ]
    )

    access_filter = build_access_filter(security)

    assert isinstance(access_filter.should, list)
    assert len(access_filter.should) == 3
    dumped = [_condition_dump(c) for c in access_filter.should[1:]]
    assert {
        "must": [
            {"key": "department", "match": {"value": "KHOA_CNTT"}},
            {"key": "access_level", "range": {"lte": 2}},
        ]
    } in dumped
    assert {
        "must": [
            {"key": "department", "match": {"value": "PHONG_DAOTAO"}},
            {"key": "access_level", "range": {"lte": 1}},
        ]
    } in dumped


def test_build_access_filter_wildcard_grants_its_level_across_every_department() -> None:
    security = AcademicSecurityContext(
        department_access=[DepartmentAccessEntry(department_id="*", access_level=2)]
    )

    access_filter = build_access_filter(security)

    assert isinstance(access_filter.should, list)
    assert len(access_filter.should) == 2
    wildcard_clause = _condition_dump(access_filter.should[1])
    assert wildcard_clause == {"key": "access_level", "range": {"lte": 2}}


def test_build_access_filter_mixes_wildcard_with_a_department_specific_grant() -> None:
    """A wildcard at a lower level plus a specific department at a higher
    level: the department gets its own higher ceiling, every other
    department is capped at the wildcard's lower one."""

    security = AcademicSecurityContext(
        department_access=[
            DepartmentAccessEntry(department_id="*", access_level=1),
            DepartmentAccessEntry(department_id="KHOA_CNTT", access_level=3),
        ]
    )

    access_filter = build_access_filter(security)

    assert isinstance(access_filter.should, list)
    dumped = [_condition_dump(c) for c in access_filter.should[1:]]
    assert {"key": "access_level", "range": {"lte": 1}} in dumped
    assert {
        "must": [
            {"key": "department", "match": {"value": "KHOA_CNTT"}},
            {"key": "access_level", "range": {"lte": 3}},
        ]
    } in dumped


def test_build_access_filter_public_clause_never_references_department_or_level() -> None:
    """Regression guard: the public clause is a bare `is_public == True`
    condition, never a `Filter(must=[...])` combining it with `department`
    or `access_level` - a public chunk must be visible regardless of
    which department it belongs to or what access_level it carries."""

    security = AcademicSecurityContext(
        department_access=[DepartmentAccessEntry(department_id="KHOA_CNTT", access_level=2)]
    )

    access_filter = build_access_filter(security)

    assert isinstance(access_filter.should, list)
    public_clause = _condition_dump(access_filter.should[0])
    assert public_clause == {"key": "is_public", "match": {"value": True}}


def test_search_chunks_passes_query_filter_to_every_named_vector_query() -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.query_points.return_value = QueryResponse(points=[])
    access_filter = build_access_filter(AcademicSecurityContext())

    search_chunks(client, query_vector=[0.1], limit=5, query_filter=access_filter)

    assert client.query_points.call_count == 3
    for call in client.query_points.call_args_list:
        assert call.kwargs["query_filter"] is access_filter
