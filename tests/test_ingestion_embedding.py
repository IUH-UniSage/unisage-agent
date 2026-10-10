import json
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app
from app.worker.celery_app import celery_app
from tests.fixtures.documents import make_pdf_bytes

celery_app.conf.update(
    broker_url="memory://",
    result_backend="cache+memory://",
    task_always_eager=True,
    task_eager_propagates=True,
    task_store_eager_result=True,
)

_EMBEDDING_PAYLOAD = {
    "document_id": "doc-1",
    "department_id": "CNTT",
    "access_level": 2,
    "object_key": "docs/handbook.pdf",
    "chunks": [{"chunk_index": 0, "content": "chunk content", "region_type": "text"}],
}

_TRUSTED_HEADERS = {
    "X-User-Department-Access": json.dumps([{"department_id": "CNTT", "access_level": 3}]),
    "X-User-Permissions": json.dumps(["DOCUMENT_ALL"]),
}


def _create_chunking_draft(
    client: TestClient,
    document_id: str,
    *,
    department_id: str = "CNTT",
    object_key: str = "docs/handbook.pdf",
    text: str = "A paragraph for chunking.",
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Chunk a tiny document (real `dispatch()`, real `validate_chunks()`)
    through the actual endpoint so the resulting draft has genuine,
    non-legacy structural metadata - what the canonical merge reads back.
    Returns the chunk list from the response."""

    with patch("app.api.v1.ingestion.minio_client.get_object_bytes") as mock_get_object_bytes:
        mock_get_object_bytes.return_value = make_pdf_bytes(text)
        response = client.post(
            "/api/v1/ingestion/chunking",
            json={
                "document_id": document_id,
                "department_id": department_id,
                "object_key": object_key,
                "strategy": "recursive",
                **({"params": params} if params else {}),
            },
            headers=_TRUSTED_HEADERS,
        )
    assert response.status_code == 200, response.text
    chunks: list[dict[str, Any]] = response.json()["data"]["chunks"]
    return chunks


def test_embedding_returns_400_when_trusted_headers_missing(client: TestClient) -> None:
    response = client.post("/api/v1/ingestion/embedding", json=_EMBEDDING_PAYLOAD)

    assert response.status_code == 400


def test_embedding_returns_403_when_missing_document_permission(client: TestClient) -> None:
    headers = {**_TRUSTED_HEADERS, "X-User-Permissions": json.dumps(["CHAT_MODEL_READ"])}

    response = client.post("/api/v1/ingestion/embedding", json=_EMBEDDING_PAYLOAD, headers=headers)

    assert response.status_code == 403


def test_embedding_returns_403_when_department_not_granted(client: TestClient) -> None:
    headers = {
        **_TRUSTED_HEADERS,
        "X-User-Department-Access": json.dumps(
            [{"department_id": "KHOA_KINH_TE", "access_level": 5}]
        ),
    }

    response = client.post("/api/v1/ingestion/embedding", json=_EMBEDDING_PAYLOAD, headers=headers)

    assert response.status_code == 403


def test_embedding_returns_403_when_access_level_exceeds_grant(client: TestClient) -> None:
    headers = {
        **_TRUSTED_HEADERS,
        "X-User-Department-Access": json.dumps([{"department_id": "CNTT", "access_level": 1}]),
    }

    response = client.post("/api/v1/ingestion/embedding", json=_EMBEDDING_PAYLOAD, headers=headers)

    assert response.status_code == 403


def test_embedding_returns_404_when_no_draft_exists_for_the_document(client: TestClient) -> None:
    """`/ingestion/embedding` reads the stored draft as canonical
    metadata - there is nothing to merge from if the document was never
    chunked (or the draft was never created)."""

    response = client.post(
        "/api/v1/ingestion/embedding",
        json={**_EMBEDDING_PAYLOAD, "document_id": "doc-never-chunked"},
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 404


@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embedding_returns_202_and_task_id_for_valid_request(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    client: TestClient,
) -> None:
    mock_embedder_cls.return_value.embed_tracked = AsyncMock(return_value=[[0.1], [0.2], [0.3]])
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()
    draft_chunks = _create_chunking_draft(client, "doc-embed-202")

    response = client.post(
        "/api/v1/ingestion/embedding",
        json={
            **_EMBEDDING_PAYLOAD,
            "document_id": "doc-embed-202",
            "chunks": draft_chunks,
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 202
    assert response.json()["data"]["task_id"]


@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_valid_request_dispatches_task_with_request_department_and_level(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    client: TestClient,
) -> None:
    mock_embedder_cls.return_value.embed_tracked = AsyncMock(return_value=[[0.1], [0.2], [0.3]])
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()
    draft_chunks = _create_chunking_draft(client, "doc-embed-level")

    client.post(
        "/api/v1/ingestion/embedding",
        json={
            **_EMBEDDING_PAYLOAD,
            "document_id": "doc-embed-level",
            "chunks": draft_chunks,
        },
        headers=_TRUSTED_HEADERS,
    )

    point_kwargs = mock_qdrant_store.ChunkPoint.call_args.kwargs
    assert point_kwargs["department"] == "CNTT"
    assert point_kwargs["access_level"] == 2


@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_successful_embedding_dispatch_marks_draft_as_embedding(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    client: TestClient,
) -> None:
    mock_embedder_cls.return_value.embed_tracked = AsyncMock(return_value=[[0.1], [0.2], [0.3]])
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    draft_chunks = _create_chunking_draft(client, "doc-embed-1")
    job_before = client.get("/api/v1/ingestion/jobs/doc-embed-1", headers=_TRUSTED_HEADERS)
    assert job_before.status_code == 200

    payload = {**_EMBEDDING_PAYLOAD, "document_id": "doc-embed-1", "chunks": draft_chunks}
    response = client.post(
        "/api/v1/ingestion/embedding",
        json=payload,
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 202
    task_id = response.json()["data"]["task_id"]

    job = client.get("/api/v1/ingestion/jobs/doc-embed-1", headers=_TRUSTED_HEADERS)
    assert job.status_code == 200
    body = job.json()["data"]
    assert body["current_step"] == "embedding"
    assert body["task_id"] == task_id
    # The job endpoint carries the live task state inline, so the client's
    # reconciliation sweep needs only this one authorized call.
    assert body["task_state"] == "SUCCESS"
    assert body["task_percent"] == 100


@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embedding_uses_canonical_metadata_and_only_client_content(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    client: TestClient,
) -> None:
    """A client tampering with structural
    fields (page_start/source_locator/...) must be ignored - only `content`
    from the request payload is applied, everything else comes from the
    stored draft."""

    mock_embedder_cls.return_value.embed_tracked = AsyncMock(return_value=[[0.1], [0.2], [0.3]])
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    draft_chunks = _create_chunking_draft(client, "doc-embed-tamper")
    tampered_chunks = [
        {
            **chunk,
            "content": "EDITED BY USER",
            "page_start": 9999,
            "source_locator": {"table_id": "forged-table"},
            "block_index": 9999,
        }
        for chunk in draft_chunks
    ]

    response = client.post(
        "/api/v1/ingestion/embedding",
        json={
            **_EMBEDDING_PAYLOAD,
            "document_id": "doc-embed-tamper",
            "chunks": tampered_chunks,
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 202
    embed_call_chunks = mock_embedder_cls.return_value.embed_tracked.call_args_list
    # The content actually embedded must reflect the client's edit...
    assert any("EDITED BY USER" in call.args[0][0] for call in embed_call_chunks)
    # ...but the structural metadata sent to Qdrant must be the canonical
    # one, not the client's forged values.
    point_kwargs = mock_qdrant_store.ChunkPoint.call_args.kwargs
    assert point_kwargs["block_index"] != 9999
    assert point_kwargs["source_locator"] != {"table_id": "forged-table"}


def test_embedding_rejects_department_id_mismatch_with_draft(client: TestClient) -> None:
    _create_chunking_draft(client, "doc-embed-mismatch-dept")

    response = client.post(
        "/api/v1/ingestion/embedding",
        json={
            **_EMBEDDING_PAYLOAD,
            "document_id": "doc-embed-mismatch-dept",
            "department_id": "KHOA_KINH_TE",
        },
        headers={
            **_TRUSTED_HEADERS,
            "X-User-Department-Access": json.dumps(
                [
                    {"department_id": "CNTT", "access_level": 3},
                    {"department_id": "KHOA_KINH_TE", "access_level": 3},
                ]
            ),
        },
    )

    assert response.status_code == 400


def test_embedding_rejects_object_key_mismatch_with_draft(client: TestClient) -> None:
    draft_chunks = _create_chunking_draft(client, "doc-embed-mismatch-key")

    response = client.post(
        "/api/v1/ingestion/embedding",
        json={
            **_EMBEDDING_PAYLOAD,
            "document_id": "doc-embed-mismatch-key",
            "object_key": "docs/some-other-file.pdf",
            "chunks": draft_chunks,
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 400


def test_embedding_rejects_chunk_count_mismatch(client: TestClient) -> None:
    draft_chunks = _create_chunking_draft(client, "doc-embed-count-mismatch")

    response = client.post(
        "/api/v1/ingestion/embedding",
        json={
            **_EMBEDDING_PAYLOAD,
            "document_id": "doc-embed-count-mismatch",
            "chunks": [*draft_chunks, {"chunk_index": 999, "content": "x", "region_type": "text"}],
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 400


@patch("app.api.v1.ingestion.embed_chunks")
def test_embedding_accepts_a_subset_when_the_user_deleted_chunks(
    mock_embed_chunks: MagicMock, client: TestClient
) -> None:
    """The wizard lets the user delete chunks before embedding: the request
    then carries only the kept ones, and only those are dispatched."""

    mock_embed_chunks.delay.return_value = MagicMock(id="task-subset")
    draft_chunks = _create_chunking_draft(
        client,
        "doc-embed-subset",
        text=" ".join(["First paragraph here.", "Second paragraph here.", "Third paragraph here."]),
        params={"chunk_size": 30, "overlap": 0},
    )
    assert len(draft_chunks) >= 3
    kept = [draft_chunks[0], draft_chunks[-1]]

    response = client.post(
        "/api/v1/ingestion/embedding",
        json={**_EMBEDDING_PAYLOAD, "document_id": "doc-embed-subset", "chunks": kept},
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 202, response.text
    dispatched = mock_embed_chunks.delay.call_args.args[2]
    assert [chunk["chunk_index"] for chunk in dispatched] == [
        kept[0]["chunk_index"],
        kept[1]["chunk_index"],
    ]


@patch("app.api.v1.ingestion.embed_chunks")
def test_embedding_dispatch_forwards_is_public_defaulting_to_false(
    mock_embed_chunks: MagicMock, client: TestClient
) -> None:
    mock_embed_chunks.delay.return_value = MagicMock(id="task-is-public")
    draft_chunks = _create_chunking_draft(client, "doc-embed-is-public")

    client.post(
        "/api/v1/ingestion/embedding",
        json={**_EMBEDDING_PAYLOAD, "document_id": "doc-embed-is-public", "chunks": draft_chunks},
        headers=_TRUSTED_HEADERS,
    )

    # (document_id, object_key, chunks, department_id, access_level, is_public)
    assert mock_embed_chunks.delay.call_args.args[5] is False


def test_embedding_rejects_duplicate_chunk_index(client: TestClient) -> None:
    draft_chunks = _create_chunking_draft(client, "doc-embed-dup-index")
    duplicated = [draft_chunks[0], draft_chunks[0]]

    response = client.post(
        "/api/v1/ingestion/embedding",
        json={
            **_EMBEDDING_PAYLOAD,
            "document_id": "doc-embed-dup-index",
            "chunks": duplicated,
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 400


def test_embedding_rejects_chunk_index_set_mismatch(client: TestClient) -> None:
    draft_chunks = _create_chunking_draft(client, "doc-embed-index-mismatch")
    swapped = [{**chunk, "chunk_index": chunk["chunk_index"] + 100} for chunk in draft_chunks]

    response = client.post(
        "/api/v1/ingestion/embedding",
        json={
            **_EMBEDDING_PAYLOAD,
            "document_id": "doc-embed-index-mismatch",
            "chunks": swapped,
        },
        headers=_TRUSTED_HEADERS,
    )

    assert response.status_code == 400


def test_embedding_rejects_legacy_draft_with_409(client: TestClient) -> None:
    """[v5] A draft created before structural metadata existed
    (`chunking_version == "legacy"`) can never be embedded - it must be
    re-chunked first. `/ingestion/chunking` always produces non-legacy
    chunks now, so a legacy draft is simulated here by stubbing `get_draft`
    to return one directly - this is exactly the shape a pre-migration
    Postgres row deserializes into (see the `chunking_version` default)."""

    from app.database.models import DocumentProcessStep
    from app.database.repositories.ingestion_job import DraftDTO
    from app.schemas.ingestion import Chunk, RegionType

    legacy_draft = DraftDTO(
        object_key="docs/handbook.pdf",
        department_id="CNTT",
        current_step=DocumentProcessStep.CHUNKED,
        chunking_strategy="recursive",
        chunking_params={},
        chunks=[Chunk(chunk_index=0, content="legacy content", region_type=RegionType.TEXT)],
        celery_task_id=None,
    )

    with patch("app.api.v1.ingestion.get_draft", return_value=legacy_draft):
        response = client.post(
            "/api/v1/ingestion/embedding",
            json={**_EMBEDDING_PAYLOAD, "document_id": "doc-embed-legacy"},
            headers=_TRUSTED_HEADERS,
        )

    assert response.status_code == 409


def test_embedding_rejects_a_draft_chunked_under_an_older_version_with_409(
    client: TestClient,
) -> None:
    """A structurally complete draft produced by the PREVIOUS chunking version
    is refused too (not only `"legacy"`), so a table chunked before pages were
    merged is re-chunked rather than embedded; the current version is accepted."""

    from app.core.config import settings
    from app.database.models import DocumentProcessStep
    from app.database.repositories.ingestion_job import DraftDTO
    from app.schemas.ingestion import Chunk, RegionType, SourceType

    def draft_with(version: str) -> DraftDTO:
        return DraftDTO(
            object_key="docs/handbook.pdf",
            department_id="CNTT",
            current_step=DocumentProcessStep.CHUNKED,
            chunking_strategy="recursive",
            chunking_params={},
            chunks=[
                Chunk(
                    chunk_index=0,
                    content="content",
                    region_type=RegionType.TEXT,
                    source_type=SourceType.PDF,
                    block_index=0,
                    chunking_version=version,
                )
            ],
            celery_task_id=None,
        )

    with patch("app.api.v1.ingestion.get_draft", return_value=draft_with("2026-09-structural-v1")):
        outdated = client.post(
            "/api/v1/ingestion/embedding",
            json={**_EMBEDDING_PAYLOAD, "document_id": "doc-embed-old"},
            headers=_TRUSTED_HEADERS,
        )

    assert settings.INGEST_CHUNKING_VERSION != "2026-09-structural-v1"
    assert outdated.status_code == 409


def _canned_stream(
    *frames: dict[str, Any],
) -> Callable[[], AsyncIterator[dict[str, Any]]]:
    async def _stream() -> AsyncIterator[dict[str, Any]]:
        for frame in frames:
            yield frame

    return _stream


def test_events_ws_only_forwards_frames_for_the_callers_departments(client: TestClient) -> None:
    own = {"type": "completed", "document_id": "doc-a", "department_id": "CNTT", "state": "SUCCESS"}
    other = {
        "type": "completed",
        "document_id": "doc-b",
        "department_id": "KHOA_KT",
        "state": "SUCCESS",
    }

    with patch("app.api.v1.ingestion.ingestion_event_stream", _canned_stream(other, own, other)):
        with client.websocket_connect("/api/v1/ingestion/events", headers=_TRUSTED_HEADERS) as ws:
            received = ws.receive_json()

    assert received == own


def test_events_ws_rejects_a_connection_without_the_internal_secret() -> None:
    bare_client = TestClient(app)

    with pytest.raises(WebSocketDisconnect):
        with bare_client.websocket_connect(
            "/api/v1/ingestion/events", headers=_TRUSTED_HEADERS
        ) as ws:
            ws.receive_json()
