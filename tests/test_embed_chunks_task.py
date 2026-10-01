import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.errors.error_codes import ErrorCode
from app.core.errors.provider_errors import EmbeddingProviderError
from app.core.registry.errors import NoAvailableCredentialError
from app.core.registry.model_registry import (
    CredentialConfig,
    ModelRegistryError,
    ModelRegistrySnapshot,
)
from app.worker.celery_app import celery_app
from app.worker.embedding_job_errors import IngestionJobFailedError
from app.worker.tasks.ingestion import embed_chunks

celery_app.conf.update(
    broker_url="memory://",
    result_backend="cache+memory://",
    task_always_eager=True,
    task_eager_propagates=True,
    task_store_eager_result=True,
)


def _chunk_payload(index: int) -> dict[str, object]:
    return {"chunk_index": index, "content": f"chunk content {index}", "region_type": "text"}


@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embed_chunks_runs_enrich_embed_upsert_in_order(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
) -> None:
    mock_embedder = mock_embedder_cls.return_value
    mock_embedder.embed_tracked = AsyncMock(return_value=[[0.1], [0.2], [0.3]])
    mock_enricher = mock_enricher_cls.return_value
    mock_enricher.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_client = MagicMock()
    mock_qdrant_store.get_client.return_value = mock_client

    result = embed_chunks.apply(
        args=(
            "doc-1",
            "docs/handbook.pdf",
            [_chunk_payload(0), _chunk_payload(1)],
            "CNTT",
            2,
        )
    ).get()

    assert result["percent"] == 100
    assert [r["status"] for r in result["results"]] == ["SUCCESS", "SUCCESS"]
    mock_qdrant_store.ensure_collection.assert_called_once_with(mock_client)
    assert mock_enricher.enrich_tracked.call_count == 2
    assert mock_embedder.embed_tracked.call_count == 2
    assert mock_qdrant_store.upsert_chunk.call_count == 2

    enrich_order = [
        call.args[0].chunk_index for call in mock_enricher.enrich_tracked.call_args_list
    ]
    assert enrich_order == [0, 1]

    point_ids = [call.kwargs["point_id"] for call in mock_qdrant_store.ChunkPoint.call_args_list]
    for point_id in point_ids:
        uuid.UUID(point_id)  # Qdrant requires an unsigned int or a UUID; raises if invalid
    assert len(set(point_ids)) == len(point_ids)
    chunk_ids = [call.kwargs["chunk_id"] for call in mock_qdrant_store.ChunkPoint.call_args_list]
    assert chunk_ids == ["doc-1:0", "doc-1:1"]


@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embed_chunks_reports_strictly_increasing_progress_to_100(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed_tracked = AsyncMock(return_value=[[0.1], [0.2], [0.3]])
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    seen_percents: list[int] = []
    task = embed_chunks
    original_update_state = task.update_state

    def _tracking_update_state(*, task_id: str, state: str, meta: dict[str, int]) -> None:
        seen_percents.append(meta["percent"])
        original_update_state(task_id=task_id, state=state, meta=meta)

    with patch.object(task, "update_state", side_effect=_tracking_update_state):
        result = task.apply(
            args=(
                "doc-1",
                "docs/handbook.pdf",
                [_chunk_payload(i) for i in range(4)],
                "CNTT",
                2,
            )
        ).get()

    assert seen_percents == sorted(seen_percents)
    assert seen_percents[-1] == 100
    assert result["percent"] == 100


@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embed_chunks_records_per_chunk_failure_without_aborting_batch(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed_tracked = AsyncMock(
        side_effect=[
            RuntimeError("embedding provider unavailable"),
            [[0.1], [0.2], [0.3]],
        ]
    )
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    result = embed_chunks.apply(
        args=(
            "doc-1",
            "docs/handbook.pdf",
            [_chunk_payload(0), _chunk_payload(1)],
            "CNTT",
            2,
        )
    ).get()

    assert result["results"][0]["status"] == "FAILED"
    assert result["results"][1]["status"] == "SUCCESS"
    assert mock_qdrant_store.upsert_chunk.call_count == 1
    assert result["failed_chunk_count"] == 1
    assert result["total_chunk_count"] == 2


@patch("app.worker.tasks.ingestion.publish_ingestion_event")
@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embed_chunks_publishes_completed_failure_when_a_chunk_failed(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    mock_publish: MagicMock,
) -> None:
    """A per-chunk failure never aborts the batch (Celery ends SUCCESS), but the
    terminal WS event must still say FAILURE - otherwise the web wizard shows a false
    "100% done" for a job where a chunk never actually got embedded."""

    mock_embedder_cls.return_value.embed_tracked = AsyncMock(
        side_effect=[RuntimeError("boom"), [[0.1], [0.2], [0.3]]]
    )
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    embed_chunks.apply(
        args=("doc-1", "docs/handbook.pdf", [_chunk_payload(0), _chunk_payload(1)], "CNTT", 2)
    ).get()

    completed_frames = [
        call.args[0] for call in mock_publish.call_args_list if call.args[0]["type"] == "completed"
    ]
    assert len(completed_frames) == 1
    assert completed_frames[0]["state"] == "FAILURE"
    assert completed_frames[0]["failed_chunk_count"] == 1
    assert completed_frames[0]["total_chunk_count"] == 2


@patch("app.worker.tasks.ingestion.publish_ingestion_event")
@patch("app.worker.tasks.ingestion.BackendJavaClient")
@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embed_chunks_aborts_and_raises_on_embedding_provider_error(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    mock_backend_client_cls: MagicMock,
    mock_publish: MagicMock,
) -> None:
    """A provider-level embedding failure (todo.md Task 13a) is not a per-chunk data problem -
    it must escape the loop entirely: no third chunk gets embedded, the one `completed`
    event published carries `state=FAILURE`, and the task itself ends up FAILED rather
    than SUCCESS."""

    credential = CredentialConfig(
        id="embed-cred",
        revision=1,
        source_type="CLOUD_API",
        provider="openai",
        model_name="text-embedding-3-small",
        api_base_url="https://api.openai.com/v1",
        priority=1,
        max_rpm=500,
        api_key="sk-test",
    )
    mock_embedder = mock_embedder_cls.return_value
    mock_embedder.embed_tracked = AsyncMock(
        side_effect=[
            [[0.1], [0.2], [0.3]],
            EmbeddingProviderError("provider auth failed", credential=credential),
        ]
    )
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()
    mock_backend_client_cls.return_value.report_health = AsyncMock(return_value={"applied": True})

    with (
        patch(
            "app.worker.tasks.ingestion.get_current_snapshot",
            return_value=ModelRegistrySnapshot(
                version=1, generated_at=None, purposes={}, embedding_index_identity=None
            ),
        ),
        pytest.raises(IngestionJobFailedError) as exc_info,
    ):
        embed_chunks.apply(
            args=(
                "doc-1",
                "docs/handbook.pdf",
                [_chunk_payload(0), _chunk_payload(1), _chunk_payload(2)],
                "CNTT",
                2,
            )
        )

    # Only 2 embed_tracked() calls happened - chunk 2 (index 2) was never reached.
    assert mock_embedder.embed_tracked.call_count == 2
    assert mock_qdrant_store.upsert_chunk.call_count == 1

    # One terminal "completed" event with state=FAILURE - not a separate "failed" type
    # (that type was silently dropped by the web wizard's schema; see celery_app.py).
    completed_frames = [
        call.args[0] for call in mock_publish.call_args_list if call.args[0]["type"] == "completed"
    ]
    assert len(completed_frames) == 1
    assert completed_frames[0]["state"] == "FAILURE"
    assert completed_frames[0]["message"] == exc_info.value.message
    assert isinstance(exc_info.value.__cause__, EmbeddingProviderError)

    mock_backend_client_cls.return_value.report_health.assert_awaited_once()
    health_kwargs = mock_backend_client_cls.return_value.report_health.call_args.kwargs
    assert health_kwargs["credential_id"] == "embed-cred"
    assert health_kwargs["error_type"] == "TRANSIENT"


@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embed_chunks_passes_structural_fields_through_to_chunk_point(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed_tracked = AsyncMock(return_value=[[0.1], [0.2], [0.3]])
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    structural_chunk = {
        "chunk_index": 0,
        "content": "Điều 5. Nội dung.",
        "region_type": "table",
        "source_type": "pdf",
        "block_index": 3,
        "heading_path": ["Chương 1", "Điều 5"],
        "page_start": 5,
        "page_end": 5,
        "source_locator": {
            "table_id": "table-3",
            "row_start": 1,
            "row_end": 2,
            "row_count": 2,
        },
        "column_names": ["Tên", "Điểm"],
        "has_header": True,
        "header_source": "inferred",
        "header_confidence": 0.6,
        "chunking_version": "2026-09-structural-v1",
        "structure_confidence": 0.3,
        "parse_warnings": ["garbled_text_raw_kept"],
    }

    embed_chunks.apply(args=("doc-1", "docs/handbook.pdf", [structural_chunk], "CNTT", 2)).get()

    point_kwargs = mock_qdrant_store.ChunkPoint.call_args.kwargs
    assert point_kwargs["source_type"] == "pdf"
    assert point_kwargs["block_index"] == 3
    assert point_kwargs["heading_path"] == ["Chương 1", "Điều 5"]
    assert point_kwargs["page_start"] == 5
    assert point_kwargs["page_end"] == 5
    assert point_kwargs["source_locator"]["table_id"] == "table-3"
    assert point_kwargs["column_names"] == ["Tên", "Điểm"]
    assert point_kwargs["has_header"] is True
    assert point_kwargs["header_source"] == "inferred"
    assert point_kwargs["chunking_version"] == "2026-09-structural-v1"
    assert point_kwargs["structure_confidence"] == 0.3
    assert point_kwargs["parse_warnings"] == ["garbled_text_raw_kept"]


@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embed_chunks_defaults_is_public_to_false_when_omitted(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed_tracked = AsyncMock(return_value=[[0.1], [0.2], [0.3]])
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    embed_chunks.apply(args=("doc-1", "docs/handbook.pdf", [_chunk_payload(0)], "CNTT", 2)).get()

    assert mock_qdrant_store.ChunkPoint.call_args.kwargs["is_public"] is False


@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embed_chunks_passes_is_public_true_through_to_chunk_point(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed_tracked = AsyncMock(return_value=[[0.1], [0.2], [0.3]])
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    embed_chunks.apply(
        args=("doc-1", "docs/handbook.pdf", [_chunk_payload(0)], "CNTT", 2, True)
    ).get()

    assert mock_qdrant_store.ChunkPoint.call_args.kwargs["is_public"] is True


@patch("app.worker.tasks.ingestion.publish_ingestion_event")
@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_embed_chunks_aborts_with_specific_reason_when_extraction_is_unusable(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    mock_publish: MagicMock,
) -> None:
    """No usable EXTRACTION credential fails every chunk the same way - the job must stop at
    the first one and say why (here: not configured), not report "N/M đoạn thất bại"."""

    try:
        raise ModelRegistryError("no ACTIVE EXTRACTION credential")
    except ModelRegistryError as cause:
        no_extraction = NoAvailableCredentialError("EXTRACTION")
        no_extraction.__cause__ = cause
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(side_effect=no_extraction)
    mock_qdrant_store.get_client.return_value = MagicMock()

    with pytest.raises(IngestionJobFailedError) as exc_info:
        embed_chunks.apply(
            args=("doc-1", "docs/handbook.pdf", [_chunk_payload(0), _chunk_payload(1)], "CNTT", 2)
        ).get()

    assert mock_enricher_cls.return_value.enrich_tracked.await_count == 1
    assert mock_embedder_cls.return_value.embed_tracked.call_count == 0
    completed_frames = [
        call.args[0] for call in mock_publish.call_args_list if call.args[0]["type"] == "completed"
    ]
    assert len(completed_frames) == 1
    assert completed_frames[0]["state"] == "FAILURE"
    assert completed_frames[0]["reason"] == "LLM_NOT_CONFIGURED"
    assert completed_frames[0]["error_code"] == ErrorCode.LLM_NOT_CONFIGURED.code
    assert "Extraction" in completed_frames[0]["message"]
    assert exc_info.value.message == completed_frames[0]["message"]


@patch("app.worker.tasks.ingestion.publish_ingestion_event")
@patch("app.worker.tasks.ingestion.qdrant_store")
@patch("app.worker.tasks.ingestion.MultiRepresentationEnricher")
@patch("app.worker.tasks.ingestion.build_embedder")
def test_partial_failure_message_names_the_first_reason(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
    mock_publish: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed_tracked = AsyncMock(
        side_effect=[KeyError("bug"), [[0.1], [0.2], [0.3]]]
    )
    mock_enricher_cls.return_value.enrich_tracked = AsyncMock(
        return_value=MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    embed_chunks.apply(
        args=("doc-1", "docs/handbook.pdf", [_chunk_payload(0), _chunk_payload(1)], "CNTT", 2)
    ).get()

    completed = next(
        call.args[0] for call in mock_publish.call_args_list if call.args[0]["type"] == "completed"
    )
    assert completed["message"].startswith("1/2 đoạn nạp liệu thất bại.")
    assert "đoạn #0" in completed["message"]
    assert "KeyError" in completed["message"]
