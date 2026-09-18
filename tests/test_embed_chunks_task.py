import uuid
from unittest.mock import MagicMock, patch

from app.worker.celery_app import celery_app, embed_chunks

celery_app.conf.update(
    broker_url="memory://",
    result_backend="cache+memory://",
    task_always_eager=True,
    task_eager_propagates=True,
    task_store_eager_result=True,
)


def _chunk_payload(index: int) -> dict[str, object]:
    return {"chunk_index": index, "content": f"chunk content {index}", "region_type": "text"}


@patch("app.worker.celery_app.qdrant_store")
@patch("app.worker.celery_app.MultiRepresentationEnricher")
@patch("app.worker.celery_app.OpenAIEmbedder")
def test_embed_chunks_runs_enrich_embed_upsert_in_order(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
) -> None:
    mock_embedder = mock_embedder_cls.return_value
    mock_embedder.embed.return_value = [[0.1], [0.2], [0.3]]
    mock_enricher = mock_enricher_cls.return_value
    mock_enricher.enrich.return_value = MagicMock(summary="a summary", questions=["Q1?", "Q2?"])
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
    assert mock_enricher.enrich.call_count == 2
    assert mock_embedder.embed.call_count == 2
    assert mock_qdrant_store.upsert_chunk.call_count == 2

    enrich_order = [call.args[0].chunk_index for call in mock_enricher.enrich.call_args_list]
    assert enrich_order == [0, 1]

    point_ids = [call.kwargs["point_id"] for call in mock_qdrant_store.ChunkPoint.call_args_list]
    for point_id in point_ids:
        uuid.UUID(point_id)  # Qdrant requires an unsigned int or a UUID; raises if invalid
    assert len(set(point_ids)) == len(point_ids)
    chunk_ids = [call.kwargs["chunk_id"] for call in mock_qdrant_store.ChunkPoint.call_args_list]
    assert chunk_ids == ["doc-1:0", "doc-1:1"]


@patch("app.worker.celery_app.qdrant_store")
@patch("app.worker.celery_app.MultiRepresentationEnricher")
@patch("app.worker.celery_app.OpenAIEmbedder")
def test_embed_chunks_reports_strictly_increasing_progress_to_100(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed.return_value = [[0.1], [0.2], [0.3]]
    mock_enricher_cls.return_value.enrich.return_value = MagicMock(
        summary="a summary", questions=["Q1?", "Q2?"]
    )
    mock_qdrant_store.get_client.return_value = MagicMock()

    seen_percents: list[int] = []
    task = embed_chunks
    original_update_state = task.update_state

    def _tracking_update_state(*, state: str, meta: dict[str, int]) -> None:
        seen_percents.append(meta["percent"])
        original_update_state(state=state, meta=meta)

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


@patch("app.worker.celery_app.qdrant_store")
@patch("app.worker.celery_app.MultiRepresentationEnricher")
@patch("app.worker.celery_app.OpenAIEmbedder")
def test_embed_chunks_records_per_chunk_failure_without_aborting_batch(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed.side_effect = [
        RuntimeError("embedding provider unavailable"),
        [[0.1], [0.2], [0.3]],
    ]
    mock_enricher_cls.return_value.enrich.return_value = MagicMock(
        summary="a summary", questions=["Q1?", "Q2?"]
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


@patch("app.worker.celery_app.qdrant_store")
@patch("app.worker.celery_app.MultiRepresentationEnricher")
@patch("app.worker.celery_app.OpenAIEmbedder")
def test_embed_chunks_passes_structural_fields_through_to_chunk_point(
    mock_embedder_cls: MagicMock,
    mock_enricher_cls: MagicMock,
    mock_qdrant_store: MagicMock,
) -> None:
    mock_embedder_cls.return_value.embed.return_value = [[0.1], [0.2], [0.3]]
    mock_enricher_cls.return_value.enrich.return_value = MagicMock(
        summary="a summary", questions=["Q1?", "Q2?"]
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
    }

    embed_chunks.apply(
        args=("doc-1", "docs/handbook.pdf", [structural_chunk], "CNTT", 2)
    ).get()

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
