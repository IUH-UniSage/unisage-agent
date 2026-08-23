import pytest
from pydantic import ValidationError

from app.schemas.ingestion import (
    Chunk,
    ChunkingRequest,
    ChunkingStrategyName,
    EmbeddingRequest,
    PreviewRequest,
    RegionType,
)


@pytest.mark.parametrize("strategy", list(ChunkingStrategyName))
def test_chunking_request_accepts_all_five_strategies(strategy: ChunkingStrategyName) -> None:
    request = ChunkingRequest(
        document_id="doc-1", object_key="docs/handbook.pdf", strategy=strategy
    )

    assert request.strategy == strategy


def test_chunking_request_rejects_missing_document_id() -> None:
    with pytest.raises(ValidationError):
        ChunkingRequest.model_validate({"object_key": "docs/handbook.pdf", "strategy": "recursive"})


def test_chunking_request_rejects_unknown_strategy() -> None:
    with pytest.raises(ValidationError):
        ChunkingRequest.model_validate(
            {"object_key": "docs/handbook.pdf", "strategy": "not_a_strategy"}
        )


@pytest.mark.parametrize("region_type", list(RegionType))
def test_chunk_accepts_all_region_types(region_type: RegionType) -> None:
    chunk = Chunk(chunk_index=0, content="text", region_type=region_type)

    assert chunk.region_type == region_type


def test_chunk_rejects_unknown_region_type() -> None:
    with pytest.raises(ValidationError):
        Chunk.model_validate({"chunk_index": 0, "content": "text", "region_type": "not_a_region"})


def test_preview_request_rejects_empty_object_key() -> None:
    with pytest.raises(ValidationError):
        PreviewRequest(object_key="")


def test_embedding_request_requires_at_least_one_chunk() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest(document_id="doc-1", object_key="docs/handbook.pdf", chunks=[])
