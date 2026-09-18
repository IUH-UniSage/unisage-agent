import pytest
from pydantic import ValidationError

from app.schemas.ingestion import (
    Chunk,
    ChunkingRequest,
    ChunkingStrategyName,
    EmbeddingRequest,
    HeaderSource,
    PreviewRequest,
    RegionType,
    SourceLocator,
    SourceType,
)


@pytest.mark.parametrize("strategy", list(ChunkingStrategyName))
def test_chunking_request_accepts_all_five_strategies(strategy: ChunkingStrategyName) -> None:
    request = ChunkingRequest(
        document_id="doc-1",
        department_id="CNTT",
        object_key="docs/handbook.pdf",
        strategy=strategy,
    )

    assert request.strategy == strategy


def test_chunking_request_rejects_missing_document_id() -> None:
    with pytest.raises(ValidationError):
        ChunkingRequest.model_validate(
            {"department_id": "CNTT", "object_key": "docs/handbook.pdf", "strategy": "recursive"}
        )


def test_chunking_request_rejects_missing_department_id() -> None:
    with pytest.raises(ValidationError):
        ChunkingRequest.model_validate(
            {"document_id": "doc-1", "object_key": "docs/handbook.pdf", "strategy": "recursive"}
        )


def test_chunking_request_rejects_unknown_strategy() -> None:
    with pytest.raises(ValidationError):
        ChunkingRequest.model_validate(
            {
                "department_id": "CNTT",
                "object_key": "docs/handbook.pdf",
                "strategy": "not_a_strategy",
            }
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
        PreviewRequest(department_id="CNTT", object_key="")


def test_preview_request_rejects_missing_department_id() -> None:
    with pytest.raises(ValidationError):
        PreviewRequest.model_validate({"object_key": "docs/handbook.pdf"})


def test_embedding_request_requires_at_least_one_chunk() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest(
            document_id="doc-1",
            department_id="CNTT",
            access_level=1,
            object_key="docs/handbook.pdf",
            chunks=[],
        )


def test_embedding_request_rejects_missing_department_id() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest.model_validate(
            {
                "document_id": "doc-1",
                "access_level": 1,
                "object_key": "docs/handbook.pdf",
                "chunks": [{"chunk_index": 0, "content": "text", "region_type": "text"}],
            }
        )


def test_embedding_request_rejects_missing_access_level() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest.model_validate(
            {
                "document_id": "doc-1",
                "department_id": "CNTT",
                "object_key": "docs/handbook.pdf",
                "chunks": [{"chunk_index": 0, "content": "text", "region_type": "text"}],
            }
        )


def test_embedding_request_rejects_negative_access_level() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest.model_validate(
            {
                "document_id": "doc-1",
                "department_id": "CNTT",
                "access_level": -1,
                "object_key": "docs/handbook.pdf",
                "chunks": [{"chunk_index": 0, "content": "text", "region_type": "text"}],
            }
        )


def test_chunk_still_constructs_with_only_the_original_three_fields() -> None:
    """Phase 0 must not break existing call sites - every new field needs a
    safe default so `Chunk(chunk_index=.., content=.., region_type=..)`
    (used throughout the test suite and by chunkers before Phase 3) keeps
    constructing without a ValidationError."""

    chunk = Chunk(chunk_index=0, content="text", region_type=RegionType.TEXT)

    assert chunk.source_type is None
    assert chunk.block_index is None
    assert chunk.heading_path == []
    assert chunk.page_start is None
    assert chunk.page_end is None
    assert chunk.source_locator is None
    assert chunk.column_names is None
    assert chunk.has_header is False
    assert chunk.header_source == HeaderSource.MISSING
    assert chunk.header_confidence == 0.0
    assert chunk.chunking_version == "legacy"


def test_chunk_block_index_zero_is_distinct_from_unset() -> None:
    chunk = Chunk(chunk_index=0, content="text", region_type=RegionType.TEXT, block_index=0)

    assert chunk.block_index == 0
    assert chunk.block_index is not None


def test_source_locator_defaults_are_all_none_or_false() -> None:
    locator = SourceLocator()

    assert locator.section is None
    assert locator.sheet_name is None
    assert locator.row_start is None
    assert locator.row_end is None
    assert locator.row_count is None
    assert locator.table_id is None
    assert locator.row_part is None
    assert locator.row_part_count is None
    assert locator.is_partial_row is False


def test_chunk_accepts_full_structural_metadata() -> None:
    chunk = Chunk(
        chunk_index=0,
        content="heading\n\nrow",
        region_type=RegionType.TABLE,
        source_type=SourceType.HTML,
        block_index=2,
        heading_path=["A", "B"],
        source_locator=SourceLocator(table_id="table-2", row_start=1, row_end=1, row_count=1),
        column_names=["Name"],
        has_header=True,
        header_source=HeaderSource.EXPLICIT,
        header_confidence=1.0,
        chunking_version="2026-09-structural-v1",
    )

    assert chunk.source_type == SourceType.HTML
    assert chunk.block_index == 2
    assert chunk.source_locator is not None
    assert chunk.source_locator.table_id == "table-2"


def test_embedding_request_accepts_access_level_above_five() -> None:
    request = EmbeddingRequest.model_validate(
        {
            "document_id": "doc-1",
            "department_id": "CNTT",
            "access_level": 6,
            "object_key": "docs/handbook.pdf",
            "chunks": [{"chunk_index": 0, "content": "text", "region_type": "text"}],
        }
    )

    assert request.access_level == 6
