import pytest

from app.core.errors.exceptions import ChunkValidationException
from app.rag.chunking.validation import validate_chunks
from app.schemas.ingestion import Chunk, HeaderSource, RegionType, SourceLocator, SourceType


def _valid_text_chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "chunk_index": 0,
        "content": "Nội dung",
        "region_type": RegionType.TEXT,
        "source_type": SourceType.HTML,
        "block_index": 0,
        "chunking_version": "2026-09-structural-v1",
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


def _valid_table_chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "chunk_index": 0,
        "content": "| a | b |\n| --- | --- |\n| 1 | 2 |",
        "region_type": RegionType.TABLE,
        "source_type": SourceType.HTML,
        "block_index": 0,
        "column_names": ["a", "b"],
        "has_header": True,
        "header_source": HeaderSource.EXPLICIT,
        "header_confidence": 1.0,
        "source_locator": SourceLocator(table_id="table-0", row_start=1, row_end=1, row_count=1),
        "chunking_version": "2026-09-structural-v1",
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


def test_valid_text_chunk_passes() -> None:
    validate_chunks([_valid_text_chunk()])


def test_valid_table_chunk_passes() -> None:
    validate_chunks([_valid_table_chunk()])


def test_pdf_text_chunk_missing_page_start_fails() -> None:
    chunk = _valid_text_chunk(source_type=SourceType.PDF, page_start=None, page_end=None)

    with pytest.raises(ChunkValidationException) as excinfo:
        validate_chunks([chunk])

    assert "0" in excinfo.value.errors
    assert "page_start" in excinfo.value.errors["0"] or "page_end" in excinfo.value.errors["0"]


def test_pdf_text_chunk_with_valid_pages_passes() -> None:
    validate_chunks([_valid_text_chunk(source_type=SourceType.PDF, page_start=1, page_end=1)])


def test_pdf_page_end_before_page_start_fails() -> None:
    chunk = _valid_text_chunk(source_type=SourceType.PDF, page_start=5, page_end=3)

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_missing_source_type_fails() -> None:
    chunk = _valid_text_chunk(source_type=None)

    with pytest.raises(ChunkValidationException) as excinfo:
        validate_chunks([chunk])

    assert "source_type" in excinfo.value.errors["0"]


def test_missing_block_index_fails() -> None:
    chunk = _valid_text_chunk(block_index=None)

    with pytest.raises(ChunkValidationException) as excinfo:
        validate_chunks([chunk])

    assert "block_index" in excinfo.value.errors["0"]


def test_negative_block_index_fails() -> None:
    chunk = _valid_text_chunk(block_index=-1)

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_table_chunk_missing_row_count_fails_v5() -> None:
    """[v5] row_start/row_end/row_count are ALL mandatory for TABLE/EXCEL_ROW
    - unlike v4, which only checked consistency IF row_count happened to be present."""

    chunk = _valid_table_chunk(
        source_locator=SourceLocator(table_id="table-0", row_start=1, row_end=1, row_count=None)
    )

    with pytest.raises(ChunkValidationException) as excinfo:
        validate_chunks([chunk])

    assert "row_start" in excinfo.value.errors["0"] or "row_count" in excinfo.value.errors["0"]


def test_table_chunk_missing_source_locator_fails() -> None:
    chunk = _valid_table_chunk(source_locator=None)

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_table_chunk_row_range_inconsistent_with_row_count_fails() -> None:
    chunk = _valid_table_chunk(
        source_locator=SourceLocator(table_id="table-0", row_start=1, row_end=3, row_count=5)
    )

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_table_chunk_missing_table_id_fails() -> None:
    chunk = _valid_table_chunk(
        source_locator=SourceLocator(table_id=None, row_start=1, row_end=1, row_count=1)
    )

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_excel_row_chunk_same_rules_as_table() -> None:
    chunk = _valid_table_chunk(
        region_type=RegionType.EXCEL_ROW,
        source_locator=SourceLocator(table_id="table-0", row_start=1, row_end=1, row_count=None),
    )

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_partial_row_missing_row_part_count_fails() -> None:
    chunk = _valid_table_chunk(
        source_locator=SourceLocator(
            table_id="table-0",
            row_start=1,
            row_end=1,
            row_count=1,
            is_partial_row=True,
            row_part=1,
            row_part_count=None,
        )
    )

    with pytest.raises(ChunkValidationException) as excinfo:
        validate_chunks([chunk])

    assert "row_part" in excinfo.value.errors["0"]


def test_partial_row_with_row_part_count_1_fails_v5_rule_l() -> None:
    """[v5] rule (l): is_partial_row=True but row_part_count < 2 is a
    meaningless fallback - a row split into "1 of 1" should never have been
    marked partial in the first place."""

    chunk = _valid_table_chunk(
        source_locator=SourceLocator(
            table_id="table-0",
            row_start=1,
            row_end=1,
            row_count=1,
            is_partial_row=True,
            row_part=1,
            row_part_count=1,
        )
    )

    with pytest.raises(ChunkValidationException) as excinfo:
        validate_chunks([chunk])

    assert "row_part_count" in excinfo.value.errors["0"]


def test_partial_row_with_valid_2_parts_passes() -> None:
    chunk = _valid_table_chunk(
        source_locator=SourceLocator(
            table_id="table-0",
            row_start=1,
            row_end=1,
            row_count=1,
            is_partial_row=True,
            row_part=1,
            row_part_count=2,
        )
    )

    validate_chunks([chunk])


def test_non_partial_row_with_row_part_set_fails_inverse_rule_i() -> None:
    chunk = _valid_table_chunk(
        source_locator=SourceLocator(
            table_id="table-0",
            row_start=1,
            row_end=1,
            row_count=1,
            is_partial_row=False,
            row_part=1,
            row_part_count=2,
        )
    )

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_header_source_missing_but_has_header_true_fails() -> None:
    chunk = _valid_table_chunk(
        header_source=HeaderSource.MISSING, has_header=True, column_names=["a", "b"]
    )

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_header_source_missing_but_column_names_set_fails() -> None:
    chunk = _valid_table_chunk(
        header_source=HeaderSource.MISSING,
        has_header=False,
        column_names=["a", "b"],
    )

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_header_source_missing_but_confidence_nonzero_fails() -> None:
    chunk = _valid_table_chunk(
        header_source=HeaderSource.MISSING,
        has_header=False,
        column_names=None,
        header_confidence=0.5,
    )

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_header_source_missing_with_all_consistent_fields_passes() -> None:
    chunk = _valid_table_chunk(
        header_source=HeaderSource.MISSING,
        has_header=False,
        column_names=None,
        header_confidence=0.0,
    )

    validate_chunks([chunk])


def test_header_source_inferred_but_has_header_false_fails_v5_rule_k() -> None:
    """[v5] rule (k): a determined header_source (EXPLICIT/INFERRED) implies
    has_header=True - the inverse of rule (f)."""

    chunk = _valid_table_chunk(
        header_source=HeaderSource.INFERRED,
        has_header=False,
        column_names=None,
    )

    with pytest.raises(ChunkValidationException) as excinfo:
        validate_chunks([chunk])

    assert "has_header" in excinfo.value.errors["0"]


def test_has_header_false_but_column_names_set_fails() -> None:
    chunk = _valid_table_chunk(
        header_source=HeaderSource.EXPLICIT, has_header=False, column_names=["a"]
    )

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_has_header_true_but_column_names_empty_fails() -> None:
    chunk = _valid_table_chunk(has_header=True, column_names=[])

    with pytest.raises(ChunkValidationException):
        validate_chunks([chunk])


def test_multiple_invalid_chunks_are_all_reported() -> None:
    bad_chunk_1 = _valid_text_chunk(chunk_index=0, source_type=None)
    bad_chunk_2 = _valid_text_chunk(chunk_index=1, block_index=None)

    with pytest.raises(ChunkValidationException) as excinfo:
        validate_chunks([bad_chunk_1, bad_chunk_2])

    assert set(excinfo.value.errors.keys()) == {"0", "1"}


def test_empty_chunk_list_passes() -> None:
    validate_chunks([])
