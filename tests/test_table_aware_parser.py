import pytest

from app.core.exceptions import UnsupportedFileTypeException
from app.rag.ingestion.table_aware_parser import split_regions
from app.schemas.ingestion import RegionType
from tests.fixtures.documents import make_pdf_bytes, make_pdf_bytes_with_table


def test_split_regions_produces_table_and_text_in_order() -> None:
    content = make_pdf_bytes_with_table("Intro paragraph.", "Outro paragraph.")

    regions = split_regions(content, "handbook.pdf", "pdf")

    region_types = [region.region_type for region in regions]
    assert RegionType.TABLE in region_types
    assert RegionType.TEXT in region_types
    table_index = region_types.index(RegionType.TABLE)
    assert region_types[:table_index] == [RegionType.TEXT] * table_index
    assert "Alice" in regions[table_index].content
    assert region_types[table_index + 1 :] == [RegionType.TEXT] * (
        len(region_types) - table_index - 1
    )


def test_split_regions_produces_only_text_when_no_tables() -> None:
    content = make_pdf_bytes("Just a plain paragraph, no tables here.")

    regions = split_regions(content, "handbook.pdf", "pdf")

    assert regions
    assert all(region.region_type == RegionType.TEXT for region in regions)


def test_split_regions_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeException):
        split_regions(b"whatever", "handbook.doc", "doc")
