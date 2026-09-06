import pytest

from app.core.exceptions import UnsupportedFileTypeException
from app.rag.ingestion.table_aware_parser import split_regions
from app.schemas.ingestion import RegionType
from tests.fixtures.documents import (
    make_docx_bytes,
    make_docx_bytes_with_table,
    make_pdf_bytes,
    make_pdf_bytes_with_table,
)


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


def test_split_regions_produces_table_and_text_in_order_for_docx() -> None:
    content = make_docx_bytes_with_table("Intro paragraph.", "Outro paragraph.")

    regions = split_regions(content, "handbook.docx", "docx")

    region_types = [region.region_type for region in regions]
    assert region_types == [RegionType.TEXT, RegionType.TABLE, RegionType.TEXT]
    assert regions[0].content == "Intro paragraph."
    assert regions[2].content == "Outro paragraph."
    table_content = regions[1].content
    assert table_content.startswith("|")
    assert "Alice" in table_content
    assert "90" in table_content


def test_split_regions_produces_only_text_when_no_tables_in_docx() -> None:
    content = make_docx_bytes("Just a plain paragraph, no tables here.")

    regions = split_regions(content, "handbook.docx", "docx")

    assert regions
    assert all(region.region_type == RegionType.TEXT for region in regions)


def _html(body: str) -> bytes:
    return f"<html><body>{body}</body></html>".encode()


def test_split_regions_html_keeps_sibling_sections_in_separate_regions() -> None:
    """Two sections under different h2 headings must never end up in the
    same region - gluing them together loses which paragraph belongs to
    which topic."""

    content = _html(
        "<h1>Quy che dao tao</h1>"
        "<h2>Dieu kien tot nghiep</h2>"
        "<p>Sinh vien phai hoan thanh.</p>"
        "<p>Dong thoi sinh vien phai dat.</p>"
        "<h2>Ho so tot nghiep</h2>"
        "<p>Sinh vien can nop.</p>"
    )

    regions = split_regions(content, "handbook.html", "html")

    assert [r.region_type for r in regions] == [RegionType.TEXT, RegionType.TEXT]
    assert "Sinh vien phai hoan thanh." in regions[0].content
    assert "Dong thoi sinh vien phai dat." in regions[0].content
    assert "Sinh vien can nop." not in regions[0].content
    assert "Sinh vien can nop." in regions[1].content


def test_split_regions_html_prefixes_each_region_with_its_heading_path() -> None:
    content = _html("<h1>Quy che dao tao</h1><h2>Dieu kien tot nghiep</h2><p>Noi dung.</p>")

    regions = split_regions(content, "handbook.html", "html")

    assert regions[0].content.startswith("Quy che dao tao > Dieu kien tot nghiep\n\n")


def test_split_regions_html_pops_heading_stack_back_to_sibling_level() -> None:
    """h3 under one h2, then a second h2 sibling - the second section's path
    must not still carry the first section's h3."""

    content = _html(
        "<h1>Quy che</h1>"
        "<h2>Muc A</h2>"
        "<h3>Chi tiet A</h3>"
        "<p>Noi dung A.</p>"
        "<h2>Muc B</h2>"
        "<p>Noi dung B.</p>"
    )

    regions = split_regions(content, "handbook.html", "html")

    assert regions[0].content.startswith("Quy che > Muc A > Chi tiet A\n\n")
    assert regions[1].content.startswith("Quy che > Muc B\n\n")


def test_split_regions_html_keeps_a_list_as_one_region() -> None:
    content = _html(
        "<h2>Ho so can chuan bi</h2><ul><li>Don xin.</li><li>Ban sao.</li><li>Anh.</li></ul>"
    )

    regions = split_regions(content, "handbook.html", "html")

    assert len(regions) == 1
    assert regions[0].region_type == RegionType.TEXT
    assert "- Don xin." in regions[0].content
    assert "- Ban sao." in regions[0].content
    assert "- Anh." in regions[0].content


def test_split_regions_html_table_is_its_own_markdown_table_region() -> None:
    content = _html(
        "<h2>Hoc phi</h2>"
        "<table>"
        "<tr><th>Loai hoc phi</th><th>Muc thu</th></tr>"
        "<tr><td>Chinh quy</td><td>15tr/nam</td></tr>"
        "</table>"
    )

    regions = split_regions(content, "handbook.html", "html")

    assert [r.region_type for r in regions] == [RegionType.TABLE]
    assert regions[0].content.startswith("Hoc phi\n\n")
    assert "| Loai hoc phi | Muc thu |" in regions[0].content
    assert "| Chinh quy | 15tr/nam |" in regions[0].content


def test_split_regions_html_strips_script_style_nav_footer() -> None:
    content = _html(
        "<nav>Menu</nav>"
        "<script>doEvilThings()</script>"
        "<style>.x{color:red}</style>"
        "<h2>Noi dung chinh</h2>"
        "<p>Van ban that.</p>"
        "<footer>Copyright 2026</footer>"
    )

    regions = split_regions(content, "handbook.html", "html")

    joined = "\n".join(r.content for r in regions)
    assert "Van ban that." in joined
    assert "Menu" not in joined
    assert "doEvilThings" not in joined
    assert "color:red" not in joined
    assert "Copyright 2026" not in joined


def test_split_regions_html_recurses_through_generic_wrapper_tags() -> None:
    """A <div>/<section> wrapper with no semantic meaning of its own must not
    stop content nested inside it from being found."""

    content = _html("<section><div><h2>Muc A</h2><p>Noi dung A.</p></div></section>")

    regions = split_regions(content, "handbook.html", "html")

    assert len(regions) == 1
    assert "Noi dung A." in regions[0].content


def test_split_regions_html_no_heading_has_no_prefix() -> None:
    content = _html("<p>Chi co mot doan van, khong co heading.</p>")

    regions = split_regions(content, "handbook.html", "html")

    assert regions[0].content == "Chi co mot doan van, khong co heading."


def test_split_regions_htm_extension_uses_the_same_html_parser() -> None:
    content = _html("<h2>Tieu de</h2><p>Noi dung.</p>")

    regions = split_regions(content, "handbook.htm", "htm")

    assert regions[0].content == "Tieu de\n\nNoi dung."
