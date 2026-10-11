from unittest.mock import patch

import pytest

from app.core.errors.exceptions import UnsupportedFileTypeException
from app.rag.ingestion.table_aware_parser import _markdown_row_cells, split_regions
from app.schemas.ingestion import RegionType, SourceType
from tests.fixtures.documents import (
    make_docx_bytes,
    make_docx_bytes_with_heading,
    make_docx_bytes_with_table,
    make_pdf_bytes,
    make_pdf_bytes_with_heading_across_pages,
    make_pdf_bytes_with_table,
)


def _pages(*texts: str) -> list[dict[str, object]]:
    """Build a synthetic `pymupdf4llm.to_markdown(..., page_chunks=True)`
    return value - one dict per page, 1-indexed - so tests can pin exact
    markdown content (including a literal `<u>` span, which real PDF
    fixtures can't reliably reproduce since `pymupdf.Page.insert_text` has
    no simple way to draw genuine underlined text pymupdf4llm's own
    span-flag detection would pick up)."""

    return [
        {"metadata": {"page_number": index + 1}, "text": text} for index, text in enumerate(texts)
    ]


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


def test_split_regions_html_records_heading_path_without_prefixing_content() -> None:
    content = _html("<h1>Quy che dao tao</h1><h2>Dieu kien tot nghiep</h2><p>Noi dung.</p>")

    regions = split_regions(content, "handbook.html", "html")

    assert regions[0].heading_path == ["Quy che dao tao", "Dieu kien tot nghiep"]
    assert regions[0].content == "Noi dung."
    assert not regions[0].content.startswith("Quy che")


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

    assert regions[0].heading_path == ["Quy che", "Muc A", "Chi tiet A"]
    assert regions[1].heading_path == ["Quy che", "Muc B"]
    assert regions[0].content == "Noi dung A."
    assert regions[1].content == "Noi dung B."


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
    assert regions[0].heading_path == ["Hoc phi"]
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
    assert regions[0].heading_path == []


def test_split_regions_htm_extension_uses_the_same_html_parser() -> None:
    content = _html("<h2>Tieu de</h2><p>Noi dung.</p>")

    regions = split_regions(content, "handbook.htm", "htm")

    assert regions[0].heading_path == ["Tieu de"]
    assert regions[0].content == "Noi dung."


def test_split_regions_html_assigns_increasing_block_index_and_source_type() -> None:
    content = _html("<h2>A</h2><p>Text A.</p><h2>B</h2><p>Text B.</p>")

    regions = split_regions(content, "handbook.html", "html")

    assert [r.block_index for r in regions] == [0, 1]
    assert all(r.source_type == SourceType.HTML for r in regions)


def test_split_regions_docx_records_heading_path_without_prefixing_content() -> None:
    content = make_docx_bytes_with_heading("Chuong Mot", "Noi dung chuong mot.")

    regions = split_regions(content, "handbook.docx", "docx")

    assert len(regions) == 1
    assert regions[0].heading_path == ["Chuong Mot"]
    assert regions[0].content == "Noi dung chuong mot."
    assert regions[0].source_type == SourceType.DOCX
    assert regions[0].block_index == 0


def test_split_regions_docx_table_gets_block_index_and_source_type() -> None:
    content = make_docx_bytes_with_table("Intro paragraph.", "Outro paragraph.")

    regions = split_regions(content, "handbook.docx", "docx")

    assert [r.block_index for r in regions] == [0, 1, 2]
    assert all(r.source_type == SourceType.DOCX for r in regions)


def test_split_regions_pdf_heading_persists_across_pages() -> None:
    """A TEXT region with no heading/type change spanning a page break is
    kept as ONE region (not split per page) - this is what lets
    `RecursiveChunker`'s char-overlap actually carry content across the
    page break; if page 1's and page 2's text became two separate regions
    (the old behavior), no chunker overlap would ever bridge them, so a
    list/paragraph split by a page break would lose all continuity."""

    content = make_pdf_bytes_with_heading_across_pages(
        "Chuong Mot", "Noi dung trang mot.", "Noi dung trang hai."
    )

    regions = split_regions(content, "handbook.pdf", "pdf")

    assert regions
    assert all(r.source_type == SourceType.PDF for r in regions)
    assert len(regions) == 1
    region = regions[0]
    assert region.heading_path == ["Chuong Mot"]
    assert region.page_start == 1
    assert region.page_end == 2
    assert "Noi dung trang mot." in region.content
    assert "Noi dung trang hai." in region.content
    assert "Chuong Mot" not in region.content


def test_split_regions_pdf_block_index_increases_across_document() -> None:
    content = make_pdf_bytes_with_heading_across_pages(
        "Chuong Mot", "Noi dung trang mot.", "Noi dung trang hai."
    )

    regions = split_regions(content, "handbook.pdf", "pdf")

    indices = [r.block_index for r in regions]
    assert all(index is not None for index in indices)
    non_null_indices = [index for index in indices if index is not None]
    assert non_null_indices == sorted(non_null_indices)
    assert len(set(non_null_indices)) == len(non_null_indices)


def test_markdown_row_cells_preserves_a_genuinely_empty_leading_cell() -> None:
    """`||content||` is a 3-cell row (empty, content, empty), not 1 cell.

    Naive `line.strip("|")` removes an unbounded run of `|` from each end,
    collapsing the two delimiter pipes AND the empty cell's own boundary
    together - silently dropping a real column and desyncing the row's
    cell count from the header's. Confirmed against a real PDF
    (`_to_delete/BAS.pdf`) where this exact shape made `TableRowChunker`
    raise `TableStructureError` on an otherwise-valid table."""

    assert _markdown_row_cells("||content||") == ["", "content", ""]
    assert _markdown_row_cells("|a|b|c|") == ["a", "b", "c"]


def test_pdf_numbered_line_with_underline_is_promoted_to_heading() -> None:
    """A numbered line at body font size (so pymupdf4llm's own `#`-heading
    detection never fires on it) is still promoted to a heading when it
    carries a `<u>...</u>` span - the pattern found in a real PDF
    (`DuongHoangHuy_DeCuongTTDN.pdf`'s "6. Ke hoach cong viec...") where a
    table's own intro line needed this to become part of `heading_path`
    instead of being silently lost as ordinary paragraph text."""

    pages = _pages(
        "# Tieu de tai lieu\n\n6. Ke hoach cong viec <u>(12 tuan)</u>\n\n|A|B|\n|---|---|\n|1|2|\n"
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "handbook.pdf", "pdf")

    table_regions = [r for r in regions if r.region_type == RegionType.TABLE]
    assert table_regions
    assert table_regions[0].heading_path == [
        "Tieu de tai lieu",
        "6. Ke hoach cong viec (12 tuan)",
    ]
    assert not any("Ke hoach cong viec" in r.content for r in regions)


def test_pdf_numbered_line_before_a_table_is_promoted_without_underline() -> None:
    """The second, independent promotion signal: no `<u>` span needed if the
    very next non-blank line starts a table."""

    pages = _pages("# Tieu de\n\n5. Gioi thieu bang\n\n|A|B|\n|---|---|\n|1|2|\n")

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "handbook.pdf", "pdf")

    table_regions = [r for r in regions if r.region_type == RegionType.TABLE]
    assert table_regions
    assert table_regions[0].heading_path == ["Tieu de", "5. Gioi thieu bang"]


def test_pdf_overlong_heading_candidate_stays_body_text() -> None:
    """A form's fill-in line ("1. Ngành: ....... Mã ngành: ....") right before a table
    passes the promotion signals, but no real section title is hundreds of characters
    long - as a heading it would be prefixed into every chunk below it and overflow
    `chunk_size`. Same for an over-long `#` line."""

    fill_in = "1. Nganh: " + "." * 300 + " Ma nganh: " + "." * 100
    long_hash = "## " + "Noi dung in dam rat dai " * 20
    pages = _pages(f"# Tieu de\n\n{fill_in}\n\n|A|B|\n|---|---|\n|1|2|\n\n{long_hash}\n\nKet thuc.\n")

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "form.pdf", "pdf")

    assert all(region.heading_path == ["Tieu de"] for region in regions)
    text = "\n".join(r.content for r in regions if r.region_type == RegionType.TEXT)
    assert "Ma nganh:" in text
    assert "Noi dung in dam rat dai" in text
    assert "##" not in text


def test_pdf_plain_numbered_clause_is_not_promoted_to_heading() -> None:
    """A numbered line with neither `<u>` nor a table/list right after it
    (just another paragraph) stays ordinary text - promoting every
    "<number>. ..." line would misfire on plain numbered sentences like
    "1. Ho ten sinh vien: ..." (seen in the real fixture PDF), which are
    not section titles."""

    pages = _pages("# Tieu de\n\n1. Ho ten: Nguyen Van A\n\nNoi dung binh thuong.\n")

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "handbook.pdf", "pdf")

    assert len(regions) == 1
    assert regions[0].heading_path == ["Tieu de"]
    assert "Ho ten: Nguyen Van A" in regions[0].content


def test_pdf_heading_title_strips_markdown_emphasis_and_html_tags() -> None:
    """`**bold**` markers and any leftover HTML tag are stripped from
    `heading_path` entries - both from a real `#`-heading (pymupdf4llm
    renders bold heading text as `# **Title**`) and from a promoted
    numbered line (which may still carry its source `<u>` wrapper)."""

    pages = _pages("# **Tieu De In Dam**\n\n6. Ke hoach <u>(chi tiet)</u>\n\n|A|\n|---|\n|1|\n")

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "handbook.pdf", "pdf")

    table_regions = [r for r in regions if r.region_type == RegionType.TABLE]
    assert table_regions[0].heading_path == ["Tieu De In Dam", "6. Ke hoach (chi tiet)"]
    assert "**" not in table_regions[0].heading_path[0]
    assert "<u>" not in table_regions[0].heading_path[1]


def test_pdf_repeated_page_letterhead_is_dropped_not_turned_into_a_chunk() -> None:
    """A plain-text line repeated verbatim across 2+ pages (a running
    header/footer, e.g. an institution name printed on every page) is
    boilerplate, not content - dropped entirely rather than becoming its
    own tiny region once a numbered heading right after it gets promoted
    and `flush()`es. Confirmed against the real fixture PDF, where
    "Truong DH Cong nghiep Tp.HCM" / "Thuc Tap Doanh Nghiep" repeat on
    every page right before the promoted "6. Ke hoach cong viec..." line."""

    letterhead = "Truong Dai Hoc Vi Du\n\n"
    pages = _pages(
        "# Tieu de\n\nNoi dung trang mot.\n\n" + letterhead,
        letterhead + "6. Ke hoach <u>(x)</u>\n\n|A|\n|---|\n|1|\n",
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "handbook.pdf", "pdf")

    assert not any("Truong Dai Hoc Vi Du" in r.content for r in regions)
    table_regions = [r for r in regions if r.region_type == RegionType.TABLE]
    assert table_regions[0].heading_path == ["Tieu de", "6. Ke hoach (x)"]


def test_pdf_roman_numeral_section_survives_a_same_level_arabic_subsection() -> None:
    """`pymupdf4llm` renders both a Roman-numeral top section and an
    Arabic-numeral subsection as the EXACT SAME markdown level (`##`) when
    they share a font size - naively trusting that raw level would let the
    subsection pop the Roman section off `heading_stack` as if it were a
    sibling, permanently losing it. Confirmed against a real PDF
    (`BAS.pdf`) where "II. Phan tich nghiep vu..." vanished from every
    later chunk's `heading_path` once "1. Module..." (same `##` level)
    was pushed. `_heading_level_for_title` fixes this by keying level off
    the numbering scheme (Roman -> shallower, Arabic -> deeper) instead of
    the raw markdown level."""

    pages = _pages(
        "## II. Phan tich nghiep vu\n\n"
        "## 1. Module dau tien\n\n"
        "Noi dung module 1.\n\n"
        "## 2. Module thu hai\n\n"
        "Noi dung module 2.\n"
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "handbook.pdf", "pdf")

    assert [r.heading_path for r in regions] == [
        ["II. Phan tich nghiep vu", "1. Module dau tien"],
        ["II. Phan tich nghiep vu", "2. Module thu hai"],
    ]


def test_pdf_bold_wrapped_arabic_subsection_is_promoted_as_sibling() -> None:
    """A sub-section rendered fully bold (`**4. ...**`) at body font size -
    so `pymupdf4llm` never marks it up as a heading at all - is still
    promoted when it introduces a bullet list, and nests at the same
    level as its `##`-detected Arabic-numbered siblings (not under
    whichever one happened to be active last). Confirmed against
    `BAS.pdf`'s "**4. Nghiep vu Chat...**", which sat right after
    "3. Giai phap..." content and needed to become its OWN heading, not
    just more text glued under "3."."""

    pages = _pages(
        "## 3. Muc thu ba\n\nNoi dung muc 3.\n\n**4. Muc thu tu**\n\n- Gach dau dong dau tien.\n"
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "handbook.pdf", "pdf")

    assert [r.heading_path for r in regions] == [
        ["3. Muc thu ba"],
        ["4. Muc thu tu"],
    ]


def test_pdf_all_caps_roman_numeral_line_is_promoted_without_underline_or_list() -> None:
    """A bold-but-not-larger Roman-numeral section title followed by
    ordinary prose (neither underlined nor immediately before a table/
    list) is still promoted because it is entirely upper-case - confirmed
    against `BAS.pdf`'s "**III. SO DO CAC ACTORS...**" and
    "**IV. QUY TRINH VAN HANH...**", each followed by plain paragraph
    text, which the underline/table-list signals alone would have missed."""

    pages = _pages(
        "# Tieu de\n\n"
        "**III. SO DO TONG QUAN HE THONG**\n\n"
        "Doan van thuong mo ta so do, khong phai bang hay list.\n"
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "handbook.pdf", "pdf")

    assert regions[0].heading_path == ["Tieu de", "III. SO DO TONG QUAN HE THONG"]


def test_pdf_mixed_case_ordinal_line_is_not_promoted_by_all_caps_signal() -> None:
    """Guard against over-triggering the all-caps signal: a mixed-case
    ordinal line with no underline and no table/list right after it stays
    ordinary text, same as the existing plain-numbered-clause case."""

    pages = _pages(
        "# Tieu de\n\nIII. Doan van thuong, khong viet hoa toan bo.\n\nTiep tuc doan van.\n"
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "handbook.pdf", "pdf")

    assert len(regions) == 1
    assert regions[0].heading_path == ["Tieu de"]
