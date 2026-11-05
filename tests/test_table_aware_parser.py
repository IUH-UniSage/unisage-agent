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
    pages = _pages(
        f"# Tieu de\n\n{fill_in}\n\n|A|B|\n|---|---|\n|1|2|\n\n{long_hash}\n\nKet thuc.\n"
    )

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


def test_pdf_page_number_between_continued_tables_is_not_a_region() -> None:
    pages = _pages(
        "# Tieu de\n\nNoi dung.\n\n|A|B|\n|---|---|\n|1|2|\n\n5\n",
        "|A|B|\n|---|---|\n|3|4|\n\n6\n",
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    assert [r.content.strip() for r in regions if r.region_type == RegionType.TEXT] == ["Noi dung."]


def test_pdf_page_number_at_the_end_of_running_text_is_dropped() -> None:
    pages = _pages(
        "# Tieu de\n\nDoan van trang mot.\n\n1\n",
        "Doan van trang hai.\n\n2\n",
        "Doan van trang ba co 3 muc.\n\n3\n",
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    text = "\n".join(r.content for r in regions)
    assert [line.strip() for line in text.splitlines() if line.strip()] == [
        "Doan van trang mot.",
        "Doan van trang hai.",
        "Doan van trang ba co 3 muc.",
    ]


def test_pdf_heading_level_follows_the_depth_of_its_number() -> None:
    # the extractor marks `5.2` as `#` and leaves `6.` as plain bold text; only
    # the numbers say `6.` closes `5.2` instead of nesting inside it
    pages = _pages(
        "# **5. Doi ngu**\n\n# **5.2 Co so vat chat**\n\nNoi dung 5.2.\n\n"
        "**6. Chien luoc giang day**\n\n- Gach dau dong.\n"
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    assert [r.heading_path for r in regions] == [
        ["5. Doi ngu", "5.2 Co so vat chat"],
        ["6. Chien luoc giang day"],
    ]


def test_pdf_numbered_heading_never_nests_under_a_number_that_is_not_its_prefix() -> None:
    pages = _pages(
        "# 3. Chuc nang\n\n# 3.1. Danh sach\n\nNoi dung 3.1.\n\n"
        "# _4.1.4. Tra cuu_\n\nNoi dung 4.1.4.\n\n# a. Muc con\n\nNoi dung a.\n"
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    assert [r.heading_path for r in regions] == [
        ["3. Chuc nang", "3.1. Danh sach"],
        ["4.1.4. Tra cuu"],
        ["4.1.4. Tra cuu", "a. Muc con"],
    ]


def test_pdf_numbered_heading_resumes_the_numbering_an_unnumbered_heading_interrupted() -> None:
    pages = _pages(
        "# Chuong trinh\n\n## 1. Muc tieu\n\nNoi dung 1.\n\n# Trinh do ngoai ngu:\n\nNoi dung.\n\n"
        "## 2. Chuan dau ra\n\nNoi dung 2.\n\n# Phu luc\n\n## 1. Bang diem\n\nNoi dung phu luc.\n"
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    assert [r.heading_path for r in regions] == [
        ["Chuong trinh", "1. Muc tieu"],
        ["Trinh do ngoai ngu:"],
        ["Chuong trinh", "2. Chuan dau ra"],
        # numbering that starts over is a new context, not a resumption
        ["Phu luc", "1. Bang diem"],
    ]


def test_pdf_table_row_holding_only_the_running_header_is_dropped() -> None:
    pages = _pages(
        "<u>KL7.0a</u>\n\n|A|B|C|\n|---|---|---|\n|1|x|y|\n",
        "<u>KL7.0a</u>\n\n|||KL7.0a|\n|---|---|---|\n|2|z|w|\n",
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    assert all("KL7.0a" not in r.content for r in regions if r.region_type == RegionType.TABLE)


# Page 14 of "Bản mô tả chương trình đào tạo đại học chính quy ngành Kiểm toán năm 2020.pdf",
# exactly as pymupdf4llm reads it: section 7.1 is a borderless list of `label <tab> : value`
# lines that the extractor turned into a 2-column table, with the section heading as its
# header row and the next heading (`7.2`) swallowed as its last row (03/11/2026).
_KIEM_TOAN_2020_P14 = (
    "|**7.1 Khối lượng kiến thức toàn khóa**||\n"
    "|---|---|\n"
    "|- Tổng số tín chỉ của chương trình đào tạo|: 139 TC|\n"
    "|- Tổng số tín chỉ không tính vào trung bình chung tích lũy|: 18 TC|\n"
    "|- Tổng số tín chỉ phải tích lũy|: 139 TC|\n"
    "|**Khối kiến thức giáo dục đại cương**|**: 48 TC**|\n"
    "|**+**Bắt buộc|: 39 TC|\n"
    "|+ Tự chọn|: 09 TC|\n"
    "|**Kiến thức giáo dục chuyên nghiệp**|**: 91 TC**|\n"
    "|-**_Khối kiến thức cơ sở ngành_**|**_: 32 TC_**|\n"
    "|**+**Bắt buộc|: 21 TC|\n"
    "|+ Tự chọn|: 11 TC|\n"
    "|- Số tín chỉ thực hành: 40 TC-28%<br>Số tín chỉ lý thuyết: 98|TC-72%|\n"
    "|**7.2 Kế hoạch đào tạo**||\n"
    "\n\n\n"
    "|**TT**|**Mã môn**|**Tên môn học**|**Số tín chỉ**|\n"
    "|---|---|---|---|\n"
    "|||**Học kỳ 1**|**17**|\n"
    "|1|2112001|Triết học Mác - Lênin|3|\n"
    "\n\n14 \n"
)


def test_pdf_key_value_pseudo_table_becomes_text_lines_under_its_heading() -> None:
    pages = _pages("# 7. NỘI DUNG CHƯƠNG TRÌNH ĐÀO TẠO\n\nGiới thiệu.\n", _KIEM_TOAN_2020_P14)

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    tables = [r for r in regions if r.region_type == RegionType.TABLE]
    assert all("Tổng số tín chỉ" not in r.content for r in tables)
    section = [
        r
        for r in regions
        if r.region_type == RegionType.TEXT
        and r.heading_path[-1:] == ["7.1 Khối lượng kiến thức toàn khóa"]
    ]
    assert len(section) == 1
    assert section[0].heading_path == [
        "7. NỘI DUNG CHƯƠNG TRÌNH ĐÀO TẠO",
        "7.1 Khối lượng kiến thức toàn khóa",
    ]
    assert section[0].page_start == 2
    assert [line for line in section[0].content.splitlines() if line] == [
        "- Tổng số tín chỉ của chương trình đào tạo: 139 TC",
        "- Tổng số tín chỉ không tính vào trung bình chung tích lũy: 18 TC",
        "- Tổng số tín chỉ phải tích lũy: 139 TC",
        "Khối kiến thức giáo dục đại cương: 48 TC",
        "+ Bắt buộc: 39 TC",
        "+ Tự chọn: 09 TC",
        "Kiến thức giáo dục chuyên nghiệp: 91 TC",
        "- Khối kiến thức cơ sở ngành: 32 TC",
        "+ Bắt buộc: 21 TC",
        "+ Tự chọn: 11 TC",
        "- Số tín chỉ thực hành: 40 TC-28%",
        "Số tín chỉ lý thuyết: 98 TC-72%",
    ]
    # no heading text inside any region, and no `col_N` header anywhere
    assert all("7.2 Kế hoạch đào tạo" not in r.content for r in regions)
    assert all(
        not any(name.startswith("col_") for name in (r.table.header_row or []))
        for r in tables
        if r.table is not None
    )


def test_pdf_heading_row_trailing_a_key_value_pseudo_table_heads_the_next_table() -> None:
    pages = _pages("# 7. NỘI DUNG CHƯƠNG TRÌNH ĐÀO TẠO\n\nGiới thiệu.\n", _KIEM_TOAN_2020_P14)

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    tables = [r for r in regions if r.region_type == RegionType.TABLE]
    assert len(tables) == 1
    assert tables[0].heading_path == [
        "7. NỘI DUNG CHƯƠNG TRÌNH ĐÀO TẠO",
        "7.2 Kế hoạch đào tạo",
    ]
    assert tables[0].table is not None
    assert tables[0].table.header_row == ["TT", "Mã môn", "Tên môn học", "Số tín chỉ"]


def test_pdf_key_value_pseudo_table_without_a_heading_row_keeps_its_first_line() -> None:
    # CNKT máy tính 2018: the extractor took the first `label : value` line as the header
    pages = _pages(
        "# 3. Khối lượng kiến thức toàn khoá\n\n"
        "|- Tổng số tín chỉ của chương trình đào|:146 TC|\n"
        "|---|---|\n"
        "|- Tổng số tín chỉ không tính vào trung bình chung tích lũy|: 18 TC|\n"
        "|- Tổng số tín chỉ phải tích lũy|: 128 TC|\n"
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    assert [r.region_type for r in regions] == [RegionType.TEXT]
    assert regions[0].heading_path == ["3. Khối lượng kiến thức toàn khoá"]
    assert regions[0].content.splitlines() == [
        "- Tổng số tín chỉ của chương trình đào: 146 TC",
        "- Tổng số tín chỉ không tính vào trung bình chung tích lũy: 18 TC",
        "- Tổng số tín chỉ phải tích lũy: 128 TC",
    ]


@pytest.mark.parametrize(
    "markdown",
    [
        # a real two-column table with a header
        "|Học phần|Số tín chỉ|\n|---|---|\n|Triết học|: 3|\n|Kinh tế chính trị|2|\n|Toán|3|\n",
        # a key/value table whose values do not start with `:`
        "|Course Outline||\n|---|---|\n|Credits|3|\n|Course description|Basic C|\n"
        "|Pre-requisites||\n",
        # too few rows to tell a list from a table
        "|Mục||\n|---|---|\n|Tổng|: 139 TC|\n",
        # three columns
        "|A|B|C|\n|---|---|---|\n|x|: 1|y|\n|z|: 2|w|\n|u|: 3|v|\n",
    ],
)
def test_pdf_ordinary_tables_are_not_read_as_key_value_lists(markdown: str) -> None:
    pages = _pages("# 1. Muc\n\n" + markdown)

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    assert [r.region_type for r in regions] == [RegionType.TABLE]


def test_pdf_key_value_lines_that_match_a_ruled_table_stay_a_table() -> None:
    # geometry found a ruled table with this very text: the document drew it as a table
    from app.rag.ingestion.table_evidence import EvidenceCell, EvidenceRow, TableEvidence

    rows = [("Tổng số tín chỉ", ": 139 TC"), ("Bắt buộc", ": 39 TC"), ("Tự chọn", ": 09 TC")]
    evidence = TableEvidence(
        page_number=1,
        bbox=(0, 0, 100, 100),
        page_height=800,
        rows=[
            EvidenceRow(
                cells=[EvidenceCell((0, i, 50, i + 1), a), EvidenceCell((50, i, 100, i + 1), b)],
                y0=i,
                y1=i + 1,
            )
            for i, (a, b) in enumerate(rows)
        ],
    )
    markdown = "".join(
        f"|{a}|{b}|\n" + ("|---|---|\n" if i == 0 else "") for i, (a, b) in enumerate(rows)
    )
    pages = _pages("# 1. Muc\n\n" + markdown)

    with (
        patch("pymupdf4llm.to_markdown", return_value=pages),
        patch(
            "app.rag.ingestion.table_aware_parser.collect_table_evidence",
            return_value={1: [evidence]},
        ),
    ):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    assert [r.region_type for r in regions] == [RegionType.TABLE]


def test_pdf_key_value_pseudo_table_pairs_lines_the_extractor_glued_into_one_row() -> None:
    # CNKT hóa học 2022 glues two lines into one row; Tiến sĩ KHMT 2022 has a label that
    # carries its own `:` before the value column's `:` (03/11/2026)
    pages = _pages(
        "# 3. Khối lượng kiến thức toàn khoá\n\n"
        "|- Tổng số tín chỉ của chương trình đào tạo|:**162TC**|\n"
        "|---|---|\n"
        "|- Tổng số tín chỉ không tính vào trung bình chung tích lũy<br>"
        "- Tổng số tín chỉ phải tích lũy|:**18TC**<br>:**162TC**|\n"
        "|- Tổng số tín chỉ không tính vào trung bình chung tích lũy :|: 0 TC|\n"
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "doc.pdf", "pdf")

    assert regions[0].content.splitlines() == [
        "- Tổng số tín chỉ của chương trình đào tạo: 162TC",
        "- Tổng số tín chỉ không tính vào trung bình chung tích lũy: 18TC",
        "- Tổng số tín chỉ phải tích lũy: 162TC",
        "- Tổng số tín chỉ không tính vào trung bình chung tích lũy: 0 TC",
    ]


def test_pdf_running_header_that_cites_a_document_is_kept_once() -> None:
    """A running header can be the only place a fact is written (the form's issuing
    letter, d0146): its first occurrence stays as text, the repeats are dropped."""

    header = "Ban hành kèm theo Công văn số: 32 /HĐGSNN ngày 20/5/2021 của Chủ tịch HĐGS\n\n"
    pages = _pages(
        header + "# Mẫu số 01\n\nNội dung trang một.\n",
        header + "Nội dung trang hai.\n",
        header + "Nội dung trang ba.\n",
    )

    with patch("pymupdf4llm.to_markdown", return_value=pages):
        regions = split_regions(make_pdf_bytes("x"), "form.pdf", "pdf")

    text = "\n".join(r.content for r in regions)
    assert text.count("32 /HĐGSNN") == 1
    assert all("HĐGSNN" not in "".join(r.heading_path) for r in regions)


def _table_region(header: list[str], rows: list[list[str]], page: int):
    from app.rag.ingestion.canonical_table import build_table_from_plain_rows
    from app.rag.ingestion.table_aware_parser import ParsedRegion
    from app.schemas.ingestion import HeaderSource

    table = build_table_from_plain_rows(header, rows, HeaderSource.INFERRED, 0.6, page=page)
    return ParsedRegion(RegionType.TABLE, "", page_start=page, page_end=page, table=table)


def test_a_banner_taken_for_the_header_gives_way_to_the_real_header_row() -> None:
    """q0167: the page banner became the header (`col_4` for its empty cells) and the real
    header was read as data, repeated on each page."""

    from app.rag.ingestion.canonical_table import check_table_invariants
    from app.rag.ingestion.table_aware_parser import _inherit_continuation_headers

    real = ["STT", "Mã HP", "Tên học phần", "Học kỳ", "Số TC"]
    region = _table_region(
        ["Ngành: Ô tô", "Ngành: Ô tô", "Ngành: Ô tô", "Loại: Chính quy", "Loại: Chính quy"],
        [real, ["1", "2113420", "Toán A1", "1", "2"], real, ["40", "2116406", "Động cơ", "5", "3"]],
        page=1,
    )

    (fixed,) = _inherit_continuation_headers([region])

    assert fixed.table.header_row == real
    assert [row.cells[0] for row in fixed.table.rows] == ["Ngành: Ô tô", "1", "40"]
    assert fixed.table.rows[2].cells[4] == "3"
    assert check_table_invariants(fixed.table) == []


def test_a_continuation_without_its_header_takes_the_header_of_the_table_before() -> None:
    from app.rag.ingestion.canonical_table import check_table_invariants
    from app.rag.ingestion.table_aware_parser import _inherit_continuation_headers

    first = _table_region(["STT", "Đơn vị", "Người liên hệ"], [["1", "Phòng Đào tạo", "A"]], page=1)
    cont = _table_region(
        ["26", "Phòng Quản lý KTX", "0908657978"], [["27", "Phòng Dịch vụ", "B"]], page=2
    )
    far = _table_region(["9", "Phòng X", "0903"], [["10", "Phòng Y", "C"]], page=5)

    _, fixed, untouched = _inherit_continuation_headers([first, cont, far])

    assert fixed.table.header_row == ["STT", "Đơn vị", "Người liên hệ"]
    assert [row.cells[1] for row in fixed.table.rows] == ["Phòng Quản lý KTX", "Phòng Dịch vụ"]
    assert "header_inherited" in fixed.table.warnings
    assert check_table_invariants(fixed.table) == []
    # three pages away: not a continuation, left as it was
    assert untouched.table.header_row == ["9", "Phòng X", "0903"]
