"""End-to-end test matrix for the PDF table pipeline (evidence -> canonical
table -> merge -> hierarchy -> chunks). Complements the per-module tests with
the combinations a real document produces."""

from dataclasses import replace

import pymupdf
import pymupdf4llm
import pytest

from app.rag.chunking.table_row import TableRowChunker
from app.rag.ingestion.canonical_table import (
    RowDisposition,
    RowSignals,
    build_table_from_plain_rows,
    check_table_invariants,
)
from app.rag.ingestion.table_aware_parser import split_regions
from app.rag.ingestion.table_evidence import EvidenceCell, EvidenceRow, TableEvidence
from app.rag.ingestion.table_hierarchy import HIERARCHY_SCORING, infer_hierarchy
from app.rag.ingestion.table_normalizer import build_page_table
from app.schemas.ingestion import HeaderSource, RegionType
from tests.fixtures.documents import make_docx_bytes_with_table, make_pdf_bytes_with_table
from tests.test_table_evidence import SAMPLE, draw_table


def _tables(content: bytes, extension: str = "pdf"):
    regions = split_regions(content, f"t.{extension}", extension)
    return [r for r in regions if r.region_type == RegionType.TABLE and r.table]


def _pdf(
    *pages: list[tuple[list[list[str]], float]], col_x=(50, 120, 300, 450), footer=True
) -> bytes:
    doc = pymupdf.open()
    for number, (rows, top) in enumerate(pages, start=1):
        page = doc.new_page()
        draw_table(page, rows, top=top, col_x=col_x)
        if footer:
            page.insert_text((250, 825), f"Page {number}/{len(pages)}", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


# --- headers and column counts ---------------------------------------------


def _evidence_row(texts: list[str | None], y0: float, y1: float) -> EvidenceRow:
    cells = [
        None if text is None else EvidenceCell((50.0 + i * 100, y0, 150.0 + i * 100, y1), text)
        for i, text in enumerate(texts)
    ]
    return EvidenceRow(cells=cells, y0=y0, y1=y1)


def test_three_tier_header_is_flattened_from_geometry() -> None:
    evidence = TableEvidence(
        page_number=1,
        bbox=(50.0, 100.0, 450.0, 300.0),
        page_height=842.0,
        rows=[
            _evidence_row(["TT", "Fees", None, None], 100, 120),
            _evidence_row([None, "Regular", "Part-time", "Notes"], 120, 140),
            _evidence_row([None, "Per credit", "Per year", None], 140, 160),
            _evidence_row(["1", "Alpha", "10", "n1"], 160, 180),
            _evidence_row(["2", "Beta", "20", "n2"], 180, 200),
        ],
        column_bounds=[50.0, 150.0, 250.0, 350.0],
        header_row_count=3,
    )
    lines = [
        "|TT|Fees|||",
        "|---|---|---|---|",
        "||Regular|Part-time|Notes|",
        "||Per credit|Per year||",
        "|1|Alpha|10|n1|",
        "|2|Beta|20|n2|",
    ]

    table = build_page_table(lines, 1, evidence)

    assert len(table.header_levels) == 3
    assert table.header_row == [
        "TT",
        "Fees > Regular > Per credit",
        "Fees > Part-time > Per year",
        "Fees > Notes",
    ]
    assert table.data_rows == [["1", "Alpha", "10", "n1"], ["2", "Beta", "20", "n2"]]
    assert check_table_invariants(table) == []


@pytest.mark.parametrize("columns", [3, 4, 5])
def test_tables_with_3_4_and_5_columns_keep_every_cell(columns: int) -> None:
    header = [f"H{i}" for i in range(columns)]
    rows = [[f"r{r}c{c}" for c in range(columns)] for r in range(3)]
    lines = ["|" + "|".join(header) + "|", "|" + "|".join(["---"] * columns) + "|"]
    lines += ["|" + "|".join(row) + "|" for row in rows]

    table = build_page_table(lines, 1, None)

    assert table.header_row == header
    assert table.data_rows == rows
    assert check_table_invariants(table) == []


def test_empty_cells_and_short_rows_are_preserved_and_flagged() -> None:
    table = build_page_table(
        ["|A|B|C|D|", "|---|---|---|---|", "|1||3||", "|4|5|", "||||"], 1, None
    )

    assert table.data_rows == [["1", "", "3", ""], ["4", "5", "", ""]]
    assert "padded_cells" in table.rows[1].warnings
    dropped = [s for s in table.source_rows if s.disposition == RowDisposition.DROPPED_EMPTY]
    assert len(dropped) == 1
    assert check_table_invariants(table) == []


# --- non-tuition hierarchical PDFs -----------------------------------------


NUMBERED_ROWS = [
    ["No", "Item", "Budget", "Note"],
    ["I", "Operating costs", "", ""],
    ["1", "Personnel", "", ""],
    ["a)", "Lecturers", "500", "x"],
    ["b)", "Assistants", "200", "y"],
    ["2", "Utilities", "300", "z"],
    ["II", "Capital costs", "", ""],
    ["1", "Equipment", "900", "w"],
]


def test_numbered_budget_pdf_recovers_the_full_ancestor_stack() -> None:
    tables = _tables(_pdf((NUMBERED_ROWS, 100)))

    assert len(tables) == 1
    table = tables[0].table
    by_item = {row.cells[1]: row.ancestors for row in table.rows}
    assert by_item["Operating costs"] == []
    assert by_item["Lecturers"] == ["I Operating costs", "1 Personnel"]
    assert by_item["Assistants"] == ["I Operating costs", "1 Personnel"]
    assert by_item["Utilities"] == ["I Operating costs"]
    assert by_item["Equipment"] == ["II Capital costs"]
    assert check_table_invariants(table) == []


def test_flat_pdf_table_gets_no_ancestors_and_no_uncertainty_warning() -> None:
    flat = [
        ["Name", "Qty", "Price", "Stock"],
        ["Alpha", "1", "10", "5"],
        ["Beta", "", "20", "6"],
        ["Gamma", "3", "30", ""],
        ["Delta", "4", "40", "8"],
    ]

    table = _tables(_pdf((flat, 100)))[0].table

    assert all(row.ancestors == [] for row in table.rows)
    assert not any("hierarchy_uncertain" in row.warnings for row in table.rows)


def test_hierarchy_carries_across_a_page_break_in_a_generated_pdf() -> None:
    first_page = NUMBERED_ROWS[:5]
    filler = [[str(100 + i), f"Filler {i}", "1", "f"] for i in range(26)]
    page1 = [*first_page, *filler]
    page2 = [NUMBERED_ROWS[0], ["b)", "Assistants", "200", "y"], ["2", "Utilities", "300", "z"]]

    tables = _tables(_pdf((page1, 100), (page2, 50)))

    assert len(tables) == 1
    rows = tables[0].table.rows
    assistants = [row for row in rows if row.cells[1] == "Assistants" and row.page_start == 2]
    assert assistants, "the continuation row must be on page 2 of the merged table"


# --- thresholds, tie-break and fallbacks -----------------------------------


def _hierarchical_plain():
    return build_table_from_plain_rows(
        ["TT", "Name", "Val"],
        [["1", "Top", ""], ["", "Row", "5"], ["", "Row2", "6"], ["2", "Second", ""]],
        HeaderSource.INFERRED,
        0.6,
    )


def test_raising_the_assign_threshold_turns_an_accepted_row_into_uncertain() -> None:
    base = _hierarchical_plain()
    strict = replace(HIERARCHY_SCORING, assign_threshold=1.01)

    default_result = infer_hierarchy(base)
    strict_result = infer_hierarchy(base, strict)

    assert default_result.rows[1].ancestors == ["1 Top"]
    assert strict_result.rows[1].ancestors == []
    assert "hierarchy_uncertain" in strict_result.rows[1].warnings


def test_an_exact_tie_between_two_depths_is_undecided_not_arbitrary() -> None:
    # indentation votes "same level as the previous row" (depth 1), the bold
    # marker votes "child of the bold parent" (depth 2), with identical weight
    scoring = replace(
        HIERARCHY_SCORING,
        weight_numbering=0.0,
        weight_indent=0.25,
        weight_font=0.25,
        weight_mask=0.0,
        weight_span=0.0,
        weight_height=0.0,
        assign_threshold=0.0,
        min_present_weight=0.1,
    )
    base = build_table_from_plain_rows(
        ["TT", "Name", "Val"],
        [
            ["1", "Top", ""],
            ["", "Bold parent", ""],
            ["", "Child", "5"],
            ["", "Child two", "6"],
            ["2", "Second", ""],
        ],
        HeaderSource.INFERRED,
        0.6,
    )
    x0s = [50.0, 60.0, 60.0, 60.0, 50.0]
    bold = [False, True, False, False, False]
    rows = [
        replace(
            row,
            signals=RowSignals(
                cell_x0=[None, x0, None],
                cell_bold=[False, is_bold, False],
                cell_size=[None, 9.0, None],
                cell_merged=[False] * 3,
                height=17.0,
            ),
        )
        for row, x0, is_bold in zip(base.rows, x0s, bold, strict=True)
    ]
    table = replace(base, rows=rows)

    first = infer_hierarchy(table, scoring)
    second = infer_hierarchy(table, scoring)

    assert "hierarchy_uncertain" in first.rows[2].warnings
    assert first.rows[2].ancestors == []
    assert [r.ancestors for r in first.rows] == [r.ancestors for r in second.rows]


def test_a_table_without_geometry_or_numbering_is_left_untouched() -> None:
    table = build_table_from_plain_rows(
        ["Name", "Qty"], [["a", "1"], ["b", ""], ["c", "3"]], HeaderSource.INFERRED, 0.6
    )

    result = infer_hierarchy(table)

    assert [row.ancestors for row in result.rows] == [[], [], []]
    assert all(row.warnings == [] for row in result.rows)


# --- invariants over every source format ------------------------------------


def _all_fixture_tables():
    yield "pdf-simple", _tables(make_pdf_bytes_with_table("Intro.", "Outro."))
    yield "docx", _tables(make_docx_bytes_with_table("Intro.", "Outro."), "docx")
    yield (
        "html",
        _tables(
            b"<html><body><table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
            b"</body></html>",
            "html",
        ),
    )
    yield "pdf-numbered", _tables(_pdf((NUMBERED_ROWS, 100)))
    if SAMPLE.exists():
        yield "pdf-sample", _tables(SAMPLE.read_bytes())


@pytest.mark.parametrize("name", ["pdf-simple", "docx", "html", "pdf-numbered", "pdf-sample"])
def test_invariants_hold_for_every_fixture_format(name: str) -> None:
    groups = dict(_all_fixture_tables())
    if name not in groups:
        pytest.skip("fixture unavailable")

    assert groups[name]
    for region in groups[name]:
        assert check_table_invariants(region.table) == []


def test_every_non_blank_markdown_table_row_of_the_sample_is_accounted_for() -> None:
    if not SAMPLE.exists():
        pytest.skip("sample PDF not available")
    document = pymupdf.open(SAMPLE)
    pages = pymupdf4llm.to_markdown(document, page_chunks=True)
    document.close()
    source_lines = [
        line
        for page in pages
        for line in page["text"].splitlines()
        if line.strip().startswith("|") and not set(line.strip()) <= set("|-: ")
    ]

    table = _tables(SAMPLE.read_bytes())[0].table

    ledger_raw = [source.raw_text for source in table.source_rows]
    assert sorted(ledger_raw) == sorted(line.strip() for line in source_lines)
    # explained: 4 header rows are on three pages, the 2 later pages repeat both tiers
    kinds = [source.disposition for source in table.source_rows]
    assert kinds.count(RowDisposition.HEADER) == 2
    assert kinds.count(RowDisposition.REPEATED_HEADER) == 4
    assert kinds.count(RowDisposition.DATA) == len(table.rows) == 113


def test_the_sample_produces_chunks_that_cover_every_row_exactly_once() -> None:
    if not SAMPLE.exists():
        pytest.skip("sample PDF not available")
    regions = split_regions(SAMPLE.read_bytes(), SAMPLE.name, "pdf")

    chunks = TableRowChunker(max_tokens=400).split(regions)

    covered: list[int] = []
    for chunk in chunks:
        locator = chunk.source_locator
        covered.extend(range(locator.row_start, locator.row_end + 1))
    assert covered == list(range(1, 114))
    assert all(chunk.chunking_version for chunk in chunks)
