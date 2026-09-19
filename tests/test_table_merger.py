from dataclasses import replace

import pymupdf
import pytest

from app.rag.ingestion.canonical_table import (
    EventKind,
    RowDisposition,
    build_table_from_plain_rows,
    check_table_invariants,
)
from app.rag.ingestion.table_aware_parser import split_regions
from app.rag.ingestion.table_evidence import EvidenceRow, TableEvidence
from app.rag.ingestion.table_merger import (
    MERGE_SCORING,
    MergeCandidate,
    decide_merge,
    furniture_key,
    merge_page_tables,
)
from app.schemas.ingestion import HeaderSource, RegionType
from tests.test_table_evidence import SAMPLE, draw_table

HEADER = ["TT", "Name", "Fee", "Year"]


def _plain(header: list[str] | None, rows: list[list[str]], page: int):
    return build_table_from_plain_rows(header, rows, HeaderSource.INFERRED, 0.6, page=page)


def _evidence(page: int, *, top: float, bottom: float, height: float = 842.0) -> TableEvidence:
    return TableEvidence(
        page_number=page,
        bbox=(50.0, top, 545.0, bottom),
        page_height=height,
        rows=[EvidenceRow(cells=[], y0=top, y1=bottom)],
        column_bounds=[50.0, 120.0, 300.0, 450.0],
    )


def _candidate(table, *, page: int, evidence=None, heading: list[str] | None = None):
    return MergeCandidate(table, page, page, heading or [], evidence)


def test_scoring_constants_are_pinned() -> None:
    s = MERGE_SCORING

    weights = (s.weight_header, s.weight_columns, s.weight_bounds, s.weight_position)
    assert weights == (0.35, 0.20, 0.20, 0.25)
    assert sum(weights) == pytest.approx(1.0)
    assert (s.threshold_with_header, s.threshold_without_header) == (0.75, 0.80)
    assert (s.borderline_floor, s.min_present_weight) == (0.65, 0.50)


def test_furniture_key_masks_digits() -> None:
    assert furniture_key("Trang 2/4") == furniture_key("Trang 3/4") == "Trang #/#"


def test_repeated_header_and_continuous_geometry_merge() -> None:
    first = _plain(HEADER, [["1", "a", "1", "2"]], 1)
    second = _plain(HEADER, [["2", "b", "3", "4"]], 2)

    decision = decide_merge(
        _candidate(first, page=1, evidence=_evidence(1, top=100, bottom=800)),
        _candidate(second, page=2, evidence=_evidence(2, top=50, bottom=300)),
        only_furniture_between=True,
    )

    assert decision.merge and decision.header_present
    assert decision.score == pytest.approx(1.0)


def test_two_different_tables_with_the_same_header_are_not_merged_when_position_breaks() -> None:
    first = _plain(HEADER, [["1", "a", "1", "2"]], 1)
    second = _plain(HEADER, [["2", "b", "3", "4"]], 2)

    decision = decide_merge(
        _candidate(first, page=1, evidence=_evidence(1, top=100, bottom=300)),  # ends mid-page
        _candidate(second, page=2, evidence=_evidence(2, top=50, bottom=300)),
        only_furniture_between=True,
    )

    assert not decision.merge
    assert decision.reason == "position_not_continuous"


def test_second_table_starting_low_on_the_page_is_not_a_continuation() -> None:
    first = _plain(HEADER, [["1", "a", "1", "2"]], 1)
    second = _plain(HEADER, [["2", "b", "3", "4"]], 2)

    decision = decide_merge(
        _candidate(first, page=1, evidence=_evidence(1, top=100, bottom=800)),
        _candidate(second, page=2, evidence=_evidence(2, top=400, bottom=700)),
        only_furniture_between=True,
    )

    assert not decision.merge and decision.reason == "position_not_continuous"


def test_hard_blockers_prevent_a_merge() -> None:
    first = _plain(HEADER, [["1", "a", "1", "2"]], 1)
    same = _plain(HEADER, [["2", "b", "3", "4"]], 2)
    narrower = _plain(["TT", "Name", "Fee"], [["2", "b", "3"]], 2)

    not_next = decide_merge(
        _candidate(first, page=1), _candidate(same, page=3), only_furniture_between=True
    )
    heading = decide_merge(
        _candidate(first, page=1),
        _candidate(same, page=2, heading=["Other"]),
        only_furniture_between=True,
    )
    content = decide_merge(
        _candidate(first, page=1), _candidate(same, page=2), only_furniture_between=False
    )
    columns = decide_merge(
        _candidate(first, page=1), _candidate(narrower, page=2), only_furniture_between=True
    )

    assert [d.merge for d in (not_next, heading, content, columns)] == [False] * 4
    assert not_next.reason == "not_next_page"
    assert heading.reason == "heading_between"
    assert content.reason == "content_between"
    assert columns.reason == "column_count_differs"


def test_without_geometry_and_without_header_there_is_not_enough_evidence() -> None:
    first = _plain(HEADER, [["1", "a", "1", "2"]], 1)
    second = _plain(["9", "x", "y", "z"], [["2", "b", "3", "4"]], 2)  # first row is really data

    decision = decide_merge(
        _candidate(first, page=1), _candidate(second, page=2), only_furniture_between=True
    )

    assert not decision.merge
    assert decision.reason == "not_enough_evidence"


def test_headerless_continuation_with_geometry_merges_at_the_higher_threshold() -> None:
    first = _plain(HEADER, [["1", "a", "1", "2"]], 1)
    second = _plain(["2", "b", "3", "4"], [["3", "c", "5", "6"]], 2)

    decision = decide_merge(
        _candidate(first, page=1, evidence=_evidence(1, top=100, bottom=800)),
        _candidate(second, page=2, evidence=_evidence(2, top=50, bottom=300)),
        only_furniture_between=True,
    )

    assert decision.merge and not decision.header_present
    assert decision.score == pytest.approx(1.0)


def test_partial_header_match_is_scored_and_renormalized() -> None:
    first = _plain(HEADER, [["1", "a", "1", "2"]], 1)
    second = _plain(["TT", "Name", "Fee", "Other"], [["2", "b", "3", "4"]], 2)  # 3 of 4 match

    decision = decide_merge(
        _candidate(first, page=1, evidence=_evidence(1, top=100, bottom=800)),
        _candidate(second, page=2, evidence=_evidence(2, top=50, bottom=300)),
        only_furniture_between=True,
    )

    # (0.35*0.75 + 0.20 + 0.20 + 0.25) / 1.0
    assert decision.merge and decision.score == pytest.approx(0.9125)


def test_borderline_score_is_reported_but_not_merged() -> None:
    first = _plain(HEADER, [["1", "a", "1", "2"]], 1)
    second = _plain(HEADER, [["2", "b", "3", "4"]], 2)
    skewed = replace(
        _evidence(2, top=50, bottom=300), column_bounds=[50.0, 200.0, 380.0, 520.0]
    )  # boundaries far off -> bounds signal 0

    decision = decide_merge(
        _candidate(first, page=1, evidence=_evidence(1, top=100, bottom=800)),
        _candidate(second, page=2, evidence=skewed),
        only_furniture_between=True,
    )

    # (0.20 + 0.35 + 0.0 + 0.25) / 1.0 = 0.80 -> merged; make it borderline by dropping header
    assert decision.merge  # the header alone lifts this above threshold
    weak = _plain(["TT", "X", "Y", "Z"], [["2", "b", "3", "4"]], 2)
    weak_decision = decide_merge(
        _candidate(first, page=1, evidence=_evidence(1, top=100, bottom=800)),
        _candidate(weak, page=2, evidence=skewed),
        only_furniture_between=True,
    )
    # header_match 0.25 < 0.5 -> header absent: (0.20 + 0.0 + 0.25) / 0.65 = 0.6923
    assert not weak_decision.merge
    assert weak_decision.borderline
    assert weak_decision.score == pytest.approx(0.45 / 0.65)


def test_merge_with_repeated_header_drops_it_and_renumbers_rows_and_ledger() -> None:
    first = _plain(HEADER, [["1", "a", "1", "2"], ["2", "b", "3", "4"]], 1)
    second = _plain(HEADER, [["3", "c", "5", "6"]], 2)

    merged = merge_page_tables(first, second, header_present=True)

    assert merged.data_rows == [["1", "a", "1", "2"], ["2", "b", "3", "4"], ["3", "c", "5", "6"]]
    assert [row.row_index for row in merged.rows] == [1, 2, 3]
    assert [row.page_start for row in merged.rows] == [1, 1, 2]
    repeated = [s for s in merged.source_rows if s.disposition == RowDisposition.REPEATED_HEADER]
    assert len(repeated) == 1
    assert any(e.kind == EventKind.DROP_REPEATED_HEADER for e in merged.normalization_events)
    assert merged.header_confidence == MERGE_SCORING.header_confidence_when_repeated
    assert check_table_invariants(merged) == []
    assert second.rows[0].row_index == 1  # inputs are not mutated


def test_merge_without_repeated_header_restores_the_taken_header_as_data() -> None:
    first = _plain(HEADER, [["1", "a", "1", "2"]], 1)
    # page 2 has no header: its first data row was read as a header by the page parse
    second = _plain(["2", "b", "3", "4"], [["3", "c", "5", "6"]], 2)
    for source in second.source_rows:
        source.cells = source.raw_text.split(" | ")

    merged = merge_page_tables(first, second, header_present=False)

    assert merged.data_rows == [["1", "a", "1", "2"], ["2", "b", "3", "4"], ["3", "c", "5", "6"]]
    assert "headerless_continuation" in merged.rows[1].warnings
    assert check_table_invariants(merged) == []


def _pdf_with_tables(*pages: dict) -> bytes:
    doc = pymupdf.open()
    for spec in pages:
        page = doc.new_page()
        if "heading" in spec:
            page.insert_text((50, 60), spec["heading"], fontsize=20)
        if "rows" in spec:
            draw_table(page, spec["rows"], top=spec["top"])
        if "footer" in spec:
            page.insert_text((250, 825), spec["footer"], fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def _rows(count: int, start: int = 1, header: bool = True) -> list[list[str]]:
    rows = [HEADER] if header else []
    rows += [[str(i), f"item {i}", str(i * 10), str(i * 20)] for i in range(start, start + count)]
    return rows


def _tables(content: bytes):
    regions = split_regions(content, "t.pdf", "pdf")
    return [r for r in regions if r.region_type == RegionType.TABLE and r.table]


def test_pdf_table_running_to_the_page_bottom_merges_with_the_next_page_and_keeps_pages() -> None:
    # 33 rows of 20pt from y=100 end at y=760 of 842 (bottom gap ~10%)
    content = _pdf_with_tables(
        {"rows": _rows(32), "top": 100, "footer": "Page 1/2"},
        {"rows": _rows(5, start=33), "top": 50, "footer": "Page 2/2"},
    )

    tables = _tables(content)

    assert len(tables) == 1
    table = tables[0].table
    assert table.table_id == "table-0"
    assert len(table.rows) == 37
    assert [row.row_index for row in table.rows] == list(range(1, 38))
    assert {row.page_start for row in table.rows} == {1, 2}
    assert (tables[0].page_start, tables[0].page_end) == (1, 2)
    assert check_table_invariants(table) == []


def test_pdf_table_continuing_without_a_repeated_header_is_merged_without_losing_a_row() -> None:
    content = _pdf_with_tables(
        {"rows": _rows(32), "top": 100},
        {"rows": _rows(5, start=33, header=False), "top": 50},
    )

    tables = _tables(content)

    assert len(tables) == 1
    table = tables[0].table
    assert len(table.rows) == 37  # the first row of page 2 was restored as data
    assert table.rows[32].cells == ["33", "item 33", "330", "660"]
    assert "headerless_continuation" in table.rows[32].warnings
    assert check_table_invariants(table) == []


def test_two_short_tables_with_the_same_header_stay_separate() -> None:
    content = _pdf_with_tables(
        {"rows": _rows(3), "top": 100},
        {"rows": _rows(3, start=4), "top": 100},
    )

    tables = _tables(content)

    assert len(tables) == 2
    assert [t.table.table_id for t in tables] == ["table-0", "table-1"]
    assert [row.row_index for row in tables[1].table.rows] == [1, 2, 3]


def test_a_heading_between_two_tables_prevents_the_merge() -> None:
    content = _pdf_with_tables(
        {"rows": _rows(32), "top": 100},
        {"heading": "New Section", "rows": _rows(3, start=33), "top": 120},
    )

    assert len(_tables(content)) == 2


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample PDF not available")
def test_sample_pdf_is_one_logical_table_across_three_pages() -> None:
    tables = _tables(SAMPLE.read_bytes())

    assert len(tables) == 1
    region = tables[0]
    table = region.table
    assert table.table_id == "table-0"
    assert (region.page_start, region.page_end) == (2, 4)
    assert len(table.rows) == 31 + 41 + 41
    assert [row.row_index for row in table.rows] == list(range(1, 114))
    assert {row.page_start for row in table.rows} == {2, 3, 4}
    assert (
        sum(s.disposition == RowDisposition.REPEATED_HEADER for s in table.source_rows) == 4
    )  # 2 tiers x 2 pages
    assert check_table_invariants(table) == []
    assert not any(row.cells[2] == "Mức thu 01 tín chỉ" for row in table.rows)
