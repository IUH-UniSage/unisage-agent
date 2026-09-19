import pytest

from app.rag.ingestion.canonical_table import (
    EventKind,
    NormalizationEvent,
    RowDisposition,
    SourceRow,
    TableBlock,
    TableRow,
    build_table_from_plain_rows,
    check_table_invariants,
)
from app.schemas.ingestion import HeaderSource


def _plain_table() -> TableBlock:
    return build_table_from_plain_rows(
        ["Name", "Score"],
        [["Alice", "90"], ["Bob", "85"]],
        HeaderSource.EXPLICIT,
        1.0,
        page=2,
    )


def _table_with(
    *,
    source_rows: list[SourceRow],
    rows: list[TableRow],
    header_levels: list[list[str]] | None = None,
    events: list[NormalizationEvent] | None = None,
) -> TableBlock:
    return TableBlock(
        header_row=["A", "B"],
        data_rows=[row.cells for row in rows],
        header_source=HeaderSource.INFERRED,
        header_confidence=0.6,
        header_levels=header_levels if header_levels is not None else [["A", "B"]],
        rows=rows,
        source_rows=source_rows,
        normalization_events=events or [],
    )


def _header_source() -> SourceRow:
    return SourceRow("p1-r1", 1, "A | B", 2, RowDisposition.HEADER)


def _row(index: int, cells: list[str], sources: list[str], source_cells: int) -> TableRow:
    return TableRow(
        cells=cells,
        raw_text=" | ".join(cells),
        row_index=index,
        page_start=1,
        page_end=1,
        source_row_ids=sources,
        source_cell_count=source_cells,
        canonical_cell_count=len(cells),
    )


def test_plain_table_builder_satisfies_all_invariants() -> None:
    table = _plain_table()

    assert check_table_invariants(table) == []
    assert table.data_rows == [["Alice", "90"], ["Bob", "85"]]
    assert [row.row_index for row in table.rows] == [1, 2]
    assert all(row.page_start == 2 and row.page_end == 2 for row in table.rows)
    assert table.header_levels == [["Name", "Score"]]
    assert [source.disposition for source in table.source_rows] == [
        RowDisposition.HEADER,
        RowDisposition.DATA,
        RowDisposition.DATA,
    ]


def test_headerless_plain_table_has_no_header_levels() -> None:
    table = build_table_from_plain_rows(None, [["x", "1"]], HeaderSource.MISSING, 0.0)

    assert table.header_levels == []
    assert check_table_invariants(table) == []


def test_hand_built_table_falls_back_to_synthesized_rows() -> None:
    table = TableBlock(
        header_row=["A"],
        data_rows=[["1"], ["2"]],
        header_source=HeaderSource.EXPLICIT,
        header_confidence=1.0,
    )

    rows = table.canonical_rows()

    assert [row.cells for row in rows] == [["1"], ["2"]]
    assert [row.row_index for row in rows] == [1, 2]
    assert check_table_invariants(table) == []


def test_i1_detects_source_row_that_produced_no_canonical_row() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow("p1-r2", 1, "x | 1", 2, RowDisposition.DATA, canonical_row_ids=[1]),
            SourceRow("p1-r3", 1, "lost | 2", 2, RowDisposition.DATA, canonical_row_ids=[]),
        ],
        rows=[_row(1, ["x", "1"], ["p1-r2"], 2)],
    )

    violations = check_table_invariants(table)

    assert any("p1-r3" in message and "no canonical row" in message for message in violations)


def test_i1_detects_non_blank_source_row_marked_dropped() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow("p1-r2", 1, "x | 1", 2, RowDisposition.DROPPED_EMPTY),
        ],
        rows=[],
    )

    assert any("dropped" in message for message in check_table_invariants(table))


def test_blank_source_row_can_be_dropped_empty() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow("p1-r2", 1, "| |", 2, RowDisposition.DROPPED_EMPTY),
            SourceRow("p1-r3", 1, "x | 1", 2, RowDisposition.DATA, canonical_row_ids=[1]),
        ],
        rows=[_row(1, ["x", "1"], ["p1-r3"], 2)],
    )

    assert check_table_invariants(table) == []


def test_i1_detects_canonical_row_without_source() -> None:
    table = _table_with(
        source_rows=[_header_source()],
        rows=[_row(1, ["x", "1"], [], 2)],
    )

    assert any("have no source row" in message for message in check_table_invariants(table))


def test_i1_repeated_header_requires_drop_event() -> None:
    source_rows = [
        _header_source(),
        SourceRow("p2-r1", 2, "A | B", 2, RowDisposition.REPEATED_HEADER),
        SourceRow("p2-r2", 2, "x | 1", 2, RowDisposition.DATA, canonical_row_ids=[1]),
    ]
    rows = [_row(1, ["x", "1"], ["p2-r2"], 2)]

    without_event = _table_with(source_rows=source_rows, rows=rows)
    with_event = _table_with(
        source_rows=source_rows,
        rows=rows,
        events=[NormalizationEvent(EventKind.DROP_REPEATED_HEADER, ["p2-r1"], "repeat")],
    )

    assert any("drop_repeated_header" in m for m in check_table_invariants(without_event))
    assert check_table_invariants(with_event) == []


def test_i1_header_count_must_match_header_levels() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow("p1-r2", 1, "A2 | B2", 2, RowDisposition.HEADER),
            SourceRow("p1-r3", 1, "x | 1", 2, RowDisposition.DATA, canonical_row_ids=[1]),
        ],
        rows=[_row(1, ["x", "1"], ["p1-r3"], 2)],
        header_levels=[["A", "B"]],
    )

    assert any("header levels" in message for message in check_table_invariants(table))


def test_i2_detects_canonical_row_descending_from_header() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow("p1-r2", 1, "x | 1", 2, RowDisposition.DATA, canonical_row_ids=[1]),
        ],
        rows=[_row(1, ["A", "B"], ["p1-r1", "p1-r2"], 2)],
    )

    assert any(message.startswith("I2") for message in check_table_invariants(table))


def test_i3_detects_lost_cell_without_event() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow("p1-r2", 1, "x | 1 | extra", 3, RowDisposition.DATA, canonical_row_ids=[1]),
        ],
        rows=[_row(1, ["x", "1"], ["p1-r2"], 3)],
    )

    messages = check_table_invariants(table)

    assert any("cell count 3 -> 2" in message for message in messages)
    assert any("lost" in message for message in messages)


def test_i3_cell_split_is_allowed_with_event() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow("p1-r2", 1, "3.2<br>Khoa hoc", 1, RowDisposition.DATA, canonical_row_ids=[1]),
        ],
        rows=[_row(1, ["3.2", "Khoa hoc"], ["p1-r2"], 1)],
        events=[NormalizationEvent(EventKind.SPLIT_CELL, ["p1-r2"], "split TT and label")],
    )

    assert check_table_invariants(table) == []


def test_i3_detects_invented_or_changed_text() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow("p1-r2", 1, "x | 1", 2, RowDisposition.DATA, canonical_row_ids=[1]),
        ],
        rows=[_row(1, ["x", "9"], ["p1-r2"], 2)],
    )

    assert any("content differs" in message for message in check_table_invariants(table))


def test_i3_recover_text_event_explains_text_change() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow(
                "p1-r2", 1, "~~o~~ | ~~1 130 000~~", 2, RowDisposition.DATA, canonical_row_ids=[1]
            ),
        ],
        rows=[_row(1, ["Môn thực hành", "1.130.000"], ["p1-r2"], 2)],
        events=[NormalizationEvent(EventKind.RECOVER_TEXT, ["p1-r2"], "recovered from geometry")],
    )

    assert check_table_invariants(table) == []


def test_i3_merged_source_rows_compare_against_their_concatenation() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow("p1-r2", 1, "Khối kinh tế | |", 3, RowDisposition.MERGED, [1]),
            SourceRow("p1-r3", 1, "(Ngôn ngữ Anh) | 35.110.000", 2, RowDisposition.MERGED, [1]),
        ],
        rows=[_row(1, ["Khối kinh tế (Ngôn ngữ Anh)", "35.110.000"], ["p1-r2", "p1-r3"], 5)],
        events=[NormalizationEvent(EventKind.MERGE_CELLS, ["p1-r2", "p1-r3"], "wrapped row")],
    )

    assert check_table_invariants(table) == []


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        ([1, 1, 2], False),
        ([2, 1], True),
    ],
)
def test_i4_pages_must_not_decrease(pages: list[int], expected: bool) -> None:
    rows = []
    source_rows = [_header_source()]
    for index, page in enumerate(pages, start=1):
        sid = f"p{page}-r{index + 1}"
        source_rows.append(
            SourceRow(sid, page, f"v{index} | 1", 2, RowDisposition.DATA, canonical_row_ids=[index])
        )
        row = _row(index, [f"v{index}", "1"], [sid], 2)
        row.page_start = row.page_end = page
        rows.append(row)

    violations = check_table_invariants(_table_with(source_rows=source_rows, rows=rows))

    assert any("earlier page" in message for message in violations) is expected


def test_i4_row_index_must_be_contiguous() -> None:
    table = _table_with(
        source_rows=[
            _header_source(),
            SourceRow("p1-r2", 1, "x | 1", 2, RowDisposition.DATA, canonical_row_ids=[2]),
        ],
        rows=[_row(2, ["x", "1"], ["p1-r2"], 2)],
    )

    assert any("not contiguous" in message for message in check_table_invariants(table))
