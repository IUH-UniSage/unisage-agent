"""Decide whether a table on one page continues the table on the previous page
and, if so, merge the two into one logical table.

The decision is a deterministic weighted score, not a
string comparison of headers:

    header 0.35 | column count 0.20 | column boundaries 0.20 | page position 0.25

A signal with no data (`None`) is dropped and the score is renormalized over
the signals that are present; if too little weight is present the answer is
"not enough evidence", never a guess. A few conditions are hard blockers no
score can override (different column count, a heading or real text between the
tables, not the next page, or - when geometry is known - the first table not
reaching the bottom of its page / the second not starting at the top).

All constants live in `MERGE_SCORING` so tests can pin them.
"""

import re
from dataclasses import dataclass, replace

from app.rag.ingestion.canonical_table import (
    EventKind,
    NormalizationEvent,
    RowDisposition,
    SourceRow,
    TableBlock,
    TableRow,
    normalize_for_compare,
)
from app.rag.ingestion.table_evidence import TableEvidence


@dataclass(frozen=True)
class MergeScoring:
    weight_header: float = 0.35
    weight_columns: float = 0.20
    weight_bounds: float = 0.20
    weight_position: float = 0.25
    # A header counts as "repeated on the next page" when at least this share
    # of its columns match; below it the next table is treated as a
    # continuation that did NOT repeat the header (header signal is None).
    header_present_min_match: float = 0.5
    threshold_with_header: float = 0.75
    threshold_without_header: float = 0.80
    borderline_floor: float = 0.65
    min_present_weight: float = 0.50
    # Relative column-boundary tolerance, as a share of the table width.
    bounds_tolerance: float = 0.03
    # First table must end within this share of the page height from the
    # bottom, the second must start within this share from the top.
    position_bottom_fraction: float = 0.25
    position_top_fraction: float = 0.15
    header_confidence_when_repeated: float = 0.9
    # A first table that ends mid-page still continues on the next page when
    # the next page repeats its header this closely and the column borders
    # line up this well (the page was simply broken early); position is then
    # a missing signal instead of a blocker. See docs/specs/known-gaps.md.
    early_break_min_header: float = 0.9
    early_break_min_bounds: float = 0.8
    # A column is the "running text" column - whose tail a page break can leave at
    # the top of the next page - when its cells average at least this many chars.
    continuation_min_mean_length: float = 20.0
    # Without geometry the position/boundary signals are absent; then, when the
    # header was not repeated, how well the next page's first row fits the rows
    # above stands in for them. Not part of the four weights that sum to 1.0.
    weight_shape: float = 0.45
    shape_short_max_length: int = 12
    shape_text_max_length: int = 45


MERGE_SCORING = MergeScoring()

_DIGITS = re.compile(r"\d+")


def furniture_key(line: str) -> str:
    """A line with its digits masked, so `Trang 2/4` and `Trang 3/4` compare
    equal - the shape of a page number/running footer, without naming any
    particular wording."""

    return _DIGITS.sub("#", line.strip())


@dataclass(frozen=True)
class MergeCandidate:
    table: TableBlock
    page_start: int
    page_end: int
    heading_path: list[str]
    evidence: TableEvidence | None


@dataclass(frozen=True)
class MergeDecision:
    merge: bool
    score: float | None
    header_present: bool
    borderline: bool
    reason: str


def _header_match(first: TableBlock, second: TableBlock) -> float:
    first_names = [normalize_for_compare(name) for name in first.header_row or []]
    second_names = [normalize_for_compare(name) for name in second.header_row or []]
    if not first_names or len(first_names) != len(second_names):
        return 0.0
    # a continuation header may add or drop a suffix: `... (Tiếp theo)`,
    # `Tự chọn` for `Tự chọn (chọn 1)`
    same = sum(
        1
        for a, b in zip(first_names, second_names, strict=True)
        if a == b or (a and b and (a.startswith(b) or b.startswith(a)))
    )
    return same / len(first_names)


def _bounds_match(first: TableEvidence, second: TableEvidence, tolerance: float) -> float | None:
    """1 - mean relative deviation of the column boundaries (relative to each
    table's own width), clamped to 0..1; None when the boundary counts differ
    (the columns are not comparable)."""

    def relative(evidence: TableEvidence) -> list[float]:
        left, _top, right, _bottom = evidence.bbox
        width = max(right - left, 1.0)
        return [(bound - left) / width for bound in evidence.column_bounds]

    a, b = relative(first), relative(second)
    if not a or len(a) != len(b):
        return None
    deviation = sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)
    return max(0.0, 1.0 - deviation / tolerance) if tolerance > 0 else 0.0


def _position_ok(first: TableEvidence, second: TableEvidence, scoring: MergeScoring) -> bool:
    bottom_gap = first.page_height - first.bbox[3]
    top_gap = second.bbox[1]
    return (
        bottom_gap <= scoring.position_bottom_fraction * first.page_height
        and top_gap <= scoring.position_top_fraction * second.page_height
    )


def _starts_at_top(second: TableEvidence, scoring: MergeScoring) -> bool:
    return second.bbox[1] <= scoring.position_top_fraction * second.page_height


def _cell_kind(text: str, scoring: MergeScoring) -> int:
    """0 empty, 1 short (a code, a number), 2 a phrase, 3 running text."""

    length = len(text.strip())
    if length == 0:
        return 0
    if length <= scoring.shape_short_max_length:
        return 1
    return 2 if length <= scoring.shape_text_max_length else 3


def _first_row_fit(first: TableBlock, second: TableBlock, scoring: MergeScoring) -> float | None:
    """How well the row the next page opens with (read as a header by the page
    parse, but a data row when the header was not repeated) fits the rows above:
    per column, the kind of cell it holds against the commonest kind in `first`
    (1 same kind, 0.5 neighbouring kind, else 0; an empty cell says nothing).
    A row that only carries the tail of a text cell fits by definition. None
    when there is nothing to compare."""

    header_sources = [s for s in second.source_rows if s.disposition == RowDisposition.HEADER]
    if not header_sources or not first.rows:
        return None
    candidate = header_sources[0].cells
    heavy = _text_heavy_column(first.rows, scoring)
    if _is_cell_continuation(candidate, heavy):
        return 1.0
    fits: list[float] = []
    for column, text in enumerate(candidate):
        kind = _cell_kind(text, scoring)
        if kind == 0:
            continue
        kinds = [
            _cell_kind(row.cells[column], scoring)
            for row in first.rows
            if column < len(row.cells) and row.cells[column].strip()
        ]
        if not kinds:
            continue
        common = max(set(kinds), key=kinds.count)
        fits.append(1.0 if kind == common else 0.5 if abs(kind - common) == 1 else 0.0)
    return sum(fits) / len(fits) if fits else None


def decide_merge(
    first: MergeCandidate,
    second: MergeCandidate,
    *,
    only_furniture_between: bool,
    scoring: MergeScoring = MERGE_SCORING,
) -> MergeDecision:
    if second.page_start != first.page_end + 1:
        return MergeDecision(False, None, False, False, "not_next_page")
    if first.heading_path != second.heading_path:
        return MergeDecision(False, None, False, False, "heading_between")
    if not only_furniture_between:
        return MergeDecision(False, None, False, False, "content_between")
    first_columns = len(first.table.header_row or [])
    if first_columns == 0 or first_columns != len(second.table.header_row or []):
        return MergeDecision(False, None, False, False, "column_count_differs")

    a_ev, b_ev = first.evidence, second.evidence
    position: float | None = None
    bounds: float | None = None
    header_match = _header_match(first.table, second.table)
    if a_ev is not None and b_ev is not None:
        bounds = _bounds_match(a_ev, b_ev, scoring.bounds_tolerance)
        if _position_ok(a_ev, b_ev, scoring):
            position = 1.0
        elif not (
            _starts_at_top(b_ev, scoring)
            and header_match >= scoring.early_break_min_header
            and bounds is not None
            and bounds >= scoring.early_break_min_bounds
        ):
            return MergeDecision(False, None, False, False, "position_not_continuous")
    header_present = header_match >= scoring.header_present_min_match

    signals: list[tuple[float, float]] = [(scoring.weight_columns, 1.0)]
    if header_present:
        signals.append((scoring.weight_header, header_match))
    if bounds is not None:
        signals.append((scoring.weight_bounds, bounds))
    if position is not None:
        signals.append((scoring.weight_position, position))
    if not header_present and position is None:
        shape = _first_row_fit(first.table, second.table, scoring)
        if shape is not None:
            signals.append((scoring.weight_shape, shape))

    present = sum(weight for weight, _ in signals)
    if present < scoring.min_present_weight:
        return MergeDecision(False, None, header_present, False, "not_enough_evidence")
    score = sum(weight * value for weight, value in signals) / present
    threshold = (
        scoring.threshold_with_header if header_present else scoring.threshold_without_header
    )
    if score >= threshold:
        return MergeDecision(True, score, header_present, False, "merged")
    borderline = scoring.borderline_floor <= score < threshold
    return MergeDecision(False, score, header_present, borderline, "below_threshold")


@dataclass
class _Incoming:
    """A row of the continuation page waiting to be appended, with the keys
    that identify it in the source ledger (so ledger ids can be remapped)."""

    row: TableRow
    keys: list[tuple[str, object]]


def _text_heavy_column(rows: list[TableRow], scoring: MergeScoring) -> int | None:
    """The column that holds the running text (highest mean cell length), if
    any column is text-heavy enough to be one."""

    if not rows:
        return None
    width = len(rows[0].cells)
    means = [
        sum(len(row.cells[column]) for row in rows if column < len(row.cells)) / len(rows)
        for column in range(width)
    ]
    best = max(range(width), key=lambda column: means[column])
    return best if means[best] >= scoring.continuation_min_mean_length else None


def _is_cell_continuation(cells: list[str], heavy: int | None) -> bool:
    """A row that fills only the text-heavy column: what is left of a cell whose
    row was cut by the page break, not a row of its own."""

    if heavy is None or heavy >= len(cells) or not cells[heavy].strip():
        return False
    return all(not cell.strip() for column, cell in enumerate(cells) if column != heavy)


def merge_page_tables(
    first: TableBlock,
    second: TableBlock,
    *,
    header_present: bool,
    scoring: MergeScoring = MERGE_SCORING,
) -> TableBlock:
    """Concatenate `second` under `first` as one logical table.

    - When `second` repeated the header, its header source rows become
      `repeated_header` (with a `drop_repeated_header` event). When it did not,
      the rows the page-level parse took for a header are data: they are
      restored as data rows (one per logical row) so no content is lost.
    - Rows at the top of `second` that only carry the tail of the previous
      row's text cell (the row was cut by the page break) are appended to that
      row instead of standing as rows of their own.
    """

    first_rows = [
        replace(
            row,
            cells=list(row.cells),
            warnings=list(row.warnings),
            source_row_ids=list(row.source_row_ids),
        )
        for row in first.rows
    ]
    first_sources = [
        replace(source, canonical_row_ids=list(source.canonical_row_ids))
        for source in first.source_rows
    ]
    second_sources = [
        replace(source, canonical_row_ids=list(source.canonical_row_ids))
        for source in second.source_rows
    ]
    header_sources = [s for s in second_sources if s.disposition == RowDisposition.HEADER]
    events = [*first.normalization_events, *second.normalization_events]
    header_confidence = first.header_confidence
    offset = len(first_rows)

    incoming: list[_Incoming] = []
    lead_ids: set[str] = set()
    if header_present:
        for source in header_sources:
            source.disposition = RowDisposition.REPEATED_HEADER
        events.append(
            NormalizationEvent(
                EventKind.DROP_REPEATED_HEADER,
                [source.source_row_id for source in header_sources],
                "header repeated on the continuation page",
            )
        )
        header_confidence = max(header_confidence, scoring.header_confidence_when_repeated)
    else:
        by_logical: dict[object, list[SourceRow]] = {}
        for source in header_sources:
            group_key = (
                source.logical_row if source.logical_row is not None else source.source_row_id
            )
            by_logical.setdefault(group_key, []).append(source)
        for group in by_logical.values():
            if not any(normalize_for_compare(source.raw_text) for source in group):
                for source in group:
                    source.disposition = RowDisposition.DROPPED_EMPTY  # a blank "header"
                continue
            for source in group:
                source.disposition = (
                    RowDisposition.MERGED if len(group) > 1 else RowDisposition.DATA
                )
                lead_ids.add(source.source_row_id)
            incoming.append(
                _Incoming(
                    TableRow(
                        cells=list(group[0].cells),
                        raw_text=" ".join(source.raw_text for source in group),
                        row_index=0,
                        page_start=group[0].page,
                        page_end=group[0].page,
                        confidence=0.8,
                        warnings=["headerless_continuation"],
                        source_row_ids=[source.source_row_id for source in group],
                        source_cell_count=sum(source.source_cell_count for source in group),
                        canonical_cell_count=len(group[0].cells),
                    ),
                    [("lead", source.source_row_id) for source in group],
                )
            )
    for row in second.rows:
        incoming.append(
            _Incoming(
                replace(
                    row,
                    cells=list(row.cells),
                    warnings=list(row.warnings),
                    source_row_ids=list(row.source_row_ids),
                ),
                [("row", row.row_index)],
            )
        )

    key_to_index: dict[tuple[str, object], int] = {}
    heavy = _text_heavy_column(first_rows, scoring)
    while incoming and first_rows and _is_cell_continuation(incoming[0].row.cells, heavy):
        item = incoming.pop(0)
        target = first_rows[-1]
        assert heavy is not None
        target.cells[heavy] = f"{target.cells[heavy]} {item.row.cells[heavy]}".strip()
        target.raw_text = f"{target.raw_text} {item.row.raw_text}"
        target.page_end = item.row.page_end
        target.source_cell_count += item.row.source_cell_count
        target.confidence = min(target.confidence, 0.8)
        if "continued_across_pages" not in target.warnings:
            target.warnings.append("continued_across_pages")
        events.append(
            NormalizationEvent(
                EventKind.MERGE_CELLS,
                [*target.source_row_ids, *item.row.source_row_ids],
                "text cell cut by the page break continued into the previous row",
            )
        )
        target.source_row_ids.extend(item.row.source_row_ids)
        for key in item.keys:
            key_to_index[key] = target.row_index
        for source in [*first_sources, *second_sources]:
            if source.source_row_id in target.source_row_ids:
                source.disposition = RowDisposition.MERGED

    for position, item in enumerate(incoming, start=1):
        item.row.row_index = offset + position
        for key in item.keys:
            key_to_index[key] = item.row.row_index

    for source in second_sources:
        if source.source_row_id in lead_ids:
            source.canonical_row_ids = [key_to_index[("lead", source.source_row_id)]]
        else:
            source.canonical_row_ids = [
                key_to_index[("row", old)] for old in source.canonical_row_ids
            ]

    rows = [*first_rows, *[item.row for item in incoming]]
    return TableBlock(
        header_row=first.header_row,
        data_rows=[row.cells for row in rows],
        header_source=first.header_source,
        header_confidence=header_confidence,
        table_id=first.table_id,
        header_levels=first.header_levels,
        rows=rows,
        source_rows=[*first_sources, *second_sources],
        normalization_events=events,
        warnings=[*first.warnings, *[w for w in second.warnings if w not in first.warnings]],
    )
