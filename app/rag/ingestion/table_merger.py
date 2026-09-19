"""Decide whether a table on one page continues the table on the previous page
and, if so, merge the two into one logical table.

The decision is a deterministic weighted score (see AD4/AD9 in the plan), not a
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
    same = sum(1 for a, b in zip(first_names, second_names, strict=True) if a == b)
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
    if a_ev is not None and b_ev is not None:
        if not _position_ok(a_ev, b_ev, scoring):
            return MergeDecision(False, None, False, False, "position_not_continuous")
        position = 1.0
        bounds = _bounds_match(a_ev, b_ev, scoring.bounds_tolerance)

    header_match = _header_match(first.table, second.table)
    header_present = header_match >= scoring.header_present_min_match

    signals: list[tuple[float, float]] = [(scoring.weight_columns, 1.0)]
    if header_present:
        signals.append((scoring.weight_header, header_match))
    if bounds is not None:
        signals.append((scoring.weight_bounds, bounds))
    if position is not None:
        signals.append((scoring.weight_position, position))

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


def merge_page_tables(
    first: TableBlock,
    second: TableBlock,
    *,
    header_present: bool,
    scoring: MergeScoring = MERGE_SCORING,
) -> TableBlock:
    """Concatenate `second` under `first` as one logical table.

    When `second` repeated the header, its header source rows become
    `repeated_header` (with a `drop_repeated_header` event). When it did not,
    the rows the page-level parse took for a header are data: they are
    restored as data rows so no content is lost.
    """

    offset = len(first.rows)
    second_sources = [
        replace(source, canonical_row_ids=list(source.canonical_row_ids))
        for source in second.source_rows
    ]
    header_sources = [s for s in second_sources if s.disposition == RowDisposition.HEADER]
    events = [*first.normalization_events, *second.normalization_events]
    header_confidence = first.header_confidence

    lead_rows: list[TableRow] = []
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
        for position, source in enumerate(header_sources, start=1):
            source.disposition = RowDisposition.DATA
            source.canonical_row_ids = [offset + position]
            lead_rows.append(
                TableRow(
                    cells=list(source.cells),
                    raw_text=source.raw_text,
                    row_index=offset + position,
                    page_start=source.page,
                    page_end=source.page,
                    confidence=0.8,
                    warnings=["headerless_continuation"],
                    source_row_ids=[source.source_row_id],
                    source_cell_count=source.source_cell_count,
                    canonical_cell_count=len(source.cells),
                )
            )

    shift = offset + len(lead_rows)
    lead_ids = {row.source_row_ids[0] for row in lead_rows}
    for source in second_sources:
        if source.source_row_id not in lead_ids:
            source.canonical_row_ids = [row_id + shift for row_id in source.canonical_row_ids]
    shifted_rows = [replace(row, row_index=row.row_index + shift) for row in second.rows]

    rows = [*first.rows, *lead_rows, *shifted_rows]
    return TableBlock(
        header_row=first.header_row,
        data_rows=[row.cells for row in rows],
        header_source=first.header_source,
        header_confidence=header_confidence,
        table_id=first.table_id,
        header_levels=first.header_levels,
        rows=rows,
        source_rows=[*first.source_rows, *second_sources],
        normalization_events=events,
        warnings=[*first.warnings, *[w for w in second.warnings if w not in first.warnings]],
    )
