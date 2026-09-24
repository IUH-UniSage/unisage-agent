"""Canonical table model shared by every table source (PDF/DOCX/HTML).

`TableBlock` is the one table abstraction of the ingestion pipeline. Besides
the flat `header_row`/`data_rows` view that the chunkers and older tests use,
it carries per-row provenance (page, row index, ancestors, confidence,
warnings) and a *source ledger*: one `SourceRow` for every row read from the
source before any normalization. The ledger is what lets
`check_table_invariants` prove that no source content was lost, instead of
only checking canonical data that was already produced.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from app.schemas.ingestion import HeaderSource


class RowDisposition(StrEnum):
    """What became of one source row after normalization."""

    DATA = "data"  # became exactly one canonical data row
    HEADER = "header"  # became (part of) a header level
    REPEATED_HEADER = "repeated_header"  # a header repeated on a later page, dropped
    MERGED = "merged"  # combined with other source rows into one canonical row
    SPLIT = "split"  # produced more than one canonical row
    DROPPED_EMPTY = "dropped_empty"  # had no content (blank/separator only)
    # text outside the table the extractor read as a table row; it is emitted
    # as running text next to the table (`TableBlock.text_before`)
    MOVED_TO_TEXT = "moved_to_text"


class EventKind(StrEnum):
    SPLIT_CELL = "split_cell"
    MERGE_CELLS = "merge_cells"
    FILL_MERGED_HEADER = "fill_merged_header"
    FILL_MERGED_CELL = "fill_merged_cell"
    DROP_REPEATED_HEADER = "drop_repeated_header"
    RECOVER_TEXT = "recover_text"
    PAD_MISSING_CELL = "pad_missing_cell"
    REORDER = "reorder"


# Events that legitimately change a row's cell count or cell text.
_CELL_CHANGING_EVENTS = frozenset(
    {
        EventKind.SPLIT_CELL,
        EventKind.MERGE_CELLS,
        EventKind.PAD_MISSING_CELL,
        EventKind.RECOVER_TEXT,
    }
)


@dataclass
class SourceRow:
    """One row exactly as read from the source, before normalization."""

    source_row_id: str
    page: int | None
    raw_text: str
    source_cell_count: int
    disposition: RowDisposition = RowDisposition.DATA
    canonical_row_ids: list[int] = field(default_factory=list)
    # The cells this row ended up with after cell-level normalization (before
    # any header/data decision). Lets a header row be re-read as data when the
    # table turns out to continue on a page without repeating its header.
    cells: list[str] = field(default_factory=list)
    # Position of the logical row (one geometry row) this source row belongs to;
    # several markdown rows share it when a wrapped cell was split over lines.
    logical_row: int | None = None


@dataclass
class NormalizationEvent:
    kind: EventKind
    source_row_ids: list[str]
    detail: str = ""


@dataclass
class RowSignals:
    """Per-cell geometry of one row, read from PDF evidence (absent for
    HTML/DOCX and for pages whose evidence did not align). Consumed only by
    hierarchy inference."""

    cell_x0: list[float | None]  # left edge of each cell's first text span
    cell_bold: list[bool]
    cell_size: list[float | None]
    cell_merged: list[bool]  # True where the cell is covered by a merged cell
    height: float = 0.0
    # Left edge of each cell. Indentation is the text's offset FROM this edge:
    # an absolute x would read a column that merely sits further right on part
    # of a page as a deeper level. Empty for hand-built rows (offset = x0).
    cell_left: list[float | None] = field(default_factory=list)


@dataclass
class TableRow:
    """One canonical data row with its provenance."""

    cells: list[str]
    raw_text: str = ""
    row_index: int = 0  # 1-based within the logical table, header not counted
    page_start: int | None = None
    page_end: int | None = None
    ancestors: list[str] = field(default_factory=list)
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    source_row_ids: list[str] = field(default_factory=list)
    source_cell_count: int = 0
    canonical_cell_count: int = 0
    signals: RowSignals | None = None
    # Columns whose text was copied down from a cell merged over several rows
    # (the source only holds it once, on the first row of the span).
    inherited_cells: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class TableBlock:
    """A table parsed directly from its source format - the canonical table.

    `header_row` is the flattened column names (one per column) and
    `data_rows` the plain cells of `rows`; both are kept as first-class fields
    so existing consumers keep working. When `rows` is empty (a hand-built
    table) `canonical_rows()` synthesizes rows from `data_rows`. Builders that
    populate `rows` must keep `data_rows` equal to `[r.cells for r in rows]`.
    """

    header_row: list[str] | None
    data_rows: list[list[str]]
    header_source: HeaderSource
    header_confidence: float
    table_id: str | None = None
    header_levels: list[list[str]] = field(default_factory=list)
    rows: list[TableRow] = field(default_factory=list)
    source_rows: list[SourceRow] = field(default_factory=list)
    normalization_events: list[NormalizationEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Text outside the table that the extractor had glued into its rows; the
    # parser emits it as running text right after the table.
    spilled_text: list[str] = field(default_factory=list)
    # Same, for text above the table read into its first rows.
    text_before: list[str] = field(default_factory=list)

    @property
    def column_names(self) -> list[str] | None:
        return self.header_row

    def canonical_rows(self) -> list[TableRow]:
        if self.rows:
            return self.rows
        return [
            TableRow(
                cells=list(cells),
                raw_text=" | ".join(cells),
                row_index=index,
                source_cell_count=len(cells),
                canonical_cell_count=len(cells),
            )
            for index, cells in enumerate(self.data_rows, start=1)
        ]


def build_table_from_plain_rows(
    header_row: list[str] | None,
    data_rows: list[list[str]],
    header_source: HeaderSource,
    header_confidence: float,
    *,
    page: int | None = None,
) -> TableBlock:
    """Build a canonical table from an already-structured source (HTML/DOCX,
    or a PDF pipe table that needed no normalization). The header row, when
    present, is source row 1; each data row follows in order. Nothing is
    normalized, so no events are emitted and the invariants hold trivially -
    they still run over this ledger in tests to keep that true."""

    source_rows: list[SourceRow] = []
    rows: list[TableRow] = []
    position = 0

    def source_id() -> str:
        nonlocal position
        position += 1
        prefix = f"p{page}" if page is not None else "p0"
        return f"{prefix}-r{position}"

    header_levels: list[list[str]] = []
    if header_row is not None:
        header_levels = [list(header_row)]
        source_rows.append(
            SourceRow(
                source_row_id=source_id(),
                page=page,
                raw_text=" | ".join(header_row),
                source_cell_count=len(header_row),
                disposition=RowDisposition.HEADER,
            )
        )

    for row_index, cells in enumerate(data_rows, start=1):
        sid = source_id()
        raw = " | ".join(cells)
        source_rows.append(
            SourceRow(
                source_row_id=sid,
                page=page,
                raw_text=raw,
                source_cell_count=len(cells),
                disposition=RowDisposition.DATA,
                canonical_row_ids=[row_index],
            )
        )
        rows.append(
            TableRow(
                cells=list(cells),
                raw_text=raw,
                row_index=row_index,
                page_start=page,
                page_end=page,
                source_row_ids=[sid],
                source_cell_count=len(cells),
                canonical_cell_count=len(cells),
            )
        )

    return TableBlock(
        header_row=header_row,
        data_rows=[list(cells) for cells in data_rows],
        header_source=header_source,
        header_confidence=header_confidence,
        header_levels=header_levels,
        rows=rows,
        source_rows=source_rows,
    )


_NON_ALNUM = re.compile(r"[\W_]+", re.UNICODE)
# Inline markup the markdown extractor wraps around text; it is not content.
_MARKUP_TAG = re.compile(r"</?(?:br|sup|sub|b|i|u|em|strong)\s*/?>", re.IGNORECASE)


def normalize_for_compare(text: str) -> str:
    """Reduce text to its alphanumeric characters (NFC, casefolded) so that a
    source row and its canonical cells compare equal despite whitespace,
    `<br>`, `~~` strike markers, pipes or a cell split/merge - while any lost
    or invented character still shows up."""

    without_tags = _MARKUP_TAG.sub("", text)
    return _NON_ALNUM.sub("", unicodedata.normalize("NFC", without_tags)).casefold()


def _is_blank(raw_text: str) -> bool:
    return not normalize_for_compare(raw_text)


def check_table_invariants(table: TableBlock) -> list[str]:
    """Return every violated invariant as a message (empty list = all hold).

    - I1 row conservation: every non-blank source row has a consistent
      disposition; canonical rows and the ledger reference each other.
    - I2: no canonical row descends from a header/repeated-header source row.
    - I3 cell conservation: a row's characters and cell count match its
      source unless a normalization event explains the difference.
    - I4: `row_index` contiguous from 1; `page_start <= page_end`; pages never
      decrease down the table.
    (I5 - never silently drop - is what I1 enforces: an unresolved row must
    still be present as a canonical row, there is no quarantine.)
    """

    violations: list[str] = []
    rows = table.canonical_rows()
    ledger = {source.source_row_id: source for source in table.source_rows}
    row_by_index = {row.row_index: row for row in rows}

    # I4
    if [row.row_index for row in rows] != list(range(1, len(rows) + 1)):
        violations.append("I4: row_index is not contiguous 1..N")
    last_page: int | None = None
    for row in rows:
        if (
            row.page_start is not None
            and row.page_end is not None
            and row.page_start > row.page_end
        ):
            violations.append(f"I4: row {row.row_index} has page_start > page_end")
        if row.page_start is not None:
            if last_page is not None and row.page_start < last_page:
                violations.append(f"I4: row {row.row_index} goes back to an earlier page")
            last_page = row.page_end if row.page_end is not None else row.page_start

    if not table.source_rows:
        return violations  # hand-built table without a ledger: only canonical checks

    dropped_header_ids = {
        source_id
        for event in table.normalization_events
        if event.kind == EventKind.DROP_REPEATED_HEADER
        for source_id in event.source_row_ids
    }
    events_by_source: dict[str, set[EventKind]] = {}
    for event in table.normalization_events:
        for source_id in event.source_row_ids:
            events_by_source.setdefault(source_id, set()).add(event.kind)

    # I1: per source row
    header_rows: set[object] = set()
    for source in table.source_rows:
        blank = _is_blank(source.raw_text)
        disposition = source.disposition
        if blank:
            continue
        if disposition == RowDisposition.DROPPED_EMPTY:
            violations.append(f"I1: non-blank source row {source.source_row_id} was dropped")
        elif disposition in (
            RowDisposition.DATA,
            RowDisposition.MERGED,
            RowDisposition.SPLIT,
        ):
            if not source.canonical_row_ids:
                violations.append(
                    f"I1: source row {source.source_row_id} is {disposition} "
                    "but produced no canonical row"
                )
            for canonical_id in source.canonical_row_ids:
                canonical = row_by_index.get(canonical_id)
                if canonical is None:
                    violations.append(
                        f"I1: source row {source.source_row_id} points to missing "
                        f"canonical row {canonical_id}"
                    )
                elif source.source_row_id not in canonical.source_row_ids:
                    violations.append(
                        f"I1: canonical row {canonical_id} does not list source row "
                        f"{source.source_row_id}"
                    )
            if disposition == RowDisposition.SPLIT and len(source.canonical_row_ids) < 2:
                violations.append(f"I1: source row {source.source_row_id} is split into <2 rows")
        elif disposition == RowDisposition.HEADER:
            # several markdown rows may make up one header tier
            header_rows.add(
                (source.page, source.logical_row)
                if source.logical_row is not None
                else source.source_row_id
            )
        elif disposition == RowDisposition.REPEATED_HEADER:
            if source.source_row_id not in dropped_header_ids:
                violations.append(
                    f"I1: repeated header {source.source_row_id} has no drop_repeated_header event"
                )

    if len(header_rows) != len(table.header_levels):
        violations.append(
            f"I1: {len(header_rows)} header rows but {len(table.header_levels)} header levels"
        )

    referenced_rows: set[int] = set()
    for source in table.source_rows:
        referenced_rows.update(source.canonical_row_ids)
    unreferenced = [row.row_index for row in rows if row.row_index not in referenced_rows]
    if unreferenced:
        violations.append(f"I1: canonical rows {unreferenced} have no source row")

    # I2 + I3: per canonical row
    for row in rows:
        source_texts: list[str] = []
        explained: set[EventKind] = set()
        for source_id in row.source_row_ids:
            cited = ledger.get(source_id)
            if cited is None:
                violations.append(f"I1: row {row.row_index} cites unknown source row {source_id}")
                continue
            if cited.disposition in (RowDisposition.HEADER, RowDisposition.REPEATED_HEADER):
                violations.append(f"I2: row {row.row_index} descends from a header row {source_id}")
            source_texts.append(cited.raw_text)
            explained |= events_by_source.get(source_id, set())

        if row.canonical_cell_count != len(row.cells):
            violations.append(f"I3: row {row.row_index} canonical_cell_count != len(cells)")

        cell_changing = explained & _CELL_CHANGING_EVENTS
        source_cells = sum(
            ledger[source_id].source_cell_count
            for source_id in row.source_row_ids
            if source_id in ledger
        )
        if len(row.source_row_ids) == 1 and len(row.cells) != source_cells and not cell_changing:
            violations.append(
                f"I3: row {row.row_index} cell count {source_cells} -> {len(row.cells)} "
                "with no normalization event"
            )
        if EventKind.RECOVER_TEXT in explained:
            continue
        expected = Counter(normalize_for_compare(" ".join(source_texts)))
        # text copied down from a merged cell above is not this row's source
        own_cells = [
            cell for column, cell in enumerate(row.cells) if column not in row.inherited_cells
        ]
        actual = Counter(normalize_for_compare(" ".join(own_cells)))
        if expected != actual:
            lost = expected - actual
            invented = actual - expected
            violations.append(
                f"I3: row {row.row_index} content differs from source "
                f"(lost={dict(lost)}, invented={dict(invented)})"
            )

    return violations
