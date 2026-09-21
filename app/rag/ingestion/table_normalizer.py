"""Normalize one page's markdown pipe table (plus optional geometric evidence)
into a canonical `TableBlock`.

Two sources are combined, each for what it is good at:

- the markdown lines are the *source ledger* (`SourceRow.raw_text`), always
  written first and never discarded;
- the geometric evidence (`table_evidence`), when it aligns row-for-row with
  the markdown, supplies clean cell text, the true column count and the
  header tier count.

Anything the normalizer changes (a cell split or merged, a merged header
cell filled across its columns, a missing cell padded) is recorded as a
`NormalizationEvent` so `check_table_invariants` can prove nothing was lost.
Nothing here depends on the wording of a document: only cell geometry, cell
counts and whether a cell is empty.
"""

import re
import unicodedata
from collections import Counter

from app.rag.ingestion.canonical_table import (
    EventKind,
    NormalizationEvent,
    RowDisposition,
    RowSignals,
    SourceRow,
    TableBlock,
    TableRow,
    build_table_from_plain_rows,
    normalize_for_compare,
)
from app.rag.ingestion.table_evidence import EvidenceRow, TableEvidence
from app.schemas.ingestion import HeaderSource

# A separator line has at least one dash; a bare `|||` is an empty data row and
# must reach the ledger (as dropped_empty) rather than vanish.
MARKDOWN_TABLE_SEPARATOR = re.compile(r"^[\s|:]*-[\s|:-]*$")
_NUMERIC_CELL = re.compile(r"^[\d.,\s%+-]+$")
_STRIKE_MARK = re.compile(r"~~")

# Header confidence: geometry that lines up with the markdown is a stronger
# structural signal than the bare "first row is the header" assumption.
_HEADER_CONFIDENCE_MARKDOWN_ONLY = 0.6
_HEADER_CONFIDENCE_WITH_EVIDENCE = 0.75

# Row confidence levels (see AD5/AD9 in the plan; hierarchy scoring later
# multiplies into these, it never raises a row above what is set here).
_CONFIDENCE_OK = 1.0
_CONFIDENCE_TEXT_MISMATCH = 0.7
_CONFIDENCE_PADDED = 0.8
_CONFIDENCE_GARBLED = 0.3


def markdown_row_cells(line: str) -> list[str]:
    """Split a GFM pipe-table row into cells.

    Strips exactly the single leading/trailing `|` delimiter a pipe-table
    row is framed in - NOT `str.strip("|")`, which removes an unbounded run
    of `|` characters from each end. A row with a genuinely empty first or
    last cell renders as a *double* pipe at that edge (`"||content||"`,
    empty cell + delimiter); `strip("|")` collapses both away and silently
    drops that cell, desyncing the row's cell count from the header's.
    """

    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [cell.strip() for cell in trimmed.split("|")]


_MARKUP_TAG = re.compile(r"</?(?:sup|sub|b|i|u|em|strong)>", re.IGNORECASE)
_WHOLE_EMPHASIS = re.compile(r"^\s*(?:_+|\*+)(.+?)(?:_+|\*+)\s*$", re.DOTALL)


def strip_markup(cell: str) -> str:
    """Drop the emphasis the markdown extractor wraps around text (`**bold**`,
    `_italic_`, `~~strike~~`, `<sup>`): it is formatting, not content, and it
    would otherwise leak into column names and chunk text (`**STT**`). The text is
    also put in NFC: PDFs often store Vietnamese with combining marks, which
    would make the same word compare unequal and split it in retrieval. A `<br>`
    is kept - it is a line break inside the cell. The source ledger keeps the
    raw text, so nothing is lost."""

    text = unicodedata.normalize("NFC", cell)
    text = _MARKUP_TAG.sub("", text).replace("**", "").replace("__", "").replace("~~", "")
    whole = _WHOLE_EMPHASIS.match(text)
    if whole:
        text = whole.group(1)
    return text.strip()


def _char_multiset(text: str) -> Counter[str]:
    return Counter(normalize_for_compare(text))


def _looks_garbled(raw_line: str, cells: list[str]) -> bool:
    """Text the PDF itself failed to encode (strike-through markers from the
    extractor, or a cell that is only a scatter of 1-2 letter fragments)."""

    if _STRIKE_MARK.search(raw_line):
        return True
    for cell in cells:
        tokens = re.split(r"[\s]+|<br>", cell.strip())
        tokens = [token for token in tokens if token]
        if (
            len(tokens) >= 3
            and all(len(token) <= 2 for token in tokens)
            and not any(char.isdigit() for char in cell)
        ):
            return True
    return False


def _split_numeric_fragments(tier: list[str]) -> tuple[list[str], list[int]]:
    """A merged header cell cut at the wrong place by the markdown extractor
    (`"Năm học 2" | "025 - 2026"`): two adjacent header cells where the left
    ends with a digit and the right starts with one are one cell. Returns the
    repaired tier and the indexes of cells that were swallowed by a merge."""

    repaired = list(tier)
    swallowed: list[int] = []
    for index in range(len(repaired) - 1):
        left, right = repaired[index], repaired[index + 1]
        if left and right and left[-1].isdigit() and right[0].isdigit():
            repaired[index] = left + right
            repaired[index + 1] = ""
            swallowed.append(index + 1)
    return repaired, swallowed


def _is_header_continuation(first: list[str], second: list[str], body: list[list[str]]) -> bool:
    """Markdown-only detection of a 2nd header tier: the row right under the
    header has an empty cell in the table's label column (the column that
    holds the most text in the body - a vertically merged header cell there),
    still carries some text, and holds no pure-number cell (data rows do)."""

    if len(second) != len(first) or not body:
        return False
    column_count = len(first)
    totals = [
        sum(len(row[index]) for row in body if index < len(row)) for index in range(column_count)
    ]
    label_column = max(range(column_count), key=lambda index: totals[index])
    non_empty = [cell for cell in second if cell]
    if not non_empty or second[label_column]:
        return False
    if any(_NUMERIC_CELL.match(cell) for cell in non_empty):
        return False
    return any(any(char.isalpha() for char in cell) for cell in non_empty)


def _fill_header_tiers(tiers: list[list[str]]) -> list[list[str]]:
    """Fill the cells a merged header cell covers, from the tier text alone:

    - a cell empty in tier k but non-empty in tier k+1 is covered by a
      group cell to its left (horizontal span) - it takes that text;
    - a cell empty in tier k+1 is covered by the cell above it (vertical
      span) - the flattening step de-duplicates it.
    """

    filled = [list(tier) for tier in tiers]
    for tier_index in range(len(filled) - 1):
        below = filled[tier_index + 1]
        tier = filled[tier_index]
        for column in range(len(tier)):
            if tier[column] or not below[column]:
                continue
            for left in range(column - 1, -1, -1):
                if tier[left]:
                    tier[column] = tier[left]
                    break
    return filled


def _flatten_header(filled_tiers: list[list[str]]) -> list[str]:
    column_count = len(filled_tiers[0])
    names: list[str] = []
    for column in range(column_count):
        parts: list[str] = []
        for tier in filled_tiers:
            value = tier[column].strip()
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        names.append(" > ".join(parts) if parts else f"col_{column + 1}")
    return names


def _row_signals(row: EvidenceRow) -> RowSignals:
    cells = row.cells
    return RowSignals(
        cell_x0=[cell.text_x0 if cell is not None else None for cell in cells],
        cell_bold=[bool(cell.bold) if cell is not None else False for cell in cells],
        cell_size=[cell.font_size if cell is not None else None for cell in cells],
        cell_merged=[cell is None for cell in cells],
        height=max(row.y1 - row.y0, 0.0),
        cell_left=[cell.bbox[0] if cell is not None else None for cell in cells],
    )


def _partition_event(md_cells: list[str], new_cells: list[str]) -> EventKind | None:
    """How the cell partition of a row changed between the markdown and the
    cells actually used (None when the cells are the same)."""

    if [normalize_for_compare(c) for c in md_cells] == [
        normalize_for_compare(c) for c in new_cells
    ]:
        return None
    if len(new_cells) > len(md_cells):
        return EventKind.SPLIT_CELL
    return EventKind.MERGE_CELLS


def _align_rows(md_cells: list[list[str]], evidence: TableEvidence) -> list[list[int]] | None:
    """For each evidence row, the markdown rows that carry its text, in order.

    The extractor often breaks a cell that wraps over several lines into extra
    markdown rows (`| | | | continuation text |`) while the geometry sees ONE
    row. Consecutive markdown rows are consumed until their characters equal
    the evidence row's; a markdown row that would add a character the evidence
    row does not have, or text left over at the end, means the two do not
    describe the same table and there is no alignment (None)."""

    md_chars = [_char_multiset(" ".join(cells)) for cells in md_cells]
    count = len(md_cells)
    groups: list[list[int]] = []
    position = 0
    for row in evidence.rows:
        target = _char_multiset(" ".join(cell.text for cell in row.cells if cell is not None))
        group: list[int] = []
        accumulated: Counter[str] = Counter()
        while accumulated != target:
            while position < count and not md_chars[position] and not group:
                position += 1  # blank markdown row between logical rows
            if position >= count:
                return None
            candidate = accumulated + md_chars[position]
            if candidate - target:  # the row brings text the evidence row lacks
                return None
            accumulated = candidate
            group.append(position)
            position += 1
        groups.append(group)
    if any(md_chars[index] for index in range(position, count)):
        return None
    return groups


def build_page_table(lines: list[str], page: int, evidence: TableEvidence | None) -> TableBlock:
    """Build the canonical table for one page's contiguous markdown table
    lines.

    `evidence`, when given, is used only if it aligns with the markdown - row
    for row when the counts match, or by merging consecutive markdown rows into
    one geometry row (see `_align_rows`); otherwise the table falls back to
    markdown alone with an `evidence_unaligned` warning. Work is done per
    LOGICAL row: one evidence row and the markdown row(s) it came from."""

    md_lines = [
        line.strip()
        for line in lines
        if line.strip() and not MARKDOWN_TABLE_SEPARATOR.match(line.strip())
    ]
    if not md_lines:
        return build_table_from_plain_rows(None, [], HeaderSource.MISSING, 0.0)

    md_cells = [[strip_markup(cell) for cell in markdown_row_cells(line)] for line in md_lines]
    source_rows = [
        SourceRow(f"p{page}-r{index + 1}", page, line, len(cells))
        for index, (line, cells) in enumerate(zip(md_lines, md_cells, strict=True))
    ]
    warnings: list[str] = []
    events: list[NormalizationEvent] = []

    groups: list[list[int]] | None = None
    if evidence is not None:
        if len(evidence.rows) == len(md_lines):
            groups = [[index] for index in range(len(md_lines))]
        else:
            groups = _align_rows(md_cells, evidence)
        if groups is None:
            warnings.append("evidence_unaligned")
    aligned = groups is not None and evidence is not None
    logical = groups if groups is not None else [[index] for index in range(len(md_lines))]
    column_count = evidence.column_count if aligned and evidence else len(md_cells[0])

    row_cells: list[list[str]] = []
    row_signals: list[RowSignals | None] = []
    row_warnings: list[list[str]] = []
    row_confidence: list[float] = []
    for position, group in enumerate(logical):
        sids = [source_rows[index].source_row_id for index in group]
        md_text = " ".join(" ".join(md_cells[index]) for index in group)
        used = list(md_cells[group[0]]) if group else []
        row_warn: list[str] = []
        confidence = _CONFIDENCE_OK
        if aligned and evidence is not None:
            geometry = [
                cell.text if cell is not None else "" for cell in evidence.rows[position].cells
            ]
            if _char_multiset(" ".join(geometry)) == _char_multiset(md_text):
                used = [strip_markup(cell) for cell in geometry]
                if len(group) > 1:
                    events.append(
                        NormalizationEvent(
                            EventKind.MERGE_CELLS,
                            sids,
                            f"{len(group)} markdown rows are one geometry row (wrapped cell)",
                        )
                    )
                elif group:
                    kind = _partition_event(md_cells[group[0]], geometry)
                    if kind is not None:
                        events.append(
                            NormalizationEvent(
                                kind,
                                sids,
                                f"markdown {len(md_cells[group[0]])} cells -> "
                                f"geometry {len(geometry)}",
                            )
                        )
            else:
                row_warn.append("geometry_text_mismatch")
                confidence = _CONFIDENCE_TEXT_MISMATCH
        if len(used) < column_count:
            events.append(
                NormalizationEvent(
                    EventKind.PAD_MISSING_CELL,
                    sids,
                    f"padded {column_count - len(used)} missing cell(s)",
                )
            )
            used = used + [""] * (column_count - len(used))
            row_warn.append("padded_cells")
            confidence = min(confidence, _CONFIDENCE_PADDED)
        elif len(used) > column_count:
            overflow = " ".join(cell for cell in used[column_count - 1 :] if cell)
            used = [*used[: column_count - 1], overflow]
            events.append(
                NormalizationEvent(
                    EventKind.MERGE_CELLS, sids, "extra cells merged into the last column"
                )
            )
            row_warn.append("merged_extra_cells")
            confidence = min(confidence, _CONFIDENCE_PADDED)
        if _looks_garbled(" ".join(md_lines[index] for index in group), used):
            row_warn.append("garbled_text_raw_kept")
            confidence = min(confidence, _CONFIDENCE_GARBLED)
        row_cells.append(used)
        row_signals.append(
            _row_signals(evidence.rows[position]) if aligned and evidence is not None else None
        )
        row_warnings.append(row_warn)
        row_confidence.append(confidence)

    if aligned and evidence is not None:
        tier_count = max(1, min(evidence.header_row_count, len(logical)))
    else:
        tier_count = 1
        if len(row_cells) >= 3 and _is_header_continuation(
            row_cells[0], row_cells[1], row_cells[2:]
        ):
            tier_count = 2

    def group_ids(position: int) -> list[str]:
        return [source_rows[index].source_row_id for index in logical[position]]

    tiers = [list(row_cells[position]) for position in range(tier_count)]
    if not aligned and tier_count >= 2:
        tiers[0], swallowed = _split_numeric_fragments(tiers[0])
        if swallowed:
            events.append(
                NormalizationEvent(
                    EventKind.MERGE_CELLS,
                    group_ids(0),
                    "rejoined a header cell cut by the extractor",
                )
            )
    filled = _fill_header_tiers(tiers)
    if filled != tiers:
        events.append(
            NormalizationEvent(
                EventKind.FILL_MERGED_HEADER,
                [sid for position in range(tier_count) for sid in group_ids(position)],
                "merged header cell filled across the columns it spans",
            )
        )
    column_names = _flatten_header(filled)

    for position, group in enumerate(logical):
        for index in group:
            source_rows[index].cells = list(row_cells[position])
            source_rows[index].logical_row = position
            if position < tier_count:
                source_rows[index].disposition = RowDisposition.HEADER
    covered = {index for group in logical for index in group}
    for index, source in enumerate(source_rows):
        if index not in covered:
            source.disposition = RowDisposition.DROPPED_EMPTY  # blank row, no logical row

    rows: list[TableRow] = []
    for position in range(tier_count, len(logical)):
        group = logical[position]
        if not group:
            continue
        sources = [source_rows[index] for index in group]
        raw = " ".join(source.raw_text for source in sources)
        if not normalize_for_compare(raw):
            for source in sources:
                source.disposition = RowDisposition.DROPPED_EMPTY
            continue
        row_index = len(rows) + 1
        for source in sources:
            source.canonical_row_ids = [row_index]
            if len(sources) > 1:
                source.disposition = RowDisposition.MERGED
        rows.append(
            TableRow(
                cells=row_cells[position],
                raw_text=raw,
                row_index=row_index,
                page_start=page,
                page_end=page,
                confidence=row_confidence[position],
                warnings=row_warnings[position],
                source_row_ids=[source.source_row_id for source in sources],
                source_cell_count=sum(source.source_cell_count for source in sources),
                canonical_cell_count=len(row_cells[position]),
                signals=row_signals[position],
            )
        )

    return TableBlock(
        header_row=column_names,
        data_rows=[row.cells for row in rows],
        header_source=HeaderSource.INFERRED,
        header_confidence=(
            _HEADER_CONFIDENCE_WITH_EVIDENCE if aligned else _HEADER_CONFIDENCE_MARKDOWN_ONLY
        ),
        header_levels=filled,
        rows=rows,
        source_rows=source_rows,
        normalization_events=events,
        warnings=warnings,
    )
