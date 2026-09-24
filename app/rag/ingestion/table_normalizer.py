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
from dataclasses import replace

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

# A geometry row whose text differs from its markdown row is still the same row
# when this share of characters is common to both. The markdown extractor
# drops glyphs (`HỆ ĐO TẠO` for `HỆ ĐÀO TẠO`), glues words and moves the text of
# a cell merged over several rows to whichever row it is centred on; the
# geometry reads each cell from its own rectangle, so it is the better text.
_GEOMETRY_ACCEPT_SIMILARITY = 0.8
# points; a cell whose bottom is further below its row's bottom is merged down
_SPAN_EPSILON = 1.0
# A fuzzy row alignment is kept only when at most this share of the markdown
# text, and of the geometry text, is left without a counterpart.
_FUZZY_ALIGN_MAX_LOSS = 0.1


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


def _geometry_text(text: str) -> str:
    """Cell text read from the PDF's own words is plain text, not markdown: a
    `**` in it is the document's own footnote mark (`KT Phần mềm**`), not bold
    markup, so only the Unicode form is normalized."""

    return unicodedata.normalize("NFC", text).strip()


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


def _vertical_origin(evidence: TableEvidence, position: int, column: int) -> int | None:
    """The row above whose cell, merged downwards, covers this cell (None when
    the cell is its own or is covered from the left)."""

    covered = evidence.rows[position].covered_by
    origin = covered[column] if column < len(covered) else None
    if origin is None or origin[0] >= position or origin[1] != column:
        return None
    return origin[0]


def _row_texts(evidence: TableEvidence, position: int) -> tuple[list[str], list[str]]:
    """(own, spanning) cell texts of a geometry row. A cell merged down over
    several rows - starting on this row or covering it from above - is
    "spanning": the markdown puts its text (or shards of its glyphs) on
    whichever row of the span it happens to be centred on."""

    row = evidence.rows[position]
    own: list[str] = []
    spanning: list[str] = []
    for column, cell in enumerate(row.cells):
        if cell is not None:
            reaches_below = cell.bbox[3] > row.y1 + _SPAN_EPSILON
            (spanning if reaches_below else own).append(cell.text)
            continue
        origin = _vertical_origin(evidence, position, column)
        if origin is not None:
            above = evidence.rows[origin].cells[column]
            spanning.append(above.text if above is not None else "")
    return own, spanning


def _geometry_matches(md_text: str, evidence: TableEvidence, positions: list[int]) -> bool:
    """The markdown row(s) and the geometry row(s) describe the same content:
    the markdown holds (almost) all the rows' own text, and (almost) nothing
    that is neither their own text nor the text of a cell merged over them.
    Several positions when one markdown row carries several geometry rows."""

    own_chars: Counter[str] = Counter()
    pool: Counter[str] = Counter()
    for position in positions:
        own, spanning = _row_texts(evidence, position)
        own_chars += _char_multiset(" ".join(own))
        pool += _char_multiset(" ".join(own)) + _char_multiset(" ".join(spanning))
    md_chars = _char_multiset(md_text)
    recall = sum((md_chars & own_chars).values()) / max(sum(own_chars.values()), 1)
    precision = sum((md_chars & pool).values()) / max(sum(md_chars.values()), 1)
    return recall >= _GEOMETRY_ACCEPT_SIMILARITY and precision >= _GEOMETRY_ACCEPT_SIMILARITY


# Largest block of rows matched together by the fuzzy alignment: this many
# markdown rows for one geometry row (a wrapped cell split over lines), or one
# markdown row for this many geometry rows (two rows glued with `<br>`).
_MAX_BLOCK_ROWS = 4


class _RowText:
    """Character multisets of every markdown and geometry row, shared by the
    alignment and its quality check."""

    def __init__(self, md_cells: list[list[str]], evidence: TableEvidence) -> None:
        self.md = [_char_multiset(" ".join(cells)) for cells in md_cells]
        self.own: list[Counter[str]] = []
        self.pool: list[Counter[str]] = []
        for position in range(len(evidence.rows)):
            own, spanning = _row_texts(evidence, position)
            self.own.append(_char_multiset(" ".join(own)))
            self.pool.append(self.own[-1] + _char_multiset(" ".join(spanning)))

    def block(self, md: range, geometry: range) -> tuple[int, int, int]:
        """(matched, extra, missing) characters of one block."""

        chars: Counter[str] = Counter()
        pool: Counter[str] = Counter()
        own: Counter[str] = Counter()
        for index in md:
            chars += self.md[index]
        for position in geometry:
            pool += self.pool[position]
            own += self.own[position]
        return (
            sum((chars & pool).values()),
            sum((chars - pool).values()),
            sum((own - chars).values()),
        )


def _blocks(groups: list[list[int]]) -> list[tuple[range, range]]:
    """Groups (markdown rows per geometry row) back to blocks; geometry rows
    that share one markdown row form one block."""

    blocks: list[tuple[range, range]] = []
    for position, group in enumerate(groups):
        md = range(group[0], group[-1] + 1) if group else range(0)
        if blocks and group and list(blocks[-1][0]) == group:
            blocks[-1] = (md, range(blocks[-1][1].start, position + 1))
        else:
            blocks.append((md, range(position, position + 1)))
    return blocks


def _alignment_ok(text: _RowText, blocks: list[tuple[range, range]]) -> bool:
    matched = extra = missing = 0
    for md, geometry in blocks:
        m, e, x = text.block(md, geometry)
        matched, extra, missing = matched + m, extra + e, missing + x
    total_md = max(matched + extra, 1)
    total_own = max(sum(sum(own.values()) for own in text.own), 1)
    return (
        extra / total_md <= _FUZZY_ALIGN_MAX_LOSS and missing / total_own <= _FUZZY_ALIGN_MAX_LOSS
    )


def _positional_alignment(
    md_cells: list[list[str]], evidence: TableEvidence
) -> list[list[int]] | None:
    """Row i to row i - only when the counts agree AND the text agrees: equal
    counts alone can hide two opposite slips (a header split over two markdown
    rows, two body rows glued into one) that shift every row in between."""

    if len(md_cells) != len(evidence.rows):
        return None
    groups = [[index] for index in range(len(md_cells))]
    return groups if _alignment_ok(_RowText(md_cells, evidence), _blocks(groups)) else None


def _align_rows_fuzzy(md_cells: list[list[str]], evidence: TableEvidence) -> list[list[int]] | None:
    """Fallback for `_align_rows` when the markdown shuffles text between rows:
    cut both sides into consecutive blocks - several markdown rows for one
    geometry row (a wrapped cell's tail on its own row) or one markdown row for
    several geometry rows (`1<br>2`) - maximising the characters each block
    shares minus those it lacks or brings extra. Geometry rows of a shared
    block all point to the same markdown row. Kept only when the blocks account
    for nearly all the text on both sides; otherwise the tables differ."""

    count, rows = len(md_cells), len(evidence.rows)
    if rows == 0 or count == 0:
        return None
    text = _RowText(md_cells, evidence)
    minus_infinity = float("-inf")
    # best[i][k]: best score for markdown rows [0, i) against geometry rows [0, k)
    best = [[minus_infinity] * (rows + 1) for _ in range(count + 1)]
    back: list[list[tuple[int, int]]] = [[(0, 0)] * (rows + 1) for _ in range(count + 1)]
    best[0][0] = 0.0
    for i in range(count + 1):
        for k in range(rows + 1):
            if best[i][k] == minus_infinity:
                continue
            for a in range(1, _MAX_BLOCK_ROWS + 1):
                for b in range(1, _MAX_BLOCK_ROWS + 1):
                    if min(a, b) != 1 or i + a > count or k + b > rows:
                        continue
                    matched, extra, missing = text.block(range(i, i + a), range(k, k + b))
                    candidate = best[i][k] + matched - extra - missing
                    if candidate > best[i + a][k + b]:
                        best[i + a][k + b] = candidate
                        back[i + a][k + b] = (i, k)
    if best[count][rows] == minus_infinity:
        return None

    blocks: list[tuple[range, range]] = []
    i, k = count, rows
    while i or k:
        pi, pk = back[i][k]
        blocks.append((range(pi, i), range(pk, k)))
        i, k = pi, pk
    blocks.reverse()
    if not _alignment_ok(text, blocks):
        return None
    groups: list[list[int]] = []
    for md, geometry in blocks:
        groups.extend(list(md) for _ in geometry)
    return groups


# `find_tables()` can draw its box one or two rows past the table (a heading
# right above it, a line of text below); at most this many rows are dropped
# from each end, and only when what is left aligns with the markdown.
_MAX_EDGE_ROWS = 2


def _slice_evidence(evidence: TableEvidence, lead: int, trail: int) -> TableEvidence:
    kept = evidence.rows[lead : len(evidence.rows) - trail]
    rows = [
        replace(
            row,
            covered_by=[
                None if origin is None or origin[0] < lead else (origin[0] - lead, origin[1])
                for origin in row.covered_by
            ],
        )
        for row in kept
    ]
    header = evidence.header_row_count - lead if lead < evidence.header_row_count else 1
    return replace(evidence, rows=rows, header_row_count=max(1, header))


def _align_without_edge_rows(
    md_cells: list[list[str]], evidence: TableEvidence
) -> tuple[TableEvidence, list[list[int]]] | None:
    """Alignment after dropping geometry rows the markdown table does not have
    at its top/bottom - text outside the table that the geometric box took in.
    Nothing is lost: the markdown ledger is the source of the table's text."""

    options = sorted(
        (
            (lead, trail)
            for lead in range(_MAX_EDGE_ROWS + 1)
            for trail in range(_MAX_EDGE_ROWS + 1)
            if 0 < lead + trail < len(evidence.rows)
        ),
        key=sum,
    )
    for lead, trail in options:
        sliced = _slice_evidence(evidence, lead, trail)
        groups = _align_rows(md_cells, sliced) or _align_rows_fuzzy(md_cells, sliced)
        if groups is not None:
            return sliced, groups
    return None


def _spilled_tail(md_cells: list[str], geometry: list[str]) -> str | None:
    """The text of a line below the table that the markdown extractor glued
    onto the table's last row, cut at the column borders (`14<br>4. HƯ` |
    `… giao dịch<br>ỚNG DẪN SỬ DỤN` | …). Each markdown cell must start with
    exactly its geometry cell's lines; what follows, read left to right, is the
    line (None when the row does not have that shape)."""

    if len(md_cells) != len(geometry):
        return None
    pieces: list[str] = []
    for md, cell in zip(md_cells, geometry, strict=True):
        lines = re.split(r"<br\s*/?>", md, flags=re.IGNORECASE)
        target = normalize_for_compare(cell)
        taken = 0
        while normalize_for_compare(" ".join(lines[:taken])) != target:
            taken += 1
            if taken > len(lines):
                return None
        pieces.append(" ".join(line.strip() for line in lines[taken:] if line.strip()))
    tail = "".join(pieces).strip()
    return tail or None


# A markdown row belongs to the table only when at least this share of its
# characters is text of the geometry row(s) it was aligned with.
_OWN_TEXT_MIN_SHARE = 0.5


def _split_leading_text(
    md_cells: list[list[str]], evidence: TableEvidence, group: list[int]
) -> tuple[list[int], list[int]]:
    """Markdown rows at the start of the first group that are not the table's
    text - a title or info line above the table that the extractor read as a
    first row. Returns (rows kept in the group, rows moved out as text)."""

    # Words, not characters: any two Vietnamese lines share most of their
    # letters, but an info line above the table shares few words with the header.
    own, spanning = _row_texts(evidence, 0)
    pool = _word_multiset(" ".join(own + spanning))
    moved: list[int] = []
    for index in group[:-1]:
        words = _word_multiset(" ".join(md_cells[index]))
        total = sum(words.values())
        if total and sum((words & pool).values()) / total >= _OWN_TEXT_MIN_SHARE:
            break
        pool = pool - words  # the words it does share are not the table's any more
        moved.append(index)
    return group[len(moved) :], moved


def _word_multiset(text: str) -> Counter[str]:
    plain = unicodedata.normalize("NFC", re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE))
    return Counter(re.findall(r"\w+", plain.casefold()))


def _moved_text(cells: list[str], captions: list[str]) -> str:
    """The text of a markdown row moved out of the table. The markdown cut it
    at the column borders, sometimes mid-word, so the same line read from the
    page (a caption the geometry trimmed off) is preferred when it holds the
    same characters; otherwise the cells are read back line by line."""

    chars = _char_multiset(" ".join(cells))
    total = max(sum(chars.values()), 1)
    for caption in captions:
        caption_chars = _char_multiset(caption)
        larger = max(total, sum(caption_chars.values()))
        if sum((chars & caption_chars).values()) / larger >= _GEOMETRY_ACCEPT_SIMILARITY:
            return _geometry_text(caption)
    lines = [re.split(r"<br\s*/?>", cell, flags=re.IGNORECASE) for cell in cells]
    depth = max((len(parts) for parts in lines), default=0)
    rebuilt = [
        " ".join(
            parts[level].strip() for parts in lines if level < len(parts) and parts[level].strip()
        )
        for level in range(depth)
    ]
    return " ".join(line for line in rebuilt if line)


def _geometry_header(evidence: TableEvidence, tier_count: int) -> list[list[str]]:
    """Header tiers straight from the cell rectangles: a cell merged across
    columns gives its text to every column under it (the parent title of each
    sub-column), a cell merged down gives its text to the tiers below (the
    flattening step de-duplicates it)."""

    tiers: list[list[str]] = []
    for position in range(tier_count):
        row = evidence.rows[position]
        tier: list[str] = []
        for column, cell in enumerate(row.cells):
            if cell is not None:
                tier.append(_geometry_text(cell.text))
                continue
            origin = row.covered_by[column] if column < len(row.covered_by) else None
            if origin is None:
                tier.append("")
            elif origin[0] == position:
                tier.append(tier[origin[1]])
            else:
                tier.append(
                    tiers[origin[0]][origin[1]] if origin[1] < len(tiers[origin[0]]) else ""
                )
        tiers.append(tier + [""] * (evidence.column_count - len(tier)))
    return tiers


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
        groups = (
            _positional_alignment(md_cells, evidence)
            or _align_rows(md_cells, evidence)
            or _align_rows_fuzzy(md_cells, evidence)
        )
        if groups is None:
            trimmed = _align_without_edge_rows(md_cells, evidence)
            if trimmed is not None:
                evidence, groups = trimmed
        if groups is None:
            warnings.append("evidence_unaligned")
    moved_before: list[int] = []
    if groups is not None and evidence is not None and len(groups[0]) > 1:
        groups[0], moved_before = _split_leading_text(md_cells, evidence, groups[0])
    aligned = groups is not None and evidence is not None
    logical = groups if groups is not None else [[index] for index in range(len(md_lines))]
    column_count = evidence.column_count if aligned and evidence else len(md_cells[0])
    geometry_tiers = (
        max(1, min(evidence.header_row_count, len(logical))) if aligned and evidence else 0
    )

    row_cells: list[list[str]] = []
    spilled: list[str] = []
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
                used = [_geometry_text(cell) for cell in geometry]
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
            elif (
                position == len(logical) - 1
                and len(group) == 1
                and (tail := _spilled_tail(md_cells[group[0]], geometry)) is not None
            ):
                # the line right under the table was read into its last row
                used = [_geometry_text(cell) for cell in geometry]
                spilled.append(tail)
                events.append(
                    NormalizationEvent(
                        EventKind.RECOVER_TEXT,
                        sids,
                        "text below the table split off the last row",
                    )
                )
            elif position < geometry_tiers or _geometry_matches(
                md_text, evidence, [p for p, other in enumerate(logical) if other == group]
            ):
                # the header is always read from geometry: that is where the
                # markdown mangles merged cells the most
                used = [_geometry_text(cell) for cell in geometry]
                events.append(
                    NormalizationEvent(
                        EventKind.RECOVER_TEXT,
                        sids,
                        "cell text read from the cell rectangles instead of the markdown row",
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
        tier_count = geometry_tiers
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
    # covered cells whose merged cell could not be located fall back to the
    # text-only fill
    filled = _fill_header_tiers(
        _geometry_header(evidence, tier_count) if aligned and evidence is not None else tiers
    )
    if filled != tiers:
        events.append(
            NormalizationEvent(
                EventKind.FILL_MERGED_HEADER,
                [sid for position in range(tier_count) for sid in group_ids(position)],
                "merged header cell filled across the columns it spans",
            )
        )
    column_names = _flatten_header(filled)

    # A cell merged down over several data rows (a group label, a shared value)
    # holds for every row it covers, so each row carries it. A cell merged
    # ACROSS columns stays once, in its first column - repeating a note or a
    # total in every column it spans would read as separate values.
    inherited: list[list[int]] = [[] for _ in logical]
    if aligned and evidence is not None:
        for position in range(tier_count, len(logical)):
            for column in range(min(column_count, len(row_cells[position]))):
                origin = _vertical_origin(evidence, position, column)
                if origin is None or origin < tier_count or row_cells[position][column]:
                    continue
                value = row_cells[origin][column] if column < len(row_cells[origin]) else ""
                if not value:
                    continue
                row_cells[position][column] = value
                inherited[position].append(column)
                events.append(
                    NormalizationEvent(
                        EventKind.FILL_MERGED_CELL,
                        group_ids(position),
                        f"column {column + 1} copied from the merged cell of logical row "
                        f"{origin + 1}",
                    )
                )

    for position, group in enumerate(logical):
        for index in group:
            source_rows[index].cells = list(row_cells[position])
            source_rows[index].logical_row = position
            if position < tier_count:
                source_rows[index].disposition = RowDisposition.HEADER
    for index in moved_before:
        source_rows[index].disposition = RowDisposition.MOVED_TO_TEXT
    covered = {index for group in logical for index in group} | set(moved_before)
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
            source.canonical_row_ids.append(row_index)
            if len(source.canonical_row_ids) > 1:
                # one markdown row carried several geometry rows (`1<br>2`)
                source.disposition = RowDisposition.SPLIT
            elif len(sources) > 1:
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
                inherited_cells=inherited[position],
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
        spilled_text=spilled,
        text_before=[
            _moved_text(md_cells[index], evidence.captions if evidence else [])
            for index in moved_before
        ],
    )
