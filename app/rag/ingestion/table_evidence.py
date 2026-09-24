"""Geometric evidence for PDF tables, read from `page.find_tables()`.

`pymupdf4llm` produces good reading-order markdown but loses cell geometry: a
merged header cell is cut at the wrong place, a wrapped label loses its space,
and a page can even drop a whole column. `find_tables()` knows the real cell
rectangles (a merged cell shows up as `None` in the cells it swallowed), and
`get_text(clip=<cell>)` gives clean text for each cell. This module collects
that evidence while the PDF is still open, so `table_aware_parser` can align it
row by row with the markdown and use it for structure only - it never
replaces the markdown's role as the raw source ledger.
"""

import logging
import statistics
from dataclasses import dataclass, field, replace
from typing import Any

import pymupdf

logger = logging.getLogger(__name__)

_BOLD_FLAG = 16  # pymupdf span flag bit for bold
_SAME_LINE = 3.0  # points; spans whose tops differ by less are on the same text line
_MIN_OVERLAP = 0.5  # points; overlap needed to attach an off-centre glyph to a cell
_HEADER_EPSILON = 1.0  # points; tolerance when deciding a row starts inside the header block
# A trailing row this many times taller than the table's typical row, with at
# most this many populated cells, is treated as non-tabular content the
# table's bounding box overshot into (see `_trim_trailing_garbage_rows`).
_GARBAGE_ROW_HEIGHT_MULTIPLIER = 3.0
_GARBAGE_ROW_MAX_FILLED_CELLS = 1

_Box = tuple[float, float, float, float]
_Word = tuple[float, float, float, float, str, int, int, int]


@dataclass(frozen=True)
class EvidenceCell:
    bbox: tuple[float, float, float, float]
    text: str
    text_x0: float | None = None  # left edge of the first text span (indentation signal)
    bold: bool = False
    font_size: float | None = None


@dataclass(frozen=True)
class EvidenceRow:
    """One table row; `cells[i] is None` means column i is covered by a merged
    cell that started in another cell of this row (horizontal span) or in the
    row above (vertical span)."""

    cells: list[EvidenceCell | None]
    y0: float
    y1: float
    # For each `None` cell, the (row, column) of the merged cell that covers it
    # - same row for a horizontal span, a row above for a vertical one. `None`
    # where the cell is its own, or where no covering cell could be found.
    # Empty for hand-built rows (no span information).
    covered_by: list[tuple[int, int] | None] = field(default_factory=list)


@dataclass(frozen=True)
class TableEvidence:
    page_number: int
    bbox: tuple[float, float, float, float]
    page_height: float
    rows: list[EvidenceRow]
    column_bounds: list[float] = field(default_factory=list)
    header_row_count: int = 1
    # Text of the rows above the table that the geometric box took in and
    # were trimmed off (`_trim_leading_caption_rows`), top to bottom.
    captions: list[str] = field(default_factory=list)

    @property
    def column_count(self) -> int:
        return max((len(row.cells) for row in self.rows), default=0)


@dataclass(frozen=True)
class _PageText:
    """The words and styled spans of a whole page, read ONCE.

    Reading text per cell (`get_text(clip=cell)` twice per cell) dominated the
    cost of evidence collection - about 60x slower than one page-level read -
    so cells are filled by assigning words/spans to them by their centre."""

    words: list[_Word]
    spans: list[dict[str, Any]]


def _read_page_text(page: pymupdf.Page) -> _PageText:
    words = [tuple(word) for word in page.get_text("words")]  # type: ignore[no-untyped-call]
    spans: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):  # type: ignore[no-untyped-call]
        for line in block.get("lines", []):
            spans.extend(span for span in line.get("spans", []) if span.get("text", "").strip())
    return _PageText(words=words, spans=spans)


def _inside(
    bbox: tuple[float, float, float, float], x0: float, y0: float, x1: float, y1: float
) -> bool:
    """A word/span belongs to a cell when its centre is inside, or when it
    genuinely overlaps the cell (at least `_MIN_OVERLAP` points both ways) -
    the same reach as clipping the page to the cell, which is what catches a
    row cut by a page break whose glyphs sit mostly outside its short cell."""

    if bbox[0] <= (x0 + x1) / 2 < bbox[2] and bbox[1] <= (y0 + y1) / 2 < bbox[3]:
        return True
    return (
        min(x1, bbox[2]) - max(x0, bbox[0]) >= _MIN_OVERLAP
        and min(y1, bbox[3]) - max(y0, bbox[1]) >= _MIN_OVERLAP
    )


def _text_in_band(text: _PageText, cells: list[Any]) -> _PageText:
    """Only the words/spans that can touch this row: one pass per row instead
    of every cell scanning the whole page (the per-cell scan was the largest
    Python-side cost of collecting evidence)."""

    boxes = [cell for cell in cells if cell is not None]
    if not boxes:
        return _PageText([], [])
    top = min(float(box[1]) for box in boxes) - _MIN_OVERLAP
    bottom = max(float(box[3]) for box in boxes) + _MIN_OVERLAP
    return _PageText(
        words=[word for word in text.words if word[3] >= top and word[1] <= bottom],
        spans=[span for span in text.spans if span["bbox"][3] >= top and span["bbox"][1] <= bottom],
    )


def _contains_centre(box: _Box, word: _Word) -> bool:
    x, y = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _overlap(box: _Box, word: _Word) -> float:
    width = min(word[2], box[2]) - max(word[0], box[0])
    height = min(word[3], box[3]) - max(word[1], box[1])
    if width < _MIN_OVERLAP or height < _MIN_OVERLAP:
        return 0.0
    return width * height


def _assign_words(
    words: list[_Word], row_boxes: list[_Box | None], table_boxes: list[_Box]
) -> list[list[_Word]]:
    """Give every word to exactly ONE cell of the row: the cell holding its
    centre, else - for a glyph cut by a page break that sits mostly outside
    its short cell - the cell it overlaps most, unless its centre lies in
    another cell of the table. A word straddling a border would otherwise be
    read into both cells (`STT` | `STT`)."""

    owned: list[list[_Word]] = [[] for _ in row_boxes]
    for word in words:
        home = next(
            (
                i
                for i, box in enumerate(row_boxes)
                if box is not None and _contains_centre(box, word)
            ),
            None,
        )
        if home is None:
            if any(_contains_centre(box, word) for box in table_boxes):
                continue
            overlaps = [0.0 if box is None else _overlap(box, word) for box in row_boxes]
            best = max(range(len(row_boxes)), key=lambda i: overlaps[i], default=None)
            if best is None or overlaps[best] <= 0.0:
                continue
            home = best
        owned[home].append(word)
    return owned


def _cell_text(text: _PageText, bbox: _Box, words: list[_Word]) -> EvidenceCell:
    # Words in extraction order (block, line, word) keep the spaces implied by
    # glyph gaps; joining span text directly would glue words together
    # (`HỆ ĐÀO TẠO` -> `HỆĐÀO TẠO`). Spans are only used for the style signals
    # (indent, bold, size) of the cell's first line.
    inside = sorted(words, key=lambda word: (word[5], word[6], word[7]))
    cell_spans = [span for span in text.spans if _inside(bbox, *(float(v) for v in span["bbox"]))]
    text_x0: float | None = None
    bold = False
    font_size: float | None = None
    if cell_spans:
        # first line = spans starting within a few points of the topmost one
        # (accented glyphs make spans on one line differ slightly in top edge);
        # the cell's style is its leftmost span on that line.
        line_top = min(float(span["bbox"][1]) for span in cell_spans)
        first_line = [
            span for span in cell_spans if float(span["bbox"][1]) <= line_top + _SAME_LINE
        ]
        first = min(first_line, key=lambda span: float(span["bbox"][0]))
        text_x0 = float(first["bbox"][0])
        font_size = float(first.get("size", 0.0)) or None
        bold = (
            bool(first.get("flags", 0) & _BOLD_FLAG) or "bold" in str(first.get("font", "")).lower()
        )
    return EvidenceCell(
        bbox=bbox,
        text=" ".join(word[4] for word in inside),
        text_x0=text_x0,
        bold=bold,
        font_size=font_size,
    )


def _header_row_count(rows: list[EvidenceRow]) -> int:
    """Rows that start inside the vertical extent of row 0's tallest cell are
    header tiers (a cell of row 0 spanning down over them). Pure geometry: no
    keyword is involved, and a single-tier header yields 1."""

    if not rows:
        return 0
    header_bottom = rows[0].y1
    count = 1
    for row in rows[1:]:
        if row.y0 < header_bottom - _HEADER_EPSILON:
            count += 1
        else:
            break
    return count


def _trim_trailing_garbage_rows(rows: list[EvidenceRow]) -> list[EvidenceRow]:
    """Drop a trailing row that isn't really a table row: `find_tables()`'s
    bounding box can overshoot the table's real bottom border and absorb the
    next section's heading/paragraph text as one final "row" (confirmed on a
    real PDF: a 3-column, 4-data-row ruled table gained a 5th row nearly 20x
    taller than every other row, holding one populated cell of unrelated
    running text with every other cell in that row empty).

    Such a row is recognized by pure geometry - it is both anomalously tall
    AND almost entirely empty - never by reading its text, so this can't
    mistake a genuine tall row (e.g. a real full-width note inside the table,
    which is at most a couple of wrapped lines taller than its neighbours,
    not an order of magnitude taller) for garbage. Only trims from the end,
    one row at a time, and never below 2 rows, so a real small table is never
    hollowed out.
    """

    trimmed = list(rows)
    while len(trimmed) > 2:
        *rest, last = trimmed
        heights = [row.y1 - row.y0 for row in rest]
        median_height = statistics.median(heights)
        filled = sum(1 for cell in last.cells if cell is not None and cell.text.strip())
        is_anomalously_tall = (
            median_height > 0
            and (last.y1 - last.y0) > _GARBAGE_ROW_HEIGHT_MULTIPLIER * median_height
        )
        if is_anomalously_tall and filled <= _GARBAGE_ROW_MAX_FILLED_CELLS:
            trimmed = rest
            continue
        break
    return trimmed


def _trim_leading_caption_rows(
    rows: list[EvidenceRow],
) -> tuple[list[EvidenceRow], list[str]]:
    """Drop rows above the table's real first row that `find_tables()` took in:
    a title or info line drawn over the full width (one cell with text spanning
    most of the columns) or an empty band between it and the table. They stop
    at the first row that has two cells with text - the header. Pure geometry;
    never trims the table down to fewer than 2 rows."""

    columns = max((len(row.cells) for row in rows), default=0)
    trimmed = list(rows)
    captions: list[str] = []
    while len(trimmed) > 2:
        cells = [cell for cell in trimmed[0].cells if cell is not None]
        filled = [cell for cell in cells if cell.text.strip()]
        if len(filled) > 1:
            break
        if filled:
            spanned = sum(1 for cell in trimmed[0].cells if cell is None) + 1
            if spanned * 2 < columns:
                break
        captions.extend(cell.text for cell in filled)
        trimmed = trimmed[1:]
    return trimmed, captions


def _collapse_empty_columns(
    rows: list[EvidenceRow], bounds: list[float], right_edge: float
) -> tuple[list[EvidenceRow], list[float]]:
    """Fold grid columns that never hold a column of their own into their
    neighbour. Double borders and cell padding make `find_tables()` cut thin
    slivers (`'' | STT | ''` over a data cell spanning all three), which would
    otherwise become phantom columns. A boundary between two grid columns is
    real only when some row - the header included - has two different cells
    with text on either side of it; every other boundary is dropped."""

    edges = [*bounds, right_edge]
    count = len(bounds)
    if count < 2:
        return rows, bounds

    def owner(row: EvidenceRow, column: int) -> EvidenceCell | None:
        x = (edges[column] + edges[column + 1]) / 2
        return next(
            (c for c in row.cells if c is not None and c.bbox[0] <= x < c.bbox[2]),
            None,
        )

    keep = [True] + [False] * (count - 1)  # keep[i]: a boundary starts column i
    for column in range(1, count):
        for row in rows:
            left, right = owner(row, column - 1), owner(row, column)
            if (
                left is not None
                and right is not None
                and left is not right
                and left.text.strip()
                and right.text.strip()
            ):
                keep[column] = True
                break
    if all(keep):
        return rows, bounds

    starts = [column for column in range(count) if keep[column]]
    new_edges = [edges[column] for column in starts] + [right_edge]
    collapsed: list[EvidenceRow] = []
    for row in rows:
        cells: list[EvidenceCell | None] = []
        for index in range(len(starts)):
            low, high = new_edges[index], new_edges[index + 1]
            parts = [c for c in row.cells if c is not None and low - 0.5 <= c.bbox[0] < high - 0.5]
            if not parts:
                cells.append(None)
                continue
            if len(parts) == 1:
                cells.append(parts[0])
                continue
            styled = next((c for c in parts if c.text.strip()), parts[0])
            cells.append(
                EvidenceCell(
                    bbox=(
                        min(c.bbox[0] for c in parts),
                        min(c.bbox[1] for c in parts),
                        max(c.bbox[2] for c in parts),
                        max(c.bbox[3] for c in parts),
                    ),
                    text=" ".join(c.text for c in parts if c.text.strip()),
                    text_x0=styled.text_x0,
                    bold=styled.bold,
                    font_size=styled.font_size,
                )
            )
        collapsed.append(EvidenceRow(cells=cells, y0=row.y0, y1=row.y1))
    return collapsed, [edges[column] for column in starts]


def _resolve_spans(
    rows: list[EvidenceRow], bounds: list[float], right_edge: float
) -> list[EvidenceRow]:
    """Find the merged cell behind every `None` cell: the real cell, in this row
    or a row above, whose rectangle holds the centre of the empty grid slot
    (column band x row band). Pure geometry, so a header cell spanning two
    columns and a row label spanning ten rows are found the same way."""

    edges = [*bounds, right_edge]
    resolved: list[EvidenceRow] = []
    for row_index, row in enumerate(rows):
        covered: list[tuple[int, int] | None] = []
        for column, cell in enumerate(row.cells):
            if cell is not None or column + 1 >= len(edges):
                covered.append(None)
                continue
            x = (edges[column] + edges[column + 1]) / 2
            y = (row.y0 + row.y1) / 2
            origin: tuple[int, int] | None = None
            for above in range(row_index, -1, -1):
                for left, candidate in enumerate(rows[above].cells[: column + 1]):
                    if candidate is None:
                        continue
                    x0, y0, x1, y1 = candidate.bbox
                    if x0 <= x < x1 and y0 <= y < y1:
                        origin = (above, left)
                        break
                if origin is not None:
                    break
            covered.append(origin)
        resolved.append(EvidenceRow(cells=row.cells, y0=row.y0, y1=row.y1, covered_by=covered))
    return resolved


def _collect_page_tables(page: pymupdf.Page) -> list[TableEvidence]:
    # Line-based detection only. Asking the layout model as well is ~5x slower and
    # gave the same grid on the pages where the markdown did not line up.
    finder = page.find_tables(use_layout=False)  # type: ignore[no-untyped-call]
    page_text = _read_page_text(page) if finder.tables else _PageText([], [])
    evidence: list[TableEvidence] = []
    for table in sorted(finder.tables, key=lambda t: t.bbox[1]):
        rows: list[EvidenceRow] = []
        table_boxes: list[_Box] = [
            (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
            for row in table.rows
            for raw in row.cells
            if raw is not None
        ]
        for row in table.rows:
            cells: list[EvidenceCell | None] = []
            ys: list[float] = []
            row_text = _text_in_band(page_text, row.cells)
            row_boxes: list[_Box | None] = [
                None
                if raw is None
                else (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
                for raw in row.cells
            ]
            owned = _assign_words(row_text.words, row_boxes, table_boxes)
            for bbox, words in zip(row_boxes, owned, strict=True):
                if bbox is None:
                    cells.append(None)
                    continue
                cells.append(_cell_text(row_text, bbox, words))
                ys.extend((bbox[1], bbox[3]))
            if not ys:
                continue
            # a row's y-extent is its own cells' - use the SHORTEST cell's bottom so a
            # tall header cell that spans down does not stretch the row itself
            row_y0 = min(cell.bbox[1] for cell in cells if cell is not None)
            row_y1 = min(cell.bbox[3] for cell in cells if cell is not None)
            rows.append(EvidenceRow(cells=cells, y0=row_y0, y1=row_y1))
        if not rows:
            continue
        rows, captions = _trim_leading_caption_rows(_trim_trailing_garbage_rows(rows))
        bounds = sorted(
            {round(cell.bbox[0], 1) for row in rows for cell in row.cells if cell is not None}
        )
        rows, bounds = _collapse_empty_columns(rows, bounds, float(table.bbox[2]))
        # spans are resolved on each row's own (shortest-cell) extent, before
        # row 0 is stretched below
        rows = _resolve_spans(rows, bounds, float(table.bbox[2]))
        # header extent uses the tallest cell of row 0, not the shortest
        first_cells = [cell for cell in rows[0].cells if cell is not None]
        tall_bottom = max(cell.bbox[3] for cell in first_cells)
        rows[0] = replace(rows[0], y1=tall_bottom)
        evidence.append(
            TableEvidence(
                page_number=page.number + 1,
                bbox=(
                    float(table.bbox[0]),
                    float(table.bbox[1]),
                    float(table.bbox[2]),
                    float(table.bbox[3]),
                ),
                page_height=float(page.rect.height),
                rows=rows,
                column_bounds=bounds,
                header_row_count=_header_row_count(rows),
                captions=captions,
            )
        )
    return evidence


def collect_table_evidence(document: pymupdf.Document) -> dict[int, list[TableEvidence]]:
    """Evidence for every page of `document`, keyed by 1-based page number.

    Best effort: a page whose tables cannot be read simply has no evidence
    (the parser then falls back to markdown alone for it), so a `find_tables`
    failure can never break ingestion."""

    result: dict[int, list[TableEvidence]] = {}
    for page in document:  # type: ignore[attr-defined]
        try:
            tables = _collect_page_tables(page)
        except Exception:
            logger.warning("find_tables failed on page %s; using markdown only", page.number + 1)
            tables = []
        if tables:
            result[page.number + 1] = tables
    return result
