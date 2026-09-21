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
from dataclasses import dataclass, field
from typing import Any

import pymupdf

logger = logging.getLogger(__name__)

_BOLD_FLAG = 16  # pymupdf span flag bit for bold
_SAME_LINE = 3.0  # points; spans whose tops differ by less are on the same text line
_MIN_OVERLAP = 0.5  # points; overlap needed to attach an off-centre glyph to a cell
_HEADER_EPSILON = 1.0  # points; tolerance when deciding a row starts inside the header block


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


@dataclass(frozen=True)
class TableEvidence:
    page_number: int
    bbox: tuple[float, float, float, float]
    page_height: float
    rows: list[EvidenceRow]
    column_bounds: list[float] = field(default_factory=list)
    header_row_count: int = 1

    @property
    def column_count(self) -> int:
        return max((len(row.cells) for row in self.rows), default=0)


@dataclass(frozen=True)
class _PageText:
    """The words and styled spans of a whole page, read ONCE.

    Reading text per cell (`get_text(clip=cell)` twice per cell) dominated the
    cost of evidence collection - about 60x slower than one page-level read -
    so cells are filled by assigning words/spans to them by their centre."""

    words: list[tuple[float, float, float, float, str, int, int, int]]
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


def _cell_text(text: _PageText, bbox: tuple[float, float, float, float]) -> EvidenceCell:
    # Words in extraction order (block, line, word) keep the spaces implied by
    # glyph gaps; joining span text directly would glue words together
    # (`HỆ ĐÀO TẠO` -> `HỆĐÀO TẠO`). Spans are only used for the style signals
    # (indent, bold, size) of the cell's first line.
    inside = sorted(
        (word for word in text.words if _inside(bbox, word[0], word[1], word[2], word[3])),
        key=lambda word: (word[5], word[6], word[7]),
    )
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


def _collect_page_tables(page: pymupdf.Page) -> list[TableEvidence]:
    # Line-based detection only. Asking the layout model as well is ~5x slower and
    # gave the same grid on the pages where the markdown did not line up.
    finder = page.find_tables(use_layout=False)  # type: ignore[no-untyped-call]
    page_text = _read_page_text(page) if finder.tables else _PageText([], [])
    evidence: list[TableEvidence] = []
    for table in sorted(finder.tables, key=lambda t: t.bbox[1]):
        rows: list[EvidenceRow] = []
        for row in table.rows:
            cells: list[EvidenceCell | None] = []
            ys: list[float] = []
            row_text = _text_in_band(page_text, row.cells)
            for raw in row.cells:
                if raw is None:
                    cells.append(None)
                    continue
                bbox = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
                cells.append(_cell_text(row_text, bbox))
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
        # header extent uses the tallest cell of row 0, not the shortest
        first_cells = [cell for cell in rows[0].cells if cell is not None]
        tall_bottom = max(cell.bbox[3] for cell in first_cells)
        rows[0] = EvidenceRow(cells=rows[0].cells, y0=rows[0].y0, y1=tall_bottom)
        bounds = sorted(
            {round(cell.bbox[0], 1) for row in rows for cell in row.cells if cell is not None}
        )
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
