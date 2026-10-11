import logging
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import cache
from io import BytesIO
from pathlib import Path

import pymupdf
import pymupdf4llm
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from docx import Document as open_docx
from docx.document import Document
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.table import _Row as DocxRow
from docx.text.paragraph import Paragraph as DocxParagraph

from app.core.config import settings
from app.core.errors.exceptions import UnsupportedFileTypeException
from app.rag.ingestion.canonical_table import (
    TableBlock,
    build_table_from_plain_rows,
    normalize_for_compare,
)
from app.rag.ingestion.table_evidence import TableEvidence, collect_table_evidence
from app.rag.ingestion.table_hierarchy import infer_hierarchy
from app.rag.ingestion.table_merger import (
    MergeCandidate,
    column_edges,
    decide_merge,
    furniture_key,
    merge_page_tables,
    reproject_columns,
)
from app.rag.ingestion.table_normalizer import build_page_table
from app.rag.ingestion.table_normalizer import (
    markdown_row_cells as _markdown_row_cells,  # noqa: F401  (kept importable from here for existing callers/tests)
)
from app.schemas.ingestion import HeaderSource, RegionType, SourceType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedRegion:
    """One contiguous block of a document, tagged as running text or a table.

    All fields beyond `region_type`/`content` have safe defaults and are
    placed after those two original fields so that
    `ParsedRegion(RegionType.TEXT, text)` (positional, as used throughout
    the existing test suite) keeps constructing unchanged. `table` is set for
    TABLE regions only. `content` never has a heading prefixed into it -
    heading text lives only in `heading_path`.
    """

    region_type: RegionType
    content: str
    heading_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    block_index: int | None = None
    source_type: SourceType | None = None
    table: TableBlock | None = None


def split_regions(content: bytes, filename: str, extension: str) -> list[ParsedRegion]:
    """Split a PDF/DOCX/HTML into ordered text/table regions for chunking.

    `.txt` and `.xlsx` are not routed through this parser: `.txt` has no
    table structure to preserve, and `.xlsx` is chunked row-by-row by its own
    dedicated strategy (see `chunking/excel_rows.py`).
    """

    if extension == "txt":
        text = content.decode("utf-8").strip()
        if not text:
            return []
        return [
            ParsedRegion(
                RegionType.TEXT,
                text,
                heading_path=[],
                block_index=0,
                source_type=SourceType.TXT,
            )
        ]
    if extension == "docx":
        return _with_hierarchy(_regions_from_docx(content))
    if extension in ("html", "htm"):
        return _with_hierarchy(_regions_from_html(content))
    if extension != "pdf":
        raise UnsupportedFileTypeException(filename)

    document = pymupdf.open(stream=content, filetype=extension)
    try:
        pages: list[dict[str, object]] = pymupdf4llm.to_markdown(
            document, page_chunks=True, ocr_language=ocr_language(settings.INGEST_OCR_LANGUAGE)
        )
        # Geometric evidence must be read while the document is still open;
        # the parser only uses it for structure (see table_evidence).
        evidence = collect_table_evidence(document)
    finally:
        document.close()
    return _with_hierarchy(_regions_from_pdf_pages(pages, evidence))


@cache
def ocr_language(wanted: str) -> str:
    """`wanted` ("vie+eng") minus the languages whose Tesseract data is not installed, so a
    machine without `vie.traineddata` still OCRs in English instead of failing the ingest."""

    try:
        tessdata = Path(pymupdf.get_tessdata())
    except RuntimeError:
        return wanted  # no Tesseract at all: pymupdf4llm skips OCR anyway
    languages = [lang for lang in wanted.split("+") if lang]
    installed = [lang for lang in languages if (tessdata / f"{lang}.traineddata").is_file()]
    for missing in sorted(set(languages) - set(installed)):
        logger.warning("OCR language %r has no traineddata in %s; skipping it", missing, tessdata)
    return "+".join(installed) or "eng"


def _with_hierarchy(regions: list["ParsedRegion"]) -> list["ParsedRegion"]:
    """Attach each table row's ancestors (see `table_hierarchy`)."""

    return [
        replace(region, table=infer_hierarchy(region.table))
        if region.region_type == RegionType.TABLE and region.table is not None
        else region
        for region in regions
    ]


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_EMPHASIS_MARKS = re.compile(r"[*_`]")
_DIGIT_RUN = re.compile(r"\d+")
# A page-number line is short (`Trang 12/120`); a sentence that merely holds a
# number is not one.
_PAGE_NUMBER_MAX_CHARS = 20
# An ordinal line candidate for heading promotion: Roman numeral (I./II./III.)
# or Arabic numeral (1./2./3.), optionally wrapped in `**` (pymupdf4llm
# renders a fully-bold line as `**...**`, which would otherwise hide the
# ordinal from a regex anchored at the true start of the line).
_ORDINAL_LINE = re.compile(r"^\*{0,2}(?:[IVXLCDM]+|\d+)\.\s+\S.*$")
_ROMAN_ORDINAL_TITLE = re.compile(r"^[IVXLCDM]+\.\s")
# `6.`, `5.2`, `2.1.2`, `2.2.4.` - a single number needs its dot, so a title
# that merely starts with a year (`2026 ...`) is not read as numbered.
_ARABIC_ORDINAL_TITLE = re.compile(r"^(?:(\d+(?:\.\d+)+)\.?|(\d+)\.)(?=[\s_*]|$)")
# `a.` / `a)` - a lettered item sits under the numbered heading it follows.
_LETTER_ORDINAL_TITLE = re.compile(r"^[a-z][.)](?=[\s_*]|$)")
_LETTER_HEADING_LEVEL = 9
_TITLE_LEAD_MARKS = re.compile(r"^[\s_*]+")
_ORDINAL_PREFIX = re.compile(r"^\*{0,2}(?:[IVXLCDM]+|\d+)\.\s+")
_UNDERLINE_TAG = re.compile(r"<u>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
# `**bold**`, `__bold__`, and `_italic_` (a lone `_` at a word edge; `snake_case`
# keeps its underscore)
_MARKDOWN_EMPHASIS = re.compile(r"\*\*|__|(?<!\w)_|_(?!\w)")
_PDF_PROMOTED_HEADING_FALLBACK_LEVEL = 2
# Share of characters a geometric table must have in common with a markdown table
# to be taken for the same table.
_EVIDENCE_MATCH_MIN = 0.9
# The geometric box may take in a row or two of text around the table (a heading
# right above it); the match is scored with up to this many rows left off each end.
_EVIDENCE_EDGE_ROWS = 2


def _match_score(markdown_chars: Counter[str], rows: list[Counter[str]]) -> float:
    candidate: Counter[str] = Counter()
    for chars in rows:
        candidate += chars
    larger = max(sum(markdown_chars.values()), sum(candidate.values()), 1)
    return sum((markdown_chars & candidate).values()) / larger


def _heading_number(title: str) -> str | None:
    """The Arabic number a heading title opens with (`5.2`, `6`), or None."""

    arabic = _ARABIC_ORDINAL_TITLE.match(_TITLE_LEAD_MARKS.sub("", title))
    return (arabic.group(1) or arabic.group(2)) if arabic else None


def _continues(above: str, number: str) -> bool:
    """`number` is a child of `above` (`7.1` under `7`) or the sibling right
    after it (`8` after `7`, `2.3` after `2.2`)."""

    if number.startswith(f"{above}."):
        return True
    a, n = above.split("."), number.split(".")
    return len(a) == len(n) and a[:-1] == n[:-1] and int(n[-1]) == int(a[-1]) + 1


def _push_heading(
    stack: list[tuple[int, str]],
    interrupted: list[tuple[int, str]] | None,
    level: int,
    title: str,
) -> list[tuple[int, str]] | None:
    """Put a heading on `stack`; return the numbered context an unnumbered
    heading interrupted, if any.

    An unnumbered heading (`Trình độ Ngoại ngữ:`, a table caption) gets the
    extractor's raw level and can close a numbered section it actually sits
    inside. When a later numbered heading continues that numbering (`2.` after
    `1.`, `8.` after `7.1`), the interrupted context is put back; a numbering
    that starts over (`1.` in an appendix) is a new context and is left alone."""

    number = _heading_number(title)
    if number and interrupted and not any(_heading_number(t) for _l, t in stack):
        for index, (_level, above_title) in enumerate(interrupted):
            above = _heading_number(above_title)
            if above and _continues(above, number):
                child = number.startswith(f"{above}.")
                stack[:] = interrupted[: index + 1 if child else index]
                interrupted = None
                break
    before = list(stack)
    while stack and stack[-1][0] >= level:
        stack.pop()
    # `4.1.4` belongs under `4.` / `4.1`, never under `3.1` - even when
    # the extractor missed the `4.` and `4.1` headings in between
    while number and stack:
        above = _heading_number(stack[-1][1])
        if above is None or number.startswith(f"{above}."):
            break
        stack.pop()
    if not number and any(_heading_number(t) for _l, t in before[len(stack) :]):
        interrupted = before
    if title:
        stack.append((level, title))
    return interrupted


def _heading_level_for_title(title: str, fallback_level: int) -> int:
    """Override a heading's nesting level based on its own numbering
    scheme, regardless of what raw level `pymupdf4llm`'s font-size
    heuristic (or the promotion heuristic) assigned it.

    Vietnamese business documents commonly mix Roman-numeral top-level
    sections (`I.`/`II.`/`III.`) with Arabic-numeral subsections
    (`1.`/`2.`/`3.`) that `pymupdf4llm` renders at the EXACT SAME markdown
    heading level - it only looks at font size, with no notion that "1."
    is semantically nested inside "II.". Trusting that raw level naively
    would let a `##`-level "1. Module..." pop `##`-level "II. Phân tích..."
    off `heading_stack` as if they were siblings, permanently losing "II."
    from every subsequent chunk's `heading_path` (confirmed against a real
    PDF, `BAS.pdf`, where this happened). Forcing Roman-numbered titles to
    a shallower level than Arabic-numbered ones - independent of the raw
    markdown level - keeps the Roman section on the stack while Arabic
    subsections correctly nest under it.

    Arabic numbering is ranked by its depth (`6.` < `5.2` < `2.1.2`): the
    extractor gives `# 5.2 ...` and a bold `6. ...` unrelated raw levels, and
    only the number says that `6.` closes `5.2` rather than nesting in it.
    A lowercase lettered title (`a.`, `b)`) nests under the numbered heading
    before it."""

    bare = _TITLE_LEAD_MARKS.sub("", title)
    if _ROMAN_ORDINAL_TITLE.match(bare):
        return 2
    number = _heading_number(title)
    if number:
        return 3 + number.count(".")
    if _LETTER_ORDINAL_TITLE.match(bare):
        return _LETTER_HEADING_LEVEL
    return fallback_level


def _is_all_caps_title(stripped_line: str) -> bool:
    """True if the line, once its ordinal prefix/markup is removed, is
    entirely upper-case letters (Unicode-aware, so Vietnamese diacritics
    count correctly) - a strong, font-size-independent signal that a line
    is a section title rather than a sentence, used for a Roman-numeral
    heading that is bold but NOT larger than body text (so neither
    `pymupdf4llm`'s own heading detection nor the underline/table-list
    promotion signals fire on it)."""

    remainder = _HTML_TAG.sub("", _ORDINAL_PREFIX.sub("", stripped_line)).strip("* ")
    return bool(remainder) and remainder.isupper()


def _clean_heading_title(text: str) -> str:
    """Strip HTML tags (e.g. `<u>`) and markdown emphasis markers (`**`/`__`)
    from a heading title, so `heading_path`/`SourceLocator.section` carry
    plain text - `pymupdf4llm` renders bold spans as `**text**` even inside
    a detected heading line, and this same cleanup also runs on a
    numbered-line title promoted to a heading by `_looks_like_pdf_heading`
    (which may still carry its source `<u>...</u>` wrapper)."""

    return _MARKDOWN_EMPHASIS.sub("", _HTML_TAG.sub("", text)).strip()


def _find_pdf_page_boilerplate(pages: list[dict[str, object]]) -> set[str]:
    """Find plain-text lines that repeat verbatim across 2+ distinct pages.

    A running header/footer (e.g. an institution name printed on every
    page) is not real document content - if a numbered line right after it
    gets promoted to a heading by `_looks_like_pdf_heading`, that boilerplate
    would otherwise become its own tiny "chunk rác" (a region of nothing but
    repeated letterhead text) once the heading promotion `flush()`es right
    after it. Only plain TEXT lines are considered (not table rows or lines
    already recognized as `#`-headings) to avoid ever discarding a
    genuinely repeated table header or heading."""

    line_pages: dict[str, set[int]] = {}
    for page in pages:
        page_number = int(page["metadata"]["page_number"])  # type: ignore[index]
        page_text: str = page["text"]  # type: ignore[assignment]
        seen_on_this_page: set[str] = set()
        for line in page_text.splitlines():
            stripped = line.strip()
            if (
                stripped
                and stripped not in seen_on_this_page
                and not stripped.startswith("|")
                and not _MARKDOWN_HEADING.match(stripped)
            ):
                line_pages.setdefault(stripped, set()).add(page_number)
                seen_on_this_page.add(stripped)
    return {line for line, page_numbers in line_pages.items() if len(page_numbers) >= 2}


def _plain_text(text: str) -> str:
    return " ".join(_MARKDOWN_EMPHASIS.sub("", _HTML_TAG.sub("", text)).split())


def _only_boilerplate_cells(row: str, boilerplate_text: set[str]) -> bool:
    """A table row whose only text is a running header/footer the extractor
    pulled into the table (`|||...|KL7.0a|` under a page that prints `KL7.0a`
    at the top of every page)."""

    cells = [_plain_text(cell) for cell in _markdown_row_cells(row)]
    filled = [cell for cell in cells if cell]
    return bool(filled) and all(cell in boilerplate_text for cell in filled)


def _looks_like_pdf_heading(
    stripped: str, flat_lines: list[tuple[int, str]], index: int, boilerplate: set[str]
) -> bool:
    """Decide whether an ordinal body-text line (same font size as the
    surrounding paragraph, so `pymupdf4llm`'s own `#`-heading detection
    never fires on it) should still be promoted to a heading.

    Three independent signals, any one sufficient: (1) the line carries a
    `<u>...</u>` underline span - a real PDF a human would recognize as an
    emphasized section title despite plain font size; (2) the line, once
    its ordinal prefix is stripped, is entirely upper-case - a Roman-
    numeral section title (`"III. SO DO CAC ACTORS..."`) rendered bold-but-
    not-larger, so it clears no font-size threshold at all - see
    `_is_all_caps_title`; or (3) the very next non-blank, non-boilerplate
    line starts a table (`|`) or a bullet-list item (`-`) - an ordinal line
    that introduces a table/bullet-list is, in practice, always a section
    title rather than a mid-paragraph sentence (deliberately NOT extended
    to a numbered-list item right after, e.g. `"1. "` - that would promote
    every item of an ordinary numbered list into its own heading). Plain
    numbered clauses in body text (most of this pattern's matches, e.g.
    "1. Ho ten sinh vien: ...") are mixed-case and followed by another
    paragraph or another numbered line, so none of the three signals fire
    and they stay as ordinary text - this is a document-specific
    heuristic (not a general "ordinal line == heading" rule), confirmed
    against 2 real PDFs where a table's own intro line (underline-styled),
    a bold sibling subsection sitting one level below a `##` (bullet-
    list-styled), and 2 bold-but-not-larger Roman-numeral section titles
    (all-caps-styled) each needed exactly one of these 3 signals."""

    if _UNDERLINE_TAG.search(stripped) or _is_all_caps_title(stripped):
        return True

    for _page_number, next_line in flat_lines[index + 1 :]:
        next_stripped = next_line.strip()
        if not next_stripped or next_stripped in boilerplate:
            continue
        return next_stripped.startswith("|") or next_stripped.startswith("-")
    return False


def _regions_from_pdf_pages(
    pages: list[dict[str, object]],
    evidence: dict[int, list[TableEvidence]] | None = None,
) -> list[ParsedRegion]:
    """Group each page's markdown into heading-aware text/table regions.

    `pymupdf4llm.to_markdown(document, page_chunks=True)` returns one dict
    per page (each with a `text` markdown string and `metadata.page_number`)
    instead of one big string for the whole document - walking page by page
    (instead of the previous single `to_markdown(document)` call) is what
    lets each region carry a real `page_start`/`page_end`. The heading stack
    itself is NOT reset between pages: a heading on page 1 still applies to
    text on page 2 until a same-or-higher-level heading replaces it, exactly
    like the HTML heading stack.

    `pymupdf4llm` renders detected headings as markdown `#`..`######` lines
    (font-size heuristic) and detected tables as markdown pipe tables - both
    read the same way `_regions_from_html`/`_regions_from_docx` do it for
    their own formats, just sourced from markdown instead of tags/XML.
    """

    regions: list[ParsedRegion] = []
    heading_stack: list[tuple[int, str]] = []
    interrupted: list[tuple[int, str]] | None = None
    buffer: list[str] = []
    current_type: RegionType | None = None
    block_index = 0
    region_start_page: int | None = None
    evidence = evidence or {}
    used_evidence: dict[int, set[int]] = {}
    evidence_by_block: dict[int, TableEvidence | None] = {}

    def evidence_for_table(page_number: int, lines: list[str]) -> TableEvidence | None:
        """The page's geometric table that holds the same text as these markdown
        lines. Matched by CONTENT, not by order or count: the layout-driven
        markdown and the line-driven geometry may disagree on how many tables a
        page has (a side box, a split table), but they never disagree on the
        text of the table they both found."""

        candidates = evidence.get(page_number, [])
        if not candidates:
            return None
        taken = used_evidence.setdefault(page_number, set())
        markdown_chars = Counter(normalize_for_compare(" ".join(lines)))
        best_index, best_score = None, 0.0
        for index, candidate in enumerate(candidates):
            if index in taken:
                continue
            row_chars = [
                Counter(normalize_for_compare(" ".join(cell.text for cell in row.cells if cell)))
                for row in candidate.rows
            ]
            score = max(
                _match_score(markdown_chars, row_chars[lead : len(row_chars) - trail])
                for lead in range(_EVIDENCE_EDGE_ROWS + 1)
                for trail in range(_EVIDENCE_EDGE_ROWS + 1)
                if lead + trail < len(row_chars)
            )
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None or best_score < _EVIDENCE_MATCH_MIN:
            return None
        taken.add(best_index)
        return candidates[best_index]

    def flush(end_page: int) -> None:
        nonlocal current_type, block_index, region_start_page
        text = "\n".join(buffer).strip()
        lines_snapshot = list(buffer)
        buffer.clear()
        if text and current_type is not None:
            table_page = region_start_page or end_page
            table = None
            if current_type == RegionType.TABLE:
                table_evidence = evidence_for_table(table_page, lines_snapshot)
                table = build_page_table(lines_snapshot, table_page, table_evidence)
                # Geometry that did not line up with the markdown is not evidence
                # about this table - the merge must not lean on it either.
                aligned_evidence = (
                    None if "evidence_unaligned" in table.warnings else table_evidence
                )
                evidence_by_block[block_index] = aligned_evidence
            for before in table.text_before if table is not None else []:
                regions.append(
                    ParsedRegion(
                        RegionType.TEXT,
                        before,
                        heading_path=[title for _level, title in heading_stack],
                        page_start=table_page,
                        page_end=table_page,
                        block_index=block_index,
                        source_type=SourceType.PDF,
                    )
                )
                block_index += 1
            regions.append(
                ParsedRegion(
                    current_type,
                    text,
                    heading_path=[title for _level, title in heading_stack],
                    page_start=region_start_page,
                    page_end=end_page,
                    block_index=block_index,
                    source_type=SourceType.PDF,
                    table=table,
                )
            )
            block_index += 1
            for spilled in table.spilled_text if table is not None else []:
                regions.append(
                    ParsedRegion(
                        RegionType.TEXT,
                        spilled,
                        heading_path=[title for _level, title in heading_stack],
                        page_start=end_page,
                        page_end=end_page,
                        block_index=block_index,
                        source_type=SourceType.PDF,
                    )
                )
                block_index += 1
        current_type = None
        region_start_page = None

    boilerplate = _find_pdf_page_boilerplate(pages)
    flat_lines: list[tuple[int, str]] = []
    for page in pages:
        page_number = int(page["metadata"]["page_number"])  # type: ignore[index]
        page_text: str = page["text"]  # type: ignore[assignment]
        flat_lines.extend((page_number, line) for line in page_text.splitlines())

    page_number_lines = _find_pdf_page_number_lines(flat_lines, boilerplate)
    boilerplate_text = {_plain_text(line) for line in boilerplate}

    last_page_number = 0
    for index, (page_number, line) in enumerate(flat_lines):
        stripped = line.strip()
        last_page_number = page_number
        if stripped in boilerplate or index in page_number_lines:
            continue
        if stripped.startswith("|") and _only_boilerplate_cells(stripped, boilerplate_text):
            continue

        heading_match = _MARKDOWN_HEADING.match(stripped)
        is_promoted_heading = bool(_ORDINAL_LINE.match(stripped)) and _looks_like_pdf_heading(
            stripped, flat_lines, index, boilerplate
        )
        raw_title = heading_match.group(2) if heading_match else stripped
        title = _clean_heading_title(raw_title)
        if (heading_match or is_promoted_heading) and len(title) > settings.INGEST_MAX_HEADING_CHARS:
            # No real section title is this long (a form's fill-in line, a bold paragraph):
            # as a heading it would be prefixed into every chunk below it. Keep it as text.
            heading_match, is_promoted_heading = None, False
            line = raw_title
        if heading_match or is_promoted_heading:
            flush(page_number)
            raw_level = (
                len(heading_match.group(1))
                if heading_match
                else _PDF_PROMOTED_HEADING_FALLBACK_LEVEL
            )
            level = _heading_level_for_title(title, raw_level)
            interrupted = _push_heading(heading_stack, interrupted, level, title)
            continue

        line_type = RegionType.TABLE if stripped.startswith("|") else RegionType.TEXT
        if current_type is None:
            current_type = line_type
            region_start_page = page_number
        elif line_type != current_type:
            flush(page_number)
            current_type = line_type
            region_start_page = page_number
        buffer.append(line)

        # A TABLE region is always flushed at the page boundary, even with
        # no heading/type change - pymupdf4llm re-detects table structure
        # per page, so a table "continuing" onto the next page would often
        # re-emit its own header/separator row, which `build_page_table`
        # would misread as a data row if the two pages' lines were simply
        # concatenated. A TEXT region, in contrast, is plain prose/list
        # content with no such per-page structure, so it is allowed to run
        # across a page break uninterrupted - a bullet list split by a page
        # break otherwise became two separate regions with no heading
        # repeated and no `RecursiveChunker` overlap carried between them.
        is_last_line_of_page = (
            index + 1 == len(flat_lines) or flat_lines[index + 1][0] != page_number
        )
        if is_last_line_of_page and current_type == RegionType.TABLE:
            flush(page_number)
    flush(last_page_number)
    return _finalize_pdf_tables(regions, evidence_by_block, _find_pdf_furniture(pages))


def _find_pdf_page_number_lines(
    flat_lines: list[tuple[int, str]], boilerplate: set[str]
) -> set[int]:
    """Indexes of the page-number lines left in the text (`10`, `Trang 2`,
    `2/24`): a short line on the first or last content line of its page whose
    number runs with the page - number minus page is the same on 2+ pages for
    lines of the same shape. A code or year that happens to end a page (`7340301`)
    does not run with the page and stays."""

    content: dict[int, list[int]] = {}
    for index, (page_number, line) in enumerate(flat_lines):
        if line.strip() and line.strip() not in boilerplate:
            content.setdefault(page_number, []).append(index)
    candidates: list[tuple[int, tuple[str, int], int]] = []
    for page_number, indices in content.items():
        for index in {indices[0], indices[-1]}:
            stripped = flat_lines[index][1].strip()
            if stripped.startswith("|") or _MARKDOWN_HEADING.match(stripped):
                continue
            plain = _EMPHASIS_MARKS.sub("", stripped).strip()
            number = _DIGIT_RUN.search(plain)
            if number is None or len(plain) > _PAGE_NUMBER_MAX_CHARS:
                continue
            shape = (furniture_key(plain), int(number.group()) - page_number)
            candidates.append((index, shape, page_number))
    pages_by_shape: dict[tuple[str, int], set[int]] = {}
    for _index, shape, page_number in candidates:
        pages_by_shape.setdefault(shape, set()).add(page_number)
    return {index for index, shape, _page in candidates if len(pages_by_shape[shape]) >= 2}


def _find_pdf_furniture(pages: list[dict[str, object]]) -> set[str]:
    """Shapes of plain-text lines that recur across 2+ pages once digits are
    masked (`Trang 2/4`, `Trang 3/4` -> `Trang #/#`): running page furniture.
    Unlike `_find_pdf_page_boilerplate` this is NOT used to drop text - only to
    let such a line sit between two tables without breaking their continuity."""

    key_pages: dict[str, set[int]] = {}
    for page in pages:
        page_number = int(page["metadata"]["page_number"])  # type: ignore[index]
        for line in str(page["text"]).splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("|") and not _MARKDOWN_HEADING.match(stripped):
                key_pages.setdefault(furniture_key(stripped), set()).add(page_number)
    return {key for key, numbers in key_pages.items() if len(numbers) >= 2}


def _only_furniture(regions: list[ParsedRegion], furniture: set[str]) -> bool:
    for region in regions:
        for line in region.content.splitlines():
            if line.strip() and furniture_key(line) not in furniture:
                return False
    return True


def _layout_of(
    table: TableBlock | None, evidence: TableEvidence | None, current: list[float] | None
) -> list[float] | None:
    """The column edges a table chain is compared against: the latest page whose
    geometry has as many columns as the table, else the ones already held."""

    if table is not None and evidence is not None:
        edges = column_edges(evidence)
        if len(edges) - 1 == len(table.header_row or []):
            return edges
    return current


def _finalize_pdf_tables(
    regions: list[ParsedRegion],
    evidence_by_block: dict[int, TableEvidence | None],
    furniture: set[str],
) -> list[ParsedRegion]:
    """Merge a table that continues on the next page into one logical table,
    then give every table its document-order `table_id` and renumber
    `block_index` so it stays contiguous after merging.

    Page furniture (a page number line) between the two tables is kept in the
    output - nothing is dropped - it just no longer blocks the merge.

    When the next page was read with a different number of columns (a rule the
    extractor missed or added), it is first re-read against the columns of the
    table it may continue, by geometry, so the two can be compared at all."""

    output: list[ParsedRegion] = []
    chain_evidence: dict[int, TableEvidence | None] = {}
    # column edges of the page that set each chain's columns
    chain_layout: dict[int, list[float] | None] = {}
    last_table: int | None = None
    between: list[ParsedRegion] = []

    for region in regions:
        if region.region_type != RegionType.TABLE or region.table is None:
            output.append(region)
            between.append(region)
            continue

        evidence = (
            evidence_by_block.get(region.block_index) if region.block_index is not None else None
        )
        if last_table is not None:
            previous = output[last_table]
            candidate = region.table
            layout = chain_layout.get(last_table)
            previous_columns = len(previous.table.header_row or [])  # type: ignore[union-attr]
            if (
                layout is not None
                and evidence is not None
                and len(layout) - 1 == previous_columns
                and len(region.table.header_row or []) != previous_columns
            ):
                candidate = (
                    reproject_columns(region.table, column_edges(evidence), layout)
                    or region.table
                )
            decision = decide_merge(
                MergeCandidate(
                    previous.table,  # type: ignore[arg-type]
                    previous.page_start or 0,
                    previous.page_end or 0,
                    previous.heading_path,
                    chain_evidence.get(last_table),
                ),
                MergeCandidate(
                    candidate,
                    region.page_start or 0,
                    region.page_end or 0,
                    region.heading_path,
                    evidence,
                ),
                only_furniture_between=_only_furniture(between, furniture),
            )
            if decision.merge:
                output[last_table] = replace(
                    previous,
                    content=previous.content + chr(10) + region.content,
                    page_end=region.page_end,
                    table=merge_page_tables(
                        previous.table,  # type: ignore[arg-type]
                        candidate,
                        header_present=decision.header_present,
                    ),
                )
                chain_evidence[last_table] = evidence
                chain_layout[last_table] = _layout_of(
                    output[last_table].table, evidence, layout
                )
                between = []
                continue
            if decision.borderline:
                region = replace(
                    region,
                    table=replace(
                        region.table, warnings=[*region.table.warnings, "merge_borderline"]
                    ),
                )
        output.append(region)
        last_table = len(output) - 1
        chain_evidence[last_table] = evidence
        chain_layout[last_table] = _layout_of(region.table, evidence, None)
        between = []

    table_number = 0
    finalized: list[ParsedRegion] = []
    for block_index, region in enumerate(output):
        table = region.table
        if region.region_type == RegionType.TABLE and table is not None:
            table = replace(table, table_id=f"table-{table_number}")
            table_number += 1
        finalized.append(replace(region, block_index=block_index, table=table))
    return finalized


def _iter_docx_block_items(document: Document) -> list[DocxParagraph | DocxTable]:
    """Yield a docx's paragraphs and tables in original document order.

    `python-docx`'s own `document.paragraphs` and `document.tables` are two
    separate, unordered-relative-to-each-other lists; walking the body XML
    directly is the standard way to get them interleaved correctly.
    """

    blocks: list[DocxParagraph | DocxTable] = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            blocks.append(DocxParagraph(child, document))
        elif child.tag == qn("w:tbl"):
            blocks.append(DocxTable(child, document))
    return blocks


def _docx_table_to_markdown(table: DocxTable) -> str:
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    header, *body_rows = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body_rows)
    return "\n".join(lines)


def _docx_row_is_header(row: DocxRow) -> bool:
    """Whether a docx table row's `<w:tr>` carries a real
    `<w:trPr><w:tblHeader/>` - Word's own "repeat this row as a header"
    attribute, read via raw XML (there is no `python-docx` high-level API
    for it)."""

    tr_pr = row._tr.trPr
    if tr_pr is None:
        return False
    return tr_pr.find(qn("w:tblHeader")) is not None


def _docx_table_to_block(table: DocxTable) -> TableBlock:
    """Parse a Word table directly into a `TableBlock` using
    `<w:tblHeader/>` - the row property Word itself sets when a user ticks
    "Repeat as header row" - as the real structural header signal, instead
    of assuming the first row is always a header.

    Many real-world DOCX tables never have this property set (most users
    never open the table properties dialog to tick it) - those tables
    correctly land in `MISSING`, not a guessed header. This is a deliberate
    trade-off (see plan Risks), not a bug: guessing from "row 1 looks
    header-ish" was the exact failure mode this plan set out to fix.
    """

    header_row: list[str] | None = None
    data_rows: list[list[str]] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        if _docx_row_is_header(row) and header_row is None:
            header_row = cells
            continue
        data_rows.append(cells)

    if header_row is None:
        return build_table_from_plain_rows(None, data_rows, HeaderSource.MISSING, 0.0)
    return build_table_from_plain_rows(header_row, data_rows, HeaderSource.EXPLICIT, 1.0)


_DOCX_HEADING_STYLE = re.compile(r"^Heading (\d)$")


def _regions_from_docx(content: bytes) -> list[ParsedRegion]:
    """Split a `.docx` into text/table regions by reading its real Word
    tables (`<w:tbl>`) directly via `python-docx`, not through `pymupdf4llm`.

    `pymupdf4llm.to_markdown()` flattens Word tables into plain paragraph
    text with no pipe syntax at all for `.docx` (unlike PDF, where it does
    detect tables) - there is nothing for the PDF markdown heuristic to
    find, so DOCX tables need this dedicated path instead.

    Headings are recognized by Word's own built-in paragraph style name
    ("Heading 1".."Heading 9") - this is a different concept from a table's
    *header row* (see `_docx_table_to_block`): one is document
    section structure, the other is "does this table's first row repeat as
    a header".
    """

    document = open_docx(BytesIO(content))
    regions: list[ParsedRegion] = []
    heading_stack: list[tuple[int, str]] = []
    text_buffer: list[str] = []
    block_index = 0

    def flush_text() -> None:
        nonlocal block_index
        text = "\n".join(text_buffer).strip()
        text_buffer.clear()
        if text:
            regions.append(
                ParsedRegion(
                    RegionType.TEXT,
                    text,
                    heading_path=[title for _level, title in heading_stack],
                    block_index=block_index,
                    source_type=SourceType.DOCX,
                )
            )
            block_index += 1

    for block in _iter_docx_block_items(document):
        if isinstance(block, DocxTable):
            flush_text()
            table_markdown = _docx_table_to_markdown(block)
            if table_markdown:
                regions.append(
                    ParsedRegion(
                        RegionType.TABLE,
                        table_markdown,
                        heading_path=[title for _level, title in heading_stack],
                        block_index=block_index,
                        source_type=SourceType.DOCX,
                        table=_docx_table_to_block(block),
                    )
                )
                block_index += 1
            continue

        style_name = block.style.name if block.style is not None else None
        heading_level_match = _DOCX_HEADING_STYLE.match(style_name or "")
        paragraph_text = block.text.strip()
        if heading_level_match:
            flush_text()
            level = int(heading_level_match.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            if paragraph_text:
                heading_stack.append((level, paragraph_text))
            continue

        if paragraph_text:
            text_buffer.append(paragraph_text)
    flush_text()
    return regions


_HEADING_LEVELS = {f"h{i}": i for i in range(1, 7)}
_NOISE_TAGS = ("script", "style", "nav", "footer", "header")
_LIST_TAGS = ("ul", "ol")


def _regions_from_html(content: bytes) -> list[ParsedRegion]:
    """Split HTML into heading-hierarchy-aware text/table regions.

    Chunking by raw character count ignores document structure and can glue
    unrelated sections into one chunk. Instead: strip non-content tags
    (script/style/nav/footer/header), walk the body in document order
    tracking a heading stack (h1..h6) to build a "H1 > H2 > H3" path, and
    group paragraphs/lists under that path into one region per section -
    each region is prefixed with its heading path so retrieval keeps the
    surrounding context even after a section is later split further by a
    generic size-based chunker. Tables become their own region (one
    markdown pipe-table per `<table>`, also heading-path-prefixed) so a
    table is never split mid-row by a downstream chunker that only knows
    about paragraph/token boundaries.
    """

    soup = BeautifulSoup(content, "lxml")
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    root = soup.body or soup
    regions: list[ParsedRegion] = []
    heading_stack: list[tuple[int, str]] = []
    text_buffer: list[str] = []
    block_index = 0

    def current_heading_path() -> list[str]:
        return [text for _level, text in heading_stack]

    def flush_text() -> None:
        nonlocal block_index
        text = "\n\n".join(text_buffer).strip()
        text_buffer.clear()
        if text:
            regions.append(
                ParsedRegion(
                    RegionType.TEXT,
                    text,
                    heading_path=current_heading_path(),
                    block_index=block_index,
                    source_type=SourceType.HTML,
                )
            )
            block_index += 1

    def visit(node: Tag) -> None:
        nonlocal block_index
        for child in node.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.lower() if child.name else ""

            if name in _HEADING_LEVELS:
                flush_text()
                level = _HEADING_LEVELS[name]
                heading_text = child.get_text(" ", strip=True)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                if heading_text:
                    heading_stack.append((level, heading_text))
                continue

            if name == "p":
                paragraph_text = child.get_text(" ", strip=True)
                if paragraph_text:
                    text_buffer.append(paragraph_text)
                continue

            if name in _LIST_TAGS:
                list_markdown = _html_list_to_markdown(child)
                if list_markdown:
                    text_buffer.append(list_markdown)
                continue

            if name == "table":
                flush_text()
                table_markdown = _html_table_to_markdown(child)
                if table_markdown:
                    regions.append(
                        ParsedRegion(
                            RegionType.TABLE,
                            table_markdown,
                            heading_path=current_heading_path(),
                            block_index=block_index,
                            source_type=SourceType.HTML,
                            table=_html_table_to_block(child),
                        )
                    )
                    block_index += 1
                continue

            # Not a leaf content tag (div/section/article/span/...) - recurse
            # to find content nested inside it, in document order.
            visit(child)

    visit(root)
    flush_text()
    return regions


def _html_list_to_markdown(list_tag: Tag) -> str:
    """Render a `<ul>`/`<ol>` as a markdown bullet/numbered list, one line
    per direct `<li>` (nested lists are flattened into that same line via
    `get_text` rather than indented - good enough for retrieval, which reads
    the list as prose, not for re-rendering a nested outline)."""

    ordered = list_tag.name == "ol"
    lines: list[str] = []
    for index, item in enumerate(list_tag.find_all("li", recursive=False), start=1):
        item_text = item.get_text(" ", strip=True)
        if not item_text:
            continue
        marker = f"{index}." if ordered else "-"
        lines.append(f"{marker} {item_text}")
    return "\n".join(lines)


def _html_table_to_markdown(table_tag: Tag) -> str:
    """Render an HTML `<table>` as a markdown pipe table - never split a row
    across chunks the way naively flattening cells to plain text would.

    This is a display-only rendering used for `ParsedRegion.content` (kept
    for chunkers that consume a region's raw content directly, bypassing
    `TableRowChunker`); it does not escape `|`/newlines the way
    `TableRowChunker`'s own markdown rendering does - the
    structural source of truth for chunking is `_html_table_to_block`,
    not this string.
    """

    rows: list[list[str]] = []
    for row in table_tag.find_all("tr"):
        cells = row.find_all(("th", "td"))
        row_text = [cell.get_text(" ", strip=True) for cell in cells]
        if row_text:
            rows.append(row_text)
    if not rows:
        return ""

    header, *body_rows = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body_rows)
    return "\n".join(lines)


def _html_table_to_block(table_tag: Tag) -> TableBlock:
    """Parse an HTML `<table>` directly into a `TableBlock`, using `<th>`
    presence as the real structural header signal - NOT by reading the
    markdown rendering back out (that was v1's approach and could not
    distinguish "no header" from "header row happens to look like data").

    Only the FIRST row counts: if every cell in it is a `<th>`, that row is
    the verified header (`EXPLICIT`, confidence 1.0). Any other case
    (no `<th>` anywhere, or `<th>` cells scattered outside a fully-`<th>`
    first row) is treated as `MISSING` - deliberately not guessing at a
    header from a partial/ambiguous signal (see plan Risks: no heuristic
    fallback for "looks like a header").
    """

    rows: list[list[str]] = []
    first_row_all_th = False
    for index, row in enumerate(table_tag.find_all("tr")):
        cells = row.find_all(("th", "td"))
        row_text = [cell.get_text(" ", strip=True) for cell in cells]
        if not row_text:
            continue
        if index == 0 and cells and all(cell.name == "th" for cell in cells):
            first_row_all_th = True
        rows.append(row_text)

    if not rows:
        return build_table_from_plain_rows(None, [], HeaderSource.MISSING, 0.0)

    if first_row_all_th:
        header_row, *data_rows = rows
        return build_table_from_plain_rows(header_row, data_rows, HeaderSource.EXPLICIT, 1.0)

    return build_table_from_plain_rows(None, rows, HeaderSource.MISSING, 0.0)
