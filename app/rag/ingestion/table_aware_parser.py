from dataclasses import dataclass
from io import BytesIO

import pymupdf
import pymupdf4llm
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from docx import Document as open_docx
from docx.document import Document
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from app.core.exceptions import UnsupportedFileTypeException
from app.schemas.ingestion import RegionType


@dataclass(frozen=True)
class ParsedRegion:
    """One contiguous block of a document, tagged as running text or a table."""

    region_type: RegionType
    content: str


def split_regions(content: bytes, filename: str, extension: str) -> list[ParsedRegion]:
    """Split a PDF/DOCX/HTML into ordered text/table regions for chunking.

    `.txt` and `.xlsx` are not routed through this parser: `.txt` has no
    table structure to preserve, and `.xlsx` is chunked row-by-row by its own
    dedicated strategy (see `chunking/excel_rows.py`).
    """

    if extension == "txt":
        text = content.decode("utf-8").strip()
        return [ParsedRegion(RegionType.TEXT, text)] if text else []
    if extension == "docx":
        return _regions_from_docx(content)
    if extension in ("html", "htm"):
        return _regions_from_html(content)
    if extension != "pdf":
        raise UnsupportedFileTypeException(filename)

    document = pymupdf.open(stream=content, filetype=extension)
    try:
        markdown: str = pymupdf4llm.to_markdown(document)
    finally:
        document.close()
    return _regions_from_markdown(markdown)


def _regions_from_markdown(markdown: str) -> list[ParsedRegion]:
    """Group markdown lines into text/table regions using pipe-table syntax.

    Used for PDF only: `pymupdf4llm` detects PDF tables via page vector
    graphics and emits real markdown pipe tables. It does not do the same
    for `.docx` (see `_regions_from_docx`), which reads Word tables
    (`<w:tbl>`) directly instead.
    """

    regions: list[ParsedRegion] = []
    buffer: list[str] = []
    current_type: RegionType | None = None

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text and current_type is not None:
            regions.append(ParsedRegion(current_type, text))
        buffer.clear()

    for line in markdown.splitlines():
        line_type = RegionType.TABLE if line.strip().startswith("|") else RegionType.TEXT
        if current_type is None:
            current_type = line_type
        elif line_type != current_type:
            flush()
            current_type = line_type
        buffer.append(line)
    flush()
    return regions


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


def _regions_from_docx(content: bytes) -> list[ParsedRegion]:
    """Split a `.docx` into text/table regions by reading its real Word
    tables (`<w:tbl>`) directly via `python-docx`, not through `pymupdf4llm`.

    `pymupdf4llm.to_markdown()` flattens Word tables into plain paragraph
    text with no pipe syntax at all for `.docx` (unlike PDF, where it does
    detect tables) - there is nothing for the PDF markdown heuristic to
    find, so DOCX tables need this dedicated path instead.
    """

    document = open_docx(BytesIO(content))
    regions: list[ParsedRegion] = []
    text_buffer: list[str] = []

    def flush_text() -> None:
        text = "\n".join(text_buffer).strip()
        if text:
            regions.append(ParsedRegion(RegionType.TEXT, text))
        text_buffer.clear()

    for block in _iter_docx_block_items(document):
        if isinstance(block, DocxTable):
            flush_text()
            table_markdown = _docx_table_to_markdown(block)
            if table_markdown:
                regions.append(ParsedRegion(RegionType.TABLE, table_markdown))
        else:
            paragraph_text = block.text.strip()
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

    def heading_path() -> str:
        return " > ".join(text for _level, text in heading_stack)

    def with_heading_path(body_text: str) -> str:
        prefix = heading_path()
        return f"{prefix}\n\n{body_text}" if prefix else body_text

    def flush_text() -> None:
        text = "\n\n".join(text_buffer).strip()
        text_buffer.clear()
        if text:
            regions.append(ParsedRegion(RegionType.TEXT, with_heading_path(text)))

    def visit(node: Tag) -> None:
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
                        ParsedRegion(RegionType.TABLE, with_heading_path(table_markdown))
                    )
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
    across chunks the way naively flattening cells to plain text would."""

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
