from dataclasses import dataclass
from io import BytesIO

import pymupdf
import pymupdf4llm
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
    """Split a PDF/DOCX into ordered text/table regions for chunking.

    `.txt` and `.xlsx` are not routed through this parser: `.txt` has no
    table structure to preserve, and `.xlsx` is chunked row-by-row by its own
    dedicated strategy (see `chunking/excel_rows.py`).
    """

    if extension == "txt":
        text = content.decode("utf-8").strip()
        return [ParsedRegion(RegionType.TEXT, text)] if text else []
    if extension == "docx":
        return _regions_from_docx(content)
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
