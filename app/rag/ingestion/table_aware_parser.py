from dataclasses import dataclass

import pymupdf
import pymupdf4llm

from app.core.exceptions import UnsupportedFileTypeException
from app.schemas.ingestion import RegionType

_MARKDOWN_SUPPORTED_EXTENSIONS = {"pdf", "docx"}


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
    if extension not in _MARKDOWN_SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeException(filename)

    document = pymupdf.open(stream=content, filetype=extension)
    try:
        markdown: str = pymupdf4llm.to_markdown(document)
    finally:
        document.close()
    return _regions_from_markdown(markdown)


def _regions_from_markdown(markdown: str) -> list[ParsedRegion]:
    """Group markdown lines into text/table regions using pipe-table syntax."""

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
