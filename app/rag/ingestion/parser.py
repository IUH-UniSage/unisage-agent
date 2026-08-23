from dataclasses import dataclass
from pathlib import PurePosixPath

import pymupdf

from app.core.exceptions import UnsupportedFileTypeException

_FITZ_SUPPORTED_EXTENSIONS = {"pdf", "docx"}


def get_extension(filename: str) -> str:
    """Return the lowercase file extension (without the dot) from a filename or object key."""

    return PurePosixPath(filename).suffix.lower().lstrip(".")


def extract_raw_text(content: bytes, filename: str) -> str:
    """Extract plain raw text from file bytes, dispatching on extension.

    `.doc` (legacy binary Word format) is intentionally not supported: the
    project's dependencies (`pymupdf4llm`) only parse OOXML (`.docx`), not
    the older binary format, and no converter for it is installed.
    """

    extension = get_extension(filename)
    if extension == "txt":
        return content.decode("utf-8")
    if extension in _FITZ_SUPPORTED_EXTENSIONS:
        document = pymupdf.open(stream=content, filetype=extension)
        try:
            return "\n\n".join(
                document[page_index].get_text("text") for page_index in range(document.page_count)
            )
        finally:
            document.close()
    raise UnsupportedFileTypeException(filename)


@dataclass(frozen=True)
class ParsedDocument:
    """Normalized document content ready for chunking."""

    source: str
    content: str
    metadata: dict[str, object]


def parse_text_document(
    source: str,
    content: str,
    metadata: dict[str, object] | None = None,
) -> ParsedDocument:
    """Normalize whitespace while preserving the document source."""

    normalized = "\n".join(line.strip() for line in content.splitlines() if line.strip())
    return ParsedDocument(source, normalized, metadata or {})
