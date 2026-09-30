from pathlib import PurePosixPath

import pymupdf
from bs4 import BeautifulSoup

from app.core.errors.exceptions import UnsupportedFileTypeException

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
    if extension in ("html", "htm"):
        soup = BeautifulSoup(content, "lxml")
        for tag_name in ("script", "style", "nav", "footer", "header"):
            for tag in soup.find_all(tag_name):
                tag.decompose()
        return (soup.body or soup).get_text("\n", strip=True)
    if extension in _FITZ_SUPPORTED_EXTENSIONS:
        document = pymupdf.open(stream=content, filetype=extension)
        try:
            return "\n\n".join(
                document[page_index].get_text("text") for page_index in range(document.page_count)
            )
        finally:
            document.close()
    raise UnsupportedFileTypeException(filename)
