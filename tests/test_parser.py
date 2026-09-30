import pytest

from app.core.errors.exceptions import UnsupportedFileTypeException
from app.rag.ingestion.parser import extract_raw_text
from tests.fixtures.documents import make_docx_bytes, make_pdf_bytes


def test_extract_raw_text_from_txt() -> None:
    text = extract_raw_text(b"Hello from TXT fixture.", "handbook.txt")

    assert text == "Hello from TXT fixture."


def test_extract_raw_text_from_pdf() -> None:
    content = make_pdf_bytes("Hello from PDF fixture.")

    text = extract_raw_text(content, "handbook.pdf")

    assert "Hello from PDF fixture." in text


def test_extract_raw_text_from_docx() -> None:
    content = make_docx_bytes("Hello from DOCX fixture.")

    text = extract_raw_text(content, "handbook.docx")

    assert "Hello from DOCX fixture." in text


def test_extract_raw_text_from_html_strips_tags_and_noise() -> None:
    content = (
        b"<html><body>"
        b"<nav>Menu</nav><script>doEvilThings()</script>"
        b"<h1>Tieu de</h1><p>Hello from HTML fixture.</p>"
        b"<footer>Copyright</footer>"
        b"</body></html>"
    )

    text = extract_raw_text(content, "handbook.html")

    assert "Hello from HTML fixture." in text
    assert "Menu" not in text
    assert "doEvilThings" not in text
    assert "Copyright" not in text


def test_extract_raw_text_from_htm_extension() -> None:
    text = extract_raw_text(b"<p>Hello from HTM fixture.</p>", "handbook.htm")

    assert "Hello from HTM fixture." in text


def test_extract_raw_text_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeException):
        extract_raw_text(b"whatever", "handbook.doc")


def test_extract_raw_text_rejects_unknown_extension() -> None:
    with pytest.raises(UnsupportedFileTypeException):
        extract_raw_text(b"whatever", "handbook.exe")
