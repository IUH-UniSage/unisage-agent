from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest

from app.rag.ingestion.table_aware_parser import ocr_language


@pytest.fixture
def tessdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(pymupdf, "get_tessdata", lambda: str(tmp_path))
    ocr_language.cache_clear()
    yield tmp_path
    ocr_language.cache_clear()


def test_keeps_every_installed_language(tessdata: Path) -> None:
    for lang in ("vie", "eng"):
        (tessdata / f"{lang}.traineddata").write_bytes(b"")
    assert ocr_language("vie+eng") == "vie+eng"


def test_drops_a_missing_language_instead_of_failing(tessdata: Path) -> None:
    (tessdata / "eng.traineddata").write_bytes(b"")
    assert ocr_language("vie+eng") == "eng"


def test_falls_back_to_english_when_nothing_matches(tessdata: Path) -> None:
    assert ocr_language("vie") == "eng"
