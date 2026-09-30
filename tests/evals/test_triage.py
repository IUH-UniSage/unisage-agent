import csv
from pathlib import Path

import pymupdf
import pytest

from evals.crawl.download import ERROR_FIELDS, MANIFEST_FIELDS, read_csv
from evals.crawl.triage import text_quality, triage


def _row(file_id: str, pages: int, text_chars: int) -> dict[str, str]:
    return {
        "file_id": file_id,
        "source_url": f"https://x.iuh.edu.vn/{file_id}.pdf",
        "source_page": "https://x.iuh.edu.vn/",
        "unit": "Phòng Đào tạo",
        "pages": str(pages),
        "text_chars": str(text_chars),
        "sha256": file_id * 4,
        "local_path": f"files/phong-dao-tao/{file_id}.pdf",
    }


GARBLED = "T6ng C6ng ty DAu tu ph6t tri6n D6 thi la don vf c6 b6 ddy lich sri voi hon 39 ndm. " * 8
ENGLISH = "The programme specification describes the learning outcomes of the course. " * 8


def _pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(36, 36, 560, 800), text, fontsize=8)
    content = document.tobytes()
    document.close()
    return content


def _dataset(tmp_path: Path) -> Path:
    rows = [
        _row("text", 2, 5000),
        _row("scan", 10, 30),
        _row("broken", 0, 0),
        _row("garbled", 1, 5000),
    ]
    bodies = {"text": _pdf(ENGLISH), "garbled": _pdf(GARBLED)}
    for row in rows:
        path = tmp_path / row["local_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bodies.get(row["file_id"], b"%PDF-1.7"))
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    with (tmp_path / "download_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=ERROR_FIELDS).writeheader()
    return tmp_path


def test_triage_moves_scans_and_drops_broken(tmp_path):
    dataset = _dataset(tmp_path)
    result = triage(dataset)

    assert (result.removed_broken, result.kept_text) == (1, 1)
    assert result.moved == {"scanned": 1, "garbled_ocr": 1}
    assert (dataset / "files/phong-dao-tao/text.pdf").exists()
    assert (dataset / "scanned_pdf/phong-dao-tao/scan.pdf").exists()
    assert (dataset / "scanned_pdf/phong-dao-tao/garbled.pdf").exists()
    assert not (dataset / "files/phong-dao-tao/scan.pdf").exists()
    assert not (dataset / "files/phong-dao-tao/broken.pdf").exists()

    manifest = {row["file_id"]: row for row in read_csv(dataset / "manifest.csv")}
    assert set(manifest) == {"text", "scan", "garbled"}
    assert manifest["scan"]["local_path"] == "scanned_pdf/phong-dao-tao/scan.pdf"
    assert {fid: row["quality"] for fid, row in manifest.items()} == {
        "text": "ok",
        "scan": "scanned",
        "garbled": "garbled_ocr",
    }
    errors = read_csv(dataset / "download_errors.csv")
    assert [(e["source_url"].rsplit("/", 1)[1], e["reason"]) for e in errors] == [
        ("broken.pdf", "unreadable")
    ]


def test_triage_is_idempotent(tmp_path):
    dataset = _dataset(tmp_path)
    triage(dataset)
    manifest = (dataset / "manifest.csv").read_text(encoding="utf-8")
    result = triage(dataset)

    assert (result.removed_broken, result.moved, result.kept_text) == (0, {}, 1)
    assert (dataset / "manifest.csv").read_text(encoding="utf-8") == manifest
    assert len(read_csv(dataset / "download_errors.csv")) == 1


@pytest.mark.parametrize(
    ("text", "quality"),
    [
        ("Quy chế đào tạo trình độ đại học theo hệ thống tín chỉ của trường. " * 10, "ok"),
        (ENGLISH, "ok"),
        ("A01 LE HUY TUONG PFCE155 A02 NGUYEN NGOC DINH BAO PFCE182 " * 20, "ok"),
        (
            "Lê Kim Ngân lớp DHHO18B đề tài 2026/ĐH-113 tổng hợp vật liệu nanocomposite "
            "ZnCo2O4/Fe2O3 ứng dụng xử lý môi trường, phòng B3.05 lúc 9h45 - 10h15. " * 6,
            "ok",
        ),
        (GARBLED, "garbled_ocr"),
        (
            "Thanh pho Ho Chi Minh, ngay 25 thang 4 nam 2016 cua Bo ve viec quy dinh "
            "dao tao cac truong dai hoc va sinh vien trong nam hoc. " * 6,
            "no_diacritics",
        ),
        ("\x00\x01\x02\x03\x04\x05\x06\x07\x03\x08" * 30, "broken_encoding"),
        ("", "ok"),
    ],
)
def test_text_quality(text, quality):
    assert text_quality(text) == quality


def test_triage_moves_false_positives_back(tmp_path):
    dataset = _dataset(tmp_path)
    triage(dataset)
    # Simulate an older, looser heuristic having set the good file aside.
    (dataset / "scanned_pdf/phong-dao-tao").mkdir(parents=True, exist_ok=True)
    (dataset / "files/phong-dao-tao/text.pdf").replace(
        dataset / "scanned_pdf/phong-dao-tao/text.pdf"
    )
    rows = read_csv(dataset / "manifest.csv")
    for row in rows:
        if row["file_id"] == "text":
            row["local_path"], row["quality"] = "scanned_pdf/phong-dao-tao/text.pdf", "garbled_ocr"
    with (dataset / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    result = triage(dataset)

    assert result.restored == 1
    assert (dataset / "files/phong-dao-tao/text.pdf").exists()
    manifest = {row["file_id"]: row for row in read_csv(dataset / "manifest.csv")}
    assert manifest["text"]["quality"] == "ok"
    assert manifest["text"]["local_path"] == "files/phong-dao-tao/text.pdf"
