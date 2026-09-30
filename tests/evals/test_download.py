import csv
from pathlib import Path

import httpx
import pymupdf
import pytest

from evals.crawl.discover import DISCOVERED_FIELDS, SOURCES_FIELDS
from evals.crawl.download import (
    MANIFEST_FIELDS,
    interleave_by_host,
    is_probably_scanned,
    read_csv,
    run,
)
from evals.crawl.http import PoliteClient


def _pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


PDF_A = _pdf("Quy che dao tao " * 20)
PDF_B = _pdf("Hoc phi nam hoc 2026 " * 20)

BODIES = {
    "/a.pdf": PDF_A,
    "/a-copy.pdf": PDF_A,
    "/b.pdf": PDF_B,
    "/fake.pdf": b"<html>not found page</html>",
    "/big.pdf": PDF_B + b"0" * 5000,
}


def _transport(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/robots.txt":
        return httpx.Response(404)
    body = BODIES.get(request.url.path)
    if body is None:
        return httpx.Response(404)
    return httpx.Response(200, headers={"content-type": "application/pdf"}, content=body)


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _dataset(tmp_path: Path) -> Path:
    _write(
        tmp_path / "sources.csv",
        SOURCES_FIELDS,
        [
            {"host": "x.iuh.edu.vn", "unit": "Phòng Đào tạo", "approved": "true"},
            {"host": "y.iuh.edu.vn", "unit": "Khoa Y", "approved": "false"},
        ],
    )
    pdfs = ["a.pdf", "a-copy.pdf", "b.pdf", "fake.pdf", "missing.pdf", "big.pdf"]
    rows = [
        {
            "pdf_url": f"https://x.iuh.edu.vn/{name}",
            "source_page": "https://x.iuh.edu.vn/",
            "host": "x.iuh.edu.vn",
            "unit": "Phòng Đào tạo",
            "campus": "HCM",
        }
        for name in pdfs
    ]
    rows.append(
        {
            "pdf_url": "https://y.iuh.edu.vn/a.pdf",
            "source_page": "https://y.iuh.edu.vn/",
            "host": "y.iuh.edu.vn",
            "unit": "Khoa Y",
            "campus": "HCM",
        }
    )
    _write(tmp_path / "discovered.csv", DISCOVERED_FIELDS, rows)
    return tmp_path


async def _run(dataset: Path) -> None:
    client = PoliteClient(min_interval=0, transport=httpx.MockTransport(_transport))
    try:
        await run(dataset, concurrency=2, client=client, max_bytes=len(PDF_B) + 100)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_download_builds_manifest_dedupes_and_records_errors(tmp_path):
    dataset = _dataset(tmp_path)
    await _run(dataset)

    manifest = read_csv(dataset / "manifest.csv")
    assert list(manifest[0]) == MANIFEST_FIELDS
    assert len(manifest) == 2  # a.pdf and a-copy.pdf share content; y.* not approved
    for row in manifest:
        assert (dataset / row["local_path"]).read_bytes()[:4] == b"%PDF"
        assert row["local_path"].startswith("files/phong-dao-tao/")
        assert row["pages"] == "1"
        assert int(row["text_chars"]) > 0
        assert row["access_level"] == ""

    reasons = {
        e["source_url"].rsplit("/", 1)[1]: e["reason"]
        for e in read_csv(dataset / "download_errors.csv")
    }
    assert reasons["fake.pdf"] == "not_pdf"
    assert reasons["missing.pdf"] == "status_404"
    assert reasons["big.pdf"] == "too_large"
    duplicate = next(name for name in ("a.pdf", "a-copy.pdf") if name in reasons)
    assert reasons[duplicate].startswith("duplicate_of:")


@pytest.mark.asyncio
async def test_download_rerun_is_idempotent(tmp_path):
    dataset = _dataset(tmp_path)
    await _run(dataset)
    first = (dataset / "manifest.csv").read_text(encoding="utf-8")
    await _run(dataset)
    assert (dataset / "manifest.csv").read_text(encoding="utf-8") == first
    assert len(read_csv(dataset / "download_errors.csv")) == 4


def test_scanned_heuristic():
    assert is_probably_scanned(pages=10, text_chars=50)
    assert not is_probably_scanned(pages=2, text_chars=5000)
    assert not is_probably_scanned(pages=0, text_chars=0)


def test_interleave_by_host_round_robins():
    items = [{"host": h, "n": str(i)} for i, h in enumerate("aaabbc")]
    assert [item["host"] for item in interleave_by_host(items)] == list("abcaba")
