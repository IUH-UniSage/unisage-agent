"""Download the PDFs found by `discover` and maintain `manifest.csv`.

Only hosts with `approved=true` in `sources.csv` are downloaded. Files land in
`files/<unit-slug>/<file_id>.pdf` (`file_id` = first 12 hex chars of the
sha256), identical content reached through several URLs is stored once, and
re-running skips URLs already in the manifest - so the command is resumable.

Failures (HTTP errors, non-PDF bodies, oversize files, duplicates) go to
`download_errors.csv` instead of the manifest.

Usage:
    python -m evals.crawl.download --dataset ../unisage-gateway/dataset
"""

import argparse
import asyncio
import csv
import hashlib
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx
import pymupdf

from evals.crawl.discover import slugify
from evals.crawl.http import DisallowedUrlError, PoliteClient

logger = logging.getLogger(__name__)

MAX_BYTES = 30 * 1024 * 1024
# Below this many extracted characters per page, a PDF is almost certainly a
# scan (image only) and will parse to nothing useful without OCR.
SCANNED_CHARS_PER_PAGE = 100

MANIFEST_FIELDS = [
    "file_id", "file_name", "source_url", "source_page", "unit", "campus",
    "department_id", "is_public", "access_level", "label_source",
    "pages", "text_chars", "quality", "size_bytes", "sha256", "local_path",
    "tls_verified", "crawled_at",
    "selected", "ingest_status", "document_id",
]  # fmt: skip
ERROR_FIELDS = ["source_url", "source_page", "unit", "reason"]


@dataclass
class Downloaded:
    content: bytes
    file_name: str


class DownloadError(Exception):
    pass


def file_name_from(response: httpx.Response, url: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", disposition, re.IGNORECASE)
    if match:
        return unquote(match.group(1)).strip()
    return unquote(Path(urlsplit(url).path).name)


def pdf_stats(content: bytes) -> tuple[int, int]:
    """(pages, extracted text characters); raises on an unreadable PDF."""

    document = pymupdf.open(stream=content, filetype="pdf")
    try:
        chars = sum(len(document[i].get_text().strip()) for i in range(document.page_count))
        return document.page_count, chars
    finally:
        document.close()


def is_probably_scanned(pages: int, text_chars: int) -> bool:
    return pages > 0 and text_chars < SCANNED_CHARS_PER_PAGE * pages


async def fetch_pdf(client: PoliteClient, url: str, *, max_bytes: int = MAX_BYTES) -> Downloaded:
    try:
        response = await client.get(url, stream=True)
    except DisallowedUrlError as exc:
        raise DownloadError("disallowed") from exc
    except httpx.HTTPError as exc:
        raise DownloadError(f"http_error: {type(exc).__name__}") from exc
    try:
        if response.status_code != 200:
            raise DownloadError(f"status_{response.status_code}")
        declared = int(response.headers.get("content-length") or 0)
        if declared > max_bytes:
            raise DownloadError("too_large")
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise DownloadError("too_large")
            chunks.append(chunk)
        content = b"".join(chunks)
    except httpx.HTTPError as exc:
        raise DownloadError(f"http_error: {type(exc).__name__}") from exc
    finally:
        await response.aclose()
    if not content.startswith(b"%PDF"):
        raise DownloadError("not_pdf")
    return Downloaded(content=content, file_name=file_name_from(response, url))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


@dataclass
class DownloadState:
    dataset: Path
    manifest: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.manifest = read_csv(self.dataset / "manifest.csv")
        self.errors = read_csv(self.dataset / "download_errors.csv")
        self.known_urls = {row["source_url"] for row in self.manifest}
        self.known_urls |= {row["source_url"] for row in self.errors}
        self.by_sha = {row["sha256"]: row for row in self.manifest}

    def save(self) -> None:
        self.manifest.sort(key=lambda row: (row["unit"], row["file_id"]))
        write_csv(self.dataset / "manifest.csv", MANIFEST_FIELDS, self.manifest)
        write_csv(self.dataset / "download_errors.csv", ERROR_FIELDS, self.errors)


def store(
    state: DownloadState,
    item: dict[str, str],
    downloaded: Downloaded,
    *,
    tls_verified: bool,
) -> dict[str, str] | None:
    """Write the file and append its manifest row; None if the content is a duplicate."""

    sha = hashlib.sha256(downloaded.content).hexdigest()
    url = item["pdf_url"]
    state.known_urls.add(url)
    if sha in state.by_sha:
        state.errors.append(
            {
                "source_url": url,
                "source_page": item["source_page"],
                "unit": item["unit"],
                "reason": f"duplicate_of:{state.by_sha[sha]['file_id']}",
            }
        )
        return None
    try:
        pages, text_chars = pdf_stats(downloaded.content)
    except Exception:  # pymupdf raises several unrelated types for broken files
        pages, text_chars = 0, 0
    file_id = sha[:12]
    relative = Path("files") / slugify(item["unit"]) / f"{file_id}.pdf"
    target = state.dataset / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(downloaded.content)
    row = {
        "file_id": file_id,
        "file_name": downloaded.file_name,
        "source_url": url,
        "source_page": item["source_page"],
        "unit": item["unit"],
        "campus": item["campus"],
        "department_id": "",
        "is_public": "",
        "access_level": "",
        "label_source": "",
        "pages": str(pages),
        "text_chars": str(text_chars),
        "quality": "",
        "size_bytes": str(len(downloaded.content)),
        "sha256": sha,
        "local_path": relative.as_posix(),
        "tls_verified": str(tls_verified).lower(),
        "crawled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "selected": "",
        "ingest_status": "",
        "document_id": "",
    }
    state.manifest.append(row)
    state.by_sha[sha] = row
    return row


def interleave_by_host(items: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Round-robin across hosts.

    The client allows one request per second per host, so concurrent slots only
    help when they hit different hosts; `discovered.csv` is sorted by host.
    """

    by_host: dict[str, list[dict[str, str]]] = {}
    for item in items:
        by_host.setdefault(item["host"], []).append(item)
    queues = list(by_host.values())
    ordered: list[dict[str, str]] = []
    for i in range(max((len(q) for q in queues), default=0)):
        ordered.extend(q[i] for q in queues if i < len(q))
    return ordered


async def run(
    dataset: Path,
    *,
    concurrency: int,
    client: PoliteClient | None = None,
    max_bytes: int = MAX_BYTES,
) -> DownloadState:
    approved = {
        row["host"]
        for row in read_csv(dataset / "sources.csv")
        if row.get("approved", "").strip().lower() == "true"
    }
    state = DownloadState(dataset)
    todo = interleave_by_host(
        item
        for item in read_csv(dataset / "discovered.csv")
        if item["host"] in approved and item["pdf_url"] not in state.known_urls
    )
    logger.info("%d PDFs to download (%d already done)", len(todo), len(state.known_urls))

    owned_client = client is None
    client = client or PoliteClient()
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def one(item: dict[str, str]) -> None:
        nonlocal done
        async with semaphore:
            try:
                downloaded = await fetch_pdf(client, item["pdf_url"], max_bytes=max_bytes)
            except DownloadError as exc:
                state.known_urls.add(item["pdf_url"])
                state.errors.append(
                    {
                        "source_url": item["pdf_url"],
                        "source_page": item["source_page"],
                        "unit": item["unit"],
                        "reason": str(exc),
                    }
                )
            else:
                host = urlsplit(item["pdf_url"]).hostname or ""
                store(state, item, downloaded, tls_verified=host not in client.insecure_hosts)
            done += 1
            if done % 25 == 0:
                state.save()
                logger.info("%d/%d", done, len(todo))

    try:
        await asyncio.gather(*(one(item) for item in todo))
    finally:
        state.save()
        if owned_client:
            await client.aclose()
    return state


def summarize(state: DownloadState) -> str:
    rows = state.manifest
    pages = sum(int(row["pages"] or 0) for row in rows)
    chars = sum(int(row["text_chars"] or 0) for row in rows)
    size = sum(int(row["size_bytes"] or 0) for row in rows)
    scanned = [
        row
        for row in rows
        if is_probably_scanned(int(row["pages"] or 0), int(row["text_chars"] or 0))
    ]
    lines = [
        f"files: {len(rows)}  pages: {pages}  text chars: {chars}  size: {size / 1e6:.1f} MB",
        f"probably scanned (< {SCANNED_CHARS_PER_PAGE} chars/page): {len(scanned)}",
        f"errors: {Counter(e['reason'].split(':')[0] for e in state.errors).most_common()}",
        "by unit:",
    ]
    for unit, count in Counter(row["unit"] for row in rows).most_common():
        lines.append(f"  {count:5d}  {unit}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=10, help="downloads in flight")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    state = asyncio.run(run(args.dataset, concurrency=args.concurrency))
    print(summarize(state))


if __name__ == "__main__":
    main()
