"""Discover IUH sites and the documents they link to.

1. Read the "affiliated units" page on iuh.edu.vn to get every unit
   (office, faculty, institute, centre, branch campus) and, from each unit's
   page, its own subdomain if it has one.
2. Breadth-first crawl each site (same host only, bounded by page count and
   depth), recording every PDF link and counting other document links.

Writes two files into the dataset directory:
- `sources.csv`   - one row per site, with an `approved` column the user edits
                    before download (defaults to `true`).
- `discovered.csv`- one row per distinct PDF URL, with the page it came from.

Usage:
    python -m evals.crawl.discover --dataset ../unisage-gateway/dataset
"""

import argparse
import asyncio
import csv
import logging
import re
import unicodedata
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urldefrag, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from evals.crawl.http import DisallowedUrlError, PoliteClient, is_allowed_host

logger = logging.getLogger(__name__)

UNITS_PAGE = "https://iuh.edu.vn/vi/cac-don-vi-truc-thuoc.html"
MAIN_SITE = "https://iuh.edu.vn/"

UNIT_PAGE_PATTERN = re.compile(
    r"^vi/(phong|khoa|vien|trung-tam|phan-hieu|co-so|van-phong|hoi|tap-chi)-[a-z0-9-]+\.html$"
)

# Portals behind a login, or with no documents worth indexing.
SKIP_HOSTS = frozenset(
    {
        "eoffice.iuh.edu.vn",
        "sv.iuh.edu.vn",
        "lms.iuh.edu.vn",
        "dkhp.iuh.edu.vn",
        "gv.iuh.edu.vn",
        "ask.iuh.edu.vn",
        "vr.iuh.edu.vn",
        "tvts.iuh.edu.vn",
        "icc.iuh.edu.vn",
    }
)

# Sites every unit page links to in its shared header/footer, so they can't be
# attributed by the "unit page links to its own subdomain" rule. Owner read
# from each site's <title> (checked 30-09-2026).
SHARED_SITE_UNITS: Mapping[str, str] = {
    "pdt.iuh.edu.vn": "Phòng Đào tạo",
    "ctdt.iuh.edu.vn": "Phòng Đào tạo",
    "smia.iuh.edu.vn": "Phòng Quản lý khoa học và hợp tác quốc tế",
    "csm.iuh.edu.vn": "Trung tâm Quản trị Hệ thống",
    "htsv.iuh.edu.vn": "Trung tâm Kết nối doanh nghiệp và Giới thiệu việc làm",
    "camnang.iuh.edu.vn": "Cẩm nang người học",
    "doantn.iuh.edu.vn": "Đoàn Thanh niên",
    "youth.iuh.edu.vn": "Đoàn Thanh niên",
    "cuusinhvien.iuh.edu.vn": "Chuyên trang Cựu sinh viên",
    "tuyensinh.iuh.edu.vn": "Tuyển sinh",
}

CAMPUS_BY_KEYWORD: Mapping[str, str] = {
    "quang-ngai": "Quảng Ngãi",
    "thanh-hoa": "Thanh Hóa",
    "nhon-trach": "Đồng Nai",
}

PDF_SUFFIXES = (".pdf",)
OTHER_DOC_SUFFIXES = (".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".rar", ".zip")
SKIP_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".css",
    ".js",
    ".mp4",
    ".mp3",
    ".avi",
    ".woff",
    ".woff2",
    ".ttf",
    ".xml",
    ".json",
)
SKIP_PATH_MARKERS = (
    "/en/",
    "/language/change",
    "login",
    "logout",
    "dang-nhap",
    "/search",
    "/tim-kiem",
)
DRIVE_HOSTS = ("drive.google.com", "docs.google.com")


def slugify(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def campus_for(unit_slug: str, host: str) -> str:
    haystack = f"{unit_slug} {host}"
    for keyword, campus in CAMPUS_BY_KEYWORD.items():
        if keyword in haystack:
            return campus
    if host.startswith("qn."):
        return "Quảng Ngãi"
    return "HCM"


def classify_link(url: str) -> str:
    """'pdf' | 'other_doc' | 'drive' | 'page' | 'skip' for an absolute URL."""

    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host in DRIVE_HOSTS:
        return "drive"
    if parts.scheme not in ("http", "https") or not is_allowed_host(host):
        return "skip"
    path = unquote(parts.path).lower()
    if path.endswith(PDF_SUFFIXES):
        return "pdf"
    if path.endswith(OTHER_DOC_SUFFIXES):
        return "other_doc"
    if path.endswith(SKIP_SUFFIXES):
        return "skip"
    full = (path + "?" + parts.query).lower()
    if any(marker in full for marker in SKIP_PATH_MARKERS):
        return "skip"
    return "page"


@dataclass(frozen=True)
class Link:
    url: str
    text: str


def parse_links(html: str, base_url: str) -> list[Link]:
    """Absolute, fragment-free links, honouring `<base href>`."""

    soup = BeautifulSoup(html, "lxml")
    base_tag = soup.find("base", href=True)
    base = urljoin(base_url, str(base_tag["href"])) if base_tag else base_url
    links: list[Link] = []
    seen: set[str] = set()
    for tag in soup.find_all(["a", "iframe", "embed"]):
        raw = tag.get("href") or tag.get("src")
        if not raw or not isinstance(raw, str):
            continue
        raw = raw.strip()
        if raw.startswith(("javascript:", "mailto:", "tel:", "data:", "#")):
            continue
        url, _ = urldefrag(urljoin(base, raw))
        if url in seen:
            continue
        seen.add(url)
        links.append(Link(url=url, text=" ".join(tag.get_text(" ").split())[:200]))
    return links


def own_hosts(unit_page_hosts: Mapping[str, set[str]]) -> dict[str, set[str]]:
    """Per unit page, the IUH hosts it links to that the other unit pages don't.

    Every unit page shares the same header/footer, so a host linked from all
    of them says nothing about the unit; the leftovers are the unit's own site.
    """

    if not unit_page_hosts:
        return {}
    shared = set.intersection(*unit_page_hosts.values())
    return {page: hosts - shared - SKIP_HOSTS for page, hosts in unit_page_hosts.items()}


@dataclass
class Site:
    root_url: str
    host: str
    unit: str
    campus: str
    pages_crawled: int = 0
    pdf_count: int = 0
    other_doc_count: int = 0
    drive_count: int = 0
    error: str = ""


@dataclass
class DiscoveredPdf:
    pdf_url: str
    source_page: str
    link_text: str
    host: str
    unit: str
    campus: str


@dataclass
class CrawlResult:
    sites: list[Site] = field(default_factory=list)
    pdfs: dict[str, DiscoveredPdf] = field(default_factory=dict)


def _is_html(response: httpx.Response) -> bool:
    return "html" in response.headers.get("content-type", "").lower()


async def discover_sites(client: PoliteClient) -> list[Site]:
    response = await client.get(UNITS_PAGE)
    unit_pages: dict[str, str] = {}
    for link in parse_links(response.text, UNITS_PAGE):
        parts = urlsplit(link.url)
        relative = parts.path.lstrip("/")
        if parts.hostname == "iuh.edu.vn" and UNIT_PAGE_PATTERN.match(relative) and link.text:
            unit_pages.setdefault(link.url, link.text)

    page_hosts: dict[str, set[str]] = {}
    for url in unit_pages:
        try:
            page = await client.get(url)
        except (httpx.HTTPError, DisallowedUrlError) as exc:
            logger.warning("unit page %s failed: %s", url, exc)
            continue
        page_hosts[url] = {
            (urlsplit(link.url).hostname or "").lower()
            for link in parse_links(page.text, url)
            if is_allowed_host(urlsplit(link.url).hostname or "")
        } - {"iuh.edu.vn", "www.iuh.edu.vn"}

    sites: dict[str, Site] = {}
    for page_url, hosts in own_hosts(page_hosts).items():
        unit = unit_pages[page_url]
        unit_slug = slugify(unit)
        for host in sorted(hosts):
            sites.setdefault(
                host, Site(f"https://{host}/", host, unit, campus_for(unit_slug, host))
            )
    for host, unit in SHARED_SITE_UNITS.items():
        sites.setdefault(
            host, Site(f"https://{host}/", host, unit, campus_for(slugify(unit), host))
        )
    sites.setdefault("iuh.edu.vn", Site(MAIN_SITE, "iuh.edu.vn", "Trường (trang chính)", "HCM"))
    return sorted(sites.values(), key=lambda s: s.host)


async def crawl_site(
    client: PoliteClient, site: Site, result: CrawlResult, *, max_pages: int, max_depth: int
) -> None:
    queue: deque[tuple[str, int]] = deque([(site.root_url, 0)])
    seen_pages = {site.root_url}
    seen_other: set[str] = set()
    while queue and site.pages_crawled < max_pages:
        url, depth = queue.popleft()
        try:
            response = await client.get(url)
        except DisallowedUrlError:
            continue
        except httpx.HTTPError as exc:
            if site.pages_crawled == 0:
                site.error = f"{type(exc).__name__}: {exc}"[:200]
            continue
        final_host = (response.url.host or "").lower()
        if response.status_code != 200 or not _is_html(response) or final_host != site.host:
            continue
        site.pages_crawled += 1
        for link in parse_links(response.text, str(response.url)):
            kind = classify_link(link.url)
            if kind == "pdf":
                if link.url not in result.pdfs:
                    result.pdfs[link.url] = DiscoveredPdf(
                        link.url, str(response.url), link.text, site.host, site.unit, site.campus
                    )
                    site.pdf_count += 1
            elif kind in ("other_doc", "drive"):
                if link.url not in seen_other:
                    seen_other.add(link.url)
                    if kind == "other_doc":
                        site.other_doc_count += 1
                    else:
                        site.drive_count += 1
            elif kind == "page" and depth < max_depth:
                if (
                    urlsplit(link.url).hostname or ""
                ).lower() == site.host and link.url not in seen_pages:
                    seen_pages.add(link.url)
                    queue.append((link.url, depth + 1))
    logger.info(
        "%s: %d pages, %d pdf, %d other docs, %d drive",
        site.host,
        site.pages_crawled,
        site.pdf_count,
        site.other_doc_count,
        site.drive_count,
    )


async def run(
    dataset: Path,
    *,
    max_pages: int,
    max_depth: int,
    concurrency: int,
    only_hosts: Iterable[str] = (),
) -> CrawlResult:
    result = CrawlResult()
    async with PoliteClient() as client:
        sites = await discover_sites(client)
        wanted = set(only_hosts)
        if wanted:
            sites = [s for s in sites if s.host in wanted]
        result.sites = sites
        logger.info("crawling %d sites", len(sites))
        semaphore = asyncio.Semaphore(concurrency)

        async def one(site: Site) -> None:
            async with semaphore:
                await crawl_site(client, site, result, max_pages=max_pages, max_depth=max_depth)

        await asyncio.gather(*(one(site) for site in sites))
        insecure = client.insecure_hosts

    write_sources(dataset / "sources.csv", result.sites, insecure)
    write_discovered(dataset / "discovered.csv", result.pdfs.values())
    return result


SOURCES_FIELDS = [
    "url",
    "host",
    "unit",
    "campus",
    "pages_crawled",
    "pdf_count",
    "other_doc_count",
    "drive_count",
    "tls_verified",
    "error",
    "approved",
]
DISCOVERED_FIELDS = ["pdf_url", "source_page", "link_text", "host", "unit", "campus"]


def write_sources(path: Path, sites: Iterable[Site], insecure_hosts: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCES_FIELDS)
        writer.writeheader()
        for site in sites:
            writer.writerow(
                {
                    "url": site.root_url,
                    "host": site.host,
                    "unit": site.unit,
                    "campus": site.campus,
                    "pages_crawled": site.pages_crawled,
                    "pdf_count": site.pdf_count,
                    "other_doc_count": site.other_doc_count,
                    "drive_count": site.drive_count,
                    "tls_verified": str(site.host not in insecure_hosts).lower(),
                    "error": site.error,
                    "approved": "true",
                }
            )


def write_discovered(path: Path, pdfs: Iterable[DiscoveredPdf]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DISCOVERED_FIELDS)
        writer.writeheader()
        for pdf in sorted(pdfs, key=lambda p: (p.host, p.pdf_url)):
            writer.writerow(pdf.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=300, help="per site")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8, help="sites crawled in parallel")
    parser.add_argument("--host", action="append", default=[], help="only crawl these hosts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    result = asyncio.run(
        run(
            args.dataset,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            concurrency=args.concurrency,
            only_hosts=args.host,
        )
    )
    print(f"{len(result.sites)} sites, {len(result.pdfs)} distinct PDF links")


if __name__ == "__main__":
    main()
