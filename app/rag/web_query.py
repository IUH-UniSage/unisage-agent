"""Web search policy: the keyword query a task's `WebSlots` become, which university
sites a topic trusts first, which pages are never worth showing, the academic year a
page is about, and how long a topic's results stay cached.

The query is built here, not written by the model: Tavily scores a question-form query
("Học phí ngành Công nghệ thông tin là bao nhiêu?") low on every page (0.04-0.19), and
building it from slots turns the same question into the same query every time, which is
what the result cache is keyed on. A year goes in only when the user named one: the
current year added by default pushed the latest fee decision (QĐ 1035, for 2025-2026,
the 2026-2027 one not yet posted) out of the top 10 (14-10-2026).
"""

import re
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import date
from urllib.parse import unquote, urlsplit

from app.schemas.web_search import WebSearchResult, WebSlots

SCHOOL = "IUH"

# Sites tried first per topic: a bare host (`www.` ignored) or host + path prefix. Not a
# filter - Tavily still searches every TAVILY_INCLUDE_DOMAINS site - only the order in
# which pages reach the web judge.
TOPIC_DOMAINS: dict[str, tuple[str, ...]] = {
    "hoc_phi": ("ptckt.iuh.edu.vn", "pdt.iuh.edu.vn", "iuh.edu.vn/vi/thong-bao"),
    "tuyen_sinh": ("tuyensinh.iuh.edu.vn", "pdt.iuh.edu.vn"),
    "lich_thong_bao": ("pdt.iuh.edu.vn", "iuh.edu.vn/vi/thong-bao", "ctsv.iuh.edu.vn"),
    "lien_he_co_so": ("iuh.edu.vn",),
    "quy_che_thu_tuc": ("pdt.iuh.edu.vn", "ctsv.iuh.edu.vn"),
    "chuong_trinh_dao_tao": ("pdt.iuh.edu.vn",),
    "khac": (),
}

# Campus sites, shown only when the question names that campus: a Quảng Ngãi fee page
# scored 0.39 for a question about the main campus.
CAMPUS_HOSTS: dict[str, tuple[str, ...]] = {
    "qn.iuh.edu.vn": ("quảng ngãi", "quang ngai"),
}

# Seconds a topic's search results stay cached. Fees change once a year, admission
# figures several times a day in season, notices daily, contacts hardly ever.
_HOUR = 3600
TOPIC_CACHE_TTL_SECONDS: dict[str, int] = {
    "hoc_phi": 24 * _HOUR,
    "tuyen_sinh": 6 * _HOUR,
    "lich_thong_bao": 3 * _HOUR,
    "lien_he_co_so": 7 * 24 * _HOUR,
}
DEFAULT_CACHE_TTL_SECONDS = 24 * _HOUR
ADMISSION_SEASON_TTL_SECONDS = _HOUR
ADMISSION_SEASON_MONTHS = frozenset({6, 7, 8})
# A search that found nothing is retried sooner: the page may just not be indexed yet.
EMPTY_RESULT_TTL_SECONDS = 30 * 60

# A home page, a listing or an author/tag page: many headlines, no content.
_LISTING_PATH = re.compile(
    r"/page/\d+|[@?&]p=\d+|/author/|/tag/|/category[/.@]|news\.html@\d+@"
    r"|/(?:thong-bao|tin-tuc|thong-tin-tuyen-sinh)\.html$",
    re.IGNORECASE,
)
# "2025-2026", "2025 - 2026" with an en dash, "2025_2026" (in file names).
_YEAR_RANGE = re.compile(r"(20\d{2})\s*[-\u2013/_ ]\s*(20\d{2})")

_ALIASES: dict[str, dict[str, str]] = {
    "nganh_hoc": {
        "cntt": "Công nghệ Thông tin",
        "it": "Công nghệ Thông tin",
        "công nghệ thông tin": "Công nghệ Thông tin",
        "ktpm": "Kỹ thuật Phần mềm",
        "khmt": "Khoa học Máy tính",
        "httt": "Hệ thống Thông tin",
        "qtkd": "Quản trị Kinh doanh",
    },
    "bac_dao_tao": {
        "dai_hoc": "Đại học",
        "đại học": "Đại học",
        "đh": "Đại học",
        "cao_hoc": "Thạc sĩ",
        "cao học": "Thạc sĩ",
        "thạc sĩ": "Thạc sĩ",
        "tiến sĩ": "Tiến sĩ",
        "nghiên cứu sinh": "Tiến sĩ",
        "sau_dai_hoc": "Sau đại học",
        "sau đại học": "Sau đại học",
    },
    "he_dao_tao": {
        "chinh_quy": "Chính quy",
        "chính quy": "Chính quy",
        "lien_thong": "Liên thông",
        "liên thông": "Liên thông",
        "vua_lam_vua_hoc": "Vừa làm vừa học",
        "vừa làm vừa học": "Vừa làm vừa học",
    },
}


def build_web_query(slots: WebSlots) -> str:
    """The keyword query for `slots`, by the template of its first topic."""

    topic = slots.topics[0]
    year = _year_part(topic, slots.nam_hoc)
    bac = _canonical("bac_dao_tao", slots.bac_dao_tao)
    he = _canonical("he_dao_tao", slots.he_dao_tao)
    nganh = _canonical("nganh_hoc", slots.nganh_hoc)
    co_so = _clean(slots.co_so)
    if topic == "hoc_phi":
        # No major: fee schedules are split by "Khối", the major's name is not in them.
        parts = [slots.subject, bac, he, co_so, year]
    elif topic == "tuyen_sinh":
        parts = [slots.subject, nganh, bac, he, co_so, year]
    elif topic == "lich_thong_bao":
        parts = [slots.subject, bac, he, co_so, year]
    elif topic == "chuong_trinh_dao_tao":
        parts = [slots.subject, nganh, bac, he]
    elif topic == "quy_che_thu_tuc":
        parts = [slots.subject, bac, he, co_so]
    else:
        parts = [slots.subject, co_so]
    query = " ".join(_clean(part) or "" for part in parts if _clean(part))
    if SCHOOL.lower() not in query.lower().split():
        query = f"{query} {SCHOOL}"
    return re.sub(r"\s+", " ", query).strip()


def order_candidates(
    results: Iterable[WebSearchResult],
    *,
    topic: str | None,
    co_so: str | None = None,
    floor: float,
    limit: int,
) -> list[WebSearchResult]:
    """Pages worth showing the web judge, best first: not a listing or another campus's
    site, the topic's own sites before the rest, the newest academic year first, then
    Tavily's score. Each page carries the academic year it is about.

    `floor` only drops pages off the topic's own sites: the latest fee decision on
    ptckt.iuh.edu.vn scored 0.05-0.07, right at it, while the lowest score seen at all was
    0.04 - Tavily already cuts to its 10 best, so a topic page is never junk by score alone."""

    preferred = TOPIC_DOMAINS.get(topic or "", ())
    kept: list[WebSearchResult] = []
    for result in results:
        if is_listing_page(result.url) or _other_campus(result.url, co_so):
            continue
        if result.score < floor and not _matches_any(result.url, preferred):
            continue
        kept.append(result.model_copy(update={"academic_year": academic_year_of(result)}))
    kept.sort(
        key=lambda result: (
            0 if _matches_any(result.url, preferred) else 1,
            -int((result.academic_year or "0")[:4]),
            -result.score,
        )
    )
    return kept[:limit]


def is_listing_page(url: str) -> bool:
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    if not path and not parts.query:
        return True
    return bool(_LISTING_PATH.search(unquote(url)))


def academic_year_of(result: WebSearchResult) -> str | None:
    """The first "YYYY-(YYYY+1)" in the page's title, URL or content."""

    for text in (result.title, unquote(result.url), result.content):
        for match in _YEAR_RANGE.finditer(text):
            start, end = int(match.group(1)), int(match.group(2))
            if end == start + 1:
                return f"{start}-{end}"
    return None


def cache_ttl_seconds(topic: str | None, *, today: date, empty: bool) -> int:
    if empty:
        return EMPTY_RESULT_TTL_SECONDS
    if topic == "tuyen_sinh" and today.month in ADMISSION_SEASON_MONTHS:
        return ADMISSION_SEASON_TTL_SECONDS
    return TOPIC_CACHE_TTL_SECONDS.get(topic or "", DEFAULT_CACHE_TTL_SECONDS)


def _year_part(topic: str, value: str | None) -> str | None:
    """The year the user named, as the topic's documents write it ("năm học 2025-2026", an
    admission's "năm 2026"); None when they named none - no year is guessed."""

    match = _YEAR_RANGE.search(value or "")
    if match and int(match.group(2)) == int(match.group(1)) + 1:
        if topic == "tuyen_sinh":
            return f"năm {match.group(1)}"
        return f"năm học {match.group(1)}-{match.group(2)}"
    single = re.search(r"\b(20\d{2})\b", value or "")
    return f"năm {single.group(1)}" if single else None


def _canonical(field: str, value: str | None) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    return _ALIASES.get(field, {}).get(text.lower(), text)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()
    return text or None


def _host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host.removeprefix("www.")


def _matches_any(url: str, patterns: Sequence[str]) -> bool:
    host = _host(url)
    path = urlsplit(url).path
    for pattern in patterns:
        pattern_host, _, prefix = pattern.partition("/")
        if host == pattern_host and path.startswith(f"/{prefix}" if prefix else ""):
            return True
    return False


def _other_campus(url: str, co_so: str | None) -> bool:
    names = CAMPUS_HOSTS.get(_host(url))
    if names is None:
        return False
    asked = (co_so or "").lower()
    return not any(name in asked for name in names)
