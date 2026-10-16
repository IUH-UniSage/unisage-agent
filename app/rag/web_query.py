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
from functools import partial
from typing import Any
from urllib.parse import unquote, urlsplit

from app.core.academic_calendar import current_term
from app.core.configs import load_yaml
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

# Campus sites: host -> (the campus's name, how a question names it). Their pages are for
# questions naming that campus: a Quảng Ngãi fee page scored 0.39 for a question about
# the main campus.
CAMPUS_HOSTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "qn.iuh.edu.vn": ("Phân hiệu Quảng Ngãi", ("quảng ngãi", "quang ngai")),
    "th.iuh.edu.vn": ("Cơ sở Thanh Hóa", ("thanh hóa", "thanh hoá", "thanh hoa")),
}
# Topics where a campus site reposts the University's own documents (the 2025 National
# Day notice TB-846-2025 was found only on qn.iuh.edu.vn; regulations and training
# programmes are the University's): a campus page is kept for these, after every other
# page, and the web judge is told which campus it is from. Fees, admission and contacts
# differ per campus, so their campus pages stay out.
CAMPUS_SHARED_TOPICS = frozenset({"lich_thong_bao", "quy_che_thu_tuc", "chuong_trinh_dao_tao"})

# Postgraduate pages (title or URL): a master's admission notice gave its "1.600.000
# đồng/tín chỉ" as the fee asked for an undergraduate major (14-10-2026). Kept only when
# the question is about that level. Matched on text without diacritics.
_POSTGRADUATE = re.compile(
    r"thac[\s_-]*si|tien[\s_-]*si|cao[\s_-]*hoc|sau[\s_-]*dai[\s_-]*hoc|nghien[\s_-]*cuu[\s_-]*sinh"
    r"|\bncs\b|\bsdh\.iuh\.edu\.vn",
    re.IGNORECASE,
)

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
    r"/page/\d+|[@?&/]p=\d+|/author/|/tag/|/category[/.@]|news\.html@\d+@"
    r"|/(?:thong-bao|tin-tuc|thong-tin-tuyen-sinh)\.html$",
    re.IGNORECASE,
)
_DETAIL_PAGE = re.compile(r"[/@]detail[@/]", re.IGNORECASE)
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


# The subject's vocabulary (configs/web_query.yaml): spelling, abbreviations, then
# multi-word synonyms anywhere in it, then whole-subject aliases per topic (`_all` last).
_VOCABULARY: dict[str, Any] = load_yaml("web_query.yaml")
SPELLING: dict[str, str] = _VOCABULARY["spelling"]
ABBREVIATIONS: dict[str, str] = _VOCABULARY["abbreviations"]
PHRASE_ALIASES: dict[str, str] = _VOCABULARY["phrase_aliases"]
SUBJECT_ALIASES: dict[str, dict[str, str]] = _VOCABULARY["subject_aliases"]
# Words that may come before a slot's value in the subject ("ngành Công nghệ thông tin").
_SLOT_LEAD_WORDS: dict[str, str] = {
    "nganh_hoc": r"(?:ngành\s+)?",
    "bac_dao_tao": r"(?:(?:bậc|trình độ)\s+)?",
    "he_dao_tao": r"(?:hệ\s+)?",
    "co_so": r"(?:(?:tại\s+)?(?:cơ sở|phân hiệu)\s+)?",
}


def build_web_query(slots: WebSlots) -> str:
    """The keyword query for `slots`, by the template of its first topic."""

    topic = slots.topics[0]
    written = slots.subject
    slots = _with_major_from_subject(slots)
    slots = slots.model_copy(update={"subject": _subject(slots, topic)})
    year = _year_part(topic, slots.nam_hoc)
    if year and all(digits in (slots.subject or "") for digits in re.findall(r"20\d{2}", year)):
        year = None  # "lịch nghỉ lễ Quốc khánh 2025" already has it
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
    if not query:
        query = _clean(written) or ""  # The subject was only a slot this template leaves out.
    if SCHOOL.lower() not in query.lower().split():
        query = f"{query} {SCHOOL}"
    return re.sub(r"\s+", " ", query).strip()


def order_candidates(
    results: Iterable[WebSearchResult],
    *,
    topic: str | None,
    co_so: str | None = None,
    postgraduate: bool = False,
    floor: float,
) -> list[WebSearchResult]:
    """Pages worth showing the web judge, best first: not a listing or another campus's
    site (kept last for `CAMPUS_SHARED_TOPICS`), not a postgraduate page unless the
    question is about that level (`postgraduate`), the topic's own sites before the rest,
    the newest academic year first, then Tavily's score. Each page carries the academic
    year it is about.

    `floor` only drops pages off the topic's own sites: the latest fee decision on
    ptckt.iuh.edu.vn scored 0.05-0.07, right at it, while the lowest score seen at all was
    0.04 - Tavily already cuts to its 10 best, so a topic page is never junk by score alone."""

    preferred = TOPIC_DOMAINS.get(topic or "", ())
    kept: list[WebSearchResult] = []
    for result in results:
        if is_listing_page(result.url) or (
            _other_campus(result.url, co_so) and topic not in CAMPUS_SHARED_TOPICS
        ):
            continue
        if not postgraduate and is_postgraduate(f"{result.title} {unquote(result.url)}"):
            continue
        if result.score < floor and not _matches_any(result.url, preferred):
            continue
        kept.append(result.model_copy(update={"academic_year": academic_year_of(result)}))

    def tier(url: str) -> int:
        if _other_campus(url, co_so):
            return 2
        return 0 if _matches_any(url, preferred) else 1

    kept.sort(
        key=lambda result: (
            tier(result.url),
            -int((result.academic_year or "0")[:4]),
            -result.score,
        )
    )
    return kept


def is_listing_page(url: str) -> bool:
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    if not path and not parts.query:
        return True
    # A detail page is never a listing, whatever page parameter it carries: the IT
    # certificate exam plan "itc.iuh.edu.vn/detail@91-1541-...html@p=2" was dropped as a
    # paginated listing (14-10-2026).
    if _DETAIL_PAGE.search(url):
        return False
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


# "ngành Kế toán" closing a subject, up to a year, cohort or term. Not after "chuyển",
# "đổi", "song", "hai": "điều kiện chuyển ngành" names no major.
_MAJOR_IN_SUBJECT = re.compile(
    r"(?<!chuyển )(?<!đổi )(?<!song )(?<!hai )(?<!\w)ngành\s+"
    r"(?P<major>\w[\w\s]*?)\s*(?=$|\b(?:năm|khóa|khoá|học kỳ|hk)\b|\d)",
    re.IGNORECASE,
)


def _with_major_from_subject(slots: WebSlots) -> WebSlots:
    """A major the classifier wrote into the subject but not into `nganh_hoc` (16-10-2026:
    "mức thu học phí ngành Công nghệ thông tin", no major slot) moves to that slot, so the
    topic's template decides whether the query names it, the same as when it is filled."""

    if _clean(slots.nganh_hoc):
        return slots
    match = _MAJOR_IN_SUBJECT.search(slots.subject or "")
    if match is None:
        return slots
    return slots.model_copy(update={"nganh_hoc": match.group("major").strip()})


def _subject(slots: WebSlots, topic: str) -> str:
    """The subject without what another slot already holds, in the topic's own words.

    The classifier is told to leave the major, level, mode and campus out of `subject`
    but does not always: "mức thu học phí ngành Công nghệ thông tin" put back the major
    the fee template leaves out on purpose, and its top page became a study-abroad notice
    (16-10-2026). A template adds each slot back where it belongs."""

    subject = normalize_phrases(_clean(slots.subject) or "")
    for field in _SLOT_LEAD_WORDS:
        for name in _names(field, getattr(slots, field)):
            pattern = rf"(?<!\w){_SLOT_LEAD_WORDS[field]}{re.escape(name)}(?!\w)"
            subject = re.sub(pattern, " ", subject, flags=re.IGNORECASE)
    subject = re.sub(r"\s+", " ", subject).strip(" ,.-")
    if not subject:
        return ""
    key = subject.lower()
    return SUBJECT_ALIASES.get(topic, {}).get(key) or SUBJECT_ALIASES.get("_all", {}).get(
        key, subject
    )


def normalize_phrases(text: str) -> str:
    """`text` with SPELLING, ABBREVIATIONS, then PHRASE_ALIASES replaced: whole words only,
    longest phrase first, one pass each, so a replacement is never matched again by the
    same table."""

    for table, pattern in (
        (SPELLING, _SPELLING_RE),
        (ABBREVIATIONS, _ABBREVIATIONS_RE),
        (PHRASE_ALIASES, _PHRASE_RE),
    ):
        text = pattern.sub(partial(_replacement, table), text)
    return text


def _replacement(table: dict[str, str], match: re.Match[str]) -> str:
    return table[match.group(0).lower()]


def _alternation(phrases: Iterable[str]) -> re.Pattern[str]:
    ordered = sorted(phrases, key=len, reverse=True)
    body = "|".join(re.escape(phrase) for phrase in ordered)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


_SPELLING_RE = _alternation(SPELLING)
_ABBREVIATIONS_RE = _alternation(ABBREVIATIONS)
_PHRASE_RE = _alternation(PHRASE_ALIASES)


def _names(field: str, value: str | None) -> list[str]:
    """Every way the subject may write a slot's value: as given, its canonical name, and
    each alias of that name ("CNTT", "it" for Công nghệ Thông tin). Longest first, so
    "Công nghệ Thông tin" goes before a shorter alias inside it could split it."""

    text = _clean(value)
    if text is None:
        return []
    canonical = _canonical(field, text) or text
    names = {text, canonical, normalize_phrases(text), normalize_phrases(canonical)}
    names.update(alias for alias, name in _ALIASES.get(field, {}).items() if name == canonical)
    return sorted(names, key=len, reverse=True)


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


# Relative time words (matched without diacritics) the classifier is told to turn into a
# year, for when it does not: "lịch thi chứng chỉ CNTT đợt gần nhất" was searched without
# one (14-10-2026). Not "hiện tại", too often "ngành hiện tại của tôi".
_THIS_TERM = re.compile(r"\b(?:hoc ky nay|ky nay|nam hoc nay)\b")
_THIS_YEAR = re.compile(
    r"\b(?:moi nhat|gan nhat|gan day|vua roi|vua qua|sap toi|hien nay|nam nay|thang nay)\b"
)
_LAST_YEAR = re.compile(r"\bnam ngoai\b")
_NEXT_YEAR = re.compile(r"\bnam (?:sau|toi)\b")


def relative_year(text: str, today: date) -> str | None:
    """The year (or academic year, for "kỳ này") a relative time word in `text` means
    today; None without one."""

    plain = _without_diacritics(text)
    if _THIS_TERM.search(plain):
        return current_term(today).academic_year
    if _THIS_YEAR.search(plain):
        return str(today.year)
    if _LAST_YEAR.search(plain):
        return str(today.year - 1)
    if _NEXT_YEAR.search(plain):
        return str(today.year + 1)
    return None


def is_postgraduate(text: str) -> bool:
    """`text` names a postgraduate level (master's, doctorate, "sau đại học")."""

    return bool(_POSTGRADUATE.search(_without_diacritics(text)))


def _without_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


def campus_of(url: str) -> str | None:
    """The campus whose site the page is on; None for the University's own sites."""

    campus = CAMPUS_HOSTS.get(_host(url))
    return campus[0] if campus else None


def _other_campus(url: str, co_so: str | None) -> bool:
    campus = CAMPUS_HOSTS.get(_host(url))
    if campus is None:
        return False
    asked = unicodedata.normalize("NFC", co_so or "").lower()
    return not any(name in asked for name in campus[1])
