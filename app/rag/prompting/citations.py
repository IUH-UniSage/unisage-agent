"""Build the structured `citations` list persisted with an assistant message.

The LLM only writes `[n]` markers inline (see `citation_rules.yaml`); everything
else about a source comes from the retrieved chunks (or, numbered after them,
WebSearchNode's pages), never from the model, so a source can't be invented.
`objectKey` is deliberately NOT part of the output: the conversation-history
API is public and backend-java passes `citations`
through untouched, so anything put here reaches the client.
"""

import re
import unicodedata
from collections.abc import Sequence
from typing import Any
from urllib.parse import unquote, urlsplit

from app.rag.web_query import campus_of
from app.schemas.retrieval import RetrievedChunk
from app.schemas.web_search import WebSearchResult

WEB_SOURCE_TYPE = "WEB"

_MARKER_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_UUID_PREFIX_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_"
)
_EXTENSION_PATTERN = re.compile(r"\.[A-Za-z0-9]{1,5}$")
# A line that opens with its marker ("[1] https://...", "- [2]: Thông báo ...") is
# a source list entry: in the answer itself a marker follows what it cites.
_SOURCE_LINE_PATTERN = re.compile(r"^\s*(?:[-*]\s*)?\[\d+(?:\s*,\s*\d+)*\]:?(?:\s.*)?$")
_SOURCE_HEADING_PATTERN = re.compile(
    r"^\W*(?:nguồn(?: tham khảo| trích dẫn)?|văn bản tham chiếu|tài liệu tham khảo)\W*$",
    re.IGNORECASE,
)
_RULE_PATTERN = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
# What site name Tavily's page titles end with, after a dash, bar or colon.
_SITE_SUFFIX = re.compile(
    r"\s*[-\u2013|:]\s*(?:trung tâm|trường|đại học công nghiệp|iuh\b|industrial university"
    r"|(?:tp\.?\s*)?hồ chí minh|tp\.?\s*hcm)[^-\u2013|]*$",
    re.IGNORECASE,
)
# A title that is a file name ("TB-957-2026.pdf"); not "TP.HCM".
_FILE_EXTENSION = re.compile(r"\.(?:pdf|docx?|xlsx?|pptx?|html?)$", re.IGNORECASE)
_TITLE_MIN_CHARS = 6
_TITLE_MAX_CHARS = 90
_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\((\S+?)\)")
_BARE_URL = re.compile(r"(?<!\()https?://[^\s<>()\[\]]*[^\s<>()\[\].,;:!?'\"]")


def cited_indexes(response_text: str, source_count: int) -> list[int]:
    """Distinct 1-based `[n]` indexes used in the text that exist among the
    sources shown to the model, in first-appearance order."""

    seen: list[int] = []
    for match in _MARKER_PATTERN.finditer(response_text):
        for raw in match.group(1).split(","):
            index = int(raw)
            if 1 <= index <= source_count and index not in seen:
                seen.append(index)
    return seen


def strip_source_list(response_text: str) -> str:
    """Drop a source list the model wrote at the end of its answer anyway
    (`citation_rules.yaml` rule 4): the client already shows each cited source,
    with its link, from the `[n]` markers, so the list only repeats it."""

    lines = response_text.rstrip().split("\n")
    end = len(lines)
    found = False
    while end > 0:
        line = lines[end - 1]
        if _SOURCE_LINE_PATTERN.match(line):
            found = True
        elif line.strip() and not (
            _SOURCE_HEADING_PATTERN.match(line) or _RULE_PATTERN.match(line)
        ):
            break
        end -= 1
    if not found or end == 0:
        return response_text
    return "\n".join(lines[:end]).rstrip()


def keep_known_links(text: str, pages: Sequence[WebSearchResult]) -> str:
    """Links the model wrote, pointed back at the pages it was given: a URL it retyped
    with diacritics ("quoc-khánh") goes to that page again, any other one is dropped (the
    link text stays). The model only copies URLs, so none it writes can be new."""

    known = {_fold_url(page.url): page.url for page in pages}

    def markdown(match: re.Match[str]) -> str:
        url = known.get(_fold_url(match.group(2)))
        return f"[{match.group(1)}]({url})" if url else match.group(1)

    def bare(match: re.Match[str]) -> str:
        return known.get(_fold_url(match.group(0)), "")

    return _BARE_URL.sub(bare, _MARKDOWN_LINK.sub(markdown, text))


def source_links_block(pages: Sequence[WebSearchResult], reply: str = "") -> str:
    """The links listed under `reply` for the official pages kept without their content
    that it does not link yet (pass it through `keep_known_links` first): written by code,
    so every one is the page's real address."""

    lines = []
    shown: set[str] = set()
    for page in pages:
        title = display_title(page)
        # The same notice reached by two addresses is listed once.
        if page.url in reply or title.lower() in shown:
            continue
        shown.add(title.lower())
        notes = [(urlsplit(page.url).hostname or "").removeprefix("www.")]
        if page.academic_year and page.academic_year not in title:
            notes.append(f"năm học {page.academic_year}")
        campus = campus_of(page.url)
        if campus:
            notes.append(campus)
        lines.append(f"- [{title}]({page.url}) · {' · '.join(note for note in notes if note)}")
    if not lines:
        return ""
    return "\n\nVăn bản liên quan trên website của Trường:\n" + "\n".join(lines)


def display_title(page: WebSearchResult) -> str:
    """A page's title as a short link text: the site name Tavily appends ("- Trung Tâm Tin
    Học - Đại học Công Nghiệp TP.HCM:..") dropped, the file or address name instead of a
    letterhead ("BỘ CÔNG THƯƠNG TRƯỜNG ĐẠI HỌC ...") or a stub ("2027"), at most
    `_TITLE_MAX_CHARS` long."""

    title = _FILE_EXTENSION.sub("", re.sub(r"\s+", " ", page.title).strip())
    previous = None
    while previous != title:
        previous = title
        title = _SITE_SUFFIX.sub("", title).strip(" -\u2013|:.\u2026")
    # A letterhead line in capitals ("BỘ CÔNG THƯƠNG ..."), not a document code ("TB-957").
    if len(title) < _TITLE_MIN_CHARS or (title.isupper() and title.count(" ") >= 3):
        title = _name_from_url(page.url)
    title = title.replace("[", "(").replace("]", ")")
    if len(title) > _TITLE_MAX_CHARS:
        title = title[:_TITLE_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,-\u2013:") + "\u2026"
    return title


def _name_from_url(url: str) -> str:
    """The last part of the address as a name: "1035-QD-DHCN-11-4-2025.pdf" ->
    "1035-QD-DHCN-11-4-2025"; a slug "thong-bao-v-v-dang-ky-hoc-phan-2188.html" ->
    "Thong bao v v dang ky hoc phan"."""

    parts = urlsplit(url)
    name = unquote(parts.path.rstrip("/").rsplit("/", 1)[-1]) or (parts.hostname or url)
    name = _EXTENSION_PATTERN.sub("", name)
    if re.fullmatch(r"[a-z0-9-]+", name) and name.count("-") >= 3:
        words = re.sub(r"-\d+$", "", name).replace("-", " ")
        return words[:1].upper() + words[1:]
    return name


def _fold_url(url: str) -> str:
    """A URL without diacritics, case or a trailing slash, for matching a retyped one."""

    text = unicodedata.normalize("NFD", unquote(url).replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in text if not unicodedata.combining(char)).lower().rstrip("/")


def source_title(source: str) -> str:
    """Display name from an object key: drop the uuid prefix and extension."""

    name = source.rsplit("/", 1)[-1]
    name = _UUID_PREFIX_PATTERN.sub("", name)
    return _EXTENSION_PATTERN.sub("", name).strip()


def build_citations(
    response_text: str,
    chunks: Sequence[RetrievedChunk],
    web_results: Sequence[WebSearchResult] = (),
) -> list[dict[str, Any]]:
    """`[1..len(chunks)]` are chunks; the indexes after them are web pages, in
    the order the `<websearch>` block listed them."""

    citations: list[dict[str, Any]] = []
    for index in cited_indexes(response_text, len(chunks) + len(web_results)):
        if index > len(chunks):
            citations.append(_web_citation(index, web_results[index - len(chunks) - 1]))
            continue
        citations.append(chunk_citation(index, chunks[index - 1]))
    return citations


def chunk_citation(index: int, chunk: RetrievedChunk) -> dict[str, Any]:
    document_id = chunk.metadata.get("document_id")
    return {
        "index": index,
        "documentId": str(document_id) if document_id else None,
        "title": source_title(chunk.source),
        "section": chunk.heading_path[-1] if chunk.heading_path else None,
        "pageStart": chunk.page_start,
        "pageEnd": chunk.page_end,
        "sourceType": chunk.source_type,
    }


def renumber_markers(response_text: str, mapping: dict[int, int]) -> str:
    """Rewrite `[n]` / `[n, m]` markers through `mapping`; a marker of an index the
    mapping does not know (one the model invented) is dropped."""

    def replace(match: re.Match[str]) -> str:
        indexes = [int(raw) for raw in match.group(1).split(",")]
        return "".join(f"[{mapping[index]}]" for index in indexes if index in mapping)

    return _MARKER_PATTERN.sub(replace, response_text)


def _web_citation(index: int, result: WebSearchResult) -> dict[str, Any]:
    return {
        "index": index,
        "documentId": None,
        "title": result.title,
        "section": None,
        "pageStart": None,
        "pageEnd": None,
        "sourceType": WEB_SOURCE_TYPE,
        "url": result.url,
    }
