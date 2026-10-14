"""Build the structured `citations` list persisted with an assistant message.

The LLM only writes `[n]` markers inline (see `citation_rules.yaml`); everything
else about a source comes from the retrieved chunks (or, numbered after them,
WebSearchNode's pages), never from the model, so a source can't be invented.
`objectKey` is deliberately NOT part of the output: the conversation-history
API is public and backend-java passes `citations`
through untouched, so anything put here reaches the client.
"""

import re
from collections.abc import Sequence
from typing import Any

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
