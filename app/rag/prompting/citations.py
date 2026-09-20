"""Build the structured `citations` list persisted with an assistant message.

The LLM only writes `[n]` markers inline (see `citation_rules.yaml`); everything
else about a source comes from the retrieved chunks, never from the model, so a
source can't be invented. `objectKey` is deliberately NOT part of the output:
the conversation-history API is public and backend-java passes `citations`
through untouched, so anything put here reaches the client.
"""

import re
from collections.abc import Sequence
from typing import Any

from app.schemas.retrieval import RetrievedChunk

_MARKER_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_UUID_PREFIX_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_"
)
_EXTENSION_PATTERN = re.compile(r"\.[A-Za-z0-9]{1,5}$")


def cited_indexes(response_text: str, chunk_count: int) -> list[int]:
    """Distinct 1-based `[n]` indexes used in the text that exist among the
    retrieved chunks, in first-appearance order."""

    seen: list[int] = []
    for match in _MARKER_PATTERN.finditer(response_text):
        for raw in match.group(1).split(","):
            index = int(raw)
            if 1 <= index <= chunk_count and index not in seen:
                seen.append(index)
    return seen


def source_title(source: str) -> str:
    """Display name from an object key: drop the uuid prefix and extension."""

    name = source.rsplit("/", 1)[-1]
    name = _UUID_PREFIX_PATTERN.sub("", name)
    return _EXTENSION_PATTERN.sub("", name).strip()


def build_citations(response_text: str, chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for index in cited_indexes(response_text, len(chunks)):
        chunk = chunks[index - 1]
        document_id = chunk.metadata.get("document_id")
        citations.append(
            {
                "index": index,
                "documentId": str(document_id) if document_id else None,
                "title": source_title(chunk.source),
                "section": chunk.heading_path[-1] if chunk.heading_path else None,
                "pageStart": chunk.page_start,
                "pageEnd": chunk.page_end,
                "sourceType": chunk.source_type,
            }
        )
    return citations
