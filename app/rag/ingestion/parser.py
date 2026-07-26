from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedDocument:
    """Normalized document content ready for chunking."""

    source: str
    content: str
    metadata: dict[str, object]


def parse_text_document(
    source: str,
    content: str,
    metadata: dict[str, object] | None = None,
) -> ParsedDocument:
    """Normalize whitespace while preserving the document source."""

    normalized = "\n".join(line.strip() for line in content.splitlines() if line.strip())
    return ParsedDocument(source, normalized, metadata or {})
