from typing import Any


def merge_chunk_metadata(
    document_metadata: dict[str, Any],
    *,
    chunk_index: int,
) -> dict[str, Any]:
    """Attach stable chunk position metadata without mutating the source mapping."""

    return {**document_metadata, "chunk_index": chunk_index}
