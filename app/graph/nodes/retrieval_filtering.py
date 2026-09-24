"""RetrievalFilteringNode - searches every retrieval text of the turn with
the caller's permission filter, merging several into one ranked list."""

from collections.abc import Sequence

from app.core.config import settings
from app.rag.retrieval.service import RetrievalServiceProtocol
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext


def retrieve_chunks(
    queries: Sequence[str],
    retrieval_service: RetrievalServiceProtocol,
    security: AcademicSecurityContext,
) -> list[RetrievedChunk]:
    if len(queries) == 1:
        return retrieval_service.retrieve(queries[0], security=security)

    return _merge_by_best_score(
        [retrieval_service.retrieve(query, security=security) for query in queries]
    )


def _merge_by_best_score(results: Sequence[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Keep each chunk once at its best score (earlier query wins ties),
    ranked and capped at `CHAT_RETRIEVAL_MAX_CHUNKS`."""

    best: dict[str, RetrievedChunk] = {}
    for chunks in results:
        for chunk in chunks:
            existing = best.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best[chunk.chunk_id] = chunk
    ranked = sorted(best.values(), key=lambda chunk: chunk.score, reverse=True)
    return ranked[: settings.CHAT_RETRIEVAL_MAX_CHUNKS]
