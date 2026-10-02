"""Post-retrieval rerank node.

Thin wrapper over `app.rag.reranking.cross_encoder.rerank`, using
`settings.CHAT_RERANK_SCORE_THRESHOLD` (default 0.70) rather than a hardcoded
constant, so it stays configurable. Each sub-query is reranked on its own,
so the graph knows which ones found no valid context (those go to
WebSearchNode), then the survivors are merged into the one list generation
cites from.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import settings
from app.rag.reranking.cross_encoder import RerankResult, rerank
from app.schemas.retrieval import RetrievedChunk


@dataclass(frozen=True)
class TurnRerankResult:
    """`per_query[i]` is query i's own rerank; `chunks` merges every query's
    survivors; `best_scores[i]` is query i's best score before the threshold
    (0.0 when retrieval found nothing at all)."""

    per_query: list[RerankResult]
    chunks: list[RetrievedChunk]
    best_scores: list[float]

    @property
    def has_valid_context(self) -> bool:
        return bool(self.chunks)

    @property
    def failed_query_indexes(self) -> list[int]:
        """Queries rerank left with no chunk at all, worst miss first (lowest
        best score, ties in query order) - the order web search spends its
        limited searches in."""

        failed = [
            index for index, result in enumerate(self.per_query) if not result.has_valid_context
        ]
        return sorted(failed, key=lambda index: self.best_scores[index])


def rerank_chunks(per_query_chunks: Sequence[Sequence[RetrievedChunk]]) -> TurnRerankResult:
    per_query = [rerank(chunks) for chunks in per_query_chunks]
    best_scores = [
        max((chunk.score for chunk in chunks), default=0.0) for chunks in per_query_chunks
    ]
    if len(per_query) == 1:
        return TurnRerankResult(
            per_query=per_query, chunks=per_query[0].chunks, best_scores=best_scores
        )
    return TurnRerankResult(
        per_query=per_query,
        best_scores=best_scores,
        chunks=_merge_by_best_score([result.chunks for result in per_query]),
    )


def _merge_by_best_score(results: Sequence[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Keep each chunk once at its best score (earlier query wins ties),
    ranked and capped at `CHAT_RETRIEVAL_MAX_CHUNKS`.

    Thresholding before this cap keeps exactly what capping first then
    thresholding did: both drop only the lowest scores."""

    best: dict[str, RetrievedChunk] = {}
    for chunks in results:
        for chunk in chunks:
            existing = best.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                best[chunk.chunk_id] = chunk
    ranked = sorted(best.values(), key=lambda chunk: chunk.score, reverse=True)
    return ranked[: settings.CHAT_RETRIEVAL_MAX_CHUNKS]
