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


def rerank_chunks(
    per_query_chunks: Sequence[Sequence[RetrievedChunk]],
    *,
    score_threshold: float | None = None,
) -> TurnRerankResult:
    """`score_threshold` None = `CHAT_RERANK_SCORE_THRESHOLD` (the pre-hybrid cutoff); the
    graph passes the low `CHAT_RETRIEVAL_DENSE_FLOOR` when an LLM rerank follows, which
    then decides what answers the question (docs/specs/SPEC-hybrid-retrieval.md)."""

    best_scores = [
        max((chunk.score for chunk in chunks), default=0.0) for chunks in per_query_chunks
    ]
    return turn_result(
        [rerank(chunks, score_threshold=score_threshold).chunks for chunks in per_query_chunks],
        best_scores=best_scores,
    )


def turn_result(
    per_query_kept: Sequence[list[RetrievedChunk]], *, best_scores: list[float]
) -> TurnRerankResult:
    """Builds the turn's result from each query's kept chunks - shared with
    LLMRerankNode, which narrows those lists further."""

    per_query = [
        RerankResult(has_valid_context=bool(chunks), chunks=list(chunks))
        for chunks in per_query_kept
    ]
    merged = per_query[0].chunks if len(per_query) == 1 else _merge_by_best_rank(per_query_kept)
    # `per_query` stays uncapped: it is the LLM rerank's candidate pool, which may
    # hold more than the generation prompt does.
    return TurnRerankResult(
        per_query=per_query,
        chunks=merged[: settings.CHAT_CONTEXT_MAX_CHUNKS],
        best_scores=best_scores,
    )


def _merge_by_best_rank(results: Sequence[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Keep each chunk once, ordered by its best position in any query's list (the
    retrieval order: cosine for dense-only, RRF for hybrid), ties by best score, then
    by the earlier query. The copy kept is the best-scoring one (earlier query on a
    tie). Ranking by position rather than score keeps a BM25-only hit, whose score is
    the fixed lexical constant, behind the dense hits the fusion ranked above it.

    `turn_result` caps the result after thresholding, which keeps exactly what
    capping first then thresholding would: both drop only the lowest ranks."""

    best: dict[str, tuple[int, float, int, RetrievedChunk]] = {}
    for query_index, chunks in enumerate(results):
        for rank, chunk in enumerate(chunks):
            current = best.get(chunk.chunk_id)
            if current is None:
                best[chunk.chunk_id] = (rank, chunk.score, query_index, chunk)
                continue
            best_rank, best_score, first_query, kept = current
            best[chunk.chunk_id] = (
                min(best_rank, rank),
                max(best_score, chunk.score),
                first_query,
                chunk if chunk.score > best_score else kept,
            )
    ordered = sorted(best.values(), key=lambda entry: (entry[0], -entry[1], entry[2]))
    return [entry[3] for entry in ordered]
