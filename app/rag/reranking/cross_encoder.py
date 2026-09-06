from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import settings
from app.schemas.retrieval import RetrievedChunk


@dataclass(frozen=True)
class RerankResult:
    """Rerank output: `has_valid_context` + the filtered/re-ranked chunks."""

    has_valid_context: bool
    chunks: list[RetrievedChunk]


def rerank(
    chunks: Sequence[RetrievedChunk],
    *,
    score_threshold: float | None = None,
) -> RerankResult:
    """Sort by score, then drop anything under `score_threshold`.

    `score_threshold` defaults to `settings.RERANK_SCORE_THRESHOLD` (0.70)
    rather than a hardcoded constant. A real cross-encoder (bge-reranker-base)
    is not wired up yet - this keeps the existing deterministic base ranking
    and applies the threshold on top of it, which is enough to exercise the
    has_valid_context / ticket-fallback branch correctly; swapping in a real
    cross-encoder later only changes how `chunk.score` is computed, not this
    function's contract.
    """

    threshold = score_threshold if score_threshold is not None else settings.RERANK_SCORE_THRESHOLD
    ranked = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
    filtered = [chunk for chunk in ranked if chunk.score >= threshold]
    return RerankResult(has_valid_context=bool(filtered), chunks=filtered)
