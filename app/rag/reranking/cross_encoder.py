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

    `score_threshold` defaults to `settings.CHAT_RERANK_SCORE_THRESHOLD` (0.70). With
    hybrid retrieval (docs/specs/SPEC-hybrid-retrieval.md) the graph passes the lower
    `CHAT_RETRIEVAL_DENSE_FLOOR` whenever an LLM rerank follows: this step is then a floor
    on the dense cosine (a BM25-only hit carries `CHAT_RETRIEVAL_LEXICAL_HIT_SCORE`), and the
    LLM rerank is the gate. No cross-encoder is wired up (planned as a later phase); one
    would only change how `chunk.score` is computed, not this function's contract.
    """

    threshold = (
        score_threshold if score_threshold is not None else settings.CHAT_RERANK_SCORE_THRESHOLD
    )
    ranked = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
    filtered = [chunk for chunk in ranked if chunk.score >= threshold]
    return RerankResult(has_valid_context=bool(filtered), chunks=filtered)
