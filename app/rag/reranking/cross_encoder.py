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
    """Drop anything under `score_threshold`, keeping the retrieval order.

    The order is the store's: cosine for dense-only, reciprocal rank fusion for hybrid.
    It is NOT re-sorted by score here: a BM25-only hit carries the fixed
    `CHAT_RETRIEVAL_LEXICAL_HIT_SCORE` (0.70), above most dense cosines (0.5-0.6), so
    sorting by score put keyword-only chunks first and, whenever the LLM rerank kept
    many, they filled the context cap while the right dense chunks were cut
    (seen 21/10/2026 on "học phí ngành quản trị kinh doanh").

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
    # A BM25-only hit has no cosine: its `score` is the fixed lexical stand-in, not a
    # relevance measure comparable to the threshold, so it passes the floor on its flag and
    # is left to the LLM rerank (docs/specs/SPEC-hybrid-retrieval.md, Thiết kế 4-5).
    filtered = [
        chunk
        for chunk in chunks
        if chunk.score >= threshold or chunk.metadata.get("lexical_only") is True
    ]
    return RerankResult(has_valid_context=bool(filtered), chunks=filtered)
