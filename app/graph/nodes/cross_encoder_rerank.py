"""CrossEncoderRerankNode (09a) - each sub-query's chunks re-scored by a cross-encoder.

The LLM rerank judges 16 chunks with one model call of 3-6 s. A cross-encoder
(bge-reranker-v2-m3, `reranker/` service) reads every (question, chunk) pair in a few
tens of milliseconds and ranks them; this node keeps each sub-query's top
`CHAT_CROSS_ENCODER_KEEP` by that score. It never empties a sub-query: a sub-query with
candidates keeps at least one, so the "no chunk answers it -> web search" decision
stays with the LLM rerank (mode `cross_encoder_then_llm`) or does not happen (mode
`cross_encoder`).

The chunk's `score` (cosine, or the lexical constant) is left alone - the rescue rule,
the dense floor and web search's ordering read it - and the cross-encoder score goes
to `metadata["ce_score"]` for the trace and the benchmark. Any service failure is
returned as `failure` with the input result untouched: the graph then runs the LLM
rerank as in mode `llm` (fail-over, docs/specs/SPEC-cross-encoder-rerank.md).
"""

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import settings
from app.graph.nodes.post_retrieval_rerank import TurnRerankResult, turn_result
from app.integrations.cross_encoder_client import (
    CrossEncoderClient,
    CrossEncoderUnavailableError,
)
from app.rag.prompting.citations import source_title
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

CE_SCORE_KEY = "ce_score"
# Prefix of the AI-admin warning raised when the service failed and the LLM rerank ran.
CROSS_ENCODER_SKIPPED = "Bỏ qua bước xếp hạng cross-encoder (chạy lọc bằng LLM thay thế): "


@dataclass(frozen=True)
class CrossEncoderOutcome:
    result: TurnRerankResult
    elapsed_ms: int
    # Why the input result was returned unchanged (the service failed).
    failure: str | None = None


def passage_text(chunk: RetrievedChunk) -> str:
    """What the cross-encoder reads for a chunk: its heading path, then its content -
    the heading carries the table's subject (which programme level a fee row is for)."""

    if chunk.heading_path:
        return " > ".join(chunk.heading_path) + "\n" + chunk.content
    return chunk.content


async def cross_encoder_rerank(
    client: CrossEncoderClient,
    questions: Sequence[str],
    rerank_result: TurnRerankResult,
    *,
    keep: int | None = None,
) -> CrossEncoderOutcome:
    """`questions[i]` is sub-query i's standalone question, in the order of
    `rerank_result.per_query`. Sub-queries are scored concurrently (one request each)."""

    keep = keep or settings.CHAT_CROSS_ENCODER_KEEP
    started = time.perf_counter()
    per_query_chunks = [list(result.chunks) for result in rerank_result.per_query]
    try:
        scored = await asyncio.gather(
            *(
                _score_query(client, question, chunks)
                for question, chunks in zip(questions, per_query_chunks, strict=True)
            )
        )
    except CrossEncoderUnavailableError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "cross-encoder rerank failed after %d ms - falling over to the LLM rerank: %s",
            elapsed_ms,
            exc,
        )
        return CrossEncoderOutcome(result=rerank_result, elapsed_ms=elapsed_ms, failure=str(exc))

    per_query_kept: list[list[RetrievedChunk]] = []
    for query_index, (question, chunks, scores) in enumerate(
        zip(questions, per_query_chunks, scored, strict=True)
    ):
        order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
        kept = [
            chunks[i].model_copy(
                update={"metadata": {**chunks[i].metadata, CE_SCORE_KEY: scores[i]}}
            )
            for i in order[:keep]
        ]
        per_query_kept.append(kept)
        logger.info(
            "cross-encoder SQ%d %r: kept [%s]; dropped [%s]",
            query_index + 1,
            question,
            "; ".join(_label(chunks[i], scores[i]) for i in order[:keep]),
            "; ".join(_label(chunks[i], scores[i]) for i in order[keep:]),
        )

    narrowed = turn_result(per_query_kept, best_scores=rerank_result.best_scores)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "cross-encoder rerank kept %d of %d chunk(s) in %d ms (keep=%d per sub-query)",
        sum(len(kept) for kept in per_query_kept),
        sum(len(chunks) for chunks in per_query_chunks),
        elapsed_ms,
        keep,
    )
    return CrossEncoderOutcome(result=narrowed, elapsed_ms=elapsed_ms)


async def _score_query(
    client: CrossEncoderClient, question: str, chunks: list[RetrievedChunk]
) -> list[float]:
    if not chunks:
        return []
    response = await client.rerank(question, [passage_text(chunk) for chunk in chunks])
    return response.scores


def _label(chunk: RetrievedChunk, ce_score: float) -> str:
    section = f" > {chunk.heading_path[-1]}" if chunk.heading_path else ""
    return f"{source_title(chunk.source)}{section} (ce={ce_score:.2f}, s={chunk.score:.2f})"
