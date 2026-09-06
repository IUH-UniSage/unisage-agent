"""Post-retrieval rerank node.

Thin wrapper over `app.rag.reranking.cross_encoder.rerank`, using
`settings.RERANK_SCORE_THRESHOLD` (default 0.70) rather than a hardcoded
constant, so it stays configurable.
"""

from collections.abc import Sequence

from app.rag.reranking.cross_encoder import RerankResult, rerank
from app.schemas.retrieval import RetrievedChunk


def rerank_chunks(chunks: Sequence[RetrievedChunk]) -> RerankResult:
    return rerank(chunks)
