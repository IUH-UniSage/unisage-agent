"""Node 11: `PostRetrievalRerankNode` (T1.10).

Thin wrapper over `app.rag.reranking.cross_encoder.rerank`, using
`settings.RERANK_SCORE_THRESHOLD` (default 0.70) — not a hardcoded
constant, per plan.md.
"""

from collections.abc import Sequence

from app.rag.reranking.cross_encoder import RerankResult, rerank
from app.schemas.retrieval import RetrievedChunk


def rerank_chunks(chunks: Sequence[RetrievedChunk]) -> RerankResult:
    return rerank(chunks)
