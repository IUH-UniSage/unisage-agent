"""RetrievalFilteringNode - searches every retrieval text of the turn with
the caller's permission filter, one ranked chunk list per query."""

import math
from collections.abc import Sequence

from app.core.config import settings
from app.rag.retrieval.service import RetrievalServiceProtocol
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext


async def retrieve_chunks(
    queries: Sequence[str],
    retrieval_service: RetrievalServiceProtocol,
    security: AcademicSecurityContext,
    *,
    lexical_queries: Sequence[str] | None = None,
) -> list[list[RetrievedChunk]]:
    """Results stay per query (in query order) so rerank can tell which
    sub-query found nothing; `rerank_chunks` merges them afterwards.

    `lexical_queries`: the standalone question of each query, for the BM25 side of the
    hybrid search (the dense side embeds `queries`, the HyDE texts)."""

    # A per-query quota keeps one sub-query from filling the whole top-k.
    limit = (
        None if len(queries) == 1 else math.ceil(settings.CHAT_RETRIEVAL_MAX_CHUNKS / len(queries))
    )
    return await retrieval_service.retrieve_many_async(
        list(queries),
        security=security,
        limit=limit,
        lexical_queries=list(lexical_queries) if lexical_queries is not None else None,
    )
