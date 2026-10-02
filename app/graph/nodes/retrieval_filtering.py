"""RetrievalFilteringNode - searches every retrieval text of the turn with
the caller's permission filter, one ranked chunk list per query."""

import math
from collections.abc import Sequence

from app.core.config import settings
from app.rag.retrieval.service import RetrievalServiceProtocol
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext


def retrieve_chunks(
    queries: Sequence[str],
    retrieval_service: RetrievalServiceProtocol,
    security: AcademicSecurityContext,
) -> list[list[RetrievedChunk]]:
    """Results stay per query (in query order) so rerank can tell which
    sub-query found nothing; `rerank_chunks` merges them afterwards."""

    if len(queries) == 1:
        return [retrieval_service.retrieve(queries[0], security=security)]

    # A per-query quota keeps one sub-query from filling the whole top-k.
    quota = math.ceil(settings.CHAT_RETRIEVAL_MAX_CHUNKS / len(queries))
    return [retrieval_service.retrieve(query, security=security, limit=quota) for query in queries]
