"""Node 10: `RetrievalFilteringNode` (T1.9).

Deliberately does NOT filter by `department_access`/permission - that is
Phase 4 scope (see tasks/plan.md). Thin wrapper over `RetrievalService`
(real Qdrant search), reading `RETRIEVAL_MAX_CHUNKS` from settings via the
service itself instead of a hardcoded default.
"""

from app.rag.retrieval.service import RetrievalServiceProtocol
from app.schemas.retrieval import RetrievedChunk


def retrieve_chunks(
    query: str, retrieval_service: RetrievalServiceProtocol
) -> list[RetrievedChunk]:
    return retrieval_service.retrieve(query)
