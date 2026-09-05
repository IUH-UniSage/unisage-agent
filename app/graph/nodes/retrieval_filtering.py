"""Node 10: `RetrievalFilteringNode` (T1.9).

Deliberately does NOT filter by `department_access`/permission - that is
Phase 4 scope (see tasks/plan.md). Thin wrapper over the existing
`RetrievalService` (demo corpus for now, not live Qdrant - see that
module's docstring), reading `RETRIEVAL_MAX_CHUNKS` from settings instead
of a hardcoded default.
"""

from app.rag.retrieval.service import RetrievalService
from app.schemas.retrieval import RetrievedChunk

_retrieval_service = RetrievalService()


def retrieve_chunks(
    query: str, *, user_faculty: str = "GLOBAL", user_level: int = 1
) -> list[RetrievedChunk]:
    return _retrieval_service.retrieve(query, user_faculty=user_faculty, user_level=user_level)
