"""Retrieval filtering node.

Thin wrapper over `RetrievalService` (real Qdrant search), reading
`CHAT_RETRIEVAL_MAX_CHUNKS` from settings via the service itself instead of a
hardcoded default. `security` is passed straight through to the service so
its `department_access` gates the Qdrant query itself (see
`app.rag.vectorstore.qdrant_store.build_access_filter`) - not applied as a
post-filter here.
"""

from app.rag.retrieval.service import RetrievalServiceProtocol
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext


def retrieve_chunks(
    query: str,
    retrieval_service: RetrievalServiceProtocol,
    security: AcademicSecurityContext,
) -> list[RetrievedChunk]:
    return retrieval_service.retrieve(query, security=security)
