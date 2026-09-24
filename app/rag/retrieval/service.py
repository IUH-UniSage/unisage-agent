from dataclasses import dataclass, field
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint

from app.core.config import settings
from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.vectorstore.qdrant_store import build_access_filter, get_client, search_chunks
from app.schemas.ingestion import SourceLocator
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext


class RetrievalServiceProtocol(Protocol):
    """What `GraphModels.retrieval` needs to provide - satisfied by the real
    `RetrievalService` below, and by test doubles that skip Qdrant/OpenAI
    entirely (see tests/llm_mocks.py's `FakeRetrievalService`)."""

    def retrieve(
        self,
        query: str,
        *,
        security: AcademicSecurityContext,
        limit: int | None = None,
    ) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class RetrievalService:
    """Embeds the query, then nearest-neighbor searches Qdrant's `content_vector`
    for ingested chunks (see app/rag/vectorstore/qdrant_store.py), pre-filtered
    by `security`'s `department_access` (see `build_access_filter`) - a chunk
    the caller isn't allowed to see is excluded at the Qdrant query itself,
    not filtered out afterward.

    `client`/`embedder` are built lazily on first use, not at construction
    time, so constructing a `RetrievalService()` never requires a reachable
    Qdrant or `OPENAI_API_KEY` - only actually calling `retrieve` does. This
    is what lets tests inject fakes for both without touching the network.
    """

    client: QdrantClient | None = None
    embedder: OpenAIEmbedder = field(default_factory=OpenAIEmbedder)

    def retrieve(
        self,
        query: str,
        *,
        security: AcademicSecurityContext,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        effective_limit = limit if limit is not None else settings.RETRIEVAL_MAX_CHUNKS
        (query_vector,) = self.embedder.embed([query])
        client = self.client or get_client()
        points = search_chunks(
            client,
            query_vector=query_vector,
            limit=effective_limit,
            query_filter=build_access_filter(security),
        )
        return [_to_retrieved_chunk(point) for point in points]


def _to_retrieved_chunk(point: ScoredPoint) -> RetrievedChunk:
    payload: dict[str, Any] = point.payload or {}
    raw_locator = payload.get("source_locator")
    source_locator = SourceLocator.model_validate(raw_locator) if raw_locator else None
    return RetrievedChunk(
        chunk_id=str(payload.get("chunk_id", point.id)),
        content=str(payload.get("content", "")),
        source=str(payload.get("object_key") or payload.get("document_id") or ""),
        score=max(0.0, min(1.0, point.score)),
        heading_path=list(payload.get("heading_path") or []),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        source_type=payload.get("source_type"),
        source_locator=source_locator,
        metadata={
            "document_id": payload.get("document_id"),
            "department": payload.get("department"),
            "access_level": payload.get("access_level"),
            "is_public": payload.get("is_public", False),
            "category": payload.get("category"),
            "region_type": payload.get("region_type"),
        },
    )
