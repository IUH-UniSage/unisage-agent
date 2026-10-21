import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http.models import Record, ScoredPoint

from app.core.config import settings
from app.rag.embeddings.provider import EmbeddingProvider, build_embedder
from app.rag.retrieval.lexical import sparse_query_vector
from app.rag.vectorstore import qdrant_store
from app.rag.vectorstore.qdrant_store import (
    SparseQuery,
    build_access_filter,
    fetch_chunks_by_ids,
    get_client,
    has_sparse_vector,
    search_chunks_batch,
)
from app.schemas.ingestion import SourceLocator
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

logger = logging.getLogger(__name__)


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

    def retrieve_many(
        self,
        queries: list[str],
        *,
        security: AcademicSecurityContext,
        limit: int | None = None,
        lexical_queries: list[str] | None = None,
    ) -> list[list[RetrievedChunk]]: ...

    async def retrieve_many_async(
        self,
        queries: list[str],
        *,
        security: AcademicSecurityContext,
        limit: int | None = None,
        lexical_queries: list[str] | None = None,
    ) -> list[list[RetrievedChunk]]: ...

    def fetch(
        self,
        chunk_ids: list[str],
        *,
        security: AcademicSecurityContext,
    ) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class RetrievalService:
    """Embeds the query and searches Qdrant, pre-filtered by the caller's
    permissions (see `build_access_filter`).

    `client`/`embedder` are built lazily on first use, not at construction
    time, so constructing a `RetrievalService()` never requires a reachable
    Qdrant or a loaded model registry snapshot - only actually calling
    `retrieve` does. This is what lets tests inject fakes for both without
    touching the network.
    """

    client: QdrantClient | None = None
    embedder: EmbeddingProvider = field(default_factory=build_embedder)

    def retrieve(
        self,
        query: str,
        *,
        security: AcademicSecurityContext,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        return self.retrieve_many([query], security=security, limit=limit)[0]

    def warm_up(self) -> None:
        """Opens the Qdrant client and resolves the embedder (credential, identity guard),
        which the first retrieval would otherwise do inside a chat turn (~2 s, 16-10-2026)."""

        if self.client is None:
            get_client()
        self.embedder.warm_up()

    def retrieve_many(
        self,
        queries: list[str],
        *,
        security: AcademicSecurityContext,
        limit: int | None = None,
        lexical_queries: list[str] | None = None,
    ) -> list[list[RetrievedChunk]]:
        """One embedding call and one Qdrant round trip for all `queries`. `lexical_queries`
        (one per query, the standalone question) adds the BM25 side of the hybrid search
        when the collection has it (docs/specs/SPEC-hybrid-retrieval.md)."""

        started_at = time.monotonic()
        query_vectors = self.embedder.embed(queries)
        embedded_at = time.monotonic()
        result = self._search(
            query_vectors, security=security, limit=limit, lexical_queries=lexical_queries
        )
        _log_timing(len(queries), started_at, embedded_at)
        return result

    async def retrieve_many_async(
        self,
        queries: list[str],
        *,
        security: AcademicSecurityContext,
        limit: int | None = None,
        lexical_queries: list[str] | None = None,
    ) -> list[list[RetrievedChunk]]:
        """`retrieve_many` for chat: embeds on the caller's event loop (see
        `EmbeddingProvider.embed_async`), only the synchronous Qdrant call goes to a worker
        thread. The instance is shared across chat requests (`app.api.deps`), so the
        embedder's connections must always be used from the server's loop."""

        started_at = time.monotonic()
        query_vectors = await self.embedder.embed_async(queries)
        embedded_at = time.monotonic()
        result = await asyncio.to_thread(
            self._search,
            query_vectors,
            security=security,
            limit=limit,
            lexical_queries=lexical_queries,
        )
        _log_timing(len(queries), started_at, embedded_at)
        return result

    def _search(
        self,
        query_vectors: list[list[float]],
        *,
        security: AcademicSecurityContext,
        limit: int | None,
        lexical_queries: list[str] | None = None,
    ) -> list[list[RetrievedChunk]]:
        effective_limit = limit if limit is not None else settings.CHAT_RETRIEVAL_MAX_CHUNKS
        client = self.client or get_client()
        sparse_queries: list[SparseQuery | None] | None = None
        if (
            lexical_queries
            and settings.RETRIEVAL_HYBRID_ENABLED
            and len(lexical_queries) == len(query_vectors)
            and has_sparse_vector(client)
        ):
            sparse_queries = [sparse_query_vector(text) for text in lexical_queries]
        per_query_points = search_chunks_batch(
            client,
            query_vectors=query_vectors,
            limits=[effective_limit] * len(query_vectors),
            query_filter=build_access_filter(security),
            sparse_queries=sparse_queries,
        )
        return [
            [_to_retrieved_chunk(point, point.score) for point in points]
            for points in per_query_points
        ]

    def fetch(
        self,
        chunk_ids: list[str],
        *,
        security: AcademicSecurityContext,
    ) -> list[RetrievedChunk]:
        """Chunks picked on an earlier turn, by id: no embedding, no search, no rerank.
        Still filtered by the caller's permissions; ones gone or no longer visible are
        missing. They were kept by that turn's rerank, so they score 1."""

        client = self.client or get_client()
        records = fetch_chunks_by_ids(client, chunk_ids, build_access_filter(security))
        return [_to_retrieved_chunk(record, 1.0) for record in records]


def _log_timing(query_count: int, started_at: float, embedded_at: float) -> None:
    """Always logged: where a slow retrieval spent its time."""

    finished_at = time.monotonic()
    logger.info(
        "Retrieval timing: queries=%d embed_ms=%d qdrant_ms=%d",
        query_count,
        int((embedded_at - started_at) * 1000),
        int((finished_at - embedded_at) * 1000),
    )


def _to_retrieved_chunk(point: ScoredPoint | Record, score: float) -> RetrievedChunk:
    payload: dict[str, Any] = point.payload or {}
    raw_locator = payload.get("source_locator")
    source_locator = SourceLocator.model_validate(raw_locator) if raw_locator else None
    return RetrievedChunk(
        chunk_id=str(payload.get("chunk_id", point.id)),
        content=str(payload.get("content", "")),
        source=str(payload.get("object_key") or payload.get("document_id") or ""),
        score=max(0.0, min(1.0, score)),
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
            # Hybrid: BM25 found it, no dense cosine was measured; `score` is a stand-in.
            "lexical_only": bool(payload.get(qdrant_store.LEXICAL_ONLY_FLAG, False)),
        },
    )
