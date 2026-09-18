from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models

from app.core.config import settings

# Dimension of OpenAI's `text-embedding-3-small`, the configured default model.
_EMBEDDING_DIMENSIONS = 1536

_VECTOR_NAMES = ("content_vector", "summary_vector", "questions_vector")


def get_client() -> QdrantClient:
    """Build a Qdrant client from configured settings."""

    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


def ensure_collection(client: QdrantClient) -> None:
    """Bootstrap the chunks collection with its three named vectors, if absent.

    Safe to call on every startup/task: a no-op when the collection already exists.
    """

    if client.collection_exists(settings.QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config={
            name: models.VectorParams(size=_EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE)
            for name in _VECTOR_NAMES
        },
    )


@dataclass(frozen=True)
class ChunkPoint:
    """One Qdrant point: a chunk's three named vectors plus its retrieval payload.

    The structural-metadata fields below (`source_type` through
    `chunking_version`) all default to `None`/`"legacy"` so a caller
    upserting a point for a legacy chunk (missing these fields entirely)
    doesn't have to fabricate values - see `RetrievedChunk`/
    `_to_retrieved_chunk` (Phase 6) for how a point missing them degrades
    gracefully at read time instead of erroring.
    """

    point_id: str
    document_id: str
    object_key: str
    chunk_id: str
    content: str
    summary: str
    questions: list[str]
    department: str
    access_level: int
    category: str
    region_type: str
    content_vector: list[float]
    summary_vector: list[float]
    questions_vector: list[float]
    source_type: str | None = None
    block_index: int | None = None
    heading_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    source_locator: dict[str, Any] | None = None
    column_names: list[str] | None = None
    has_header: bool = False
    header_source: str | None = None
    chunking_version: str = "legacy"


def search_chunks(
    client: QdrantClient,
    *,
    query_vector: list[float],
    limit: int,
) -> list[models.ScoredPoint]:
    """Nearest-neighbor search across all 3 named vectors (content/summary/
    questions - see app/rag/enrichment/multi_representation.py), keeping the
    best-scoring representation per chunk.

    A single-vector search on `content_vector` alone misses chunks whose raw
    text embeds poorly against a short factual query even though the same
    chunk's `summary` or one of its precomputed `questions` is a near-exact
    match - fanning out to all 3 and keeping the max score per chunk finds
    those. Uses plain per-vector queries + client-side max, not Qdrant's
    native RRF fusion: RRF scores are rank-based (~0.01-0.03), not cosine
    similarity, and would be meaningless against `RERANK_SCORE_THRESHOLD`
    (a cosine-similarity cutoff).

    Returns an empty list (not an error) when the collection doesn't exist
    yet - a fresh environment with nothing ingested is a normal state, not
    a failure, for the retrieval node calling this.
    """

    if not client.collection_exists(settings.QDRANT_COLLECTION):
        return []

    best_by_id: dict[str | int | UUID, models.ScoredPoint] = {}
    for vector_name in _VECTOR_NAMES:
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            using=vector_name,
            limit=limit,
            with_payload=True,
        )
        for point in response.points:
            existing = best_by_id.get(point.id)
            if existing is None or point.score > existing.score:
                best_by_id[point.id] = point

    ranked = sorted(best_by_id.values(), key=lambda point: point.score, reverse=True)
    return ranked[:limit]


def upsert_chunk(client: QdrantClient, point: ChunkPoint) -> None:
    """Upsert one chunk's point, with its named vectors and resolved payload."""

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=[
            models.PointStruct(
                id=point.point_id,
                vector={
                    "content_vector": point.content_vector,
                    "summary_vector": point.summary_vector,
                    "questions_vector": point.questions_vector,
                },
                payload={
                    "document_id": point.document_id,
                    "object_key": point.object_key,
                    "chunk_id": point.chunk_id,
                    "content": point.content,
                    "summary": point.summary,
                    "questions": point.questions,
                    "department": point.department,
                    "access_level": point.access_level,
                    "category": point.category,
                    "region_type": point.region_type,
                    "source_type": point.source_type,
                    "block_index": point.block_index,
                    "heading_path": point.heading_path,
                    "page_start": point.page_start,
                    "page_end": point.page_end,
                    "source_locator": point.source_locator,
                    "column_names": point.column_names,
                    "has_header": point.has_header,
                    "header_source": point.header_source,
                    "chunking_version": point.chunking_version,
                },
            )
        ],
    )
