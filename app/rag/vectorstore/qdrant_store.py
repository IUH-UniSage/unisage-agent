from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.schemas.security import AcademicSecurityContext

# Dimension of OpenAI's `text-embedding-3-small`, the configured default model.
_EMBEDDING_DIMENSIONS = 1536

_VECTOR_NAMES = ("content_vector", "summary_vector", "questions_vector")

# Grants its access_level in every department (same meaning as in app/api/deps.py).
WILDCARD_DEPARTMENT_ID = "*"


def get_client() -> QdrantClient:
    """Build a Qdrant client from configured settings."""

    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection (3 named vectors + permission payload indexes) if
    absent. An existing collection is never migrated - drop and recreate it."""

    if client.collection_exists(settings.QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config={
            name: models.VectorParams(size=_EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE)
            for name in _VECTOR_NAMES
        },
    )
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="department",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="access_level",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="is_public",
        field_schema=models.PayloadSchemaType.BOOL,
    )


def get_collection_dimension(client: QdrantClient) -> int | None:
    """The configured vector size of the collection's named vectors, or `None` if the
    collection doesn't exist yet. All 3 named vectors (`_VECTOR_NAMES`) are always created with
    the same size (`ensure_collection`), so reading `content_vector`'s is representative — used
    by the embedding identity guard (`app.core.embedding_identity`) to sanity-check a registered
    identity's `dimension` against what the collection is actually configured with."""

    if not client.collection_exists(settings.QDRANT_COLLECTION):
        return None
    info = client.get_collection(settings.QDRANT_COLLECTION)
    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict):
        content_params = vectors_config.get(_VECTOR_NAMES[0])
        return content_params.size if content_params is not None else None
    # A collection with a single unnamed vector (shouldn't happen for this collection, but
    # `VectorParams` is a valid non-dict shape for the SDK type) — defensive fallback.
    return vectors_config.size if vectors_config is not None else None


def collection_has_points(client: QdrantClient) -> bool:
    """True if the collection exists and already holds at least one point — used by the
    embedding identity guard to decide whether an empty collection may self-bootstrap its
    identity, or whether a collection with vectors but no registered identity must be refused."""

    if not client.collection_exists(settings.QDRANT_COLLECTION):
        return False
    info = client.get_collection(settings.QDRANT_COLLECTION)
    return bool(info.points_count)


def build_access_filter(security: AcademicSecurityContext) -> models.Filter:
    """Permission pre-filter, built only from the JWT-verified
    `department_access` (never from self-declared `confirmed_metadata`).

    A chunk is visible if `is_public`, or if a `department_access` entry
    covers its department (or is `*`) at an `access_level` >= the chunk's.
    Chunks missing these payload fields are denied."""

    should: list[models.Condition] = [
        models.FieldCondition(key="is_public", match=models.MatchValue(value=True))
    ]
    for entry in security.department_access:
        access_level_condition = models.FieldCondition(
            key="access_level", range=models.Range(lte=entry.access_level)
        )
        if entry.department_id == WILDCARD_DEPARTMENT_ID:
            should.append(access_level_condition)
        else:
            should.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="department",
                            match=models.MatchValue(value=entry.department_id),
                        ),
                        access_level_condition,
                    ]
                )
            )
    return models.Filter(should=should)


@dataclass(frozen=True)
class ChunkPoint:
    """One Qdrant point: a chunk's three named vectors plus its payload.

    Fields after `questions_vector` have defaults so a legacy chunk can be
    upserted without fabricating values."""

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
    is_public: bool = False
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
    structure_confidence: float | None = None
    parse_warnings: list[str] = field(default_factory=list)
    embedding_identity_key: str | None = None


def search_chunks(
    client: QdrantClient,
    *,
    query_vector: list[float],
    limit: int,
    query_filter: models.Filter | None = None,
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
    similarity, and would be meaningless against `CHAT_RERANK_SCORE_THRESHOLD`
    (a cosine-similarity cutoff).

    `query_filter` (see `build_access_filter`) is applied identically on all
    3 vector queries - a chunk the caller isn't allowed to see must never
    surface via ANY of the three representations, not just `content_vector`.

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
            query_filter=query_filter,
        )
        for point in response.points:
            existing = best_by_id.get(point.id)
            if existing is None or point.score > existing.score:
                best_by_id[point.id] = point

    ranked = sorted(best_by_id.values(), key=lambda point: point.score, reverse=True)
    return ranked[:limit]


def scroll_chunks_by_document(client: QdrantClient, document_id: str) -> list[models.Record]:
    """Return every indexed point (payload only, no vectors) for `document_id`,
    ordered by `chunk_id` (which sorts numerically-by-suffix since it's built
    as f"{document_id}:{chunk_index}").

    Used by the "indexed chunks" management view, which reads the payload
    Qdrant actually holds (including `summary`/`questions`, generated by
    `MultiRepresentationEnricher` at embed time) - distinct from
    `ChunkRepository.get_page`'s Postgres-backed chunking *draft*, which never
    has those two fields.

    Unlike `search_chunks`, this always fetches ALL matching points (a single
    document rarely has more than a few hundred chunks) rather than paginating
    at the Qdrant level, so the caller can apply simple page/limit slicing.
    Returns an empty list if the collection doesn't exist yet.
    """

    if not client.collection_exists(settings.QDRANT_COLLECTION):
        return []

    records: list[models.Record] = []
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            limit=200,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(batch)
        if next_offset is None:
            break

    records.sort(key=lambda record: (record.payload or {}).get("chunk_id", ""))
    return records


def delete_chunk_point(client: QdrantClient, document_id: str, chunk_id: str) -> None:
    """Delete one indexed point by its `chunk_id` payload field.

    Filtered by `document_id` too so a caller can never delete a point
    belonging to another document even if it somehow guessed a valid
    `chunk_id` string for it.
    """

    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    ),
                    models.FieldCondition(key="chunk_id", match=models.MatchValue(value=chunk_id)),
                ]
            )
        ),
    )


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
                    "is_public": point.is_public,
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
                    "structure_confidence": point.structure_confidence,
                    "parse_warnings": point.parse_warnings,
                    "embedding_identity_key": point.embedding_identity_key,
                },
            )
        ],
    )
