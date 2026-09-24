from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.schemas.security import AcademicSecurityContext

# Dimension of OpenAI's `text-embedding-3-small`, the configured default model.
_EMBEDDING_DIMENSIONS = 1536

_VECTOR_NAMES = ("content_vector", "summary_vector", "questions_vector")

# A department_access entry with this id grants its access_level across every
# department, not just one - mirrors `TrustedContext.granted_access_level`'s
# WILDCARD_DEPARTMENT_ID on the ingestion side (app/api/deps.py), so the same
# token means the same thing whether it's gating an upload or a chat answer.
WILDCARD_DEPARTMENT_ID = "*"


def get_client() -> QdrantClient:
    """Build a Qdrant client from configured settings."""

    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


def ensure_collection(client: QdrantClient) -> None:
    """Bootstrap the chunks collection with its three named vectors and the
    payload indexes `build_access_filter`'s conditions need, if absent.

    Safe to call on every startup/task: a no-op when the collection already exists.
    Deliberately does not migrate an existing collection - a pre-existing
    collection from before the department/access_level payload index was
    added is expected to be dropped and recreated, not patched in place.
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


def build_access_filter(security: AcademicSecurityContext) -> models.Filter:
    """The permission pre-filter (flow_design node 08, tier 1): which chunks
    `security` is allowed to see, built ONLY from the JWT-verified
    `department_access` - never from `confirmed_metadata` (a student's own
    unverified in-conversation claims), so a claim like "em là học viên cao
    học" can never widen what gets retrieved.

    A chunk is visible when:
    - `is_public == True` - a document-level flag, independent of
      `access_level`/`department` (mirrors `unisage-backend`'s
      `Document.isPublic`), that opens the chunk to everyone, including a
      guest whose `department_access` is empty. This condition is always
      present, so the `should` list is never empty even for a guest, and it
      is NOT expressed as `access_level == 0`: a guest's own identity carries
      no `access_level` at all (only per-department entries do), so
      "public" has to be its own flag, not a level a non-public chunk could
      also legitimately hold; OR
    - one of `security.department_access`'s entries grants it: either that
      entry's `department_id` matches the chunk's `department` at an
      `access_level` the chunk's `access_level` is `<=` to, or the entry's
      `department_id` is the wildcard (`*`), which grants its `access_level`
      across every department the same way `TrustedContext.
      granted_access_level` does for ingestion. This clause never applies to
      a public chunk's visibility (already covered above) - it only gates
      non-public, department-scoped chunks.

    A chunk missing `department`/`access_level`/`is_public` (pre-migration
    data) matches none of these and is denied by default - see qdrant
    collection reset note in `ensure_collection`.
    """

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
    similarity, and would be meaningless against `RERANK_SCORE_THRESHOLD`
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
                },
            )
        ],
    )
