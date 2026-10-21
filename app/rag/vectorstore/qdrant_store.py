import os
import threading
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.core.registry.model_registry import active_collection, get_current_snapshot
from app.rag.retrieval.lexical import BM25_VECTOR, rrf_merge
from app.schemas.security import AcademicSecurityContext

# Dimension of OpenAI's `text-embedding-3-small` - only used to create the live collection
# before any identity is registered; a re-index target is created with its model's dimension.
_DEFAULT_EMBEDDING_DIMENSIONS = 1536

_VECTOR_NAMES = ("content_vector", "summary_vector", "questions_vector")

# Grants its access_level in every department (same meaning as in app/api/deps.py).
WILDCARD_DEPARTMENT_ID = "*"


def chunk_point_id(chunk_id: str) -> str:
    """The Qdrant point id ingestion gives the chunk `chunk_id` (see `embed_chunks` in
    app/worker/tasks/ingestion.py, which builds the same uuid5)."""

    return str(uuid5(NAMESPACE_URL, chunk_id))


_client: tuple[int, QdrantClient] | None = None
_client_lock = threading.Lock()


def get_client() -> QdrantClient:
    """The process's Qdrant client, built from configured settings on first use.

    Shared so a chat turn reuses its pooled connection instead of paying a new connection
    and the client's server-version check every time. Keyed by pid: a Celery prefork child
    must never reuse a connection its parent opened."""

    global _client
    pid = os.getpid()
    with _client_lock:
        if _client is None or _client[0] != pid:
            _client = (pid, QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT))
        return _client[1]


def _is_missing_collection(exc: UnexpectedResponse) -> bool:
    return exc.status_code == 404


def _name(collection: str | None) -> str:
    """`collection`, or the live collection (the snapshot's, else `QDRANT_COLLECTION`)."""

    return collection or active_collection()


def _default_dimension() -> int:
    snapshot = get_current_snapshot()
    identity = snapshot.embedding_index_identity if snapshot is not None else None
    if identity is not None and identity.dimension:
        return int(identity.dimension)
    return _DEFAULT_EMBEDDING_DIMENSIONS


def ensure_collection(
    client: QdrantClient, collection: str | None = None, dimension: int | None = None
) -> None:
    """Create the collection (3 named vectors + payload indexes) if absent. An existing
    collection is never migrated - re-index into a new one instead.

    `dimension` defaults to the live identity's (1536 before one is registered)."""

    name = _name(collection)
    if client.collection_exists(name):
        return
    size = dimension or _default_dimension()
    client.create_collection(
        collection_name=name,
        vectors_config={
            vector_name: models.VectorParams(size=size, distance=models.Distance.COSINE)
            for vector_name in _VECTOR_NAMES
        },
        sparse_vectors_config=sparse_vectors_config(),
    )
    _sparse_cache[name] = True
    for field_name, schema in (
        ("department", models.PayloadSchemaType.KEYWORD),
        ("access_level", models.PayloadSchemaType.INTEGER),
        ("is_public", models.PayloadSchemaType.BOOL),
        ("document_id", models.PayloadSchemaType.KEYWORD),
    ):
        client.create_payload_index(
            collection_name=name, field_name=field_name, field_schema=schema
        )


def get_collection_dimension(client: QdrantClient, collection: str | None = None) -> int | None:
    """The configured vector size of the collection's named vectors, or `None` if the
    collection doesn't exist yet. All 3 named vectors (`_VECTOR_NAMES`) are always created with
    the same size (`ensure_collection`), so reading `content_vector`'s is representative — used
    by the embedding identity guard (`app.core.registry.embedding_identity`) to sanity-check a
    registered identity's `dimension` against what the collection is actually configured with."""

    name = _name(collection)
    if not client.collection_exists(name):
        return None
    info = client.get_collection(name)
    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict):
        content_params = vectors_config.get(_VECTOR_NAMES[0])
        return content_params.size if content_params is not None else None
    # A collection with a single unnamed vector (shouldn't happen for this collection, but
    # `VectorParams` is a valid non-dict shape for the SDK type) — defensive fallback.
    return vectors_config.size if vectors_config is not None else None


def sparse_vectors_config() -> dict[str, models.SparseVectorParams]:
    """The `bm25` sparse vector: term frequencies from `app.rag.retrieval.lexical`, IDF
    applied by Qdrant (docs/specs/SPEC-hybrid-retrieval.md)."""

    return {BM25_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)}


# Whether a collection carries the `bm25` sparse vector, by name: read once per process,
# so a chat turn never pays a `get_collection` round trip. `add_sparse_vector_config` and
# `ensure_collection` update it; `reset_sparse_cache` is for tests.
_sparse_cache: dict[str, bool] = {}


def reset_sparse_cache() -> None:
    _sparse_cache.clear()


def has_sparse_vector(client: QdrantClient, collection: str | None = None) -> bool:
    """True when the collection exists and has the `bm25` sparse vector configured - the
    condition for the hybrid query; a collection not yet backfilled/rebuilt is searched
    dense-only."""

    name = _name(collection)
    cached = _sparse_cache.get(name)
    if cached is not None:
        return cached
    if not client.collection_exists(name):
        return False
    sparse = client.get_collection(name).config.params.sparse_vectors or {}
    found = BM25_VECTOR in sparse
    _sparse_cache[name] = found
    return found


def point_with_bm25(
    record: models.Record, *, payload_update: dict[str, Any] | None = None
) -> models.PointStruct:
    """`record` (scrolled with its vectors) as a point for a collection that has the `bm25`
    sparse vector: dense vectors as they are, `bm25` computed from the payload (heading path +
    content, like ingestion), payload copied with `payload_update` applied."""

    from app.rag.retrieval.lexical import lexical_text, sparse_vector

    payload = {**(record.payload or {}), **(payload_update or {})}
    vectors = dict(record.vector) if isinstance(record.vector, dict) else {}
    dense = {name: cast(list[float], vectors[name]) for name in _VECTOR_NAMES if name in vectors}
    bm25 = sparse_vector(
        lexical_text(str(payload.get("content") or ""), list(payload.get("heading_path") or []))
    )
    return models.PointStruct(
        id=record.id,
        vector=point_vectors(
            dense.get(_VECTOR_NAMES[0], []),
            dense.get(_VECTOR_NAMES[1], []),
            dense.get(_VECTOR_NAMES[2], []),
            bm25=bm25,
        ),
        payload=payload,
    )


def copy_points_with_bm25(
    client: QdrantClient,
    source: str,
    target: str,
    *,
    batch: int = 256,
    on_batch: Callable[[int], None] | None = None,
) -> int:
    """Copy every point of `source` into `target` (dense vectors and payload as they are)
    and give each one its `bm25` sparse vector from the payload. The way a collection
    indexed before hybrid retrieval gets the sparse vector: Qdrant cannot add a vector to
    an existing collection, so the points move to one created with it (`ensure_collection`).
    No embedding call. Returns the number of points copied."""

    copied = 0
    offset: PointOffset = None
    while True:
        records, offset = client.scroll(
            collection_name=source, limit=batch, offset=offset, with_payload=True, with_vectors=True
        )
        points = [point_with_bm25(record) for record in records]
        if points:
            client.upsert(collection_name=target, points=points)
            copied += len(points)
            if on_batch is not None:
                on_batch(copied)
        if offset is None:
            return copied


def collection_has_points(client: QdrantClient, collection: str | None = None) -> bool:
    """True if the collection exists and already holds at least one point — used by the
    embedding identity guard to decide whether an empty collection may self-bootstrap its
    identity, or whether a collection with vectors but no registered identity must be refused."""

    name = _name(collection)
    if not client.collection_exists(name):
        return False
    info = client.get_collection(name)
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
    # BM25 term frequencies of the chunk (`lexical.sparse_vector`); None/empty = no sparse
    # vector written (a collection without `bm25` accepts the point all the same).
    bm25: tuple[list[int], list[float]] | None = None


def search_chunks(
    client: QdrantClient,
    *,
    query_vector: list[float],
    limit: int,
    query_filter: models.Filter | None = None,
    collection: str | None = None,
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

    return search_chunks_batch(
        client,
        query_vectors=[query_vector],
        limits=[limit],
        query_filter=query_filter,
        collection=collection,
    )[0]


SparseQuery = tuple[list[int], list[float]]


def search_chunks_batch(
    client: QdrantClient,
    *,
    query_vectors: list[list[float]],
    limits: list[int],
    query_filter: models.Filter | None = None,
    collection: str | None = None,
    sparse_queries: Sequence[SparseQuery | None] | None = None,
    candidate_multiplier: int | None = None,
    rrf_k: int | None = None,
    lexical_hit_score: float | None = None,
) -> list[list[models.ScoredPoint]]:
    """`search_chunks` for several queries at once: every query x named
    vector goes out in one `query_batch_points` round trip.

    Hybrid (docs/specs/SPEC-hybrid-retrieval.md): with `sparse_queries` (one BM25 sparse
    vector per query, None for a query without terms), each query also searches the `bm25`
    vector, every list fetches `limit x candidate_multiplier`, the 3 dense lists are merged
    (best cosine per chunk) and that ranking is fused with the BM25 ranking by reciprocal
    rank fusion, both with the same weight. The fusion only decides the order and the cut; the
    returned `score` stays the best dense cosine of the chunk (everything downstream - the
    floor, the LLM rerank's rescue rule, web search's ordering, the retrieval cache - reads
    it as cosine), or `lexical_hit_score` for a chunk only BM25 found. The caller checks
    `has_sparse_vector` first.

    `query_filter` (the caller's permissions) is applied identically to every query, sparse
    included: a chunk the caller may not see never surfaces by matching a code."""

    hybrid = sparse_queries is not None
    multiplier = (
        (candidate_multiplier or settings.CHAT_RETRIEVAL_HYBRID_CANDIDATES_MULTIPLIER)
        if hybrid
        else 1
    )
    requests: list[models.QueryRequest] = []
    # Per query: how many requests went out (3 dense, +1 when it had a sparse vector).
    request_counts: list[int] = []
    for position, (vector, limit) in enumerate(zip(query_vectors, limits, strict=True)):
        fetch = limit * multiplier
        count = 0
        for vector_name in _VECTOR_NAMES:
            requests.append(
                models.QueryRequest(
                    query=vector,
                    using=vector_name,
                    limit=fetch,
                    with_payload=True,
                    filter=query_filter,
                )
            )
            count += 1
        sparse = sparse_queries[position] if sparse_queries is not None else None
        if sparse is not None and sparse[0]:
            requests.append(
                models.QueryRequest(
                    query=models.SparseVector(indices=list(sparse[0]), values=list(sparse[1])),
                    using=BM25_VECTOR,
                    limit=fetch,
                    with_payload=True,
                    filter=query_filter,
                )
            )
            count += 1
        request_counts.append(count)
    try:
        responses = client.query_batch_points(collection_name=_name(collection), requests=requests)
    except UnexpectedResponse as exc:
        # No collection yet (nothing ingested) is a normal state, not a failure. Checked
        # here rather than with a `collection_exists` call first, which would cost every
        # chat turn an extra round trip.
        if _is_missing_collection(exc):
            return [[] for _ in query_vectors]
        raise

    dense_count = len(_VECTOR_NAMES)
    results: list[list[models.ScoredPoint]] = []
    cursor = 0
    for limit, count in zip(limits, request_counts, strict=True):
        dense_responses = responses[cursor : cursor + dense_count]
        sparse_response = responses[cursor + dense_count] if count > dense_count else None
        cursor += count
        best_by_id: dict[str | int | UUID, models.ScoredPoint] = {}
        for response in dense_responses:
            for point in response.points:
                existing = best_by_id.get(point.id)
                if existing is None or point.score > existing.score:
                    best_by_id[point.id] = point
        if sparse_response is None:
            ranked = sorted(best_by_id.values(), key=lambda point: point.score, reverse=True)
            results.append(ranked[:limit])
            continue
        # Two lists of equal weight: the dense side merged first (best cosine per chunk, as
        # the dense-only search ranks it), then BM25. Fusing the 3 dense lists separately
        # gave dense three votes to BM25's one, and a chunk matched only by its code never
        # reached the top (measured 21/10/2026).
        dense_ranked = sorted(best_by_id.values(), key=lambda point: point.score, reverse=True)
        ranked_lists: list[list[str | int | UUID]] = [
            [point.id for point in dense_ranked],
            [point.id for point in sparse_response.points],
        ]
        lexical_score = (
            lexical_hit_score
            if lexical_hit_score is not None
            else settings.CHAT_RETRIEVAL_LEXICAL_HIT_SCORE
        )
        sparse_by_id = {point.id: point for point in sparse_response.points}
        fused = rrf_merge(ranked_lists, k=rrf_k or settings.CHAT_RETRIEVAL_RRF_K)
        ranked = []
        for point_id, _rrf in fused[:limit]:
            dense_point = best_by_id.get(cast("str | int | UUID", point_id))
            if dense_point is not None:
                ranked.append(dense_point)
            else:
                lexical_point = sparse_by_id[cast("str | int | UUID", point_id)]
                ranked.append(lexical_point.model_copy(update={"score": lexical_score}))
        results.append(ranked)
    return results


def fetch_chunks_by_ids(
    client: QdrantClient,
    chunk_ids: Sequence[str],
    query_filter: models.Filter | None = None,
    collection: str | None = None,
) -> list[models.Record]:
    """The points of `chunk_ids` (payload only), looked up by point id - no embedding
    and no vector search - with `query_filter` (the caller's permissions) still applied.
    In `chunk_ids` order; a chunk no longer indexed or no longer visible is missing."""

    if not chunk_ids:
        return []
    must: list[models.Condition] = [
        models.HasIdCondition(has_id=[chunk_point_id(chunk_id) for chunk_id in chunk_ids])
    ]
    if query_filter is not None:
        must.append(query_filter)
    try:
        records, _ = client.scroll(
            collection_name=_name(collection),
            scroll_filter=models.Filter(must=must),
            limit=len(chunk_ids),
            with_payload=True,
            with_vectors=False,
        )
    except UnexpectedResponse as exc:
        if _is_missing_collection(exc):
            return []
        raise
    order = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    return sorted(
        records,
        key=lambda record: order.get(str((record.payload or {}).get("chunk_id")), len(order)),
    )


def scroll_chunks_by_document(
    client: QdrantClient, document_id: str, collection: str | None = None
) -> list[models.Record]:
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

    name = _name(collection)
    if not client.collection_exists(name):
        return []

    records: list[models.Record] = []
    next_offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=name,
            scroll_filter=_document_filter(document_id),
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


def delete_chunk_point(
    client: QdrantClient, document_id: str, chunk_id: str, collection: str | None = None
) -> None:
    """Delete one indexed point by its `chunk_id` payload field.

    Filtered by `document_id` too so a caller can never delete a point
    belonging to another document even if it somehow guessed a valid
    `chunk_id` string for it.
    """

    name = _name(collection)
    if not client.collection_exists(name):
        return
    client.delete(
        collection_name=name,
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


def upsert_chunk(client: QdrantClient, point: ChunkPoint, collection: str | None = None) -> None:
    """Upsert one chunk's point, with its named vectors and resolved payload."""

    client.upsert(
        collection_name=_name(collection),
        points=[
            models.PointStruct(
                id=point.point_id,
                vector=point_vectors(
                    point.content_vector,
                    point.summary_vector,
                    point.questions_vector,
                    bm25=point.bm25,
                ),
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


def point_vectors(
    content_vector: list[float],
    summary_vector: list[float],
    questions_vector: list[float],
    *,
    bm25: tuple[list[int], list[float]] | None = None,
) -> dict[str, Any]:
    """The named vectors of a point: the 3 dense ones, plus `bm25` when the chunk has terms."""

    vectors: dict[str, Any] = {
        "content_vector": content_vector,
        "summary_vector": summary_vector,
        "questions_vector": questions_vector,
    }
    if bm25 is not None and bm25[0]:
        vectors[BM25_VECTOR] = models.SparseVector(indices=list(bm25[0]), values=list(bm25[1]))
    return vectors


def _document_filter(document_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )


def count_points(
    client: QdrantClient, collection: str | None = None, query_filter: models.Filter | None = None
) -> int:
    """Exact number of points (matching `query_filter`); 0 when the collection doesn't exist."""

    name = _name(collection)
    if not client.collection_exists(name):
        return 0
    return client.count(collection_name=name, count_filter=query_filter, exact=True).count


def delete_document_points(
    client: QdrantClient,
    document_id: str,
    *,
    collection: str | None = None,
    keep_chunk_ids: Collection[str] | None = None,
) -> int:
    """Delete every point of `document_id` except those of `keep_chunk_ids` (after a re-embed:
    the chunks the new set no longer has). Returns how many points were deleted."""

    name = _name(collection)
    if not client.collection_exists(name):
        return 0
    selector = _document_filter(document_id)
    if keep_chunk_ids:
        selector = models.Filter(
            must=selector.must,
            must_not=[
                models.HasIdCondition(
                    has_id=[chunk_point_id(chunk_id) for chunk_id in keep_chunk_ids]
                )
            ],
        )
    stale = client.count(collection_name=name, count_filter=selector, exact=True).count
    if stale:
        client.delete(collection_name=name, points_selector=models.FilterSelector(filter=selector))
    return stale


def set_document_payload(
    client: QdrantClient,
    document_id: str,
    payload: dict[str, Any],
    *,
    collection: str | None = None,
) -> int:
    """Overwrite `payload`'s keys on every point of `document_id` (vectors untouched). Returns
    how many points it touched."""

    name = _name(collection)
    if not client.collection_exists(name):
        return 0
    selector = _document_filter(document_id)
    touched = client.count(collection_name=name, count_filter=selector, exact=True).count
    if touched:
        client.set_payload(collection_name=name, payload=payload, points=selector)
    return touched


PointOffset = str | int | UUID | None


def scroll_points(
    client: QdrantClient,
    collection: str,
    *,
    limit: int,
    offset: PointOffset = None,
    with_payload: bool = True,
    with_vectors: bool = False,
) -> tuple[list[models.Record], PointOffset]:
    """One page of `collection` (vectors only on request) in point id order, and the offset of
    the next page (None after the last one)."""

    records, next_offset = client.scroll(
        collection_name=collection,
        limit=limit,
        offset=offset,
        with_payload=with_payload,
        with_vectors=with_vectors,
    )
    return list(records), next_offset


def point_ids(client: QdrantClient, collection: str, page_size: int = 1000) -> set[str]:
    """Every point id of `collection`, as strings; empty when it doesn't exist."""

    if not client.collection_exists(collection):
        return set()
    ids: set[str] = set()
    offset: PointOffset = None
    while True:
        records, offset = scroll_points(
            client, collection, limit=page_size, offset=offset, with_payload=False
        )
        ids.update(str(record.id) for record in records)
        if offset is None:
            return ids


def retrieve_points(
    client: QdrantClient, collection: str, ids: Sequence[str], *, with_vectors: bool = False
) -> list[models.Record]:
    """The points `ids` of `collection` with their payload (vectors only on request)."""

    if not ids:
        return []
    return list(
        client.retrieve(
            collection_name=collection,
            ids=list(ids),
            with_payload=True,
            with_vectors=with_vectors,
        )
    )


def upsert_points(
    client: QdrantClient, collection: str, points: Sequence[models.PointStruct]
) -> None:
    if points:
        client.upsert(collection_name=collection, points=list(points))


def delete_points(client: QdrantClient, collection: str, ids: Sequence[str]) -> None:
    if ids:
        client.delete(
            collection_name=collection, points_selector=models.PointIdsList(points=list(ids))
        )


def delete_collection(client: QdrantClient, collection: str) -> None:
    """Drop `collection` if it exists - only ever a cancelled or failed re-index target."""

    if client.collection_exists(collection):
        client.delete_collection(collection_name=collection)
