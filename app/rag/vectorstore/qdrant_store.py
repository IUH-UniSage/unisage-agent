from dataclasses import dataclass

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
    """One Qdrant point: a chunk's three named vectors plus its retrieval payload."""

    point_id: str
    document_id: str
    object_key: str
    chunk_id: str
    content: str
    summary: str
    questions: list[str]
    department: str
    access_level: int
    region_type: str
    content_vector: list[float]
    summary_vector: list[float]
    questions_vector: list[float]


def search_chunks(
    client: QdrantClient,
    *,
    query_vector: list[float],
    limit: int,
    vector_name: str = "content_vector",
) -> list[models.ScoredPoint]:
    """Nearest-neighbor search on one named vector, with payload attached.

    Returns an empty list (not an error) when the collection doesn't exist
    yet - a fresh environment with nothing ingested is a normal state, not
    a failure, for the retrieval node calling this.
    """

    if not client.collection_exists(settings.QDRANT_COLLECTION):
        return []
    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        using=vector_name,
        limit=limit,
        with_payload=True,
    )
    return response.points


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
                    "region_type": point.region_type,
                },
            )
        ],
    )
