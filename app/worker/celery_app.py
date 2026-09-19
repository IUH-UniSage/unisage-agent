import logging
import uuid
from typing import Any

from celery import Celery

from app.core.config import settings
from app.core.events import publish_ingestion_event
from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.enrichment.multi_representation import MultiRepresentationEnricher
from app.rag.vectorstore import qdrant_store
from app.schemas.ingestion import Chunk

logger = logging.getLogger(__name__)

celery_app = Celery(
    "unisage_ingestion",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)
# Keep task results well past a wizard tab's lifetime so the client's
# reconciliation sweep can still read a terminal state days later.
celery_app.conf.result_expires = 60 * 60 * 24 * 7


@celery_app.task(bind=True, name="embed_chunks")
def embed_chunks(
    self: Any,
    document_id: str,
    object_key: str,
    chunks: list[dict[str, Any]],
    department_id: str,
    access_level: int,
    category: str = "HOC_VU",
) -> dict[str, Any]:
    """Enrich, embed, and upsert a client-approved chunk list into Qdrant.

    Reports percent-complete via `update_state` (for the client's
    reconciliation sweep) and a Redis `progress` event per chunk (for the
    live `WS /ingestion/events` relay), then a `completed` event on finish.
    One chunk's enrichment/embedding failure is recorded in the returned
    per-chunk results rather than raised, so it doesn't abort the batch.
    """

    task_id = self.request.id

    def _publish(payload: dict[str, Any]) -> None:
        publish_ingestion_event(
            {
                "task_id": task_id,
                "document_id": document_id,
                "department_id": department_id,
                **payload,
            }
        )

    embedder = OpenAIEmbedder()
    enricher = MultiRepresentationEnricher()
    client = qdrant_store.get_client()
    qdrant_store.ensure_collection(client)

    total = len(chunks)
    results: list[dict[str, Any]] = []

    for position, raw_chunk in enumerate(chunks):
        chunk = Chunk.model_validate(raw_chunk)
        try:
            enriched = enricher.enrich(chunk)
            # Empty summary/questions (the enrichment fallback) would send an
            # empty string to the embeddings API; fall back to the chunk's own
            # content so every point still gets three valid vectors.
            summary_text = enriched.summary or chunk.content
            questions_text = " ".join(enriched.questions) or chunk.content
            content_vector, summary_vector, questions_vector = embedder.embed(
                [chunk.content, summary_text, questions_text]
            )
            chunk_id = f"{document_id}:{chunk.chunk_index}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
            qdrant_store.upsert_chunk(
                client,
                qdrant_store.ChunkPoint(
                    point_id=point_id,
                    document_id=document_id,
                    object_key=object_key,
                    chunk_id=chunk_id,
                    content=chunk.content,
                    summary=enriched.summary,
                    questions=enriched.questions,
                    department=department_id,
                    access_level=access_level,
                    category=category,
                    region_type=chunk.region_type.value,
                    content_vector=content_vector,
                    summary_vector=summary_vector,
                    questions_vector=questions_vector,
                    source_type=chunk.source_type.value if chunk.source_type else None,
                    block_index=chunk.block_index,
                    heading_path=list(chunk.heading_path),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    source_locator=(
                        chunk.source_locator.model_dump(mode="json")
                        if chunk.source_locator is not None
                        else None
                    ),
                    column_names=chunk.column_names,
                    has_header=chunk.has_header,
                    header_source=chunk.header_source.value,
                    chunking_version=chunk.chunking_version,
                    structure_confidence=chunk.structure_confidence,
                    parse_warnings=list(chunk.parse_warnings),
                ),
            )
            results.append({"chunk_index": chunk.chunk_index, "status": "SUCCESS"})
        except Exception as exc:
            logger.exception("Failed to embed chunk %s of %s", chunk.chunk_index, document_id)
            results.append(
                {"chunk_index": chunk.chunk_index, "status": "FAILED", "error": str(exc)}
            )

        percent = round((position + 1) / total * 100)
        self.update_state(state="PROGRESS", meta={"percent": percent})
        _publish({"type": "progress", "percent": percent})

    _publish({"type": "completed", "state": "SUCCESS"})
    return {"percent": 100, "results": results}
