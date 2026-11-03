"""Re-embed a collection IN PLACE with the ACTIVE embedding model, from the payload each
point holds: the three dense vectors and `bm25`, with the texts ingestion now embeds (the
document title first, 29/10/2026). Same model, same dimension, same identity - only the
texts changed - so the registry pointer and identity stay as they are. Dev/eval tool; a
production collection goes through the `rebuild` re-index with EMBEDDING_REBUILD_REEMBED=true.

    task db:reembed env=eval -- --dry-run
    task db:reembed env=eval                 # snapshot first, then re-embed
    task db:reembed env=eval -- --limit 64

A Qdrant snapshot of the collection is taken first (restore it to undo). Resumable: a point
already re-embedded carries `embedding_text_version` and is skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from decimal import Decimal
from typing import Any

from qdrant_client import models
from redis import asyncio as redis_asyncio

from app.core.budget.tracker import BudgetTracker
from app.core.config import settings
from app.core.registry.model_registry import active_collection, init_model_registry
from app.core.usage.usage_recorder import UsageRecorder
from app.rag.embeddings.provider import build_embedder
from app.rag.retrieval.lexical import BM25_VECTOR
from app.rag.vectorstore import qdrant_store
from app.worker.embedding_reindex import embedding_texts, sparse_of_payload

EMBEDDING_TEXT_VERSION = "2026-10-29-title"
_DENSE = ("content_vector", "summary_vector", "questions_vector")


async def invalidate_retrieval_cache(payloads: list[dict[str, Any]]) -> int:
    """Drop the cached retrievals of every document whose points this run rewrote: the
    cache key does not change when the texts are re-embedded in place (same identity), so
    without this a turn keeps the old ranking (29/10/2026: 258/268 eval turns did)."""

    from app.services.retrieval_cache_invalidation import invalidate_after_embed

    documents = {
        str(p.get("document_id")): (str(p.get("department") or ""), bool(p.get("is_public")))
        for p in payloads
        if p.get("document_id")
    }
    for document_id, (department_id, is_public) in documents.items():
        await invalidate_after_embed(document_id, department_id=department_id, is_public=is_public)
    return len(documents)


def _pending(client: Any, collection: str, limit: int | None) -> list[models.Record]:
    out: list[models.Record] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection, limit=256, offset=offset, with_payload=True
        )
        out += [
            r
            for r in records
            if (r.payload or {}).get("content")
            and (r.payload or {}).get("embedding_text_version") != EMBEDDING_TEXT_VERSION
        ]
        if offset is None or (limit is not None and len(out) >= limit):
            break
    return out[:limit] if limit is not None else out


async def run(args: argparse.Namespace) -> int:
    await init_model_registry()
    client = qdrant_store.get_client()
    collection = args.collection or active_collection()
    points = _pending(client, collection, args.limit)
    print(f"{collection}: {len(points)} point(s) to re-embed ({len(points) * 3} texts)")
    if args.dry_run or not points:
        if points:
            print("-----\n" + embedding_texts(points[0].payload or {})[0][:400])
        return 0
    if not args.no_snapshot:
        snapshot = client.create_snapshot(collection_name=collection)
        print(f"Snapshot: {snapshot.name if snapshot else '?'} (restore it to undo)")

    embedder = build_embedder()
    redis_client = redis_asyncio.Redis.from_url(settings.REDIS_URL)
    budget = BudgetTracker(redis_client=redis_client)
    recorder = UsageRecorder(request_id=str(uuid.uuid4()), purpose="INGEST", budget_tracker=budget)
    status = "ERROR"
    done = 0
    try:
        await budget.reserve_request(
            request_id=recorder.request_id,
            purpose="INGEST",
            estimate_usd=Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD)),
        )
        for start in range(0, len(points), args.batch):
            batch = points[start : start + args.batch]
            texts = [text for record in batch for text in embedding_texts(record.payload or {})]
            vectors = await embedder.embed_tracked(texts, recorder, budget)
            updates = []
            for position, record in enumerate(batch):
                dense = vectors[3 * position : 3 * position + 3]
                indices, values = sparse_of_payload(record.payload or {})
                updates.append(
                    models.PointVectors(
                        id=record.id,
                        vector={
                            **dict(zip(_DENSE, dense, strict=True)),
                            BM25_VECTOR: models.SparseVector(indices=indices, values=values),
                        },
                    )
                )
            client.update_vectors(collection_name=collection, points=updates)
            client.set_payload(
                collection_name=collection,
                payload={"embedding_text_version": EMBEDDING_TEXT_VERSION},
                points=[record.id for record in batch],
            )
            done += len(batch)
            print(f"  {done}/{len(points)}", end="\r", flush=True)
        status = "SUCCESS"
    finally:
        await recorder.close(status=status)
        await redis_client.aclose()
    dropped = await invalidate_retrieval_cache([record.payload or {} for record in points])
    print(f"\nXong: {done} point(s) re-embedded in {collection}")
    print(f"Retrieval cache dropped for {dropped} document(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-embed a collection in place.")
    parser.add_argument("--collection", help="default: the registry's active collection")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
