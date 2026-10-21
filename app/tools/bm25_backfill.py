"""Give a collection indexed before hybrid retrieval its `bm25` sparse vector
(docs/specs/SPEC-hybrid-retrieval.md). Dev/eval tool, and the fallback path.

Qdrant cannot add a vector to an existing collection (verified on 1.15: `update_collection`
answers "Not existing vector name"), so this rebuilds the collection under the SAME name
through a temporary one: copy every point (dense vectors + payload, no embedding call) into
`<name>__bm25tmp` created with the sparse vector, drop `<name>`, recreate it with the sparse
vector, copy back, drop the temporary one. Chat on that collection is interrupted for the
seconds the swap takes and the registry pointer/identity stay untouched (same name) - fine
for dev and eval. Production goes through the "rebuild" re-index (plan Task 9), which
cuts over atomically.

    task db:bm25-backfill env=eval                 # the env's live collection
    task db:bm25-backfill env=eval -- --dry-run    # report only
    task db:bm25-backfill -- --collection unisage_chunks__20261021120000
"""

from __future__ import annotations

import argparse
import sys
import time

from app.core.config import settings
from app.rag.retrieval.lexical import BM25_VECTOR
from app.rag.vectorstore import qdrant_store


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild a collection with the bm25 sparse vector."
    )
    parser.add_argument("--collection", default=settings.QDRANT_COLLECTION)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    client = qdrant_store.get_client()
    name = args.collection
    if not client.collection_exists(name):
        print(f"Collection {name!r} does not exist.")
        return 1
    total = qdrant_store.count_points(client, name)
    dimension = qdrant_store.get_collection_dimension(client, name)
    if qdrant_store.has_sparse_vector(client, name):
        print(f"{name}: {total} point(s), {BM25_VECTOR} already present - nothing to do.")
        return 0
    print(f"{name}: {total} point(s), dimension {dimension}, {BM25_VECTOR} absent")
    if args.dry_run:
        return 0

    temp = f"{name}__bm25tmp"
    started = time.perf_counter()
    if client.collection_exists(temp):
        client.delete_collection(temp)
    qdrant_store.ensure_collection(client, temp, dimension)
    copied = qdrant_store.copy_points_with_bm25(
        client,
        name,
        temp,
        batch=args.batch,
        on_batch=lambda n: print(f"  -> temp {n}/{total}", end="\r"),
    )
    print()
    if copied != total:
        print(f"Copied {copied} of {total} point(s) into {temp}; leaving {name} untouched.")
        return 1
    # Swap: recreate the collection under its own name with the sparse vector.
    client.delete_collection(name)
    qdrant_store.reset_sparse_cache()
    qdrant_store.ensure_collection(client, name, dimension)
    back = qdrant_store.copy_points_with_bm25(
        client,
        temp,
        name,
        batch=args.batch,
        on_batch=lambda n: print(f"  <- {name} {n}/{total}", end="\r"),
    )
    print()
    if back != total:
        print(f"Copied back {back} of {total} point(s); the full copy is still in {temp}.")
        return 1
    client.delete_collection(temp)
    print(
        f"Done: {name} rebuilt with {BM25_VECTOR}, {total} point(s), {time.perf_counter() - started:.1f}s."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
