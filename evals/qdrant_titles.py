"""Give already-ingested chunks a readable citation name without re-ingesting.

The citation chip shows `source_title(object_key)` and falls back to the bare
`document_id` when a chunk has no `object_key` (chunks written by the old direct
pipeline, whose `document_id` is the manifest `file_id`). This sets
`object_key = <title>.pdf` on every point of each manifest row, matching the row
by `document_id` (backend UUID) or `file_id`. Only the chip label changes: opening
the file still needs a `documents` row in the backend, which only the upload flow
(`evals.ingest`) creates.

Reads QDRANT_HOST / QDRANT_PORT / QDRANT_COLLECTION like the agent (`task ... env=`).

Usage:
    python -m evals.qdrant_titles --dataset ../unisage-gateway/dataset/official --dry-run
"""

import argparse
import csv
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.core.config import settings
from evals.crawl.download import DownloadState
from evals.titles import upload_name


def ids_of(row: dict[str, str]) -> list[str]:
    return [value for value in (row.get("document_id"), row["file_id"]) if value]


def retitle(
    client: QdrantClient,
    collection: str,
    rows: list[dict[str, str]],
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, int]:
    """Returns {file_id: points updated (or that would be)} for rows with points."""

    touched: dict[str, int] = {}
    for row in rows:
        if not row.get("title"):
            continue
        must: list[models.Condition] = [
            models.FieldCondition(key="document_id", match=models.MatchAny(any=ids_of(row)))
        ]
        if not overwrite:
            must.append(models.IsEmptyCondition(is_empty=models.PayloadField(key="object_key")))
        selector = models.Filter(must=must)
        count = client.count(collection, count_filter=selector, exact=True).count
        if not count:
            continue
        touched[row["file_id"]] = count
        if not dry_run:
            client.set_payload(
                collection, payload={"object_key": upload_name(row["title"])}, points=selector
            )
    return touched


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--only", type=Path, help="CSV whose file_id column limits the rows")
    parser.add_argument("--collection", default=settings.QDRANT_COLLECTION)
    parser.add_argument("--dry-run", action="store_true", help="count the points, change nothing")
    parser.add_argument(
        "--overwrite", action="store_true", help="also replace an object_key that is already set"
    )
    args = parser.parse_args()

    rows = DownloadState(args.dataset).manifest
    if args.only is not None:
        with args.only.open(newline="", encoding="utf-8-sig") as handle:
            wanted = {row["file_id"] for row in csv.DictReader(handle)}
        rows = [row for row in rows if row["file_id"] in wanted]

    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    touched = retitle(client, args.collection, rows, dry_run=args.dry_run, overwrite=args.overwrite)
    verb = "would update" if args.dry_run else "updated"
    print(
        f"{args.collection}: {verb} {sum(touched.values())} points of {len(touched)} documents "
        f"({len(rows) - len(touched)} manifest rows have no matching points)"
    )


if __name__ == "__main__":
    main()
