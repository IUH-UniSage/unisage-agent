"""Re-enrich the points of a collection (summary + hypothetical questions) from the payload
they already hold - no PDF parse, no re-chunking - with the document title and heading path
in the enricher's input (29/10/2026, `enrich_input`). Dev/eval tool.

Only Gemini or Qwen may enrich: the run stops before the first call when an ACTIVE
EXTRACTION credential is anything else (a GPT row must be deactivated first). The dense
vectors are NOT touched: run the `rebuild` re-index with EMBEDDING_REBUILD_REEMBED=true
afterwards so the new summary/questions (and the title) are embedded.

    task db:reenrich env=eval -- --dry-run --limit 3      # show what would be sent
    task db:reenrich env=eval -- --backup runs/enrich-backup.jsonl
    task db:reenrich env=eval -- --limit 50 --concurrency 4

Resumable: a point already re-enriched carries `enrichment_version` and is skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

from qdrant_client import models
from redis import asyncio as redis_asyncio

from app.core.budget.tracker import BudgetTracker
from app.core.config import settings
from app.core.registry.model_registry import (
    active_collection,
    active_credentials_for,
    init_model_registry,
)
from app.core.usage.usage_recorder import UsageRecorder
from app.rag.enrichment.multi_representation import MultiRepresentationEnricher, enrich_input
from app.rag.retrieval.lexical import document_title
from app.rag.vectorstore import qdrant_store
from app.schemas.ingestion import Chunk, RegionType
from app.tools.reembed import invalidate_retrieval_cache
from app.worker.tasks.ingestion import _enrich_waiting_for_credentials

logger = logging.getLogger("reenrich")

ENRICHMENT_VERSION = "2026-10-29-title"


def allowed_extraction_model(provider: str | None, model_name: str | None) -> bool:
    """Gemini (provider google) or a Qwen model; never GPT or anything else."""

    name = (model_name or "").casefold()
    if "gpt" in name:
        return False
    return (provider or "").casefold() == "google" or "qwen" in name


def check_credentials() -> list[str]:
    """`provider/model` of every ACTIVE EXTRACTION credential that is not allowed."""

    return [
        f"{credential.provider}/{credential.model_name}"
        for credential in active_credentials_for("EXTRACTION")
        if not allowed_extraction_model(credential.provider, credential.model_name)
    ]


def chunk_of(payload: dict[str, Any]) -> Chunk:
    index = str(payload.get("chunk_id") or "0").rsplit(":", 1)[-1]
    region = payload.get("region_type") or RegionType.TEXT.value
    return Chunk(
        chunk_index=int(index) if index.isdigit() else 0,
        content=str(payload.get("content") or ""),
        region_type=RegionType(region)
        if region in RegionType._value2member_map_
        else RegionType.TEXT,
        heading_path=list(payload.get("heading_path") or []),
    )


def _points(client: Any, collection: str, limit: int | None) -> list[models.Record]:
    out: list[models.Record] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        out += [
            r for r in records if (r.payload or {}).get("enrichment_version") != ENRICHMENT_VERSION
        ]
        if offset is None or (limit is not None and len(out) >= limit):
            break
    return out[:limit] if limit is not None else out


def qwen_model(model_name: str) -> Any:
    """An OpenAI-compatible Qwen model from the judge's endpoint and key in `.env.eval`
    (`JUDGE_OPENAI_BASE_URL`, first of `JUDGE_OPENAI_API_KEYS`): used when the registry's
    Gemini EXTRACTION keys are out of quota (29/10/2026). Never a GPT model."""

    import os

    from app.core.llm.provider_models import build_model
    from app.core.registry.model_registry import CredentialConfig

    if not allowed_extraction_model("openai", model_name):
        raise SystemExit(f"Dừng: {model_name} không phải Qwen/Gemini.")
    base_url = os.environ.get("JUDGE_OPENAI_BASE_URL", "").strip()
    keys = [k.strip() for k in os.environ.get("JUDGE_OPENAI_API_KEYS", "").split(",") if k.strip()]
    if not base_url or not keys:
        raise SystemExit(
            "Dừng: thiếu JUDGE_OPENAI_BASE_URL / JUDGE_OPENAI_API_KEYS trong .env.eval."
        )
    return build_model(
        CredentialConfig(
            id="reenrich-qwen",
            revision=0,
            source_type="SELF_HOSTED",
            provider="openai",
            model_name=model_name,
            api_base_url=base_url,
            priority=0,
            max_rpm=None,
            api_key=keys[0],
        )
    )


class _Pacer:
    """At most `rpm` call starts per minute."""

    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            if now < self._next:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self._interval


async def run(args: argparse.Namespace) -> int:
    await init_model_registry()
    if args.qwen_model:
        return await run_qwen(args)
    refused = check_credentials()
    if refused:
        print(
            "Dừng: có model EXTRACTION đang bật không phải Gemini/Qwen: " + ", ".join(refused),
            file=sys.stderr,
        )
        return 2
    allowed = [f"{c.provider}/{c.model_name}" for c in active_credentials_for("EXTRACTION")]
    if not allowed:
        print("Dừng: không có model EXTRACTION nào đang bật.", file=sys.stderr)
        return 2
    print(f"Model EXTRACTION dùng để enrich: {sorted(set(allowed))}")

    client = qdrant_store.get_client()
    collection = args.collection or active_collection()
    points = _points(client, collection, args.limit)
    print(f"{collection}: {len(points)} point(s) to re-enrich")
    if args.dry_run:
        for record in points[:3]:
            payload = record.payload or {}
            print(
                "-----\n"
                + enrich_input(chunk_of(payload), document_title(payload.get("object_key")))[:600]
            )
        return 0

    # A dev tool appending a few lines: the blocking open is fine.
    backup = Path(args.backup).open("a", encoding="utf-8") if args.backup else None  # noqa: ASYNC230
    redis_client = redis_asyncio.Redis.from_url(settings.REDIS_URL)
    budget = BudgetTracker(redis_client=redis_client)
    recorder = UsageRecorder(request_id=str(uuid.uuid4()), purpose="INGEST", budget_tracker=budget)
    enricher = MultiRepresentationEnricher()
    semaphore = asyncio.Semaphore(args.concurrency)
    done = failed = 0
    status = "ERROR"

    async def one(record: models.Record) -> None:
        nonlocal done, failed
        payload = record.payload or {}
        async with semaphore:
            try:
                enriched = await _enrich_waiting_for_credentials(
                    enricher,
                    chunk_of(payload),
                    recorder,
                    budget,
                    document_title(payload.get("object_key")),
                )
            except Exception as exc:
                failed += 1
                logger.warning(
                    "re-enrich %s failed: %s", payload.get("chunk_id"), type(exc).__name__
                )
                return
        if not enriched.summary and not enriched.questions:
            failed += 1
            return
        if backup is not None:
            backup.write(
                json.dumps(
                    {
                        "id": str(record.id),
                        "summary": payload.get("summary"),
                        "questions": payload.get("questions"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            backup.flush()
        client.set_payload(
            collection_name=collection,
            payload={
                "summary": enriched.summary,
                "questions": enriched.questions,
                "enrichment_version": ENRICHMENT_VERSION,
            },
            points=[record.id],
        )
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(points)} re-enriched", flush=True)

    try:
        await budget.reserve_request(
            request_id=recorder.request_id,
            purpose="INGEST",
            estimate_usd=Decimal(str(settings.BUDGET_RESERVATION_FALLBACK_USD))
            * max(len(points), 1),
        )
        await asyncio.gather(*(one(record) for record in points))
        status = "SUCCESS" if not failed else "PARTIAL"
    finally:
        await recorder.close(status=status)
        await redis_client.aclose()
        if backup is not None:
            backup.close()
    await invalidate_retrieval_cache([record.payload or {} for record in points])
    print(f"Xong: {done} point(s) re-enriched, {failed} failed")
    return 0 if not failed else 1


async def run_qwen(args: argparse.Namespace) -> int:
    """Enrich with Qwen directly (no registry credential, no usage line in the backend)."""

    from app.rag.enrichment.multi_representation import _parse_enrichment_output

    model = qwen_model(args.qwen_model)
    print(f"Model enrich: qwen/{args.qwen_model} (≤ {args.rpm} lượt/phút)")
    client = qdrant_store.get_client()
    collection = args.collection or active_collection()
    points = _points(client, collection, args.limit)
    print(f"{collection}: {len(points)} point(s) to re-enrich")
    if args.dry_run or not points:
        return 0
    enricher = MultiRepresentationEnricher()
    agent = enricher._build_agent(model)
    pacer = _Pacer(args.rpm)
    semaphore = asyncio.Semaphore(args.concurrency)
    backup = Path(args.backup).open("a", encoding="utf-8") if args.backup else None  # noqa: ASYNC230
    done = failed = 0

    async def one(record: models.Record) -> None:
        nonlocal done, failed
        payload = record.payload or {}
        chunk = chunk_of(payload)
        text = enrich_input(chunk, document_title(payload.get("object_key")))
        enriched = None
        async with semaphore:
            for attempt in range(4):
                await pacer.wait()
                try:
                    result = await agent.run(text)
                    enriched = _parse_enrichment_output(
                        result.output or "", chunk, enricher.question_count
                    )
                    if enriched is not None:
                        break
                except Exception as exc:
                    logger.warning(
                        "qwen enrich %s attempt %d: %s",
                        payload.get("chunk_id"),
                        attempt + 1,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(10 * (attempt + 1))
        if enriched is None or (not enriched.summary and not enriched.questions):
            failed += 1
            return
        if backup is not None:
            backup.write(
                json.dumps(
                    {
                        "id": str(record.id),
                        "summary": payload.get("summary"),
                        "questions": payload.get("questions"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            backup.flush()
        client.set_payload(
            collection_name=collection,
            payload={
                "summary": enriched.summary,
                "questions": enriched.questions,
                "enrichment_version": ENRICHMENT_VERSION,
                "enrichment_model": f"qwen/{args.qwen_model}",
            },
            points=[record.id],
        )
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(points)} re-enriched", flush=True)

    try:
        await asyncio.gather(*(one(record) for record in points))
    finally:
        if backup is not None:
            backup.close()
    await invalidate_retrieval_cache([record.payload or {} for record in points])
    print(f"Xong: {done} point(s) re-enriched, {failed} failed")
    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-enrich a collection's points (Gemini/Qwen only)."
    )
    parser.add_argument("--collection", help="default: the registry's active collection")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--backup", help="JSONL file the old summary/questions are appended to")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--qwen-model",
        help="Qwen model (JUDGE_OPENAI_* of .env.eval), e.g. Qwen3.8-Max-fast-agent",
    )
    parser.add_argument("--rpm", type=int, default=55, help="Qwen calls per minute")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
