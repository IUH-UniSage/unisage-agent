"""Retrieval-only rerank benchmark straight from the question set (no recorded run).

The original harness (`candidates.py`) reads the 16 chunks a recorded eval turn retrieved.
On the official set there is no such run for every question, and the question is whether a
*wider* pool helps the cross-encoder, so this module retrieves the pool itself:

    task benchmark:pool env=eval -- collect --out benchmark_results/<name>
    task benchmark:pool env=eval -- ce      --out benchmark_results/<name>
    task benchmark:pool env=eval -- score   --out benchmark_results/<name>

1. `collect`: for every question with expected documents (both the exact `question` and the
   user-style `question_user`), embed the text once and run the graph's hybrid search
   (`RetrievalService._search`: 3 dense vectors + BM25, fused by RRF, the caller's
   permission filter from the row's persona exactly as `evals.run` builds it) at the pool
   depth (64) and at the graph's own depth (`CHAT_RETRIEVAL_MAX_CHUNKS`). No classifier, no
   HyDE, no decomposition: the question text is both the dense and the BM25 query. The
   retrieval cache sits in the graph, above this layer, so it is never read. Writes
   `pools.jsonl` (one line per query: ids in RRF order + scores) and `chunks.jsonl` (one
   line per distinct chunk: the passage the cross-encoder reads).
2. `ce`: for each query and each CE pool size, the graph's floor (`CHAT_RETRIEVAL_DENSE_FLOOR`,
   BM25-only hits pass) on the first N chunks, then one `/rerank` call with the agent's own
   passage builder (`passage_text`). Writes `ce.jsonl` (scores + latency per size).
3. `score`: doc-level metrics per variant (`none`, `ceN`), written to `metrics.json`.

Every step reads only the files the previous one wrote, so scoring is reproducible offline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

logger = logging.getLogger("evals.benchmark")

POOL_DEPTH = 64
CE_POOL_SIZES = (16, 40, 64)
KS = (1, 3, 5, 8)
Progress = Callable[[int, int], None]


# --- the queries ------------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryJob:
    question_id: str
    variant: str  # exact | user
    category: str
    persona: str
    expect_visible: bool
    text: str
    expected_document_ids: tuple[str, ...]
    expected_titles: tuple[str, ...]
    access: tuple[dict[str, Any], ...]

    @property
    def key(self) -> str:
        return f"{self.question_id}:{self.variant}"

    def security(self) -> AcademicSecurityContext:
        from evals.run.runner import security_of

        return security_of({"persona": self.persona}, list(self.access))


def load_jobs(dataset: Path, *, include_hidden: bool = True) -> list[QueryJob]:
    """Rows with expected documents, both variants. `expect_visible` false rows (the asker
    may not see the document) are kept when `include_hidden`, for the leak check only."""

    from evals.run.runner import (
        access_of,
        document_ids,
        document_titles,
        load_departments,
        load_questions,
        read_manifest,
    )

    manifest = read_manifest(dataset)
    departments = load_departments(dataset)
    jobs: list[QueryJob] = []
    for row in load_questions(dataset):
        if not row.get("expected_doc_ids"):
            continue
        visible = row.get("expect_visible") is not False
        if not visible and not include_hidden:
            continue
        for variant, field in (("exact", "question"), ("user", "question_user")):
            text = str(row.get(field) or "").strip()
            if not text:
                continue
            jobs.append(
                QueryJob(
                    question_id=str(row["id"]),
                    variant=variant,
                    category=str(row["category"]),
                    persona=str(row.get("persona") or "guest"),
                    expect_visible=visible,
                    text=text,
                    expected_document_ids=tuple(document_ids(row, manifest)),
                    expected_titles=tuple(document_titles(row, manifest)),
                    access=tuple(access_of(row, departments)),
                )
            )
    return jobs


# --- step 1: collect ----------------------------------------------------------------------------


def pool_entry(chunk: RetrievedChunk, rank: int) -> dict[str, Any]:
    meta = chunk.metadata
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": str(meta.get("document_id") or chunk.chunk_id.split(":")[0]),
        "rrf_rank": rank,
        "score": round(chunk.score, 6),  # best dense cosine, or the BM25-only stand-in
        "lexical_only": bool(meta.get("lexical_only")),
        "rrf_score": meta.get("rrf_score"),
        "dense_rank": meta.get("dense_rank"),
        "bm25_rank": meta.get("bm25_rank"),
    }


def chunk_record(chunk: RetrievedChunk) -> dict[str, Any]:
    from app.graph.nodes.cross_encoder_rerank import passage_text
    from app.rag.prompting.citations import source_title

    return {
        "chunk_id": chunk.chunk_id,
        "document_id": str(chunk.metadata.get("document_id") or ""),
        "title": source_title(chunk.source) if chunk.source else "",
        "passage": passage_text(chunk),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def collect(
    jobs: Sequence[QueryJob],
    out: Path,
    *,
    depth: int = POOL_DEPTH,
    graph_depth: int | None = None,
    batch: int = 16,
    on_progress: Progress | None = None,
) -> None:
    """Resumable: queries already in `pools.jsonl` are skipped."""

    from app.rag.retrieval.service import RetrievalService

    graph_depth = graph_depth or settings.CHAT_RETRIEVAL_MAX_CHUNKS
    await asyncio.to_thread(out.mkdir, parents=True, exist_ok=True)
    pools_path, chunks_path = out / "pools.jsonl", out / "chunks.jsonl"
    done = {row["key"] for row in _read_jsonl(pools_path)}
    known = {row["chunk_id"] for row in _read_jsonl(chunks_path)}
    todo = [job for job in jobs if job.key not in done]
    service = RetrievalService()
    finished = len(jobs) - len(todo)
    for start in range(0, len(todo), batch):
        group = todo[start : start + batch]
        started = time.perf_counter()
        vectors = await service.embedder.embed_async([job.text for job in group])
        embed_ms = (time.perf_counter() - started) * 1000 / len(group)
        rows, new_chunks = [], []
        for job, vector in zip(group, vectors, strict=True):
            security = job.security()
            t0 = time.perf_counter()
            pool = (
                await asyncio.to_thread(
                    service._search,
                    [vector],
                    security=security,
                    limit=depth,
                    lexical_queries=[job.text],
                )
            )[0]
            search_ms = (time.perf_counter() - t0) * 1000
            native = (
                await asyncio.to_thread(
                    service._search,
                    [vector],
                    security=security,
                    limit=graph_depth,
                    lexical_queries=[job.text],
                )
            )[0]
            for chunk in pool:
                if chunk.chunk_id not in known:
                    known.add(chunk.chunk_id)
                    new_chunks.append(chunk_record(chunk))
            rows.append(
                {
                    "key": job.key,
                    "id": job.question_id,
                    "variant": job.variant,
                    "category": job.category,
                    "persona": job.persona,
                    "expect_visible": job.expect_visible,
                    "text": job.text,
                    "expected_document_ids": list(job.expected_document_ids),
                    "expected_titles": list(job.expected_titles),
                    "access": list(job.access),
                    "pool": [pool_entry(chunk, rank) for rank, chunk in enumerate(pool, 1)],
                    "graph_pool_ids": [chunk.chunk_id for chunk in native],
                    "graph_depth": graph_depth,
                    "embed_ms": round(embed_ms, 1),
                    "search_ms": round(search_ms, 1),
                }
            )
        _append(chunks_path, new_chunks)
        _append(pools_path, rows)
        finished += len(group)
        if on_progress:
            on_progress(finished, len(jobs))


# --- step 2: cross-encoder ----------------------------------------------------------------------


def floored(pool: Sequence[dict[str, Any]], floor: float) -> list[dict[str, Any]]:
    """The graph's step 09 in front of any rerank: drop chunks under the dense floor, keep a
    BM25-only hit on its flag (`app.rag.reranking.cross_encoder.rerank`)."""

    return [entry for entry in pool if entry["score"] >= floor or entry["lexical_only"]]


def ce_candidates(pool: Sequence[dict[str, Any]], size: int, floor: float) -> list[dict[str, Any]]:
    return floored(pool[:size], floor)


async def score_ce(
    out: Path,
    *,
    sizes: Sequence[int] = CE_POOL_SIZES,
    floor: float | None = None,
    timeout_seconds: float = 30.0,
    only_visible: bool = True,
    on_progress: Progress | None = None,
) -> None:
    """Resumable: queries already in `ce.jsonl` are skipped. One request per (query, size),
    sequential (concurrency 1), so `took_ms` is the latency a lone chat turn would see."""

    from app.integrations.cross_encoder_client import CrossEncoderClient

    floor = settings.CHAT_RETRIEVAL_DENSE_FLOOR if floor is None else floor
    passages = {row["chunk_id"]: row["passage"] for row in _read_jsonl(out / "chunks.jsonl")}
    ce_path = out / "ce.jsonl"
    done = {row["key"] for row in _read_jsonl(ce_path)}
    pools = [
        row for row in _read_jsonl(out / "pools.jsonl") if row["expect_visible"] or not only_visible
    ]
    client = CrossEncoderClient(timeout_seconds=timeout_seconds)
    finished = 0
    for row in pools:
        finished += 1
        if row["key"] in done:
            continue
        result: dict[str, Any] = {"key": row["key"], "floor": floor, "sizes": {}}
        for size in sizes:
            candidates = ce_candidates(row["pool"], size, floor)
            ids = [entry["chunk_id"] for entry in candidates]
            started = time.perf_counter()
            response = await client.rerank(row["text"], [passages[cid] for cid in ids])
            result["sizes"][str(size)] = {
                "ids": ids,
                "scores": [round(s, 5) for s in response.scores],
                "took_ms": round((time.perf_counter() - started) * 1000, 1),
                "service_ms": response.took_ms,
                "truncated": response.truncated,
            }
        _append(ce_path, [result])
        if on_progress and (finished % 10 == 0 or finished == len(pools)):
            on_progress(finished, len(pools))


# --- step 3: metrics ---------------------------------------------------------------------------


def ce_order(ids: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Best CE score first, ties in retrieval order - as `cross_encoder_rerank` sorts."""

    order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
    return [ids[i] for i in order]


def kept_lists(
    pool_row: dict[str, Any],
    ce_row: dict[str, Any] | None,
    *,
    keep: int,
    floor: float,
    none_depth: int,
) -> dict[str, list[str]]:
    """Chunk ids each variant hands generation, best first. `none` = the floor on the graph's
    retrieval depth, cut to `keep` (what `CHAT_RERANK_MODE=none` does); `ceN` = the CE order of
    the floored first N, cut to `keep`. Also the full CE order (`ceN_full`) for rank stats."""

    lists: dict[str, list[str]] = {
        "none": [e["chunk_id"] for e in ce_candidates(pool_row["pool"], none_depth, floor)][:keep],
        "rrf_full": [e["chunk_id"] for e in floored(pool_row["pool"], floor)],
    }
    for size, entry in (ce_row or {}).get("sizes", {}).items():
        ordered = ce_order(entry["ids"], entry["scores"])
        lists[f"ce{size}"] = ordered[:keep]
        lists[f"ce{size}_full"] = ordered
    return lists


def recall_at(doc_ids: Sequence[str], expected: frozenset[str], k: int) -> float:
    return len(set(doc_ids[:k]) & expected) / len(expected) if expected else 0.0


def first_rank(doc_ids: Sequence[str], expected: frozenset[str]) -> int | None:
    return next((rank for rank, doc in enumerate(doc_ids, 1) if doc in expected), None)


def ndcg_at(doc_ids: Sequence[str], expected: frozenset[str], k: int) -> float:
    """Binary NDCG over the kept list at document level: a document counts once, at its first
    chunk (later chunks of the same document add nothing), ideal = every expected doc first."""

    seen: set[str] = set()
    dcg = 0.0
    for rank, doc in enumerate(doc_ids[:k], 1):
        if doc in expected and doc not in seen:
            seen.add(doc)
            dcg += 1.0 / math.log2(rank + 1)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(expected), k) + 1))
    return dcg / idcg if idcg else 0.0


def question_metrics(doc_ids: Sequence[str], expected: frozenset[str], keep: int) -> dict[str, Any]:
    rank = first_rank(doc_ids[:keep], expected)
    row: dict[str, Any] = {f"recall@{k}": recall_at(doc_ids, expected, k) for k in KS}
    row.update(
        {
            "hit@5": 1.0 if first_rank(doc_ids[:5], expected) else 0.0,
            "mrr": 1.0 / rank if rank else 0.0,
            f"ndcg@{keep}": ndcg_at(doc_ids, expected, keep),
            "first_rank": rank,
        }
    )
    return row


def score_all(
    out: Path, *, keep: int | None = None, floor: float | None = None, none_depth: int | None = None
) -> dict[str, Any]:
    """Per-query rows (`per_query.jsonl`) + the inputs the report aggregates."""

    keep = keep or settings.CHAT_CROSS_ENCODER_KEEP
    floor = settings.CHAT_RETRIEVAL_DENSE_FLOOR if floor is None else floor
    none_depth = none_depth or settings.CHAT_RETRIEVAL_MAX_CHUNKS
    ce_rows = {row["key"]: row for row in _read_jsonl(out / "ce.jsonl")}
    per_query: list[dict[str, Any]] = []
    for pool_row in _read_jsonl(out / "pools.jsonl"):
        expected = frozenset(pool_row["expected_document_ids"])
        doc_of = {e["chunk_id"]: e["document_id"] for e in pool_row["pool"]}
        pool_docs = [e["document_id"] for e in pool_row["pool"]]
        lists = kept_lists(
            pool_row, ce_rows.get(pool_row["key"]), keep=keep, floor=floor, none_depth=none_depth
        )
        row: dict[str, Any] = {
            key: pool_row[key]
            for key in ("key", "id", "variant", "category", "persona", "expect_visible", "text")
        }
        row["expected_document_ids"] = sorted(expected)
        row["expected_titles"] = pool_row["expected_titles"]
        row["in_pool"] = {str(n): recall_at(pool_docs, expected, n) for n in (8, 16, 40, 64)}
        row["first_rank_rrf"] = first_rank(pool_docs, expected)
        floored_docs = [doc_of[cid] for cid in lists["rrf_full"]]
        row["first_rank_rrf_floored"] = first_rank(floored_docs, expected)
        row["leak"] = bool(set(pool_docs) & expected) if not pool_row["expect_visible"] else None
        graph_ids = pool_row.get("graph_pool_ids") or []
        row["graph_pool_same_order"] = (
            graph_ids == [e["chunk_id"] for e in pool_row["pool"]][: len(graph_ids)]
        )
        row["graph_pool_overlap"] = (
            len(set(graph_ids) & {e["chunk_id"] for e in pool_row["pool"][: len(graph_ids)]})
            / len(graph_ids)
            if graph_ids
            else None
        )
        row["variants"] = {}
        for name, ids in lists.items():
            if name.endswith("_full") or name == "rrf_full":
                continue
            docs = [doc_of[cid] for cid in ids]
            metrics = question_metrics(docs, expected, keep)
            full = lists.get(f"{name}_full")
            if full is not None:
                metrics["first_rank_full"] = first_rank([doc_of[c] for c in full], expected)
            metrics["kept_ids"] = ids
            row["variants"][name] = metrics
        ce_row = ce_rows.get(pool_row["key"])
        row["ce_ms"] = (
            {size: entry["took_ms"] for size, entry in ce_row["sizes"].items()} if ce_row else {}
        )
        row["ce_candidates"] = (
            {size: len(entry["ids"]) for size, entry in ce_row["sizes"].items()} if ce_row else {}
        )
        row["ce_truncated"] = (
            {size: entry["truncated"] for size, entry in ce_row["sizes"].items()} if ce_row else {}
        )
        per_query.append(row)
    path = out / "per_query.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per_query), encoding="utf-8"
    )
    return {"keep": keep, "floor": floor, "none_depth": none_depth, "queries": len(per_query)}


# --- CLI ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(prog="python -m evals.benchmark.retrieval_pool")
    parser.add_argument("step", choices=("collect", "ce", "score"))
    parser.add_argument("--dataset", type=Path, default=Path("../unisage-gateway/dataset/official"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=POOL_DEPTH)
    parser.add_argument("--limit", type=int, default=None, help="first N queries (collect)")
    args = parser.parse_args(argv)

    def progress(done: int, total: int) -> None:
        print(f"{args.step}: {done}/{total}", flush=True)

    if args.step == "collect":
        from app.core.lifespan import lifespan

        jobs = load_jobs(args.dataset)
        if args.limit:
            jobs = jobs[: args.limit]

        async def run() -> None:
            async with lifespan(None):  # type: ignore[arg-type]
                await collect(jobs, args.out, depth=args.depth, on_progress=progress)

        asyncio.run(run())
    elif args.step == "ce":
        asyncio.run(score_ce(args.out, on_progress=progress))
    else:
        print(json.dumps(score_all(args.out), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
