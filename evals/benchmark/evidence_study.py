"""Threshold study for the evidence signals (UNISAGE-106 Phase 6, Task 10).

    task benchmark:evidence env=eval -- hybrid-on [--limit N] [--out FILE.json]

For every usable question of the run, retrieve again (hybrid, same collection, same
permission filter) so the fused chunks carry `rrf_score` / `dense_rank` / `bm25_rank`,
compute the sub-query's `EvidenceSignals`, and mark the question a *miss* when no expected
document is among the first 8 chunks. Then sweep the cosine threshold and print, for each:
how many questions `level()` calls weak, how many of the misses it catches, and how many
good retrievals it would send to the LLM rerank for nothing. The thresholds in settings are
set from this table, not by hand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.graph.evidence import signals_for
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.rag.retrieval.service import RetrievalService
from evals.benchmark.candidates import read_results, run_dir
from evals.run.runner import security_of

logger = logging.getLogger("evals.benchmark")


def _doc(chunk_id: str) -> str:
    return chunk_id.split(":")[0]


async def study(run: Path, *, limit: int | None) -> list[dict[str, Any]]:
    from app.core.lifespan import lifespan

    rows: list[dict[str, Any]] = []
    async with lifespan(None):  # type: ignore[arg-type]
        retrieval = RetrievalService()
        for row in read_results(run):
            expected = set(row.get("expected_document_ids") or [])
            retrievals = row.get("retrievals") or []
            if not expected or not retrievals:
                continue
            question = str(retrievals[0]["query"])
            per_query = await retrieve_chunks(
                [question],
                retrieval,
                security_of(row, row.get("access") or []),
                lexical_queries=[question],
            )
            chunks = per_query[0]
            signals = signals_for(chunks)
            top8 = chunks[: settings.CHAT_CONTEXT_MAX_CHUNKS]
            rows.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "miss_top8": not any(_doc(c.chunk_id) in expected for c in top8),
                    "miss_all": not any(_doc(c.chunk_id) in expected for c in chunks),
                    **signals.as_dict(),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def sweep(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    misses = sum(1 for r in rows if r["miss_top8"])
    for min_cosine in (0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75):
        for require_no_agreement in (True, False):
            weak = []
            for r in rows:
                cos = r["top1_cosine"]
                cosine_weak = cos is None or cos < min_cosine
                agree_weak = r["agreement_top3"] == 0 if require_no_agreement else True
                weak.append(r["candidates"] == 0 or (cosine_weak and agree_weak))
            flagged = sum(weak)
            caught = sum(1 for r, w in zip(rows, weak, strict=True) if w and r["miss_top8"])
            out.append(
                {
                    "min_cosine": min_cosine,
                    "and_no_agreement": require_no_agreement,
                    "flagged": flagged,
                    "flagged_pct": round(100 * flagged / len(rows), 1) if rows else 0,
                    "misses": misses,
                    "misses_caught": caught,
                    "false_alarms": flagged - caught,
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(prog="python -m evals.benchmark.evidence_study")
    parser.add_argument("run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = asyncio.run(study(run_dir(args.run), limit=args.limit))
    table = sweep(rows)
    print(
        f"questions={len(rows)} misses_top8={sum(1 for r in rows if r['miss_top8'])} "
        f"misses_all={sum(1 for r in rows if r['miss_all'])}"
    )
    print(f"{'min_cos':>8}{'&no_agree':>10}{'flagged':>9}{'%':>7}{'caught':>8}{'false':>7}")
    for t in table:
        print(
            f"{t['min_cosine']:>8}{t['and_no_agreement']!s:>10}{t['flagged']:>9}{t['flagged_pct']:>7}"
            f"{t['misses_caught']:>8}{t['false_alarms']:>7}"
        )
    print(
        "\nmisses:",
        [(r["id"], r["top1_cosine"], r["agreement_top3"]) for r in rows if r["miss_top8"]],
    )
    if args.out:
        args.out.write_text(
            json.dumps({"rows": rows, "sweep": table}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
