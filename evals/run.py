"""Run the evaluation questions through the chat graph in-process (Task 10, 10b).

For every selected question (and variant: the exact `question`, the user-style
`question_user`, or both) the graph runs once as the row's persona, against the environment
chosen with `task ... env=` (its Qdrant collection, its backend for the model registry). Each
result is appended to `runs/<name>/results.jsonl` as soon as it is known, with the
deterministic checks of `evals.metrics`; re-running with the same `--out` skips what is
already there, so an interrupted run (Ctrl+C, quota) continues where it stopped.

Needs: Postgres/Redis/Qdrant of the env, and backend-java (model registry). No agent API,
worker or gateway.

Usage (via task, from unisage-agent):
    task eval:run env=eval -- --limit 20
    task eval:run set=official env=eval -- --variant both --web off
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.graph.streaming_state import GraphModels
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry
from evals.ingest import DEPARTMENTS_FILE, is_quota
from evals.metrics import WILDCARD, score_row, summarize
from evals.recording import RecordingRetrieval, TurnResult, chunk_summary, run_question

VARIANTS = ("exact", "user")


# --- selection ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    question: dict[str, Any]
    variant: str

    @property
    def key(self) -> str:
        return f"{self.question['id']}:{self.variant}"

    @property
    def text(self) -> str:
        if self.variant == "user":
            return str(self.question["question_user"])
        return str(self.question["question"])


def load_questions(dataset: Path) -> list[dict[str, Any]]:
    lines = (dataset / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def select_jobs(
    questions: list[dict[str, Any]],
    *,
    subset: str,
    variant: str,
    categories: set[str] | None = None,
    ids: set[str] | None = None,
) -> list[Job]:
    """`subset=demo`: the demo questions plus those needing no document (they work on any
    corpus); `official`: everything. A question without `question_user` only runs exact."""

    variants = VARIANTS if variant == "both" else (variant,)
    jobs: list[Job] = []
    for question in questions:
        if subset == "demo" and not (
            question.get("source_set") == "demo" or not question.get("expected_doc_ids")
        ):
            continue
        if categories and question["category"] not in categories:
            continue
        if ids and question["id"] not in ids:
            continue
        for name in variants:
            if name == "user" and not question.get("question_user"):
                continue
            jobs.append(Job(question, name))
    return jobs


def spread(jobs: list[Job], limit: int) -> list[Job]:
    """The first `limit` turns taken round-robin across categories (file order inside each),
    so a quick `--limit 20` run covers every kind of question instead of the first block."""

    by_category: dict[str, list[Job]] = {}
    for job in jobs:
        by_category.setdefault(job.question["category"], []).append(job)
    picked: list[Job] = []
    queues = list(by_category.values())
    while len(picked) < limit and any(queues):
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


# --- personas and documents -----------------------------------------------------------------


def load_departments(dataset: Path) -> dict[str, str]:
    path = dataset / DEPARTMENTS_FILE
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def access_of(question: dict[str, Any], departments: dict[str, str]) -> list[dict[str, Any]]:
    """The persona's grants with department codes turned into the backend UUIDs Qdrant holds
    (`*` stays). A code missing from `departments` cannot match anything, so it is kept as is
    - the run reports it rather than silently widening access."""

    grants = (question.get("asker") or {}).get("department_access") or []
    return [
        {
            "department_id": grant["department_id"]
            if grant["department_id"] == WILDCARD
            else departments.get(grant["department_id"], grant["department_id"]),
            "access_level": int(grant["access_level"]),
        }
        for grant in grants
    ]


def security_of(question: dict[str, Any], access: list[dict[str, Any]]) -> AcademicSecurityContext:
    persona = question.get("persona") or "guest"
    if not access:
        return AcademicSecurityContext()
    return AcademicSecurityContext(
        user_id=f"eval-{persona}",
        role="EVAL",
        department_access=[DepartmentAccessEntry(**grant) for grant in access],
    )


def document_ids(question: dict[str, Any], manifest: dict[str, dict[str, str]]) -> list[str]:
    """Expected `file_id`s as the `document_id` stored in Qdrant: the backend UUID once
    ingested through the upload flow, else the `file_id` itself (older direct ingests)."""

    return [
        (manifest.get(file_id) or {}).get("document_id") or file_id
        for file_id in question.get("expected_doc_ids") or []
    ]


# --- one result row -------------------------------------------------------------------------


def result_row(
    job: Job,
    result: TurnResult,
    *,
    access: list[dict[str, Any]],
    expected_document_ids: list[str],
    web: str,
) -> dict[str, Any]:
    question = job.question
    output = result.output
    recording = result.recording
    row: dict[str, Any] = {
        "id": question["id"],
        "variant": job.variant,
        "asked": job.text,
        "category": question["category"],
        "case": question.get("case"),
        "source_set": question.get("source_set"),
        "persona": question.get("persona"),
        "expect_visible": question.get("expect_visible"),
        "expected_intent": question.get("expected_intent"),
        "expected_answer": question.get("expected_answer"),
        "evidence": question.get("evidence"),
        "expected_doc_ids": question.get("expected_doc_ids") or [],
        "expected_document_ids": expected_document_ids,
        "expected_numbers": question.get("expected_numbers"),
        "access": access,
        "web": web,
        "response": result.response_text,
        "citations": output.citations if output else [],
        "used_ticket_fallback": bool(output and output.used_ticket_fallback),
        "used_web_search": bool(output and output.used_web_search),
        "asked_back": bool(output and output.pending_round is not None),
        "calculation_items": output.calculation_items if output else [],
        "nodes": recording.nodes,
        "prompts": {
            name: texts
            for name, texts in recording.prompts.items()
            if name in ("03_MessageClassificationNode", "06_QueryTransformationNode")
        },
        "retrievals": recording.retrievals,
        "context": [
            {**chunk_summary(chunk), "content": chunk.content} for chunk in recording.context_chunks
        ],
        "web_results": [
            {"url": page.url, "title": page.title, "content": page.content}
            for page in recording.web_results
        ],
        "ttft_ms": recording.ttft_ms,
        "total_ms": result.total_ms,
        "usage": usage_summary(result.usage_lines),
        "error": result.error,
        "error_code": result.error_code,
    }
    row["checks"] = score_row(row)
    return row


def usage_summary(lines: list[dict[str, Any]]) -> dict[str, Any]:
    calls = len(lines)
    tokens_in = sum(int(line.get("inputTokens") or 0) for line in lines)
    tokens_out = sum(int(line.get("outputTokens") or 0) for line in lines)
    cost = 0.0
    for line in lines:
        value = (
            line.get("costUsd") if line.get("costUsd") is not None else line.get("estimatedCostUsd")
        )
        cost += float(value or 0)
    return {"calls": calls, "input_tokens": tokens_in, "output_tokens": tokens_out, "usd": cost}


# --- the run --------------------------------------------------------------------------------


def done_keys(results: Path) -> set[str]:
    if not results.is_file():
        return set()
    keys: set[str] = set()
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not row.get("quota"):
                keys.add(f"{row['id']}:{row['variant']}")
    return keys


@dataclass
class Runner:
    models: GraphModels
    departments: dict[str, str]
    manifest: dict[str, dict[str, str]]
    out: Path
    web: str
    concurrency: int = 2
    timeout_s: float = 180.0
    quota_wait_s: float = 0.0
    max_quota_waits: int = 48

    async def run(self, jobs: list[Job]) -> None:
        results = self.out / "results.jsonl"
        pending = [job for job in jobs if job.key not in done_keys(results)]
        print(f"{len(jobs)} turns selected, {len(jobs) - len(pending)} already done")
        waits = 0
        while pending:
            quota_hit = await self._pass(pending, results)
            pending = [job for job in pending if job.key not in done_keys(results)]
            if not quota_hit or not pending:
                break
            if self.quota_wait_s <= 0 or waits >= self.max_quota_waits:
                print(
                    f"quota exhausted: {len(pending)} turn(s) left - rerun the same command "
                    "(same --out) once the quota is back",
                    file=sys.stderr,
                )
                return
            waits += 1
            resume = time.strftime("%H:%M", time.localtime(time.time() + self.quota_wait_s))
            print(f"quota exhausted: waiting until ~{resume} ({waits}/{self.max_quota_waits})")
            await asyncio.sleep(self.quota_wait_s)

    async def _pass(self, jobs: list[Job], results: Path) -> bool:
        semaphore = asyncio.Semaphore(self.concurrency)
        stop = asyncio.Event()
        lock = asyncio.Lock()
        total = len(jobs)
        finished = 0

        async def one(job: Job) -> None:
            nonlocal finished
            async with semaphore:
                if stop.is_set():
                    return
                access = access_of(job.question, self.departments)
                result = await run_question(
                    job.text,
                    security_of(job.question, access),
                    self.models,
                    timeout_s=self.timeout_s,
                )
                row = result_row(
                    job,
                    result,
                    access=access,
                    expected_document_ids=document_ids(job.question, self.manifest),
                    web=self.web,
                )
                if result.error and is_quota(result.error_code, result.error):
                    row["quota"] = True
                    stop.set()
                async with lock:
                    with results.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    finished += 1
                    checks = row["checks"]
                    status = (
                        "QUOTA"
                        if row.get("quota")
                        else "ERROR"
                        if result.error
                        else {True: "pass", False: "FAIL", None: "pending"}[checks["pass"]]
                    )
                    print(
                        f"[{finished}/{total}] {job.key:<14} {status:<7} "
                        f"{','.join(checks['outcome']):<14} {result.total_ms / 1000:5.1f}s"
                    )

        await asyncio.gather(*(one(job) for job in jobs))
        return stop.is_set()


def write_summary(out: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and not json.loads(line).get("quota")
    ]
    summary = summarize(rows)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"\n{'variant/category':<28}{'rows':>6}{'pass':>6}{'fail':>6}{'pend':>6}{'err':>5}"
        f"{'recall@5':>11}{'leak':>8}"
    )
    for name, counts in summary["table"].items():
        recall = (
            f"{counts.get('recall_hit', 0)}/{counts['recall_n']}" if counts.get("recall_n") else "-"
        )
        leak = f"{counts.get('leak', 0)}/{counts['leak_n']}" if counts.get("leak_n") else "-"
        print(
            f"{name:<28}{counts['rows']:>6}{counts.get('pass', 0):>6}{counts.get('fail', 0):>6}"
            f"{counts.get('pending', 0):>6}{counts.get('error', 0):>5}{recall:>11}{leak:>8}"
        )
    print(f"latency {summary['latency_ms']}  ttft {summary['ttft_ms']}")
    if summary["leaks"]:
        print(f"LEAKAGE in {len(summary['leaks'])} row(s): {summary['leaks'][:5]}")
    return summary


def read_manifest(dataset: Path) -> dict[str, dict[str, str]]:
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return {row["file_id"]: row for row in csv.DictReader(handle)}


def prepare(args: argparse.Namespace) -> tuple[list[Job], Path, dict[str, str]]:
    """Everything that touches files before the event loop starts: the turns to run, the run
    folder (with its config.json) and the department map."""

    dataset: Path = args.dataset
    departments = load_departments(dataset)
    if not departments and not args.dry_run:
        sys.exit(
            f"no {DEPARTMENTS_FILE} in {dataset}: run `task eval:ingest env=<env> -- --limit 1` "
            "(or eval:link) once against this environment so the department UUIDs are known"
        )
    jobs = select_jobs(
        load_questions(dataset),
        subset=args.set,
        variant=args.variant,
        categories=set(args.category) if args.category else None,
        ids=set(args.ids.split(",")) if args.ids else None,
    )
    if args.limit:
        jobs = spread(jobs, args.limit)
    out: Path = args.out or Path("runs") / (
        f"{datetime.now():%Y%m%d-%H%M}-{args.set}-{args.variant}-web{args.web}"
    )
    if args.dry_run:
        return jobs, out, departments
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                **{k: str(v) for k, v in vars(args).items()},
                "app_env": settings.APP_ENV,
                "qdrant_collection": settings.QDRANT_COLLECTION,
                "started": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return jobs, out, departments


async def execute(
    args: argparse.Namespace,
    jobs: list[Job],
    out: Path,
    departments: dict[str, str],
    manifest: dict[str, dict[str, str]],
) -> None:
    """Start the app's registry/pricing lifecycle, then run every turn."""

    from app.api.deps import get_graph_models
    from app.core.lifespan import lifespan

    settings.CHAT_WEB_SEARCH_ENABLED = args.web == "on"
    async with lifespan(None):  # type: ignore[arg-type]
        models = await get_graph_models()
        models.retrieval = RecordingRetrieval(models.retrieval)
        await Runner(
            models,
            departments,
            manifest,
            out,
            web=args.web,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
            quota_wait_s=args.quota_wait * 60,
            max_quota_waits=args.max_quota_waits,
        ).run(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--set", choices=("demo", "official"), default="demo")
    parser.add_argument("--variant", choices=("exact", "user", "both"), default="both")
    parser.add_argument("--web", choices=("off", "on"), default="off")
    parser.add_argument("--category", action="append", help="only this category (repeatable)")
    parser.add_argument("--ids", help="comma-separated question ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds per turn")
    parser.add_argument("--out", type=Path, help="run folder; reuse one to continue it")
    parser.add_argument(
        "--quota-wait",
        type=float,
        default=float(os.getenv("EVAL_QUOTA_WAIT_MINUTES", "0")),
        help="minutes to wait when the model quota runs out (0 = stop, rerun later)",
    )
    parser.add_argument("--max-quota-waits", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true", help="list the turns, run nothing")
    args = parser.parse_args()
    if os.getenv("APP_ENV") != "eval":
        sys.exit("APP_ENV is not 'eval': run through `task eval:run env=eval`")
    jobs, out, departments = prepare(args)
    print(f"results -> {out}")
    if args.dry_run:
        for job in jobs:
            print(f"  {job.key:<14} {job.question['category']:<13} {job.text[:90]}")
        return
    asyncio.run(execute(args, jobs, out, departments, read_manifest(args.dataset)))
    write_summary(out)


if __name__ == "__main__":
    main()
