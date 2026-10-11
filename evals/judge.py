"""Score recorded turns with RAGAS (Task 11, `spec.md` §4-§5, §12.6).

    uv run --isolated --no-project --with-requirements evals/requirements-judge.txt \\
        python -m evals.judge --out <run folder> [--limit 20] [--metrics refusal,...]

Reads `<out>/results.jsonl` (written by `evals.run`) and turns each row into RAGAS inputs
(`spec.md` §4: `user_input` = question, `response` = answer, `retrieved_contexts` = chunk text
after rerank, `retrieved_context_ids` = their `document_id`, `reference` = `expected_answer`,
`reference_context_ids` = `expected_document_ids`):

- rows with an expected answer (`normal`, `calculation`, `access` the persona may see):
  faithfulness, answer_coverage, context_recall, answer_relevancy, id_context_precision,
  id_context_recall;
- every answered row: refusal ("did the answer refuse or hand over to a ticket?", #4);
- rows answered from the web: web_faithfulness against the fetched pages (#14), plus
  answer_coverage when the question has an expected answer.

`answer_coverage` runs RAGAS FactualCorrectness (claims of both texts, checked both ways) and
keeps both directions instead of RAGAS' single F1: coverage = share of the expected answer's
claims the answer holds (the pass/fail score of criterion #3), precision = share of the
answer's claims the expected answer backs (informative only: expected answers are terse, so
correct extra detail counts against it), and the F1 RAGAS would report. Same four calls.

Each score is appended to `<out>/judge.jsonl` the moment it is known, with the model and the
(masked) key that produced it; a rerun only scores what is missing or `NaN`. Before any call
the cost is estimated at list price and the run stops if it exceeds `--max-cost`. When every
key is out for the day the run sleeps until the reset (`--quota-wait` > 0) or stops with its
checkpoint. If more than 5% of the scored rows needed the fallback model, those rows are scored
again with the main model only. At the end the scores are merged into `results.jsonl`
(`row["ragas"]`, old file kept as `.bak`) and summarised in `<out>/judge_summary.json`.

RAGAS needs an older OpenAI client than the service, so it runs in its own environment
(`evals/requirements-judge.txt`); this module imports RAGAS only in `evals.judge_ragas`, which
`main()` loads lazily - the selection, budget and checkpoint logic here are tested without it.
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from evals.gemini_pool import QuotaExhausted
from evals.metrics import outcome_of

ANSWER_METRICS = (
    "faithfulness",
    "answer_coverage",
    "context_recall",
    "answer_relevancy",
    "id_context_precision",
    "id_context_recall",
)
METRICS = (*ANSWER_METRICS, "refusal", "web_faithfulness")
# Extra values one metric's calls also produce (recorded, summarised, never selected alone).
EXTRA_METRICS = ("answer_precision", "factual_correctness_f1")
NO_LLM = frozenset({"id_context_precision", "id_context_recall"})
# LLM calls per metric (faithfulness: statements + NLI; answer_coverage:
# claims of both texts + NLI both ways; answer_relevancy with strictness 1 also embeds).
CALLS = {
    "faithfulness": 2,
    "answer_coverage": 4,
    "context_recall": 1,
    "answer_relevancy": 1,
    "refusal": 1,
    "web_faithfulness": 2,
    "id_context_precision": 0,
    "id_context_recall": 0,
}
# USD per 1M tokens (input, output) - LiteLLM prices of 10-10-2026, thinking billed as output.
PRICES = {
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_FALLBACK = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING = "gemini-embedding-001"
PASS_SCORE = 0.7
FALLBACK_SHARE_MAX = 0.05
# Context handed to the judge: at most 8 chunks of ~500 tokens (`spec.md` §5, 504 risk).
MAX_CONTEXTS = 8
MAX_CONTEXT_CHARS = 2000
# Estimate: a RAGAS prompt adds ~600 tokens of instructions and examples; ~3 characters of
# Vietnamese per token; ~250 tokens of JSON out; 20% for retries.
PROMPT_OVERHEAD_TOKENS = 600
CHARS_PER_TOKEN = 3.0
OUTPUT_TOKENS_PER_CALL = 250
RETRY_MARGIN = 1.2


@dataclass(frozen=True)
class Task:
    key: str  # "<id>:<variant>", as in evals.run
    metric: str
    inputs: dict[str, Any]


@dataclass
class Score:
    value: float
    reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)


class Scorer(Protocol):
    async def score(self, task: Task) -> Score: ...

    def stats(self) -> dict[str, Any]: ...


# --- which metrics a row gets ---------------------------------------------------------------


def row_key(row: dict[str, Any]) -> str:
    return f"{row['id']}:{row['variant']}"


def has_reference(row: dict[str, Any]) -> bool:
    category = row.get("category")
    answerable = category in ("normal", "calculation") or (
        category == "access" and row.get("expect_visible") is True
    )
    return answerable and bool((row.get("expected_answer") or "").strip())


def _contexts(items: Iterable[dict[str, Any]]) -> list[str]:
    texts = [str(item.get("content") or "").strip() for item in items]
    return [text[:MAX_CONTEXT_CHARS] for text in texts if text][:MAX_CONTEXTS]


def _unique(values: Iterable[Any]) -> list[str]:
    return [str(value) for value in dict.fromkeys(values) if value]


def tasks_for(row: dict[str, Any], selected: Sequence[str] = METRICS) -> list[Task]:
    """The RAGAS inputs of one `results.jsonl` row (`spec.md` §4)."""

    response = (row.get("response") or "").strip()
    if row.get("error") or row.get("quota") or not response:
        return []
    key = row_key(row)
    question = row.get("asked") or ""
    reference = (row.get("expected_answer") or "").strip()
    wanted = set(selected)
    tasks: list[Task] = []

    def add(metric: str, **inputs: Any) -> None:
        if metric in wanted:
            tasks.append(Task(key, metric, inputs))

    from_web = "web" in outcome_of(row)
    if from_web:
        add(
            "web_faithfulness",
            user_input=question,
            response=response,
            retrieved_contexts=_contexts(row.get("web_results") or []),
        )
        if reference:
            add("answer_coverage", response=response, reference=reference)
    elif has_reference(row):
        contexts = _contexts(row.get("context") or [])
        add("faithfulness", user_input=question, response=response, retrieved_contexts=contexts)
        add("answer_coverage", response=response, reference=reference)
        add(
            "context_recall",
            user_input=question,
            retrieved_contexts=contexts,
            reference=reference,
        )
        add("answer_relevancy", user_input=question, response=response)
        ids = {
            "retrieved_context_ids": _unique(c.get("document_id") for c in row.get("context") or []),
            "reference_context_ids": _unique(row.get("expected_document_ids") or []),
        }
        add("id_context_precision", **ids)
        add("id_context_recall", **ids)
    add("refusal", user_input=question, response=response)
    return tasks


def tasks_of(rows: Iterable[dict[str, Any]], selected: Sequence[str]) -> list[Task]:
    return [task for row in rows for task in tasks_for(row, selected)]


# --- cost estimate and budget guard ---------------------------------------------------------


def _tokens(*texts: Any) -> int:
    chars = 0
    for text in texts:
        if isinstance(text, list):
            chars += sum(len(str(item)) for item in text)
        elif text:
            chars += len(str(text))
    return int(chars / CHARS_PER_TOKEN)


def estimate_task(task: Task, prices: tuple[float, float]) -> tuple[int, int, int, float]:
    """(calls, input tokens, output tokens, USD) one task is expected to cost."""

    calls = CALLS[task.metric]
    if not calls:
        return 0, 0, 0, 0.0
    inputs = task.inputs
    if task.metric in ("faithfulness", "web_faithfulness"):
        # statements from the answer, then each statement against the whole context
        text_in = _tokens(inputs["user_input"], inputs["response"]) * 2 + _tokens(
            inputs["retrieved_contexts"]
        )
    elif task.metric == "answer_coverage":
        text_in = (_tokens(inputs["response"]) + _tokens(inputs["reference"])) * 2
    elif task.metric == "context_recall":
        text_in = _tokens(inputs["user_input"], inputs["retrieved_contexts"], inputs["reference"])
    else:
        text_in = _tokens(inputs.get("user_input"), inputs.get("response"))
    tokens_in = int((text_in + calls * PROMPT_OVERHEAD_TOKENS) * RETRY_MARGIN)
    tokens_out = int(calls * OUTPUT_TOKENS_PER_CALL * RETRY_MARGIN)
    usd = (tokens_in * prices[0] + tokens_out * prices[1]) / 1_000_000
    return calls, tokens_in, tokens_out, usd


def estimate(tasks: Sequence[Task], model: str) -> dict[str, Any]:
    prices = PRICES.get(model, PRICES[DEFAULT_MODEL])
    calls = tokens_in = tokens_out = 0
    usd = 0.0
    for task in tasks:
        c, i, o, u = estimate_task(task, prices)
        calls, tokens_in, tokens_out, usd = calls + c, tokens_in + i, tokens_out + o, usd + u
    return {
        "tasks": len(tasks),
        "calls": calls,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "usd": round(usd, 4),
        "model": model,
    }


class BudgetExceeded(Exception):
    pass


def check_budget(estimated: dict[str, Any], max_cost: float) -> None:
    if estimated["usd"] > max_cost:
        raise BudgetExceeded(
            f"estimated {estimated['usd']:.2f} USD for {estimated['calls']} calls exceeds "
            f"--max-cost {max_cost:.2f} USD - nothing was called"
        )


# --- checkpoint -----------------------------------------------------------------------------


def is_scored(record: dict[str, Any]) -> bool:
    value = record.get("value")
    return isinstance(value, (int, float)) and not math.isnan(value)


def load_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """The latest record of every (row, metric) in `judge.jsonl`."""

    records: dict[tuple[str, str], dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[(record["key"], record["metric"])] = record
    return records


def pending_tasks(
    tasks: Sequence[Task], records: dict[tuple[str, str], dict[str, Any]]
) -> list[Task]:
    return [task for task in tasks if not is_scored(records.get((task.key, task.metric), {}))]


def _record(task: Task, score: Score) -> dict[str, Any]:
    value = score.value if score.value is not None else math.nan
    return {
        "key": task.key,
        "metric": task.metric,
        "value": None if math.isnan(value) else round(float(value), 4),
        "reason": score.reason,
        "extra": {
            name: None if extra is None or math.isnan(extra) else round(float(extra), 4)
            for name, extra in score.extra.items()
        },
        **score.usage,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# --- the run --------------------------------------------------------------------------------


@dataclass
class JudgeRun:
    out: Path
    scorer_factory: Callable[[bool], Scorer]
    fallback_model: str | None = None
    concurrency: int = 8
    quota_wait_s: float = 0.0
    max_quota_waits: int = 48
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
    clock: Callable[[], float] = time.time

    @property
    def checkpoint(self) -> Path:
        return self.out / "judge.jsonl"

    async def run(self, tasks: Sequence[Task]) -> bool:
        """Score every task; False when stopped by quota with tasks left (checkpoint kept)."""

        scorer = self.scorer_factory(False)
        if not await self._until_done(scorer, tasks):
            return False
        records = load_records(self.checkpoint)
        redo = fallback_rows(records, self.fallback_model)
        if redo:
            print(f"{len(redo)} row(s) needed the fallback model (> 5%): scoring them again")
            again = [task for task in tasks if task.key in redo and task.metric not in NO_LLM]
            primary = self.scorer_factory(True)
            return await self._until_done(primary, again)
        return True

    async def _until_done(self, scorer: Scorer, tasks: Sequence[Task]) -> bool:
        """Every task tried once (a failed metric is recorded as NaN and left for the next
        run); only the tasks a quota stop never reached are tried again after the wait."""

        waits = 0
        todo = list(tasks)
        while todo:
            reset_at, tried = await self._pass(scorer, todo)
            todo = [task for task in todo if (task.key, task.metric) not in tried]
            if reset_at is None or not todo:
                return True
            if self.quota_wait_s <= 0 or waits >= self.max_quota_waits:
                print(
                    f"quota exhausted: {len(todo)} score(s) left - rerun the same command "
                    "once the quota is back"
                )
                return False
            waits += 1
            wait = min(max(60.0, reset_at - self.clock() + 60.0), self.quota_wait_s)
            resume = time.strftime("%H:%M", time.localtime(self.clock() + wait))
            print(f"every key is out for the day: waiting until ~{resume} ({waits})")
            await self.sleep(wait)
        return True

    async def _pass(
        self, scorer: Scorer, tasks: Sequence[Task]
    ) -> tuple[float | None, set[tuple[str, str]]]:
        """One pass over `tasks`: the reset time if the quota ran out (else None) and the
        tasks that got a record."""

        gate = asyncio.Semaphore(self.concurrency)
        lock = asyncio.Lock()
        stopped: list[float] = []
        tried: set[tuple[str, str]] = set()
        done = 0

        async def one(task: Task) -> None:
            nonlocal done
            async with gate:
                if stopped:
                    return
                try:
                    score = await scorer.score(task)
                except QuotaExhausted as error:
                    stopped.append(error.reset_at)
                    return
                record = _record(task, score)
                async with lock:
                    with self.checkpoint.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    tried.add((task.key, task.metric))
                    done += 1
                    shown = "NaN" if record["value"] is None else record["value"]
                    print(f"[{done}/{len(tasks)}] {task.key} {task.metric} = {shown}")

        await asyncio.gather(*(one(task) for task in tasks))
        return (min(stopped) if stopped else None), tried


def used_fallback(record: dict[str, Any], fallback_model: str | None) -> bool:
    """Read from the models a score used, so a record written before the embedding model
    stopped counting as a fallback is read right too."""

    return bool(fallback_model and fallback_model in (record.get("models") or {}))


def fallback_rows(
    records: dict[tuple[str, str], dict[str, Any]], fallback_model: str | None
) -> set[str]:
    """Rows scored partly by the fallback model, when they are more than 5% of the rows."""

    rows = {key for key, _metric in records}
    used = {key for (key, _m), record in records.items() if used_fallback(record, fallback_model)}
    if not rows or len(used) / len(rows) <= FALLBACK_SHARE_MAX:
        return set()
    return used


# --- results --------------------------------------------------------------------------------


def merge_into_results(out: Path, records: dict[tuple[str, str], dict[str, Any]]) -> int:
    """Write the scores into `results.jsonl` as `row["ragas"]` (old file kept as .bak)."""

    results = out / "results.jsonl"
    lines = [line for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_row: dict[str, dict[str, Any]] = defaultdict(dict)
    for (key, metric), record in records.items():
        by_row[key][metric] = {
            "value": record.get("value"),
            "reason": record.get("reason"),
            "model": record.get("model"),
        }
        for name, extra in (record.get("extra") or {}).items():
            by_row[key][name] = {"value": extra, "model": record.get("model")}
    rows = []
    for line in lines:
        row = json.loads(line)
        if row_key(row) in by_row:
            row["ragas"] = by_row[row_key(row)]
        rows.append(row)
    results.with_suffix(".jsonl.bak").write_text("\n".join(lines) + "\n", encoding="utf-8")
    results.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    return sum(1 for row in rows if "ragas" in row)


def summarize_scores(
    rows: Sequence[dict[str, Any]],
    records: dict[tuple[str, str], dict[str, Any]],
    fallback_model: str | None = None,
) -> dict[str, Any]:
    """Mean and share ≥ 0.7 of every metric, overall and per category; cost and fallback use."""

    category = {row_key(row): row.get("category") for row in rows}
    values: dict[str, list[float]] = defaultdict(list)
    per_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    nan: Counter[str] = Counter()
    usage = Counter[str]()
    models: Counter[str] = Counter()
    for (key, metric), record in records.items():
        scored = [(metric, record.get("value")), *(record.get("extra") or {}).items()]
        for name, value in scored:
            if isinstance(value, (int, float)) and not math.isnan(value):
                values[name].append(float(value))
                per_category[name][category.get(key) or "?"].append(float(value))
            else:
                nan[name] += 1
        for name in ("calls", "input_tokens", "output_tokens"):
            usage[name] += int(record.get(name) or 0)
        usage["usd_micros"] += int(round(float(record.get("usd") or 0) * 1_000_000))
        if record.get("model"):
            models[record["model"]] += 1

    def stats(items: list[float]) -> dict[str, Any]:
        if not items:
            return {"n": 0}
        return {
            "n": len(items),
            "mean": round(sum(items) / len(items), 4),
            "pass_rate": round(sum(1 for v in items if v >= PASS_SCORE) / len(items), 4),
        }

    return {
        "metrics": {
            metric: {
                **stats(values[metric]),
                "nan": nan[metric],
                "by_category": {cat: stats(v) for cat, v in sorted(per_category[metric].items())},
            }
            for metric in (*METRICS, *EXTRA_METRICS)
            if values[metric] or nan[metric]
        },
        "usage": {
            "calls": usage["calls"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "usd": round(usage["usd_micros"] / 1_000_000, 4),
        },
        "models": dict(models),
        "fallback_rows": len(
            {key for (key, _m), r in records.items() if used_fallback(r, fallback_model)}
        ),
    }


# --- CLI ------------------------------------------------------------------------------------


def read_keys(env_file: Path | None) -> list[str]:
    """`GEMINI_API_KEYS` from the environment, else from `env_file` (comma/space separated)."""

    raw = os.getenv("GEMINI_API_KEYS", "")
    if not raw and env_file and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GEMINI_API_KEYS":
                raw = value.strip().strip('"').strip("'")
    return [key for key in re.split(r"[\s,;]+", raw) if key]


def load_rows(out: Path) -> list[dict[str, Any]]:
    path = out / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if not row.get("quota")]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evals.judge", description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="run folder with results.jsonl")
    parser.add_argument("--metrics", default=",".join(METRICS), help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-fallback", default=DEFAULT_FALLBACK, help="'' to disable")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--thinking", default="minimal", help="lowest thinking level the model allows")
    parser.add_argument("--rpm", type=int, default=12, help="requests per minute per key")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=4.0, help="USD at list price")
    parser.add_argument(
        "--quota-wait",
        type=float,
        default=float(os.getenv("EVAL_QUOTA_WAIT_MINUTES", "0")),
        help="minutes to sleep when every key is out for the day (0 = stop, rerun later)",
    )
    parser.add_argument("--max-quota-waits", type=int, default=48)
    parser.add_argument("--env-file", type=Path, default=Path(".env.eval"))
    parser.add_argument("--prompts-dir", type=Path, default=Path(__file__).parent / "ragas_prompts")
    parser.add_argument(
        "--adapt-prompts",
        action="store_true",
        help="translate the RAGAS prompt examples to Vietnamese once and cache them",
    )
    parser.add_argument("--dry-run", action="store_true", help="estimate the cost and stop")
    args = parser.parse_args(argv)
    args.metrics = [name.strip() for name in args.metrics.split(",") if name.strip()]
    unknown = sorted(set(args.metrics) - set(METRICS))
    if unknown:
        parser.error(f"unknown metric(s): {', '.join(unknown)} (known: {', '.join(METRICS)})")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.out)
    if args.limit:
        rows = rows[: args.limit]
    tasks = pending_tasks(tasks_of(rows, args.metrics), load_records(args.out / "judge.jsonl"))
    estimated = estimate(tasks, args.judge_model)
    print(
        f"{estimated['tasks']} score(s) to compute: ~{estimated['calls']} calls, "
        f"~{estimated['input_tokens']:,} tokens in / {estimated['output_tokens']:,} out, "
        f"~{estimated['usd']:.2f} USD at list price ({args.judge_model})"
    )
    try:
        check_budget(estimated, args.max_cost)
    except BudgetExceeded as error:
        print(error, file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    keys = read_keys(args.env_file)
    if not keys and any(task.metric not in NO_LLM for task in tasks):
        print("GEMINI_API_KEYS is not set (environment or --env-file)", file=sys.stderr)
        return 2

    from evals.judge_ragas import RagasScorerFactory  # needs the judge environment

    factory = RagasScorerFactory(
        keys=keys,
        model=args.judge_model,
        fallback=args.judge_fallback or None,
        embedding_model=args.embedding_model,
        thinking=args.thinking,
        rpm=args.rpm,
        prompts_dir=args.prompts_dir,
        adapt_prompts=args.adapt_prompts,
        metrics=args.metrics,
    )
    run = JudgeRun(
        out=args.out,
        scorer_factory=factory,
        fallback_model=args.judge_fallback or None,
        concurrency=args.concurrency,
        quota_wait_s=args.quota_wait * 60,
        max_quota_waits=args.max_quota_waits,
    )
    finished = asyncio.run(run.run(tasks))
    records = load_records(run.checkpoint)
    merged = merge_into_results(args.out, records)
    summary = summarize_scores(load_rows(args.out), records, args.judge_fallback or None)
    summary["keys"] = factory.stats()
    (args.out / "judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nscores merged into {merged} row(s) of results.jsonl")
    print(f"{'metric':<24}{'n':>6}{'mean':>8}{'≥0.7':>8}{'NaN':>6}")
    for metric, stats in summary["metrics"].items():
        print(
            f"{metric:<24}{stats.get('n', 0):>6}{stats.get('mean', '-'):>8}"
            f"{stats.get('pass_rate', '-'):>8}{stats['nan']:>6}"
        )
    print(f"usage {summary['usage']}  models {summary['models']}")
    return 0 if finished else 3


if __name__ == "__main__":
    sys.exit(main())
