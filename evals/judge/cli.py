"""Score recorded turns with RAGAS (Task 11, `spec.md` §4-§5, §12.6).

    uv run --isolated --no-project --with-requirements evals/judge/requirements.txt \\
        python -m evals.judge --out <run folder> [--limit 20] [--metrics refusal,...] \\
        [--judge-provider gemini|openai_compat]

Reads `<out>/results.jsonl` (written by `evals.run`) and turns each row into RAGAS inputs
(`tasks.py`). The judge model comes from a provider adapter (`providers/`), picked by
`JUDGE_PROVIDER` in the environment or `.env.eval` (`--judge-provider` overrides; Gemini by
default); answer_relevancy embeds through `JUDGE_EMBEDDING_PROVIDER` (default: the judge
provider when it can embed, else Gemini).

Each score is appended to `<out>/judge.jsonl` the moment it is known, with the model and the
(masked) key that produced it; a rerun only scores what is missing or `NaN`. Before any call
the cost is estimated at list price and the run stops if it exceeds `--max-cost`. When every
key is out for the day the run sleeps until the reset (`--quota-wait` > 0) or stops with its
checkpoint. If more than 5% of the scored rows needed the fallback model, those rows are scored
again with the main model only. At the end the scores are merged into `results.jsonl`
(`row["ragas"]`, old file kept as `.bak`) and summarised in `<out>/judge_summary.json`.

RAGAS needs an older OpenAI client than the service, so it runs in its own environment
(`evals/judge/requirements.txt`); RAGAS is imported only by `evals.judge.ragas_metrics` (and
Gemini's SDK only by its adapter), loaded lazily - selection, budget, checkpoint and the
OpenAI-compatible adapter are tested without them.
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.judge import providers
from evals.judge.checkpoint import (
    load_records,
    load_rows,
    merge_into_results,
    pending_tasks,
    summarize_scores,
)
from evals.judge.cost import BudgetExceeded, check_budget, estimate
from evals.judge.runner import JudgeRun
from evals.judge.tasks import METRICS, NO_LLM, tasks_of


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="evals.judge", description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="run folder with results.jsonl")
    parser.add_argument("--metrics", default=",".join(METRICS), help="comma-separated subset")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    parser.add_argument(
        "--judge-provider",
        choices=sorted(providers.ADAPTERS),
        help=f"overrides {providers.PROVIDER_ENV} (default {providers.DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=sorted(providers.ADAPTERS),
        help=f"overrides {providers.EMBEDDING_PROVIDER_ENV}",
    )
    parser.add_argument("--judge-model", help="default: the provider's (see providers/)")
    parser.add_argument("--judge-fallback", help="'' to disable; default: the provider's")
    parser.add_argument("--embedding-model", help="default: the embedding provider's")
    parser.add_argument(
        "--thinking", default="minimal", help="Gemini: lowest thinking level the model allows"
    )
    parser.add_argument("--rpm", type=int, default=12, help="requests per minute per key")
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds per call")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=4.0, help="USD at list price")
    parser.add_argument(
        "--quota-wait",
        type=float,
        default=float(os.getenv("EVAL_QUOTA_WAIT_MINUTES", "0")),
        help="minutes to sleep when every key is out for the day (0 = stop, rerun later)",
    )
    parser.add_argument("--max-quota-waits", type=int, default=48)
    parser.add_argument(
        "--rpd-recheck",
        type=float,
        default=0.0,
        help="minutes before a key hit by a per-day 429 is tried again (0 = next daily reset)",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env.eval"))
    parser.add_argument("--prompts-dir", type=Path, default=Path(__file__).parent / "prompts")
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


def _pick(flag: str | None, default: str | None) -> str | None:
    """A CLI value wins, '' meaning "none"; otherwise the provider's default."""

    if flag is not None:
        return flag or None
    return default


def resolve(args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    """Which providers and models this run uses (no adapter is built, no SDK imported
    beyond the adapter classes)."""

    name = providers.judge_provider(env, args.judge_provider)
    defaults = providers.adapter_class(name).default_models(env)
    model = _pick(args.judge_model, defaults.judge)
    if not model:
        raise providers.MissingSetting(name, ["--judge-model or the provider's model setting"])
    plan: dict[str, Any] = {
        "provider": name,
        "model": model,
        "fallback": _pick(args.judge_fallback, defaults.fallback),
        "embedding_provider": None,
        "embedding_model": None,
    }
    if "answer_relevancy" in args.metrics:
        embedder = providers.embedding_provider(env, name, defaults, args.embedding_provider)
        embed_defaults = (
            defaults if embedder == name else providers.adapter_class(embedder).default_models(env)
        )
        plan["embedding_provider"] = embedder
        plan["embedding_model"] = _pick(args.embedding_model, embed_defaults.embedding)
        if not plan["embedding_model"]:
            raise providers.MissingSetting(embedder, ["--embedding-model"])
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    env = providers.read_env(args.env_file)
    rows = load_rows(args.out)
    if args.limit:
        rows = rows[: args.limit]
    tasks = pending_tasks(tasks_of(rows, args.metrics), load_records(args.out / "judge.jsonl"))
    needs_llm = any(task.metric not in NO_LLM for task in tasks)
    try:
        plan = resolve(args, env)
    except (ValueError, providers.MissingSetting) as error:
        print(error, file=sys.stderr)
        return 2
    estimated = estimate(tasks, plan["model"])
    print(
        f"judge {plan['provider']}:{plan['model']} (fallback {plan['fallback'] or '-'})"
        + (
            f", embeddings {plan['embedding_provider']}:{plan['embedding_model']}"
            if plan["embedding_provider"]
            else ""
        )
    )
    print(
        f"{estimated['tasks']} score(s) to compute: ~{estimated['calls']} calls, "
        f"~{estimated['input_tokens']:,} tokens in / {estimated['output_tokens']:,} out, "
        f"~{estimated['usd']:.2f} USD at list price"
        + ("" if estimated["price_known"] else " (price unknown: highest known price used)")
    )
    try:
        check_budget(estimated, args.max_cost)
    except BudgetExceeded as error:
        print(error, file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    options = providers.Options(
        rpm=args.rpm,
        rpd_recheck_s=args.rpd_recheck * 60,
        timeout_s=args.timeout,
        thinking=args.thinking,
    )
    try:
        adapter = embedder = None
        if needs_llm:
            adapter = providers.adapter_class(plan["provider"]).from_env(env, options)
            if plan["embedding_provider"] == plan["provider"]:
                embedder = adapter
            elif plan["embedding_provider"]:
                embedder = providers.adapter_class(plan["embedding_provider"]).from_env(
                    env, options
                )
    except (ValueError, providers.MissingSetting) as error:
        print(error, file=sys.stderr)
        return 2

    from evals.judge.ragas_metrics import RagasScorerFactory  # needs the judge environment

    factory = RagasScorerFactory(
        adapter=adapter,  # type: ignore[arg-type]  # None only when no task calls a model
        model=plan["model"],
        fallback=plan["fallback"],
        prompts_dir=args.prompts_dir,
        adapt_prompts=args.adapt_prompts,
        embedder=embedder,
        embedding_model=plan["embedding_model"],
    )
    run = JudgeRun(
        out=args.out,
        scorer_factory=factory,
        fallback_model=plan["fallback"],
        concurrency=args.concurrency,
        quota_wait_s=args.quota_wait * 60,
        max_quota_waits=args.max_quota_waits,
    )

    async def go() -> bool:
        try:
            return await run.run(tasks)
        finally:
            await factory.aclose()

    finished = asyncio.run(go())
    records = load_records(run.checkpoint)
    merged = merge_into_results(args.out, records)
    summary = summarize_scores(load_rows(args.out), records, plan["fallback"])
    summary["judge"] = plan
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
    denied = sorted(
        {
            key
            for keys in summary["keys"].values()
            for key, counts in keys.items()
            if counts.get("denied")
        }
    )
    if denied:
        print(f"key(s) denied (401/403), remove them from the provider's keys: {', '.join(denied)}")
    return 0 if finished else 3


if __name__ == "__main__":
    sys.exit(main())
