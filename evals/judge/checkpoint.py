"""`judge.jsonl` (one record per score, appended as it is known) and what is built from it:
the merge into `results.jsonl` and `judge_summary.json`."""

import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.judge.tasks import (
    EXTRA_METRICS,
    FALLBACK_SHARE_MAX,
    METRICS,
    PASS_SCORE,
    Score,
    Task,
    row_key,
)


def load_rows(out: Path) -> list[dict[str, Any]]:
    path = out / "results.jsonl"
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return [row for row in rows if not row.get("quota")]


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


def record_of(task: Task, score: Score) -> dict[str, Any]:
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
        usage["usd_micros"] += round(float(record.get("usd") or 0) * 1_000_000)
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
