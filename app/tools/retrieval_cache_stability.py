"""How stable is the standalone question the classifier writes? (SPEC-retrieval-cache.md,
decision 5: the ceiling of the L2 hit rate.)

Runs the classifier `--runs` times on every question of the eval set and reports how
often the same question yields the same L2 key (the normalized standalone questions).
Every run is a real CHAT model call: `--limit` keeps the bill small.

    task db:retrieval-cache-stability env=eval -- --limit 50 --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from app.core.llm.provider_models import build_model
from app.core.registry.model_registry import init_model_registry, require_top_priority_credential
from app.graph.nodes.message_classification import build_classification_agent, classify_intent
from app.graph.nodes.query_transformation import extract_standalone_question
from app.integrations.retrieval_cache import normalize_question
from app.schemas.intent import ClassifiedTask

DEFAULT_DATASET = Path("../unisage-gateway/dataset/official")


def _questions(dataset: Path, limit: int | None) -> list[str]:
    path = dataset / "questions.jsonl"
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("question_user") or row.get("question") or row.get("question_exact")
        if text:
            texts.append(str(text))
    return texts[:limit] if limit else texts


def l2_key_of(tasks: list[ClassifiedTask]) -> tuple[str, ...] | None:
    """The normalized standalone questions an advisory classification leads to; None when
    the turn would not reach retrieval (no advisory task)."""

    questions: list[str] = []
    for task in tasks:
        if task.intent != "academic_advisory":
            continue
        if task.sub_queries:
            questions.extend(task.sub_queries)
        elif task.hyde_text:
            questions.append(extract_standalone_question(task.hyde_text))
        else:
            questions.append(task.query)
    if not questions:
        return None
    return tuple(normalize_question(question) for question in questions)


async def _run(args: argparse.Namespace) -> int:
    await init_model_registry()
    credential = require_top_priority_credential("CHAT")
    agent = build_classification_agent(build_model(credential))
    questions = _questions(args.dataset, args.limit)
    print(f"{len(questions)} question(s) x {args.runs} run(s) on {credential.model_name}")

    stable = 0
    partial = 0
    unstable: list[tuple[str, list[tuple[str, ...] | None]]] = []
    skipped = 0
    for index, question in enumerate(questions, 1):
        keys: list[tuple[str, ...] | None] = []
        for _ in range(args.runs):
            classification = await classify_intent(agent, question, role="KHACH")
            keys.append(l2_key_of(classification.tasks))
        if all(key is None for key in keys):
            skipped += 1
            continue
        counts = Counter(keys)
        top = counts.most_common(1)[0][1]
        if top == args.runs:
            stable += 1
        elif top >= 2:
            partial += 1
            unstable.append((question, keys))
        else:
            unstable.append((question, keys))
        print(
            f"  [{index}/{len(questions)}] {'ok ' if top == args.runs else 'DIFF'} {question[:70]}",
            flush=True,
        )

    measured = len(questions) - skipped
    if measured == 0:
        print("No advisory question measured.")
        return 1
    print()
    print(f"measured={measured} skipped(non-advisory)={skipped}")
    print(f"fully stable ({args.runs}/{args.runs} same key): {stable} = {stable / measured:.1%}")
    print(f"majority stable (>=2 same): {stable + partial} = {(stable + partial) / measured:.1%}")
    print()
    for question, keys in unstable[: args.examples]:
        print(f"- {question}")
        for key in keys:
            print(f"    {key}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="folder with questions.jsonl"
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--examples", type=int, default=10, help="unstable examples to print")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
