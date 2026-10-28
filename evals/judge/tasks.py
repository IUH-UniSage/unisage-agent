"""Which RAGAS metrics a `results.jsonl` row gets, and their inputs (`spec.md` §4)."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from evals.run.metrics import outcome_of

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
PASS_SCORE = 0.7
FALLBACK_SHARE_MAX = 0.05
# Context handed to the judge: at most 8 chunks of ~500 tokens (`spec.md` §5, 504 risk).
MAX_CONTEXTS = 8
MAX_CONTEXT_CHARS = 2000


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
    # what the judge answered at each RAGAS step (statements, verdicts and their reasons),
    # so a low score can be explained claim by claim
    steps: list[dict[str, Any]] = field(default_factory=list)


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
            "retrieved_context_ids": _unique(
                c.get("document_id") for c in row.get("context") or []
            ),
            "reference_context_ids": _unique(row.get("expected_document_ids") or []),
        }
        add("id_context_precision", **ids)
        add("id_context_recall", **ids)
    add("refusal", user_input=question, response=response)
    return tasks


def tasks_of(rows: Iterable[dict[str, Any]], selected: Sequence[str]) -> list[Task]:
    return [task for row in rows for task in tasks_for(row, selected)]
