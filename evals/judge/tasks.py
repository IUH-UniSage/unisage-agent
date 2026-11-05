"""Which RAGAS metrics a `results.jsonl` row gets, and their inputs (`spec.md` §4)."""

import re
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
    """Each chunk with its document title first: the answer names the document ("Thông báo
    số 2", "ngành Công nghệ thông tin") from the title the model saw, which the chunk text
    does not repeat (03/11/2026: such answers were scored as invented)."""

    texts = []
    for item in items:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        title = str(item.get("title") or "").strip()
        texts.append(f"[Tài liệu: {title}]\n{content}" if title else content)
    return [text[:MAX_CONTEXT_CHARS] for text in texts][:MAX_CONTEXTS]


# Lines written by code, not by the model, so they are not claims to judge (03/11/2026: each
# of them was scored as an invented statement):
# - the caution the grounding check appends ("> Lưu ý: con số 7,5 không tìm thấy ... trước
#   khi dùng.", 29/10/2026);
# - the profile update notice a reply opens with ("Mình đã cập nhật thông tin của bạn - ...");
# - a calculation block's formula description and version ("Công thức:" up to "Thay số:" or
#   "Kết quả:"), which come from the formula registry; its steps and result stay;
# - the notice that a requested calculation was rejected ("_Mình chưa tính được ..._");
# - the current-semester sentence the system prompt asks for ("Theo lịch dự kiến của
#   Trường, hiện là ...").
_CAUTION = re.compile(r"\n*> Lưu ý: con số .*? trước khi dùng\.", re.DOTALL)
_UPDATE_NOTICE = re.compile(r"^[^\n]{0,40}đã cập nhật thông tin của [^\n]*\n*", re.MULTILINE)
_FORMULA = re.compile(r"^Công thức:\n.*?(?=^(?:Thay số:|Kết quả:))", re.DOTALL | re.MULTILINE)
_REJECTED = re.compile(r"^_Mình chưa tính được .*?_\s*$\n?", re.MULTILINE)
_SEMESTER = re.compile(r"[^.\n]*Theo lịch dự kiến của Trường[^\n]*?(?:\.(?=\s|$)|$)")


def judged_response(response: str) -> str:
    text = _CAUTION.sub("", response)
    for pattern in (_UPDATE_NOTICE, _FORMULA, _REJECTED, _SEMESTER):
        text = pattern.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def with_question(question: str, contexts: list[str]) -> list[str]:
    """Faithfulness reads the question too: a detail the answer repeats from it ("Tạp chí số
    29 (2017)") is not invented, though the table of contents chunk does not repeat it
    (29/10/2026: correct answers were scored 0)."""

    return [f"Câu hỏi của người dùng: {question}", *contexts] if question else contexts


def _unique(values: Iterable[Any]) -> list[str]:
    return [str(value) for value in dict.fromkeys(values) if value]


def tasks_for(row: dict[str, Any], selected: Sequence[str] = METRICS) -> list[Task]:
    """The RAGAS inputs of one `results.jsonl` row (`spec.md` §4)."""

    response = judged_response(row.get("response") or "")
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
            retrieved_contexts=with_question(question, _contexts(row.get("web_results") or [])),
        )
        if reference:
            add("answer_coverage", user_input=question, response=response, reference=reference)
    elif has_reference(row):
        contexts = _contexts(row.get("context") or [])
        add(
            "faithfulness",
            user_input=question,
            response=response,
            retrieved_contexts=with_question(question, contexts),
        )
        add("answer_coverage", user_input=question, response=response, reference=reference)
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
