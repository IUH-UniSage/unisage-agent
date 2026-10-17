"""WebJudgeNode (09c) - which web pages answer which sub-query rerank left empty.

Tavily's score says how well a page matches the query's wording, not whether it answers
the question: a keyword query rated a scholarship article 0.90 and a master's-degree fee
page 0.72 for an undergraduate fee question. One call to the RERANK model (EXTRACTION
while none is configured) reads the candidates and sorts the ones worth keeping into:

- `answers`: the snippet holds the information -> generation cites them like a document;
- `sources`: the school's official document on that very topic, whose snippet lacks the
  figures (a scanned fee decision, "chi tiết theo phụ lục") -> the ticket fallback links
  them instead of saying nothing was found.

Runs only when a sub-query is left without a chunk, so a turn the knowledge base answers
never pays for it. Fails open to the rule from before it existed (`select_without_judge`),
and the failure is returned for the AI-admin warning.
"""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import unquote

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.academic_calendar import current_term
from app.core.config import settings
from app.core.errors.llm_failure import describe_llm_failure
from app.core.registry.model_registry import CredentialConfig
from app.core.timezone import now_ict
from app.graph.nodes.web_search import select_without_judge
from app.graph.streaming import (
    AttemptRecorder,
    BudgetContext,
    auxiliary_model_settings,
    run_agent_text_with_failover,
)
from app.rag.prompting import get_templates
from app.rag.web_query import campus_of
from app.schemas.web_search import WebSearchResult

logger = logging.getLogger(__name__)

DEFAULT_PURPOSE = "RERANK"
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_SUB_QUERY_KEY = re.compile(r"^SQ(\d+)$")
_PAGE_KEY = re.compile(r"^\[?W?(\d+)\]?$")
# A year, but not an intake ("khóa 2024", "K2024"): the fee of intake 2024 is in the
# 2025-2026 fee decision.
_YEAR = re.compile(r"(?<!\d)(?:(khóa|khoá|khoa|k)\s*)?(20\d{2})(?!\d)", re.IGNORECASE)


class MalformedWebJudgementError(ValueError):
    """The model's answer was not the `{"SQk": {"Wn": {...}}}` object asked for."""


@dataclass(frozen=True)
class WebJudgement:
    answers: list[WebSearchResult]
    sources: list[WebSearchResult] = field(default_factory=list)
    # Why the score-only rule was used instead, for the AI-admin warning.
    failure: str | None = None


def build_web_judge_agent(model: Model | str) -> Agent[None, str]:
    return Agent(
        model=model,
        # `.replace`, not `.format`: the prompt is full of literal JSON braces.
        system_prompt=get_templates().agent_web_result_judge.replace(
            "{max_answers}", str(settings.CHAT_WEB_SEARCH_MAX_ANSWERS_PER_SUB)
        ),
        model_settings=auxiliary_model_settings(model),
    )


async def judge_web_results(
    agent: Agent[None, str],
    questions: Sequence[str],
    candidates: Sequence[WebSearchResult],
    *,
    credential: CredentialConfig | None = None,
    purpose: str = DEFAULT_PURPOSE,
    snapshot_version: int | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
) -> WebJudgement:
    """`questions` are the sub-queries left without a chunk."""

    if not candidates or not questions:
        return WebJudgement(answers=[])
    try:
        output = await run_agent_text_with_failover(
            agent,
            build_judge_prompt(questions, candidates),
            purpose=purpose,
            credential=credential,
            snapshot_version=snapshot_version,
            agent_factory=build_web_judge_agent,
            on_attempt=on_attempt,
            budget=budget,
            timeout_seconds=settings.CHAT_AUX_CALL_TIMEOUT_SECONDS,
        )
        kept = parse_judgement(output, sub_query_count=len(questions), page_count=len(candidates))
    except Exception as exc:
        message = (
            describe_llm_failure(exc, purpose=purpose).message
            if not isinstance(exc, MalformedWebJudgementError)
            else f"Mô hình chấm trang web trả về kết quả không đúng định dạng ({exc})."
        )
        logger.warning("Web judge failed - keeping pages by score: %s", message)
        return WebJudgement(
            answers=select_without_judge(
                candidates, limit=settings.CHAT_WEB_SEARCH_MAX_ANSWERS_PER_SUB * len(questions)
            ),
            failure=message,
        )

    kept = _answers_of_the_year_asked(questions, candidates, kept)
    _log_decisions(questions, candidates, kept)
    # Per sub-query, the judge's own order (best first) and at most ANSWERS_PER_SUB: the
    # judge picks the pages, Tavily's score does not.
    answer_indexes: list[int] = []
    for per_query in kept:
        picked = [index for index, (_reason, is_answer) in per_query.items() if is_answer]
        for index in picked[: settings.CHAT_WEB_SEARCH_MAX_ANSWERS_PER_SUB]:
            if index not in answer_indexes:
                answer_indexes.append(index)
    # The year asked first: for "năm 2025" the 2025 notice before the 2026 one.
    asked = set().union(*(_years_in(question) for question in questions))
    source_indexes = sorted(
        {
            index
            for per_query in kept
            for index, (_reason, is_answer) in per_query.items()
            if not is_answer
        }
        - set(answer_indexes),
        key=lambda index: (not asked & _years_of(candidates[index]), index),
    )
    return WebJudgement(
        answers=[candidates[index] for index in answer_indexes],
        sources=[candidates[index] for index in source_indexes][
            : settings.CHAT_WEB_SEARCH_MAX_SOURCES
        ],
    )


def build_judge_prompt(questions: Sequence[str], candidates: Sequence[WebSearchResult]) -> str:
    question_lines = "\n".join(
        f"SQ{index}. {question}" for index, question in enumerate(questions, 1)
    )
    page_lines = "\n\n".join(
        f"[W{index}] (Tiêu đề: {page.title}; URL: {page.url}; "
        f"Năm học: {page.academic_year or 'không ghi'}{_campus_note(page.url)})\n{page.content}"
        for index, page in enumerate(candidates, 1)
    )
    academic_year = current_term(now_ict().date()).academic_year
    return (
        f"Năm học hiện tại: {academic_year}\n\n"
        f"Câu hỏi con:\n{question_lines}\n\nTrang web:\n{page_lines}"
    )


def _campus_note(url: str) -> str:
    campus = campus_of(url)
    return f"; Website của {campus}" if campus else ""


def parse_judgement(
    raw_output: str, *, sub_query_count: int, page_count: int
) -> list[dict[int, tuple[str, bool]]]:
    """Per sub-query, the kept pages' 0-based indexes -> (reason, holds the answer). A page
    counts only with a non-empty reason; a missing or non-boolean `answer` counts as false,
    so a page is only quoted when the model said its snippet answers."""

    candidate = raw_output.strip().strip("`").strip()
    match = _JSON_OBJECT_PATTERN.search(candidate)
    if match is None:
        raise MalformedWebJudgementError("no JSON object")
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise MalformedWebJudgementError("invalid JSON") from exc
    if not isinstance(loaded, dict):
        raise MalformedWebJudgementError("not a JSON object")

    kept: list[dict[int, tuple[str, bool]]] = [{} for _ in range(sub_query_count)]
    recognised = False
    for key, value in loaded.items():
        key_match = _SUB_QUERY_KEY.match(str(key))
        if key_match is None or not isinstance(value, dict):
            continue
        query_index = int(key_match.group(1)) - 1
        if not 0 <= query_index < sub_query_count:
            continue
        recognised = True
        for page_key, verdict in value.items():
            page_match = _PAGE_KEY.match(str(page_key).strip())
            if page_match is None or not isinstance(verdict, dict):
                continue
            page_index = int(page_match.group(1)) - 1
            reason = verdict.get("reason")
            if not 0 <= page_index < page_count or not isinstance(reason, str):
                continue
            if reason.strip():
                kept[query_index][page_index] = (reason.strip(), verdict.get("answer") is True)
    if not recognised:
        raise MalformedWebJudgementError("no SQk keys")
    return kept


def _answers_of_the_year_asked(
    questions: Sequence[str],
    candidates: Sequence[WebSearchResult],
    kept: list[dict[int, tuple[str, bool]]],
) -> list[dict[int, tuple[str, bool]]]:
    """A page whose title and URL name only other years than the question does is kept
    as a link, never quoted: the judge once took the 2026 National Day notice as the
    answer for 2025, noting itself "năm trong văn bản là 2026 chứ không phải 2025"."""

    checked: list[dict[int, tuple[str, bool]]] = []
    for question, per_query in zip(questions, kept, strict=True):
        asked = _years_in(question)
        checked.append(
            {
                index: (
                    reason,
                    is_answer and not _other_year(candidates[index], asked),
                )
                for index, (reason, is_answer) in per_query.items()
            }
        )
    return checked


def _other_year(page: WebSearchResult, asked: set[str]) -> bool:
    named = _years_of(page)
    return bool(asked and named and not asked & named)


def _years_of(page: WebSearchResult) -> set[str]:
    return _years_in(f"{page.title} {unquote(page.url)}")


def _years_in(text: str) -> set[str]:
    return {match.group(2) for match in _YEAR.finditer(text) if not match.group(1)}


def _log_decisions(
    questions: Sequence[str],
    candidates: Sequence[WebSearchResult],
    kept: Sequence[dict[int, tuple[str, bool]]],
) -> None:
    """Always logged, one line per sub-query: which pages were kept, as answer or source,
    and why - what to read when an answer quoted the wrong page."""

    for query_index, question in enumerate(questions):
        logger.info(
            "Web judge SQ%d %r: kept [%s]; dropped [%s]",
            query_index + 1,
            question,
            "; ".join(
                f"W{index + 1} {candidates[index].url} "
                f"({'answer' if is_answer else 'source'}): {reason}"
                for index, (reason, is_answer) in sorted(kept[query_index].items())
            ),
            "; ".join(
                f"W{index + 1} {page.url} ({page.score:.2f})"
                for index, page in enumerate(candidates)
                if index not in kept[query_index]
            ),
        )
