"""LLMRerankNode (09a) - which reranked chunks actually answer which sub-query.

PostRetrievalRerankNode only thresholds embedding similarity, so a chunk that
merely shares keywords passes: the admission table listing "Công nghệ thông
tin" passed for "chương trình khung ngành CNTT", which kept web search from
running for a sub-query no document answers and handed generation tables of
the wrong programme level. One call to the RERANK model per turn (EXTRACTION
while none is configured; never CHAT's) narrows each sub-query's chunks to the
ones that answer it; a sub-query left with none counts as failed and goes to
WebSearchNode.

Fails open: if the call or its output fails, the score-only result stands and
the failure is returned for the AI-admin warning - this node can only ever
remove noise, never take away an answer the turn would have had without it.
"""

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.config import settings
from app.core.errors.llm_failure import describe_llm_failure
from app.core.registry.model_registry import CredentialConfig
from app.graph.nodes.post_retrieval_rerank import TurnRerankResult, turn_result
from app.graph.streaming import (
    AttemptRecorder,
    BudgetContext,
    auxiliary_model_settings,
    run_agent_text_with_failover,
)
from app.rag.prompting import get_templates
from app.rag.prompting.citations import source_title
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_PURPOSE = "RERANK"
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_SUB_QUERY_KEY = re.compile(r"^SQ(\d+)$")
_CHUNK_KEY = re.compile(r"^\[?C?(\d+)\]?$")
# Table chunks carry a breadcrumb line above each group of rows, e.g.
# "[A ĐỐI VỚI TRỤ SỞ CHÍNH > 3 Đại học chính quy > 3.1 Khóa tuyển sinh năm học 2025-2026]".
_SECTION_LINE = re.compile(r"^\[([^\]]*[^\]\s-][^\]]*)\]$")
# Upper bound on the outline so a chunk with dozens of row groups can't crowd the prompt.
_MAX_OUTLINE_CHARS = 1200


class MalformedRerankOutputError(ValueError):
    """The model's answer was not the `{"SQk": [n, ...]}` object asked for."""


@dataclass(frozen=True)
class LLMRerankOutcome:
    result: TurnRerankResult
    # Why the score-only result was kept instead, for the AI-admin warning.
    failure: str | None = None


def build_llm_rerank_agent(model: Model | str) -> Agent[None, str]:
    return Agent(
        model=model,
        system_prompt=get_templates().agent_reranker_compressor,
        model_settings=auxiliary_model_settings(),
    )


async def llm_rerank(
    agent: Agent[None, str],
    questions: Sequence[str],
    rerank_result: TurnRerankResult,
    *,
    credential: CredentialConfig | None = None,
    purpose: str = DEFAULT_PURPOSE,
    snapshot_version: int | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
) -> LLMRerankOutcome:
    """`questions[i]` is sub-query i's question, in the same order as
    `rerank_result.per_query`."""

    candidates = _candidate_chunks(rerank_result)
    if not candidates:
        return LLMRerankOutcome(result=rerank_result)

    try:
        output = await run_agent_text_with_failover(
            agent,
            _build_prompt(questions, candidates),
            purpose=purpose,
            credential=credential,
            snapshot_version=snapshot_version,
            agent_factory=build_llm_rerank_agent,
            on_attempt=on_attempt,
            budget=budget,
        )
        relevant = _parse_relevance(
            output, sub_query_count=len(questions), chunk_count=len(candidates)
        )
    except Exception as exc:
        message = (
            describe_llm_failure(exc, purpose=purpose).message
            if not isinstance(exc, MalformedRerankOutputError)
            else f"Mô hình Extraction trả về kết quả lọc không đúng định dạng ({exc})."
        )
        logger.warning("LLM rerank failed - keeping the score-only ranking: %s", message)
        return LLMRerankOutcome(result=rerank_result, failure=message)

    per_query_kept = [
        sorted(
            (candidates[index] for index in relevant[query_index]),
            key=lambda chunk: chunk.score,
            reverse=True,
        )
        for query_index in range(len(questions))
    ]
    _log_decisions(questions, candidates, relevant)
    per_query_kept = _rescue_emptied_queries(per_query_kept, rerank_result)
    narrowed = turn_result(per_query_kept, best_scores=rerank_result.best_scores)
    logger.info(
        "LLM rerank kept %d of %d chunk(s); sub-queries without any: %s",
        len(narrowed.chunks),
        len(candidates),
        [index + 1 for index in narrowed.failed_query_indexes] or "none",
    )
    return LLMRerankOutcome(result=narrowed)


def _chunk_label(index: int, chunk: RetrievedChunk) -> str:
    section = f" > {chunk.heading_path[-1]}" if chunk.heading_path else ""
    return f"C{index + 1} {source_title(chunk.source)}{section} ({chunk.score:.2f})"


def _log_decisions(
    questions: Sequence[str],
    candidates: Sequence[RetrievedChunk],
    relevant: Sequence[dict[int, str]],
) -> None:
    """Always logged, one line per sub-query: which chunks were kept and why,
    and which were dropped - what to read when an answer used the wrong table."""

    for query_index, question in enumerate(questions):
        kept = relevant[query_index]
        logger.info(
            "LLM rerank SQ%d %r: kept [%s]; dropped [%s]",
            query_index + 1,
            question,
            "; ".join(
                f"{_chunk_label(index, candidates[index])}: {reason}"
                for index, reason in sorted(kept.items())
            ),
            "; ".join(
                _chunk_label(index, chunk)
                for index, chunk in enumerate(candidates)
                if index not in kept
            ),
        )


def _rescue_emptied_queries(
    per_query_kept: list[list[RetrievedChunk]], rerank_result: TurnRerankResult
) -> list[list[RetrievedChunk]]:
    """Only when the model kept nothing for ANY sub-query: each one whose best
    chunk scored high gets its top chunks back, instead of the turn ending in
    TicketFallback with the right document already retrieved. A sub-query left
    empty while others kept chunks is a deliberate drop (keyword-only match)
    and stays empty."""

    keep = settings.CHAT_LLM_RERANK_RESCUE_KEEP
    if keep == 0 or any(per_query_kept):
        return per_query_kept

    min_score = settings.CHAT_LLM_RERANK_RESCUE_MIN_SCORE
    rescued: list[list[RetrievedChunk]] = []
    for query_index, kept in enumerate(per_query_kept):
        own = rerank_result.per_query[query_index].chunks
        if not own or own[0].score < min_score:
            rescued.append(kept)
            continue
        restored = own[:keep]
        logger.info(
            "LLM rerank rescue SQ%d: model kept nothing, restoring top %d by score [%s]",
            query_index + 1,
            len(restored),
            "; ".join(f"{source_title(chunk.source)} ({chunk.score:.2f})" for chunk in restored),
        )
        rescued.append(restored)
    return rescued


def _candidate_chunks(rerank_result: TurnRerankResult) -> list[RetrievedChunk]:
    """Every chunk any sub-query kept, once - a chunk retrieved for one
    sub-query may well answer another."""

    seen: dict[str, RetrievedChunk] = {}
    for result in rerank_result.per_query:
        for chunk in result.chunks:
            seen.setdefault(chunk.chunk_id, chunk)
    return list(seen.values())


def _build_prompt(questions: Sequence[str], chunks: Sequence[RetrievedChunk]) -> str:
    max_chars = settings.CHAT_LLM_RERANK_SNIPPET_CHARS
    question_lines = "\n".join(
        f"SQ{index}. {question}" for index, question in enumerate(questions, 1)
    )
    chunk_lines = "\n\n".join(
        f"[C{index}] (Tài liệu: {chunk.source}"
        + (f"; Mục: {' > '.join(chunk.heading_path)}" if chunk.heading_path else "")
        + ")"
        + _outline_line(chunk)
        + f"\n{chunk.content[:max_chars]}"
        for index, chunk in enumerate(chunks, 1)
    )
    return f"Câu hỏi con:\n{question_lines}\n\nĐoạn văn bản:\n{chunk_lines}"


def _outline_line(chunk: RetrievedChunk) -> str:
    """Every section breadcrumb inside the chunk, in order - the model only sees
    the first SNIPPET_CHARS of the text, and the row it is looking for (e.g.
    Khối Công nghệ of the current intake) is often further down a fee table."""

    sections: list[str] = []
    for line in chunk.content.splitlines():
        match = _SECTION_LINE.match(line.strip())
        if match is not None and match.group(1) not in sections:
            sections.append(match.group(1))
    if not sections:
        return ""
    outline = "; ".join(sections)
    if len(outline) > _MAX_OUTLINE_CHARS:
        outline = outline[:_MAX_OUTLINE_CHARS].rstrip() + "…"
    return f"\nMục có trong đoạn (kể cả phần bị cắt bên dưới): {outline}"


def _parse_relevance(
    raw_output: str, *, sub_query_count: int, chunk_count: int
) -> list[dict[int, str]]:
    """Per sub-query, the kept chunks' 0-based indexes mapped to the model's
    reason. A chunk only counts when the model gave a non-empty reason for it
    (having to say why is what keeps a small model from keeping keyword-only
    matches); a bare list of numbers is still accepted, with no reasons.
    Out-of-range numbers are dropped; a sub-query left out has none."""

    candidate = raw_output.strip().strip("`").strip()
    match = _JSON_OBJECT_PATTERN.search(candidate)
    if match is None:
        raise MalformedRerankOutputError("no JSON object")
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise MalformedRerankOutputError("invalid JSON") from exc
    if not isinstance(loaded, dict):
        raise MalformedRerankOutputError("not a JSON object")

    relevant: list[dict[int, str]] = [{} for _ in range(sub_query_count)]
    recognised = False
    for key, value in loaded.items():
        key_match = _SUB_QUERY_KEY.match(str(key))
        if key_match is None:
            continue
        query_index = int(key_match.group(1)) - 1
        if not 0 <= query_index < sub_query_count:
            continue
        if isinstance(value, dict):
            entries = [(_chunk_number(chunk_key), reason) for chunk_key, reason in value.items()]
            kept = {
                number - 1: reason.strip()
                for number, reason in entries
                if number is not None and isinstance(reason, str) and reason.strip()
            }
        elif isinstance(value, list):
            kept = {number - 1: "" for number in value if isinstance(number, int)}
        else:
            continue
        recognised = True
        relevant[query_index] = {
            index: reason for index, reason in kept.items() if 0 <= index < chunk_count
        }
    if not recognised:
        raise MalformedRerankOutputError("no SQk keys")
    return relevant


def _chunk_number(key: object) -> int | None:
    match = _CHUNK_KEY.match(str(key).strip())
    return int(match.group(1)) if match else None
