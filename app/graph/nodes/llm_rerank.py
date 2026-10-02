"""LLMRerankNode (09a) - which reranked chunks actually answer which sub-query.

PostRetrievalRerankNode only thresholds embedding similarity, so a chunk that
merely shares keywords passes: the admission table listing "Công nghệ thông
tin" passed for "chương trình khung ngành CNTT", which kept web search from
running for a sub-query no document answers and handed generation tables of
the wrong programme level. One call to the EXTRACTION model per turn (not
CHAT's - it is a cheap judging task) narrows each sub-query's chunks to the
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
from app.graph.streaming import AttemptRecorder, BudgetContext, run_agent_text_with_failover
from app.rag.prompting import get_templates
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

PURPOSE = "EXTRACTION"
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_SUB_QUERY_KEY = re.compile(r"^SQ(\d+)$")


class MalformedRerankOutputError(ValueError):
    """The model's answer was not the `{"SQk": [n, ...]}` object asked for."""


@dataclass(frozen=True)
class LLMRerankOutcome:
    result: TurnRerankResult
    # Why the score-only result was kept instead, for the AI-admin warning.
    failure: str | None = None


def build_llm_rerank_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, system_prompt=get_templates().agent_reranker_compressor)


async def llm_rerank(
    agent: Agent[None, str],
    questions: Sequence[str],
    rerank_result: TurnRerankResult,
    *,
    credential: CredentialConfig | None = None,
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
            purpose=PURPOSE,
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
            describe_llm_failure(exc, purpose=PURPOSE).message
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
    narrowed = turn_result(per_query_kept, best_scores=rerank_result.best_scores)
    logger.info(
        "LLM rerank kept %d of %d chunk(s); sub-queries without any: %s",
        len(narrowed.chunks),
        len(candidates),
        [index + 1 for index in narrowed.failed_query_indexes] or "none",
    )
    return LLMRerankOutcome(result=narrowed)


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
        + f")\n{chunk.content[:max_chars]}"
        for index, chunk in enumerate(chunks, 1)
    )
    return f"Câu hỏi con:\n{question_lines}\n\nĐoạn văn bản:\n{chunk_lines}"


def _parse_relevance(raw_output: str, *, sub_query_count: int, chunk_count: int) -> list[set[int]]:
    """0-based chunk indexes per sub-query. Out-of-range numbers are dropped; a
    sub-query the model left out counts as having none."""

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

    relevant: list[set[int]] = [set() for _ in range(sub_query_count)]
    recognised = False
    for key, value in loaded.items():
        key_match = _SUB_QUERY_KEY.match(str(key))
        if key_match is None or not isinstance(value, list):
            continue
        query_index = int(key_match.group(1)) - 1
        if not 0 <= query_index < sub_query_count:
            continue
        recognised = True
        relevant[query_index] = {
            number - 1 for number in value if isinstance(number, int) and 1 <= number <= chunk_count
        }
    if not recognised:
        raise MalformedRerankOutputError("no SQk keys")
    return relevant
