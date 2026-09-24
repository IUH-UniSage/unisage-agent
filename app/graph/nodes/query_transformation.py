"""QueryTransformationNode - entry point of the advisory flow. Turns each
advisory task into sub-queries, all tasks concurrently: HyDE for a SINGLE
task, the decomposer for a MULTI (comparison) task.

HyDE first rewrites a follow-up into a self-contained question using recent
history, then writes the hypothetical document; the whole output is the
retrieval text. Confirmed attributes are folded into the input so a resumed
turn retrieves with them.
"""

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.config import settings
from app.rag.prompting import append_recent_history, get_templates
from app.schemas.chat_history import HistoryMessage
from app.schemas.intent import ClassifiedTask, RoutingMode

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class SubQuery:
    """`question` is shown to generation (`SQk. ...`); `retrieval_text` is
    what gets embedded."""

    question: str
    retrieval_text: str


def build_query_transformation_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, system_prompt=get_templates().agent_hyde_generator)


def build_decomposer_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, system_prompt=get_templates().agent_multi_query_decomposer)


def _fold_confirmed_metadata_into_query(user_query: str, confirmed_metadata: dict[str, str]) -> str:
    if not confirmed_metadata:
        return user_query
    declared = ", ".join(f"{field}={value}" for field, value in confirmed_metadata.items())
    return f"{user_query} (thông tin sinh viên đã xác nhận: {declared})"


async def transform_query(
    agent: Agent[None, str],
    user_query: str,
    *,
    confirmed_metadata: dict[str, str] | None = None,
    history: Sequence[HistoryMessage] = (),
) -> str:
    """HyDE retrieval text: the self-contained question, then the document."""

    enriched_query = _fold_confirmed_metadata_into_query(user_query, confirmed_metadata or {})
    result = await agent.run(append_recent_history(enriched_query, history))
    return result.output


def _parse_sub_queries(raw_output: str) -> list[str]:
    candidate = raw_output.strip().strip("`").strip()
    match = _JSON_OBJECT_PATTERN.search(candidate)
    for text in (candidate, match.group(0) if match else None):
        if text is None:
            continue
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            continue
        raw = loaded.get("sub_queries") if isinstance(loaded, dict) else None
        if not isinstance(raw, list):
            return []
        queries: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip() and item.strip() not in queries:
                queries.append(item.strip())
        return queries[: settings.CHAT_MAX_SUB_QUERIES]
    return []


async def decompose_query(
    agent: Agent[None, str],
    user_query: str,
    *,
    confirmed_metadata: dict[str, str] | None = None,
    history: Sequence[HistoryMessage] = (),
) -> list[str]:
    """Up to `CHAT_MAX_SUB_QUERIES` sub-queries; an unusable output gives `[]`."""

    enriched_query = _fold_confirmed_metadata_into_query(user_query, confirmed_metadata or {})
    result = await agent.run(append_recent_history(enriched_query, history))
    return _parse_sub_queries(result.output or "")


async def _transform_task(
    hyde_agent: Agent[None, str],
    decomposer_agent: Agent[None, str] | None,
    task: ClassifiedTask,
    mode: RoutingMode,
    *,
    confirmed_metadata: dict[str, str] | None,
    history: Sequence[HistoryMessage],
) -> list[SubQuery]:
    if mode == "MULTI" and decomposer_agent is not None:
        sub_queries = await decompose_query(
            decomposer_agent, task.query, confirmed_metadata=confirmed_metadata, history=history
        )
        # Fewer than 2 sub-queries is not a decomposition - fall back to HyDE.
        if len(sub_queries) >= 2:
            return [SubQuery(question=query, retrieval_text=query) for query in sub_queries]

    retrieval_text = await transform_query(
        hyde_agent, task.query, confirmed_metadata=confirmed_metadata, history=history
    )
    return [SubQuery(question=task.query, retrieval_text=retrieval_text)]


async def transform_tasks(
    hyde_agent: Agent[None, str],
    tasks: Sequence[tuple[ClassifiedTask, RoutingMode]],
    *,
    decomposer_agent: Agent[None, str] | None = None,
    confirmed_metadata: dict[str, str] | None = None,
    history: Sequence[HistoryMessage] = (),
) -> list[SubQuery]:
    """Every task's sub-queries, flattened in task order."""

    per_task = await asyncio.gather(
        *(
            _transform_task(
                hyde_agent,
                decomposer_agent,
                task,
                mode,
                confirmed_metadata=confirmed_metadata,
                history=history,
            )
            for task, mode in tasks
        )
    )
    return [sub_query for sub_queries in per_task for sub_query in sub_queries]


def extract_standalone_question(hyde_output: str) -> str:
    """The question line HyDE output leads with (falls back to the first
    line if the blank-line separator is missing)."""

    return hyde_output.split("\n\n", 1)[0].strip()
