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
from dataclasses import dataclass, field, replace

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.config import settings
from app.core.registry.model_registry import CredentialConfig
from app.core.registry.model_router import ModelRouter
from app.graph.streaming import (
    AgentFactory,
    AttemptRecorder,
    BudgetContext,
    FailoverCallback,
    auxiliary_model_settings,
    run_agent_text_with_failover,
)
from app.rag.prompting import append_current_time, append_recent_history, get_templates
from app.schemas.chat_history import HistoryMessage
from app.schemas.intent import ClassifiedTask, RoutingMode

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")


def _number_key(number: str) -> str:
    """ "01" and "1", "3.20" and "3,2" are one number."""

    integer, _, fraction = number.replace(",", ".").partition(".")
    fraction = fraction.replace(".", "").rstrip("0")
    integer = integer.lstrip("0") or "0"
    return f"{integer}.{fraction}" if fraction else integer


def strip_unstated_numbers(passage: str, question: str) -> str:
    """The HyDE passage with every number its question does not state written as "...".
    The prompt forbids them, and a small model still wrote "điểm trung bình tích lũy từ
    3.20 đến 3.59" for "tốt nghiệp loại giỏi" in 2 of 3 runs (23-11-2026) - a guess the
    dense search then leans on. Not for the table row: its years come from the cohort."""

    stated = {_number_key(number) for number in _NUMBER_PATTERN.findall(question)}
    return _NUMBER_PATTERN.sub(
        lambda match: match.group(0) if _number_key(match.group(0)) in stated else "...",
        passage,
    )


@dataclass(frozen=True)
class SubQuery:
    """`question` is shown to generation (`SQk. ...`); `retrieval_text` is
    what gets embedded."""

    question: str
    retrieval_text: str
    # More dense texts for the same sub-query (the HyDE passage as a table row): each is
    # embedded and the dense side keeps a chunk's best cosine over all of them, so they
    # share one vote against BM25 instead of adding votes of their own.
    extra_retrieval_texts: tuple[str, ...] = ()
    # Which advisory task (in `transform_tasks`' order) it came from - bookkeeping, not
    # part of what the sub-query is.
    task_index: int = field(default=0, compare=False)


def build_query_transformation_agent(model: Model | str) -> Agent[None, str]:
    return Agent(
        model=model,
        system_prompt=get_templates().agent_hyde_generator,
        model_settings=auxiliary_model_settings(model),
    )


def build_decomposer_agent(model: Model | str) -> Agent[None, str]:
    return Agent(
        model=model,
        system_prompt=get_templates().agent_multi_query_decomposer,
        model_settings=auxiliary_model_settings(model),
    )


def _fold_confirmed_metadata_into_query(user_query: str, confirmed_metadata: dict[str, str]) -> str:
    if not confirmed_metadata:
        return user_query
    declared = ", ".join(f"{field}={value}" for field, value in confirmed_metadata.items())
    # Only for what the question leaves open: "học phí ngành điện tử" from a CNTT student
    # was rewritten to the CNTT fee (14-10-2026).
    return (
        f"{user_query} (hồ sơ người hỏi đã khai, CHỈ dùng cho tiêu chí câu hỏi không tự nêu: "
        f"{declared})"
    )


async def transform_query(
    agent: Agent[None, str],
    user_query: str,
    *,
    confirmed_metadata: dict[str, str] | None = None,
    history: Sequence[HistoryMessage] = (),
    role: str | None = None,
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
    agent_factory: AgentFactory | None = None,
    router: ModelRouter | None = None,
    on_failover: FailoverCallback | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
) -> str:
    """HyDE retrieval text: the self-contained question, then the document.

    `role` (verified, from the token) tells who carries out a procedure the question
    does not name a person for - a lecturer's "điều chỉnh điểm" is not a student's.

    `purpose`/`credential`/`snapshot_version`/`agent_factory`/`router`/
    `on_failover`/`on_attempt`/`budget` are the same opt-in failover/usage/budget
    wiring as `run_agent_text_with_failover()`.
    """

    enriched_query = _fold_confirmed_metadata_into_query(user_query, confirmed_metadata or {})
    prompt = append_recent_history(enriched_query, history)
    if role:
        prompt += f"\n\n<user_role>{role}</user_role>"
    return await run_agent_text_with_failover(
        agent,
        append_current_time(prompt, confirmed_metadata),
        purpose=purpose,
        credential=credential,
        snapshot_version=snapshot_version,
        agent_factory=agent_factory,
        router=router,
        on_failover=on_failover,
        on_attempt=on_attempt,
        budget=budget,
        timeout_seconds=settings.CHAT_AUX_CALL_TIMEOUT_SECONDS,
    )


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
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
    agent_factory: AgentFactory | None = None,
    router: ModelRouter | None = None,
    on_failover: FailoverCallback | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
) -> list[str]:
    """Up to `CHAT_MAX_SUB_QUERIES` sub-queries; an unusable output gives `[]`.

    `purpose`/`credential`/`snapshot_version`/`agent_factory`/`router`/
    `on_failover`/`on_attempt`/`budget` are the same opt-in failover/usage/budget
    wiring as `run_agent_text_with_failover()`.
    """

    enriched_query = _fold_confirmed_metadata_into_query(user_query, confirmed_metadata or {})
    output = await run_agent_text_with_failover(
        agent,
        append_current_time(append_recent_history(enriched_query, history), confirmed_metadata),
        purpose=purpose,
        credential=credential,
        snapshot_version=snapshot_version,
        agent_factory=agent_factory,
        router=router,
        on_failover=on_failover,
        on_attempt=on_attempt,
        budget=budget,
        timeout_seconds=settings.CHAT_AUX_CALL_TIMEOUT_SECONDS,
    )
    return _parse_sub_queries(output)


async def _transform_task(
    hyde_agent: Agent[None, str],
    decomposer_agent: Agent[None, str] | None,
    task: ClassifiedTask,
    mode: RoutingMode,
    *,
    confirmed_metadata: dict[str, str] | None,
    history: Sequence[HistoryMessage],
    role: str | None,
    purpose: str | None,
    credential: CredentialConfig | None,
    snapshot_version: int | None,
    hyde_agent_factory: AgentFactory | None,
    decomposer_agent_factory: AgentFactory | None,
    router: ModelRouter | None,
    on_failover: FailoverCallback | None,
    on_attempt: AttemptRecorder | None,
    budget: BudgetContext | None,
) -> list[SubQuery]:
    # Classification saw the same profile (`<declared_attributes>`) and applies it by the
    # same rule as the HyDE prompt, so its text is reused: a second LLM call for a
    # logged-in student took 14s (14-10-2026).
    if mode == "MULTI" and task.sub_queries:
        return [SubQuery(question=query, retrieval_text=query) for query in task.sub_queries]
    if mode == "SINGLE" and task.hyde_text:
        rows = (task.hyde_row_text,) if task.hyde_row_text else ()
        return [
            SubQuery(question=task.query, retrieval_text=task.hyde_text, extra_retrieval_texts=rows)
        ]

    if mode == "MULTI" and decomposer_agent is not None:
        sub_queries = await decompose_query(
            decomposer_agent,
            task.query,
            confirmed_metadata=confirmed_metadata,
            history=history,
            purpose=purpose,
            credential=credential,
            snapshot_version=snapshot_version,
            agent_factory=decomposer_agent_factory,
            router=router,
            on_failover=on_failover,
            on_attempt=on_attempt,
            budget=budget,
        )
        # Fewer than 2 sub-queries is not a decomposition - fall back to HyDE.
        if len(sub_queries) >= 2:
            return [SubQuery(question=query, retrieval_text=query) for query in sub_queries]

    retrieval_text = await transform_query(
        hyde_agent,
        task.query,
        confirmed_metadata=confirmed_metadata,
        history=history,
        role=role,
        purpose=purpose,
        credential=credential,
        snapshot_version=snapshot_version,
        agent_factory=hyde_agent_factory,
        router=router,
        on_failover=on_failover,
        on_attempt=on_attempt,
        budget=budget,
    )
    passage_text, row_text = split_hyde_output(retrieval_text)
    return [
        SubQuery(
            question=task.query,
            retrieval_text=passage_text,
            extra_retrieval_texts=(row_text,) if row_text else (),
        )
    ]


async def transform_tasks(
    hyde_agent: Agent[None, str],
    tasks: Sequence[tuple[ClassifiedTask, RoutingMode]],
    *,
    decomposer_agent: Agent[None, str] | None = None,
    confirmed_metadata: dict[str, str] | None = None,
    history: Sequence[HistoryMessage] = (),
    role: str | None = None,
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
    hyde_agent_factory: AgentFactory | None = None,
    decomposer_agent_factory: AgentFactory | None = None,
    router: ModelRouter | None = None,
    on_failover: FailoverCallback | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
) -> list[SubQuery]:
    """Every task's sub-queries, flattened in task order.

    `purpose`/`credential`/`snapshot_version`/`hyde_agent_factory`/
    `decomposer_agent_factory`/`router`/`on_failover`/`on_attempt`/`budget` are
    the same opt-in failover/usage/budget wiring as `run_agent_text_with_failover()` -
    each concurrent task gets its own independent retry loop, so one task's
    failover never touches another's (or the original `hyde_agent`/
    `decomposer_agent`) mid-flight; `on_failover` still fires per task, so
    whichever task fails over first is what the caller sees update the
    shared model state with. `on_attempt` is called from every concurrent
    task's coroutine - safe because `UsageRecorder.record_attempt()` never
    awaits, so no two calls can interleave mid-append even though the tasks
    themselves run concurrently.
    """

    per_task = await asyncio.gather(
        *(
            _transform_task(
                hyde_agent,
                decomposer_agent,
                task,
                mode,
                confirmed_metadata=confirmed_metadata,
                history=history,
                role=role,
                purpose=purpose,
                credential=credential,
                snapshot_version=snapshot_version,
                hyde_agent_factory=hyde_agent_factory,
                decomposer_agent_factory=decomposer_agent_factory,
                router=router,
                on_failover=on_failover,
                on_attempt=on_attempt,
                budget=budget,
            )
            for task, mode in tasks
        )
    )
    return [
        replace(sub_query, task_index=task_index)
        for task_index, sub_queries in enumerate(per_task)
        for sub_query in sub_queries
    ]


def split_hyde_output(hyde_output: str) -> tuple[str, str | None]:
    """HyDE's `question / passage / row` blocks as (question + passage, question + row).
    The row is the last block when there are three and it is written as a table row
    (" | "); anything else is kept as the passage. The passage loses the numbers the
    question does not state (`strip_unstated_numbers`); a one-block output stays whole."""

    blocks = [block.strip() for block in hyde_output.strip().split("\n\n") if block.strip()]
    if len(blocks) < 2:
        return hyde_output, None
    question = blocks[0]
    row = blocks[-1] if len(blocks) >= 3 and " | " in blocks[-1] else None
    passages = blocks[1:-1] if row is not None else blocks[1:]
    passage = strip_unstated_numbers("\n\n".join(passages), question)
    return f"{question}\n\n{passage}", f"{question}\n\n{row}" if row is not None else None


def extract_standalone_question(hyde_output: str) -> str:
    """The question line HyDE output leads with (falls back to the first
    line if the blank-line separator is missing)."""

    return hyde_output.split("\n\n", 1)[0].strip()
