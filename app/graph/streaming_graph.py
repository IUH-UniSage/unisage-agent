"""The streaming graph orchestrator.

Implemented as a plain async function rather than a `pydantic_graph` graph:
`pydantic_graph.Graph.run()`/`iter()` return a single final output, not a
token stream, so wiring per-token streaming through it would mean fighting
the library's grain. A follow-up could port this function's branching onto
real `pydantic_graph.BaseNode` subclasses without changing its signature -
`run_graph(input, models, token_sink) -> GraphOutput` is the seam to keep.

Deliberately HTTP-agnostic: `GraphInput.is_first_turn` is computed by the
caller (via Java's message history) before this runs - this module never
calls `BackendJavaClient` itself, keeping graph logic testable without HTTP
mocks.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pydantic import JsonValue
from pydantic_ai.models import Model

from app.calculation.formulas import FORMULAS
from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace, bind_trace, unbind_trace
from app.core.registry.model_registry import CredentialConfig
from app.core.timezone import now_ict
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.address import address_for
from app.graph.calc_requests import arithmetic_blocks
from app.graph.calculation_turn import (
    CalculationTask,
    Review,
    calculation_titles,
    known_fields,
    needs_input_parts,
    number_citations,
    render_outcomes,
    review,
    trace_items,
)
from app.graph.clarification_round import (
    TaskQuestions,
    advisory_questions,
    advisory_task,
    build_round,
)
from app.graph.evidence import levels_for
from app.graph.grounding import caution_line, check_answer, looks_like_refusal, numbers_in
from app.graph.metadata_fields import (
    Scope,
    apply_updates,
    canonical_field,
    concept,
    field_scope,
    normalize_value,
    update_notice,
)
from app.graph.nodes.calculation import (
    CalculationDeps,
    Computed,
    Deferred,
    Planned,
    TaskOutcome,
    plan_calculation,
    registry_id,
    resume_calculation_task,
    run_planned_task,
)
from app.graph.nodes.cross_encoder_rerank import (
    CROSS_ENCODER_SKIPPED,
    cross_encoder_rerank,
)
from app.graph.nodes.generation_synthesis import build_generation_agent, run_generation_synthesis
from app.graph.nodes.greeting import detect_greeting, greeting_for
from app.graph.nodes.intent_routing import plan_route
from app.graph.nodes.llm_rerank import (
    LLM_RERANK_SKIPPED,
    build_llm_rerank_agent,
    llm_rerank,
)
from app.graph.nodes.message_classification import (
    build_classification_agent,
    classify_intent,
    describe_classification,
)
from app.graph.nodes.off_topic import off_topic_reply
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.graph.nodes.query_transformation import (
    SubQuery,
    build_decomposer_agent,
    build_query_transformation_agent,
    extract_standalone_question,
    transform_tasks,
)
from app.graph.nodes.retrieval_cache import (
    CachedTurn,
    L1Pending,
    TurnCacheContext,
    l1_storable,
    lookup_l1,
    lookup_l2,
    open_turn_cache,
    store_l1,
    store_l2,
)
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.graph.nodes.social_chat import build_social_chat_agent, run_social_chat
from app.graph.nodes.ticket_fallback import build_ticket_fallback_agent, run_ticket_fallback
from app.graph.nodes.web_judge import WebJudgement, build_web_judge_agent, judge_web_results
from app.graph.nodes.web_search import (
    WebSearchOutcome,
    WebSearchRequest,
    search_web,
    select_without_judge,
)
from app.graph.streaming import BudgetContext, FailoverCallback, TokenSink
from app.graph.streaming_state import (
    AdminWarning,
    GraphInput,
    GraphModels,
    GraphOutput,
    ResumeInput,
)
from app.graph.web_policy import decide_fallback
from app.integrations.cross_encoder_client import CrossEncoderClient
from app.rag.prompting.citations import (
    build_citations,
    build_context_trace,
    keep_known_links,
    source_links_block,
    strip_source_list,
)
from app.rag.web_query import build_web_query, is_postgraduate
from app.schemas.clarification import LastCalculation, PendingCalculationTask, PendingRound
from app.schemas.intent import ClassifiedTask, IntentClassification, RoutingMode
from app.schemas.web_search import WebSearchResult, WebSlots

logger = logging.getLogger(__name__)

# Prefixes of the AI-admin warnings (`event: warning`) this module raises.
_UNGROUNDED = "Câu trả lời có con số không tìm thấy trong tài liệu được trích dẫn: "
_INVALID_CITATION = "Câu trả lời trích dẫn chỉ số nguồn không tồn tại: "
_NO_CITATION = "Câu trả lời nêu số liệu nhưng không trích dẫn tài liệu nào."
_WEB_SEARCH_FAILED = (
    "Tìm kiếm web thất bại nên câu trả lời không dùng được nguồn từ website Trường: "
)
_WEB_JUDGE_SKIPPED = (
    "Bỏ qua bước chấm trang web (mô hình Rerank/Extraction), trang web được giữ theo điểm Tavily: "
)
_WEB_SOURCE_USED = (
    "Lượt này dùng trang trên website Trường vì kho văn bản không có; cân nhắc nạp các văn "
    "bản này vào kho: "
)

# Speculative web searches a turn turned out not to need keep running so their results
# land in the cache for the next turn, and a web judge started alongside rerank keeps
# running if the turn ends before it is awaited; held here so they are not
# garbage-collected, and a failure is logged rather than lost.
_background_searches: set[asyncio.Task[Any]] = set()


def _make_failover_applier(models: GraphModels) -> FailoverCallback:
    """`models.classification`/`query_transformation`/`generation` are all
    built from the SAME top-priority CHAT credential once at request start
    (see `get_graph_models()`) - without this, a credential failing in node
    03 leaves nodes 06/10/11 still holding the dead credential, forcing each
    to independently rediscover the same failure (see `streaming.py`'s
    `on_failover` docstring for the full rationale). Mutating `models` in
    place here is what makes a failover in an early node visible to every
    node that runs after it in the SAME request."""

    def _apply(credential: CredentialConfig, model: Model | str) -> None:
        models.generation_credential = credential
        models.classification = model
        models.query_transformation = model
        models.generation = model

    return _apply


# Saves the student's profile the moment the turn changes it (the graph never calls
# the database itself), so a correction is kept even if the rest of the turn fails.
ProfileSaver = Callable[[dict[str, str]], Awaitable[None]]


async def run_graph(
    graph_input: GraphInput,
    models: GraphModels,
    token_sink: TokenSink,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    budget: BudgetContext | None = None,
    on_profile_change: ProfileSaver | None = None,
) -> GraphOutput:
    # Bound for the whole run so a failover deep inside an LLM helper can log which
    # model/credential the node moved to (see `GraphTrace.model_switch`).
    token = bind_trace(trace)
    try:
        return await _run_graph(
            graph_input,
            models,
            token_sink,
            trace,
            usage_recorder,
            budget=budget,
            on_profile_change=on_profile_change,
        )
    finally:
        unbind_trace(token)


async def _run_graph(
    graph_input: GraphInput,
    models: GraphModels,
    token_sink: TokenSink,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    budget: BudgetContext | None = None,
    on_profile_change: ProfileSaver | None = None,
) -> GraphOutput:
    # A panel-submit turn resumes the claimed round - never a greeting or a new question.
    if graph_input.resume is not None:
        return await _run_resume(
            graph_input, graph_input.resume, models, token_sink, trace, usage_recorder, budget
        )

    # GreetingDetectionNode: fast path, no LLM.
    trace.node("01_GreetingDetectionNode")
    if detect_greeting(graph_input.user_message, first_turn=graph_input.is_first_turn):
        greeting = greeting_for(graph_input.security)
        await token_sink(greeting)
        return GraphOutput(
            response_text=greeting,
            confirmed_metadata=graph_input.confirmed_metadata,
        )

    confirmed_metadata = graph_input.confirmed_metadata

    # RetrievalCacheNode, L1: a turn with no history whose question was asked before
    # under the same scope replays the classifier's tasks and the reranked chunks
    # (docs/specs/SPEC-retrieval-cache.md) - nodes 03, 06, 08, 09 and 09a are skipped.
    cache_ctx = await open_turn_cache(graph_input, models)
    cached_turn: CachedTurn | None = None
    l1_pending: L1Pending | None = None
    if cache_ctx is not None and cache_ctx.l1_eligible:
        cached_turn, l1_pending = await lookup_l1(
            cache_ctx, graph_input, confirmed_metadata, models.retrieval
        )

    if cached_turn is not None:
        trace.node("03_MessageClassificationNode", cached=True)
        classification = cached_turn.classification
    else:
        # MessageClassificationNode.
        classification_agent = build_classification_agent(models.classification)
        trace.node(
            "03_MessageClassificationNode",
            agent=classification_agent,
            credential=models.generation_credential,
        )
        classification = await classify_intent(
            classification_agent,
            graph_input.user_message,
            history=graph_input.history,
            purpose="CHAT",
            credential=models.generation_credential,
            snapshot_version=models.snapshot_version,
            agent_factory=build_classification_agent,
            on_failover=_make_failover_applier(models),
            on_attempt=usage_recorder.bind("MessageClassificationNode"),
            budget=budget,
            previous_calculation=calculation_title(graph_input.last_calculation),
            profile=confirmed_metadata,
            role=graph_input.security.role,
            departments=[entry.department_id for entry in graph_input.security.department_access],
        )
        trace.prompt("03_MessageClassificationNode", describe_classification(classification))
        if l1_pending is not None:
            l1_pending = (
                replace(l1_pending, confidence=classification.confidence)
                if l1_storable(classification)
                else None
            )

    # A correction of the profile applies at once: saved now, used by every later step.
    confirmed_metadata, changed = apply_updates(
        confirmed_metadata, classification.metadata_updates, now_ict().date()
    )
    notice = ""
    if changed:
        logger.info("profile.updated fields=%s", sorted(changed))
        if on_profile_change is not None:
            await on_profile_change(dict(confirmed_metadata))
        notice = f"{update_notice(changed, address_for(graph_input.security))}\n\n"
        await token_sink(notice)
    output = await _classified_turn(
        graph_input,
        models,
        token_sink,
        trace,
        usage_recorder,
        budget,
        classification=classification,
        confirmed_metadata=confirmed_metadata,
        cache_ctx=cache_ctx,
        cached_turn=cached_turn,
        l1_pending=l1_pending,
    )
    return replace(output, response_text=notice + output.response_text) if notice else output


async def _classified_turn(
    graph_input: GraphInput,
    models: GraphModels,
    token_sink: TokenSink,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    budget: BudgetContext | None,
    *,
    classification: IntentClassification,
    confirmed_metadata: dict[str, str],
    cache_ctx: TurnCacheContext | None = None,
    cached_turn: CachedTurn | None = None,
    l1_pending: L1Pending | None = None,
) -> GraphOutput:

    # IntentRoutingNode (deterministic).
    trace.node("04_IntentRoutingNode")
    route_plan = plan_route(classification)

    if route_plan.end == "SOCIAL_CHAT":
        # SocialChatNode (streaming).
        social_agent = build_social_chat_agent(models.generation)
        trace.node(
            "04_IntentRouting_SocialChat",
            agent=social_agent,
            credential=models.generation_credential,
        )
        social_text = await run_social_chat(
            social_agent,
            graph_input.user_message,
            security=graph_input.security,
            confirmed_metadata=confirmed_metadata,
            token_sink=token_sink,
            purpose="CHAT",
            credential=models.generation_credential,
            snapshot_version=models.snapshot_version,
            on_failover=_make_failover_applier(models),
            on_attempt=usage_recorder.bind("SocialChatNode"),
            budget=budget,
        )
        return GraphOutput(
            response_text=social_text,
            confirmed_metadata=confirmed_metadata,
        )

    if route_plan.end == "OFF_TOPIC":
        trace.node("05_OffTopicRejectNode")
        off_topic_text = off_topic_reply()
        await token_sink(off_topic_text)
        return GraphOutput(
            response_text=off_topic_text,
            confirmed_metadata=confirmed_metadata,
        )

    calculation_tasks = [
        CalculationTask(task_id=f"T{index}", query=task.query)
        for index, task in enumerate(route_plan.calculation_tasks, start=1)
    ]
    deps = _calculation_deps(
        graph_input, models, usage_recorder, budget, trace, confirmed_metadata=confirmed_metadata
    )
    # The extractor runs before anything else: a task no registered formula fits joins
    # the advisory part (answered from the documents, no arithmetic by the model), so
    # its retrieval starts with the others'.
    planned = await _plan_calculations(calculation_tasks, deps, trace)
    runnable = [task for task in planned if not isinstance(task, Deferred)]
    advisory_tasks: list[tuple[ClassifiedTask, RoutingMode]] = [
        *route_plan.advisory_tasks,
        *(
            (
                ClassifiedTask(intent="academic_advisory", query=task.query, routing_mode="SINGLE"),
                "SINGLE",
            )
            for task in planned
            if isinstance(task, Deferred)
        ),
    ]
    # A mixed turn answers only the advisory question(s), so generation doesn't
    # try to answer the calculation part from regulations.
    advisory_question = " ".join(task.query for task, _mode in advisory_tasks) if runnable else None
    advisory: AdvisoryPart | None = None
    if advisory_tasks:
        advisory = AdvisoryPart(
            task_id=f"T{len(calculation_tasks) + 1}",
            tasks=advisory_tasks,
            question=advisory_question,
        )
    return await _turn_with_calculations(
        graph_input,
        models,
        token_sink,
        trace,
        usage_recorder,
        budget,
        confirmed_metadata=confirmed_metadata,
        calculations=_start_calculations(runnable, deps),
        queries={task.task_id: task.query for task in runnable},
        cache_ctx=cache_ctx,
        cached_turn=cached_turn,
        l1_pending=l1_pending if not runnable else None,
        deps=deps,
        advisory=advisory,
        original_query=advisory_question or graph_input.user_message,
        previous_round=None,
        strict_numbers=len(runnable) < len(planned),
    )


async def _run_resume(
    graph_input: GraphInput,
    resume: ResumeInput,
    models: GraphModels,
    token_sink: TokenSink,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    budget: BudgetContext | None,
) -> GraphOutput:
    """A panel-submit turn: run only the tasks the panel was asked for, with the
    answers already validated against the stored panel. Every answer goes under its
    registry name; one for a profile field (or an advisory field the registry does not
    know) also goes into the profile, normalised. Calculation answers complete the
    stored parameters - no extractor call, no retrieval - and the original advisory
    question is re-answered."""

    trace.node("02_ClarificationResume")
    pending = resume.pending_round
    confirmed_metadata = dict(graph_input.confirmed_metadata)
    answers_by_task: dict[str, dict[str, JsonValue]] = {}
    # Answers about someone else: part of the re-answered question, never the profile.
    case_answers: list[str] = []
    today = now_ict().date()
    for answer in resume.answers.values():
        question = answer.question
        name = canonical_field(question.field)
        # A calculation's choice goes on as its label: the LLM never saw the "o1".."o12"
        # ids the panel gave its options, so an id alone would make it ask again.
        value = (
            answer.display
            if question.origin == "calculation" and question.kind == "choice"
            else answer.value
        )
        answers_by_task.setdefault(question.task_id, {})[name] = value
        if question.origin == "advisory" and question.about == "other":
            case_answers.append(f"{concept(name)}: {answer.display}")
            continue
        default_scope: Scope = "profile" if question.origin == "advisory" else "calculation"
        if field_scope(name, default_scope) == "profile":
            # The intake is kept as a year: "kỳ 9" would be wrong next term.
            normalized = normalize_value(name, value, today)
            if normalized is not None:
                confirmed_metadata[name] = normalized

    outcomes: list[TaskOutcome] = []
    queries: dict[str, str] = {}
    advisory: AdvisoryPart | None = None
    deps = _calculation_deps(
        graph_input, models, usage_recorder, budget, trace, confirmed_metadata=confirmed_metadata
    )
    for task in pending.tasks:
        if isinstance(task, PendingCalculationTask):
            queries[task.task_id] = task.query
            outcomes.append(
                await resume_calculation_task(
                    task.task_id,
                    task.query,
                    task.plan,
                    task.known_params,
                    answers_by_task.get(task.task_id, {}),
                    deps,
                )
            )
        else:
            advisory = AdvisoryPart(
                task_id=task.task_id,
                tasks=[(origin, origin.routing_mode or "SINGLE") for origin in task.origin_tasks],
                question=_with_case_answers(pending.original_query, case_answers),
            )

    return await _turn_with_calculations(
        graph_input,
        models,
        token_sink,
        trace,
        usage_recorder,
        budget,
        confirmed_metadata=confirmed_metadata,
        calculations=_done(outcomes) if outcomes else None,
        queries=queries,
        deps=deps,
        advisory=advisory,
        original_query=pending.original_query,
        previous_round=pending,
    )


def _with_case_answers(question: str, case_answers: Sequence[str]) -> str:
    """The question with what the user just answered about the person it asks about -
    their attributes belong to this question, not to `<declared_attributes>`, which
    holds only the caller's own."""

    if not case_answers:
        return question
    return (
        f"{question}\n(Về trường hợp được hỏi, người dùng vừa cho biết: {'; '.join(case_answers)})"
    )


@dataclass(frozen=True)
class AdvisoryPart:
    task_id: str
    tasks: list[tuple[ClassifiedTask, RoutingMode]]
    question: str | None


def _calculation_deps(
    graph_input: GraphInput,
    models: GraphModels,
    usage_recorder: UsageRecorder,
    budget: BudgetContext | None,
    trace: GraphTrace,
    *,
    confirmed_metadata: dict[str, str] | None = None,
) -> CalculationDeps:
    return CalculationDeps(
        models=models,
        security=graph_input.security,
        on_attempt=usage_recorder.bind("CalculationNode"),
        on_failover=_make_failover_applier(models),
        budget=budget,
        previous=graph_input.last_calculation,
        history=graph_input.history,
        recent_calculations=graph_input.recent_calculations,
        confirmed_metadata=(
            confirmed_metadata if confirmed_metadata is not None else graph_input.confirmed_metadata
        ),
        trace=trace,
        user_message=graph_input.user_message if graph_input.resume is None else None,
    )


def calculation_title(last: LastCalculation | None) -> str | None:
    """What the classifier is told about the latest calculation: its name only."""

    if last is None:
        return None
    known = registry_id(last.plan.formula_id)
    if last.title or known is None:
        return last.title
    return FORMULAS[known].title


def _computed_calculations(outcomes: Sequence[TaskOutcome]) -> list[LastCalculation]:
    """The turn's computed calculations, each with its parameters (a follow-up reuses
    them instead of asking again)."""

    return [
        LastCalculation(plan=outcome.plan, params=outcome.params, title=outcome.result.title)
        for outcome in outcomes
        if isinstance(outcome, Computed)
    ]


async def _plan_calculations(
    tasks: Sequence[CalculationTask], deps: CalculationDeps, trace: GraphTrace
) -> list[Planned | Deferred]:
    """One extractor call per task, concurrently: which registered formula (and which
    numbers, unknown and target), plain arithmetic over the student's numbers, or
    `Deferred` when none fits."""

    if not tasks:
        return []
    trace.node("07_CalculationNode")
    return list(
        await asyncio.gather(*(plan_calculation(task.task_id, task.query, deps) for task in tasks))
    )


def _start_calculations(
    tasks: Sequence[Planned], deps: CalculationDeps
) -> "asyncio.Task[list[TaskOutcome]] | None":
    """Planned calculations run alongside the advisory retrieval (they are awaited only
    at the barrier right before node 10)."""

    if not tasks:
        return None

    async def run_all() -> list[TaskOutcome]:
        return list(await asyncio.gather(*(run_planned_task(task, deps) for task in tasks)))

    return asyncio.create_task(run_all())


def _done(outcomes: list[TaskOutcome]) -> "asyncio.Task[list[TaskOutcome]]":
    async def ready() -> list[TaskOutcome]:
        return outcomes

    return asyncio.ensure_future(ready())


async def _turn_with_calculations(
    graph_input: GraphInput,
    models: GraphModels,
    token_sink: TokenSink,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    budget: BudgetContext | None,
    *,
    confirmed_metadata: dict[str, str],
    calculations: "asyncio.Task[list[TaskOutcome]] | None",
    queries: dict[str, str],
    deps: CalculationDeps,
    advisory: AdvisoryPart | None,
    original_query: str,
    previous_round: PendingRound | None,
    cache_ctx: TurnCacheContext | None = None,
    cached_turn: CachedTurn | None = None,
    l1_pending: L1Pending | None = None,
    # A calculation question joined the advisory part: numbers the model works out
    # itself are flagged, since code was supposed to compute them.
    strict_numbers: bool = False,
) -> GraphOutput:
    """Stream order: calculation blocks (Python) → advisory answer (node 10, which only
    starts after the blocks were sent) → one panel for everything still missing. A
    calculation-only turn instead opens with the checked review, so its blocks wait for
    that call; a result the review says does not answer what was asked is shown under
    "Lưu ý"."""

    outcomes: list[TaskOutcome] = []
    shown: list[str] = []
    calculation_citations: list[dict[str, Any]] = []
    checked: Review | None = None

    def collect(done: Sequence[TaskOutcome], first_citation: int) -> None:
        numbered, cited = number_citations(done, first_citation)
        outcomes.extend(numbered)
        calculation_citations.extend(cited)

    async def emit(text: str) -> None:
        if text:
            shown.append(text)
            await token_sink(text)

    async def show_calculations(first_citation: int = 1) -> list[str]:
        collect(await calculations if calculations is not None else [], first_citation)
        text = render_outcomes(outcomes)
        await emit(f"{text}\n\n" if text else "")
        return calculation_titles(outcomes)

    try:
        if advisory is not None:
            output = await _run_advisory_flow(
                graph_input,
                models,
                token_sink,
                trace,
                usage_recorder,
                confirmed_metadata=confirmed_metadata,
                advisory_tasks=advisory.tasks,
                question=advisory.question,
                budget=budget,
                before_generation=show_calculations,
                cache_ctx=cache_ctx,
                cached_turn=cached_turn,
                l1_pending=l1_pending,
                strict_numbers=strict_numbers,
            )
            output = replace(
                output,
                response_text="".join(shown) + output.response_text,
                citations=output.citations + calculation_citations,
            )
        else:
            done = await calculations if calculations is not None else []
            computed = [o for o in done if isinstance(o, Computed)]
            checked = await review(computed, original_query, deps)
            collect(done, 1)
            lead = checked.lead() if checked is not None else ""
            await emit("\n\n".join(text for text in (lead, render_outcomes(outcomes)) if text))
            output = GraphOutput(
                response_text="".join(shown),
                confirmed_metadata=confirmed_metadata,
                citations=calculation_citations,
            )
    finally:
        if calculations is not None and not calculations.done():
            calculations.cancel()

    if deps.warnings:
        known = {warning.code for warning in output.admin_warnings}
        output = replace(
            output,
            admin_warnings=[
                *output.admin_warnings,
                *(warning for warning in deps.warnings if warning.code not in known),
            ],
        )
    parts = needs_input_parts(outcomes, queries)
    if advisory is not None:
        parts.append(
            TaskQuestions(
                task=advisory_task(advisory.task_id, [task for task, _mode in advisory.tasks]),
                questions=advisory_questions(
                    output.ask_forms,
                    confirmed_metadata=output.confirmed_metadata,
                    known_fields=known_fields(outcomes),
                ),
            )
        )
    if previous_round is not None:
        output = _with_follow_up_round(output, parts, previous_round)
    else:
        output = replace(
            output,
            pending_round=build_round(parts, original_query=original_query, chain_depth=1),
        )
    if outcomes:
        public, private = trace_items(
            outcomes,
            queries=queries,
            run_id=budget.request_id if budget is not None else usage_recorder.request_id,
            deps=deps,
            review=checked,
        )
        output = replace(
            output,
            calculation_items=public,
            calculation_traces=private,
            calculations=_computed_calculations(outcomes),
        )
    return output


def _with_follow_up_round(
    output: GraphOutput, parts: list[TaskQuestions], previous: PendingRound
) -> GraphOutput:
    """Whatever is still missing after a panel answer becomes the next panel (no limit:
    each one needs the student to answer, and answered fields are never asked again)."""

    follow_up = build_round(
        parts, original_query=previous.original_query, chain_depth=previous.chain_depth + 1
    )
    return replace(output, pending_round=follow_up)


@dataclass(frozen=True)
class _WebGroup:
    """One web search and whom it is for: all of a task's sub-queries (`sub_index` None)
    or the one MULTI sub-query the classifier wrote its own slots for."""

    request: WebSearchRequest
    task_index: int
    sub_index: int | None
    # A fast-changing topic: searched from the start of the advisory branch.
    dynamic: bool


def _web_groups(advisory_tasks: Sequence[tuple[ClassifiedTask, RoutingMode]]) -> list[_WebGroup]:
    """A MULTI task's sub-queries are searched one by one when the classifier wrote slots
    for each (one keyword query for "học phí ... và lịch thi chứng chỉ" found only fee
    pages, 14-10-2026); any other task by its own slots, once."""

    speculative = set(settings.CHAT_WEB_SEARCH_SPECULATIVE_TOPICS)
    groups: list[_WebGroup] = []
    for task_index, (task, _mode) in enumerate(advisory_tasks):
        if task.sub_query_web and task.sub_queries:
            for sub_index, (question, slots) in enumerate(
                zip(task.sub_queries, task.sub_query_web, strict=False)
            ):
                groups.append(
                    _WebGroup(
                        _web_request(slots, question),
                        task_index,
                        sub_index,
                        dynamic=slots is not None and bool(set(slots.topics) & speculative),
                    )
                )
        elif task.web is not None:
            groups.append(
                _WebGroup(
                    _web_request(task.web, task.query),
                    task_index,
                    None,
                    dynamic=bool(set(task.web.topics) & speculative),
                )
            )
    return groups


def _web_request(slots: WebSlots | None, question: str) -> WebSearchRequest:
    """By the slots' keyword query; without slots, by the question itself."""

    if slots is None:
        return WebSearchRequest(query=question, postgraduate=is_postgraduate(question))
    return WebSearchRequest(
        query=build_web_query(slots),
        topic=slots.topics[0],
        co_so=slots.co_so,
        postgraduate=is_postgraduate(f"{slots.bac_dao_tao or ''} {slots.subject} {question}"),
    )


def _group_questions(
    groups: Sequence[_WebGroup],
    sub_queries: Sequence[SubQuery],
    advisory_tasks: Sequence[tuple[ClassifiedTask, RoutingMode]],
) -> list[list[int]]:
    """Per group, the indexes of the sub-queries (after query transformation) it serves.
    When transformation did not keep the classifier's sub-queries, a per-sub-query group
    serves the whole task."""

    served: list[list[int]] = []
    for group in groups:
        indexes = [i for i, sub in enumerate(sub_queries) if sub.task_index == group.task_index]
        task = advisory_tasks[group.task_index][0]
        if group.sub_index is not None and len(indexes) == len(task.sub_query_web or ()):
            indexes = [indexes[group.sub_index]]
        served.append(indexes)
    return served


def _start_speculative_search(
    groups: Sequence[_WebGroup],
) -> tuple[asyncio.Task[WebSearchOutcome], list[int]] | None:
    """The fast-changing topics' searches, started now; with the positions (in `groups`)
    of the groups searched, in request order."""

    if not settings.CHAT_WEB_SEARCH_ENABLED or settings.CHAT_WEB_SEARCH_MODE != "speculative":
        return None
    dynamic = [position for position, group in enumerate(groups) if group.dynamic]
    if not dynamic:
        return None
    task = asyncio.create_task(search_web([groups[position].request for position in dynamic]))
    return task, dynamic


def _page_reference(page: WebSearchResult) -> str:
    return f"{page.url} (năm học {page.academic_year})" if page.academic_year else page.url


def _finish_in_background(task: asyncio.Task[Any]) -> None:
    _background_searches.add(task)
    task.add_done_callback(_discard_background_search)


def _discard_background_search(task: asyncio.Task[Any]) -> None:
    _background_searches.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Background web search failed", exc_info=task.exception())


async def _judge_web_results(
    models: GraphModels,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    questions: Sequence[str],
    candidates: Sequence[WebSearchResult],
    *,
    budget: BudgetContext | None,
    alongside: str | None = None,
) -> WebJudgement:
    """`alongside`: the node this judge runs next to - it is then logged as a step of
    it, neither ending that node's timing nor sending a status."""

    if not candidates:
        return WebJudgement(answers=[])
    if models.rerank is None:
        return WebJudgement(
            answers=select_without_judge(
                candidates, limit=settings.CHAT_WEB_SEARCH_MAX_ANSWERS_PER_SUB * len(questions)
            ),
            failure=models.rerank_unavailable or "chưa cấu hình mô hình Rerank/Extraction.",
        )
    agent = build_web_judge_agent(models.rerank)
    if alongside is None:
        trace.node("09c_WebJudgeNode", agent=agent, credential=models.rerank_credential)
    else:
        trace.step(
            "09c_WebJudgeNode",
            parent=alongside,
            agent=agent,
            credential=models.rerank_credential,
        )
    return await judge_web_results(
        agent,
        questions,
        candidates,
        credential=models.rerank_credential,
        purpose=models.rerank_purpose,
        snapshot_version=models.snapshot_version,
        on_attempt=usage_recorder.bind("WebJudgeNode"),
        budget=budget,
    )


# One judge call: the sub-queries (indexes into the turn's questions) and their pages.
_JudgeJob = tuple[list[int], list[WebSearchResult]]


async def _judge_jobs(
    models: GraphModels,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    questions: Sequence[str],
    jobs: Sequence[_JudgeJob],
    *,
    budget: BudgetContext | None,
    alongside: str,
) -> list[tuple[list[int], WebJudgement]]:
    """Each search's pages judged for the sub-queries it was for, all calls at once: a
    sub-query is judged on its own pages, so its at most ANSWERS_PER_SUB pages are the
    judge's pick among them."""

    judged = await asyncio.gather(
        *(
            _judge_web_results(
                models,
                trace,
                usage_recorder,
                [questions[index] for index in indexes],
                pages,
                budget=budget,
                alongside=alongside,
            )
            for indexes, pages in jobs
        )
    )
    return [(indexes, judgement) for (indexes, _pages), judgement in zip(jobs, judged, strict=True)]


def _merge_judgements(judged: Sequence[tuple[list[int], WebJudgement]]) -> WebJudgement:
    """Answers in sub-query order without repeating a page; sources after them, capped."""

    answers: list[WebSearchResult] = []
    sources: list[WebSearchResult] = []
    seen: set[str] = set()
    for _indexes, judgement in judged:
        for page in judgement.answers:
            if page.url not in seen:
                answers.append(page)
                seen.add(page.url)
    for _indexes, judgement in judged:
        for page in judgement.sources:
            if page.url not in seen:
                sources.append(page)
                seen.add(page.url)
    return WebJudgement(
        answers=answers,
        sources=sources[: settings.CHAT_WEB_SEARCH_MAX_SOURCES],
        failure=next((j.failure for _i, j in judged if j.failure is not None), None),
    )


@dataclass(frozen=True)
class _JudgedSearch:
    outcome: WebSearchOutcome
    judged: list[tuple[list[int], WebJudgement]]


async def _search_then_judge(
    models: GraphModels,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    questions: Sequence[str],
    search: asyncio.Task[WebSearchOutcome],
    served: Sequence[list[int]],
    *,
    budget: BudgetContext | None,
) -> _JudgedSearch:
    """The speculative searches' pages judged for every sub-query they were for, while
    retrieval and rerank run: needs only the questions and the pages, not the chunks.
    `served[k]`: the sub-queries of the k-th searched request."""

    outcome = await search
    jobs = [
        (indexes, pages)
        for indexes, pages in zip(served, outcome.per_request, strict=False)
        if indexes and pages
    ]
    judged = await _judge_jobs(
        models,
        trace,
        usage_recorder,
        questions,
        jobs,
        budget=budget,
        alongside="08_RetrievalFilteringNode",
    )
    return _JudgedSearch(outcome=outcome, judged=judged)


async def _search_for_failed(
    groups: Sequence[_WebGroup],
    served: Sequence[list[int]],
    failed: set[int],
    questions: Sequence[str],
    speculative: tuple[asyncio.Task[WebSearchOutcome], list[int]] | None,
) -> tuple[list[_JudgeJob], list[WebSearchOutcome]]:
    """Pages for the sub-queries left without a chunk: the speculative searches' for those
    they cover, a search now for the other groups serving one, and a search by the
    question itself for one no group serves."""

    jobs: list[_JudgeJob] = []
    outcomes: list[WebSearchOutcome] = []
    covered: set[int] = set()
    searched: set[int] = set()
    if speculative is not None:
        task, positions = speculative
        outcome = await task
        outcomes.append(outcome)
        searched.update(positions)
        for position, pages in zip(positions, outcome.per_request, strict=False):
            covered.update(served[position])
            indexes = [index for index in served[position] if index in failed]
            if indexes and pages:
                jobs.append((indexes, pages))
    owners: list[list[int]] = []
    requests: list[WebSearchRequest] = []
    for position, group in enumerate(groups):
        indexes = [index for index in served[position] if index in failed]
        if position in searched or not indexes:
            continue
        owners.append(indexes)
        requests.append(group.request)
        covered.update(served[position])
    for index in sorted(failed - covered):
        owners.append([index])
        requests.append(
            WebSearchRequest(query=questions[index], postgraduate=is_postgraduate(questions[index]))
        )
    if requests:
        outcome = await search_web(requests)
        outcomes.append(outcome)
        jobs.extend(
            (indexes, pages)
            for indexes, pages in zip(owners, outcome.per_request, strict=False)
            if pages
        )
    return jobs, outcomes


async def _run_advisory_flow(
    graph_input: GraphInput,
    models: GraphModels,
    token_sink: TokenSink,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    *,
    confirmed_metadata: dict[str, str],
    advisory_tasks: Sequence[tuple[ClassifiedTask, RoutingMode]],
    question: str | None = None,
    budget: BudgetContext | None = None,
    # Called with the first citation index the advisory answer leaves free.
    before_generation: Callable[[int], Awaitable[list[str]]] | None = None,
    cache_ctx: TurnCacheContext | None = None,
    cached_turn: CachedTurn | None = None,
    l1_pending: L1Pending | None = None,
    strict_numbers: bool = False,
) -> GraphOutput:
    """Advisory branch: query transformation → retrieval → rerank → web search
    for the sub-queries rerank left empty → generation (or ticket fallback
    when neither found anything). A task on a fast-changing topic (speculative
    search) has its pages judged for every sub-query alongside retrieval and
    rerank instead (`CHAT_WEB_JUDGE_WITH_CHUNKS`), and generation gets them next
    to the chunks. `question` is what gets answered; `None` means the whole user
    message."""

    question = question or graph_input.user_message

    # WebSearchNode, early: a task about something posted on the website first (fees,
    # admission, notices, contacts) starts its search now, alongside query transformation,
    # retrieval and rerank.
    web_groups = _web_groups(advisory_tasks)
    speculative = _start_speculative_search(web_groups)

    # QueryTransformationNode: HyDE per SINGLE task, decomposer per MULTI task.
    if cached_turn is not None:
        trace.node("06_QueryTransformationNode", cached=True)
        sub_queries = cached_turn.sub_queries
    else:
        query_transformation_agent = build_query_transformation_agent(models.query_transformation)
        trace.node(
            "06_QueryTransformationNode",
            agent=query_transformation_agent,
            credential=models.generation_credential,
        )
        sub_queries = await transform_tasks(
            query_transformation_agent,
            advisory_tasks,
            decomposer_agent=build_decomposer_agent(models.query_transformation),
            confirmed_metadata=confirmed_metadata,
            history=graph_input.history,
            purpose="CHAT",
            credential=models.generation_credential,
            snapshot_version=models.snapshot_version,
            hyde_agent_factory=build_query_transformation_agent,
            decomposer_agent_factory=build_decomposer_agent,
            on_failover=_make_failover_applier(models),
            on_attempt=usage_recorder.bind("QueryTransformationNode"),
            budget=budget,
        )
        for sub_query in sub_queries:
            trace.prompt("06_QueryTransformationNode", sub_query.retrieval_text)
    # The standalone question of each sub-query (HyDE's rewrite, or the decomposed
    # question itself) - what the LLM rerank and the web judge judge against.
    questions = [extract_standalone_question(sub_query.retrieval_text) for sub_query in sub_queries]
    served = _group_questions(web_groups, sub_queries, advisory_tasks)
    judged_search = (
        asyncio.create_task(
            _search_then_judge(
                models,
                trace,
                usage_recorder,
                questions,
                speculative[0],
                [served[position] for position in speculative[1]],
                budget=budget,
            )
        )
        if speculative is not None and settings.CHAT_WEB_JUDGE_WITH_CHUNKS
        else None
    )
    if judged_search is not None:
        _finish_in_background(judged_search)
    # The standalone rewrite only applies to a single HyDE question.
    resolved_query = (
        extract_standalone_question(sub_queries[0].retrieval_text)
        if len(sub_queries) == 1
        else None
    )

    # RetrievalCacheNode, L2: the same standalone questions under the same scope were
    # retrieved and reranked before - replay the chunk ids (fetched again with the
    # caller's filter) and skip nodes 08, 09 and 09a.
    admin_warnings: list[AdminWarning] = []
    l2_lookup = None
    cached_result = cached_turn.rerank_result if cached_turn is not None else None
    if cached_result is None and cache_ctx is not None and graph_input.resume is None:
        l2_lookup = await lookup_l2(cache_ctx, questions, models.retrieval, graph_input.security)
        cached_result = l2_lookup.result

    if cached_result is not None:
        trace.node("08_RetrievalFilteringNode", cached=True)
        rerank_result = cached_result
        rerank_ok = False  # nothing new to store
    else:
        # RetrievalFilteringNode (permission pre-filter on every query).
        trace.node("08_RetrievalFilteringNode")
        per_query_chunks = await retrieve_chunks(
            [sub_query.retrieval_text for sub_query in sub_queries],
            models.retrieval,
            graph_input.security,
            # BM25 side of the hybrid search: the standalone questions, where a course code
            # or a document name appears verbatim.
            lexical_queries=questions,
        )

        # PostRetrievalRerankNode (per sub-query, then merged). With an LLM rerank ahead,
        # only the low dense floor applies here; the LLM is the gate.
        trace.node("09_PostRetrievalRerankNode")
        llm_rerank_follows = settings.CHAT_LLM_RERANK_ENABLED and models.rerank is not None
        judge_follows = llm_rerank_follows or settings.CHAT_RERANK_MODE != "llm"
        rerank_result = rerank_chunks(
            per_query_chunks,
            score_threshold=settings.CHAT_RETRIEVAL_DENSE_FLOOR if judge_follows else None,
        )
        rerank_ok = True

    # EvidenceSignals: how confident retrieval is per sub-query, logged every turn; `auto`
    # runs the LLM rerank only when a sub-query is weak (app/graph/evidence.py).
    evidence = levels_for([result.chunks for result in rerank_result.per_query])
    evidence_rows = [
        {"sub_query": index + 1, "level": level, **signals.as_dict()}
        for index, (signals, level) in enumerate(evidence)
    ]
    logger.info(
        "evidence mode=%s %s",
        settings.CHAT_RERANK_MODE,
        "; ".join(f"SQ{i + 1} {lvl} {sig.describe()}" for i, (sig, lvl) in enumerate(evidence)),
    )
    weak_evidence = any(level == "weak" for _, level in evidence)

    # CrossEncoderRerankNode (CHAT_RERANK_MODE cross_encoder / cross_encoder_then_llm,
    # docs/specs/SPEC-cross-encoder-rerank.md): each sub-query's top CHAT_CROSS_ENCODER_KEEP
    # by cross-encoder score. `cross_encoder` stops there; `cross_encoder_then_llm` hands
    # those to the LLM rerank. A service failure falls over to the LLM rerank with a
    # warning and skips the cache store: that result is not what this mode's key stands for.
    # `none` runs neither: the dense floor + retrieval order is the whole rerank stage.
    llm_rerank_wanted = settings.CHAT_LLM_RERANK_ENABLED and (
        settings.CHAT_RERANK_MODE not in ("none", "auto")
        or (settings.CHAT_RERANK_MODE == "auto" and weak_evidence)
    )
    if (
        cached_result is None
        and settings.CHAT_RERANK_MODE in ("cross_encoder", "cross_encoder_then_llm")
        and rerank_result.has_valid_context
    ):
        trace.node("09a_CrossEncoderRerankNode")
        ce_outcome = await cross_encoder_rerank(CrossEncoderClient(), questions, rerank_result)
        if ce_outcome.failure is None:
            rerank_result = ce_outcome.result
            llm_rerank_wanted = (
                settings.CHAT_RERANK_MODE == "cross_encoder_then_llm"
                and settings.CHAT_LLM_RERANK_ENABLED
            )
        else:
            rerank_ok = False
            admin_warnings.append(
                AdminWarning(
                    code="CROSS_ENCODER_FAILED",
                    message=f"{CROSS_ENCODER_SKIPPED}{ce_outcome.failure}",
                )
            )

    # LLMRerankNode: keep only the chunks that answer each sub-query (RERANK model).
    if cached_result is None and llm_rerank_wanted and rerank_result.has_valid_context:
        if models.rerank is not None:
            rerank_agent = build_llm_rerank_agent(models.rerank)
            trace.node("09a_LLMRerankNode", agent=rerank_agent, credential=models.rerank_credential)
            llm_rerank_outcome = await llm_rerank(
                rerank_agent,
                questions,
                rerank_result,
                credential=models.rerank_credential,
                purpose=models.rerank_purpose,
                snapshot_version=models.snapshot_version,
                on_attempt=usage_recorder.bind("LLMRerankNode"),
                budget=budget,
            )
            rerank_result = llm_rerank_outcome.result
            if llm_rerank_outcome.failure is not None:
                rerank_ok = False
                admin_warnings.append(
                    AdminWarning(
                        code="LLM_RERANK_FAILED",
                        message=f"{LLM_RERANK_SKIPPED}{llm_rerank_outcome.failure}",
                    )
                )
        elif models.rerank_unavailable is not None:
            rerank_ok = False
            admin_warnings.append(
                AdminWarning(
                    code="LLM_RERANK_UNAVAILABLE",
                    message=f"{LLM_RERANK_SKIPPED}{models.rerank_unavailable}",
                )
            )

    # RetrievalCacheNode, store: a fully successful miss is written in the background
    # (write-on-second-sighting applies inside the cache); nothing waits on Redis.
    if cache_ctx is not None and rerank_ok and rerank_result.has_valid_context:
        if l2_lookup is not None:
            _finish_in_background(
                asyncio.create_task(store_l2(cache_ctx, l2_lookup, rerank_result))
            )
        if l1_pending is not None:
            _finish_in_background(
                asyncio.create_task(
                    store_l1(
                        cache_ctx,
                        l1_pending,
                        [task for task, _mode in advisory_tasks],
                        sub_queries,
                        rerank_result,
                    )
                )
            )

    # WebSearchNode + WebJudgeNode: judged alongside rerank for every sub-query (fast-
    # changing topic), else only for the sub-queries left with no chunk.
    # Fallback scope: the policy decides from the evidence levels and what the rerank
    # emptied, apart from which rerank mode ran (app/graph/web_policy.py).
    web_decision = decide_fallback(
        [level for _, level in evidence], emptied_by_rerank=rerank_result.failed_query_indexes
    )
    failed = set(web_decision.sub_queries)
    web_results: list[WebSearchResult] = []
    web_sources: list[WebSearchResult] = []
    outcomes: list[WebSearchOutcome] = []
    judged: list[tuple[list[int], WebJudgement]] = []
    mode = ""
    if judged_search is not None:
        waited_at = time.perf_counter()
        parallel = await judged_search
        wait_ms = (time.perf_counter() - waited_at) * 1000
        outcomes = [parallel.outcome]
        for indexes, verdict in parallel.judged:
            if verdict.failure is not None and not failed & set(indexes):
                # Those sub-queries have chunks: no page picked by score alone goes next to
                # them. Without a RERANK model at all, LLM_RERANK_UNAVAILABLE already says so.
                verdict = WebJudgement(
                    answers=[], failure=verdict.failure if models.rerank is not None else None
                )
            judged.append((indexes, verdict))
        mode = f"parallel wait_ms={wait_ms:.0f}"
    elif failed and settings.CHAT_WEB_SEARCH_ENABLED:
        trace.node("09b_WebSearchNode")
        jobs, outcomes = await _search_for_failed(
            web_groups, served, failed, questions, speculative
        )
        for outcome in outcomes:
            for result in outcome.results:
                # The user sees each page the web judge is about to read.
                trace.web_page(result.url)
        trace.node("09c_WebJudgeNode")
        judged = await _judge_jobs(
            models,
            trace,
            usage_recorder,
            questions,
            jobs,
            budget=budget,
            alongside="09c_WebJudgeNode",
        )
        mode = "speculative" if speculative is not None else "fallback"
    elif speculative is not None:
        _finish_in_background(speculative[0])

    if outcomes:
        judgement = _merge_judgements(judged)
        web_results, web_sources = judgement.answers, judgement.sources
        if judged_search is not None and (web_results or web_sources):
            # Judged while rerank ran (no status then, it would interleave with
            # retrieval's): the user sees the pages the answer uses.
            trace.node("09b_WebSearchNode")
            for page in [*web_results, *web_sources]:
                trace.web_page(page.url)
        search_failure = next((o.failure for o in outcomes if o.failure is not None), None)
        if search_failure is not None:
            admin_warnings.append(
                AdminWarning(
                    code=search_failure.code,
                    message=f"{_WEB_SEARCH_FAILED}{search_failure}",
                )
            )
        if judgement.failure is not None:
            admin_warnings.append(
                AdminWarning(
                    code="WEB_JUDGE_FAILED", message=f"{_WEB_JUDGE_SKIPPED}{judgement.failure}"
                )
            )
        if web_results or web_sources:
            admin_warnings.append(
                AdminWarning(
                    code="WEB_SOURCE_USED",
                    message=_WEB_SOURCE_USED
                    + "; ".join(_page_reference(page) for page in [*web_results, *web_sources]),
                )
            )
        logger.info(
            "Web search turn: mode=%s topics=%s requests=%d cache_hits=%d candidates=%d "
            "answers=%d sources=%d sub_queries_with_chunks=%d/%d",
            mode,
            [group.request.topic for group in web_groups],
            sum(len(outcome.per_request) for outcome in outcomes),
            sum(outcome.cache_hits for outcome in outcomes),
            sum(len(outcome.results) for outcome in outcomes),
            len(web_results),
            len(web_sources),
            len(questions) - len(failed),
            len(questions),
        )

    # Barrier: the calculation blocks go out before any advisory token.
    calculation_titles_shown = (
        await before_generation(len(rerank_result.chunks) + len(web_results) + 1)
        if before_generation
        else []
    )

    if not rerank_result.has_valid_context and not web_results:
        # TicketFallbackNode (streaming).
        fallback_agent = build_ticket_fallback_agent(models.generation)
        trace.node(
            "11_TicketFallbackNode", agent=fallback_agent, credential=models.generation_credential
        )
        fallback_text = await run_ticket_fallback(
            fallback_agent,
            question,
            security=graph_input.security,
            confirmed_metadata=confirmed_metadata,
            history=graph_input.history,
            web_sources=web_sources,
            token_sink=token_sink,
            purpose="CHAT",
            credential=models.generation_credential,
            snapshot_version=models.snapshot_version,
            on_failover=_make_failover_applier(models),
            on_attempt=usage_recorder.bind("TicketFallbackNode"),
            budget=budget,
        )
        _, fallback_web = build_context_trace(fallback_text, [], [], web_sources)
        return GraphOutput(
            response_text=fallback_text,
            confirmed_metadata=confirmed_metadata,
            used_ticket_fallback=True,
            admin_warnings=admin_warnings,
            context_web=fallback_web,
        )

    # GenerationSynthesisNode (streaming).
    generation_agent = build_generation_agent(models.generation)
    trace.node(
        "10_GenerationSynthesisNode",
        agent=generation_agent,
        credential=models.generation_credential,
    )
    sub_query_questions = (
        [sub_query.question for sub_query in sub_queries] if len(sub_queries) > 1 else None
    )
    generation_result = await run_generation_synthesis(
        generation_agent,
        user_query=question,
        resolved_query=resolved_query,
        security=graph_input.security,
        confirmed_metadata=confirmed_metadata,
        chunks=rerank_result.chunks,
        web_results=web_results,
        history=graph_input.history,
        token_sink=token_sink,
        trace=trace,
        sub_queries=sub_query_questions,
        calculation_titles=calculation_titles_shown,
        strict_arithmetic=strict_numbers,
        purpose="CHAT",
        credential=models.generation_credential,
        snapshot_version=models.snapshot_version,
        on_failover=_make_failover_applier(models),
        on_attempt=usage_recorder.bind("GenerationSynthesisNode"),
        budget=budget,
    )
    response_text = keep_known_links(
        strip_source_list(generation_result.response_text), [*web_results, *web_sources]
    )
    # GroundingCheck: numbers and [n] of the answer against the sources it was shown
    # (chunks, then web pages, numbered in that order). The text is already streamed, so a
    # finding adds a caution line and an AI-admin warning rather than a rewrite.
    if settings.CHAT_GROUNDING_CHECK:
        shown = [chunk.content for chunk in rerank_result.chunks] + [
            page.content for page in web_results
        ]
        # Strict when the arithmetic of this turn belongs to code (a deferred calculation
        # question, or the answer requested calc blocks): a sum the model states itself is
        # ungrounded, not "derivable".
        strict = strict_numbers or bool(generation_result.calc_requests)
        report = check_answer(response_text, shown, question=question, strict=strict)
        if report.ungrounded:
            caution = caution_line(report.ungrounded)
            await token_sink(caution)
            response_text += caution
            admin_warnings.append(
                AdminWarning(
                    code="ANSWER_UNGROUNDED_NUMBERS",
                    message=f"{_UNGROUNDED}{', '.join(report.ungrounded)}",
                )
            )
        if report.invalid_markers:
            admin_warnings.append(
                AdminWarning(
                    code="ANSWER_INVALID_CITATION",
                    message=f"{_INVALID_CITATION}{report.invalid_markers}",
                )
            )
        elif (
            report.no_citation
            and numbers_in(response_text)
            and not looks_like_refusal(response_text)
        ):
            admin_warnings.append(AdminWarning(code="ANSWER_NO_CITATION", message=_NO_CITATION))
        logger.info(
            "grounding ungrounded=%s cited=%s invalid=%s no_citation=%s",
            report.ungrounded,
            report.cited,
            report.invalid_markers,
            report.no_citation,
        )
    # Plain arithmetic the answer asked for over figures it cited: computed by code and
    # appended as a Python block (or refused with a fixed line and an AI-admin warning).
    if generation_result.calc_requests:
        computed, rejected = arithmetic_blocks(
            generation_result.calc_requests, question=question, chunks=rerank_result.chunks
        )
        if computed:
            await token_sink(computed)
            response_text += computed
        admin_warnings.extend(rejected)
    # Official pages the judge kept only as links: listed after the answer, as the
    # fallback lists them, instead of being lost because another page answered.
    links = source_links_block(web_sources, response_text)
    if links:
        await token_sink(links)
        response_text += links
    context_chunks, context_web = build_context_trace(
        response_text, rerank_result.chunks, web_results, web_sources
    )
    return GraphOutput(
        response_text=response_text,
        confirmed_metadata=confirmed_metadata,
        ask_forms=generation_result.ask_forms,
        citations=build_citations(response_text, rerank_result.chunks, web_results),
        context_chunks=context_chunks,
        context_web=context_web,
        used_web_search=bool(web_results),
        admin_warnings=admin_warnings,
        evidence=evidence_rows,
    )
