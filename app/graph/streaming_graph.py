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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace

from pydantic import JsonValue
from pydantic_ai.models import Model

from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace, bind_trace, unbind_trace
from app.core.registry.model_registry import CredentialConfig
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.calculation_turn import (
    CalculationTask,
    calculation_titles,
    commentary,
    needs_input_parts,
    render_outcomes,
    trace_items,
)
from app.graph.clarification_round import (
    MAX_CHAIN_DEPTH,
    TaskQuestions,
    advisory_questions,
    advisory_task,
    build_round,
    unanswered_note,
)
from app.graph.nodes.calculation import (
    CalculationDeps,
    Computed,
    TaskOutcome,
    resume_calculation,
    run_calculation_task,
)
from app.graph.nodes.generation_synthesis import build_generation_agent, run_generation_synthesis
from app.graph.nodes.greeting import GREETING_TEMPLATE, detect_greeting
from app.graph.nodes.intent_routing import plan_route
from app.graph.nodes.llm_rerank import build_llm_rerank_agent, llm_rerank
from app.graph.nodes.message_classification import (
    build_classification_agent,
    classify_intent,
    describe_classification,
)
from app.graph.nodes.off_topic import off_topic_reply
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.graph.nodes.query_transformation import (
    build_decomposer_agent,
    build_query_transformation_agent,
    extract_standalone_question,
    transform_tasks,
)
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.graph.nodes.social_chat import social_chat_reply
from app.graph.nodes.ticket_fallback import build_ticket_fallback_agent, run_ticket_fallback
from app.graph.nodes.web_search import search_web
from app.graph.streaming import BudgetContext, FailoverCallback, TokenSink
from app.graph.streaming_state import (
    AdminWarning,
    GraphInput,
    GraphModels,
    GraphOutput,
    ResumeInput,
)
from app.rag.prompting.citations import build_citations
from app.schemas.clarification import PendingCalculationTask, PendingRound
from app.schemas.intent import ClassifiedTask, RoutingMode
from app.schemas.web_search import WebSearchResult

# Prefixes of the AI-admin warnings (`event: warning`) this module raises.
_LLM_RERANK_SKIPPED = "Bỏ qua bước lọc độ liên quan của tài liệu (mô hình Extraction): "
_WEB_SEARCH_FAILED = (
    "Tìm kiếm web thất bại nên câu trả lời không dùng được nguồn từ website Trường: "
)


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


async def run_graph(
    graph_input: GraphInput,
    models: GraphModels,
    token_sink: TokenSink,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    budget: BudgetContext | None = None,
) -> GraphOutput:
    # Bound for the whole run so a failover deep inside an LLM helper can log which
    # model/credential the node moved to (see `GraphTrace.model_switch`).
    token = bind_trace(trace)
    try:
        return await _run_graph(
            graph_input, models, token_sink, trace, usage_recorder, budget=budget
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
) -> GraphOutput:
    # A panel-submit turn resumes the claimed round - never a greeting or a new question.
    if graph_input.resume is not None:
        return await _run_resume(
            graph_input, graph_input.resume, models, token_sink, trace, usage_recorder, budget
        )

    # GreetingDetectionNode: fast path, no LLM.
    trace.node("01_GreetingDetectionNode")
    if detect_greeting(graph_input.user_message, first_turn=graph_input.is_first_turn):
        await token_sink(GREETING_TEMPLATE)
        return GraphOutput(
            response_text=GREETING_TEMPLATE,
            confirmed_metadata=graph_input.confirmed_metadata,
        )

    confirmed_metadata = graph_input.confirmed_metadata

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
    )
    trace.prompt("03_MessageClassificationNode", describe_classification(classification))

    # IntentRoutingNode (deterministic).
    trace.node("04_IntentRoutingNode")
    route_plan = plan_route(classification)

    if route_plan.end == "SOCIAL_CHAT":
        trace.node("04_IntentRouting_SocialChat")
        social_text = social_chat_reply(graph_input.user_message)
        await token_sink(social_text)
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

    # A mixed turn answers only the advisory question(s), so generation doesn't
    # try to answer the calculation part from regulations.
    advisory_question = (
        " ".join(task.query for task, _mode in route_plan.advisory_tasks)
        if route_plan.calculation_tasks
        else None
    )
    calculation_tasks = [
        CalculationTask(task_id=f"T{index}", query=task.query)
        for index, task in enumerate(route_plan.calculation_tasks, start=1)
    ]
    deps = _calculation_deps(graph_input, models, usage_recorder, budget)
    advisory: AdvisoryPart | None = None
    if route_plan.advisory_tasks:
        advisory = AdvisoryPart(
            task_id=f"T{len(calculation_tasks) + 1}",
            tasks=list(route_plan.advisory_tasks),
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
        calculations=_start_calculations(calculation_tasks, deps, trace),
        queries={task.task_id: task.query for task in calculation_tasks},
        deps=deps,
        advisory=advisory,
        original_query=advisory_question or graph_input.user_message,
        previous_round=None,
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
    answers already validated against the stored panel. Advisory answers become
    confirmed_metadata and the original question is re-answered; calculation answers
    complete the stored parameters - no extractor call, no retrieval."""

    trace.node("02_ClarificationResume")
    pending = resume.pending_round
    confirmed_metadata = dict(graph_input.confirmed_metadata)
    answers_by_task: dict[str, dict[str, JsonValue]] = {}
    for answer in resume.answers.values():
        question = answer.question
        answers_by_task.setdefault(question.task_id, {})[question.field] = answer.value
        if question.origin == "advisory" and isinstance(answer.value, str):
            confirmed_metadata[question.field] = answer.value

    outcomes: list[TaskOutcome] = []
    queries: dict[str, str] = {}
    advisory: AdvisoryPart | None = None
    for task in pending.tasks:
        if isinstance(task, PendingCalculationTask):
            queries[task.task_id] = task.query
            outcomes.append(
                resume_calculation(
                    task.task_id,
                    task.plan,
                    task.known_params,
                    answers_by_task.get(task.task_id, {}),
                )
            )
        else:
            advisory = AdvisoryPart(
                task_id=task.task_id,
                tasks=[(origin, origin.routing_mode or "SINGLE") for origin in task.origin_tasks],
                question=pending.original_query,
            )

    deps = _calculation_deps(graph_input, models, usage_recorder, budget)
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
) -> CalculationDeps:
    return CalculationDeps(
        models=models,
        security=graph_input.security,
        on_attempt=usage_recorder.bind("CalculationNode"),
        on_failover=_make_failover_applier(models),
        budget=budget,
    )


def _start_calculations(
    tasks: Sequence[CalculationTask], deps: CalculationDeps, trace: GraphTrace
) -> "asyncio.Task[list[TaskOutcome]] | None":
    """Calculation tasks run alongside the advisory retrieval (they are awaited only at
    the barrier right before node 10)."""

    if not tasks:
        return None
    trace.node("07_CalculationNode")

    async def run_all() -> list[TaskOutcome]:
        return list(
            await asyncio.gather(
                *(run_calculation_task(task.task_id, task.query, deps) for task in tasks)
            )
        )

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
) -> GraphOutput:
    """Stream order: calculation blocks (Python) → advisory answer (node 10, which only
    starts after the blocks were sent) or, for a calculation-only turn, the checked
    note → one panel for everything still missing."""

    outcomes: list[TaskOutcome] = []
    shown: list[str] = []

    async def show_calculations() -> list[str]:
        if calculations is not None:
            outcomes.extend(await calculations)
        text = render_outcomes(outcomes)
        if text:
            text += "\n\n" if advisory is not None else ""
            shown.append(text)
            await token_sink(text)
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
            )
            output = replace(output, response_text="".join(shown) + output.response_text)
        else:
            await show_calculations()
            text = "".join(shown)
            results = [o.result for o in outcomes if isinstance(o, Computed)]
            if results:
                note = await commentary(results, graph_input.user_message, deps)
                if note:
                    note = f"\n\n{note}"
                    await token_sink(note)
                    text += note
            output = GraphOutput(response_text=text, confirmed_metadata=confirmed_metadata)
    finally:
        if calculations is not None and not calculations.done():
            calculations.cancel()

    parts = needs_input_parts(outcomes, queries)
    if advisory is not None:
        parts.append(
            TaskQuestions(
                task=advisory_task(advisory.task_id, [task for task, _mode in advisory.tasks]),
                questions=advisory_questions(
                    output.ask_forms, confirmed_metadata=output.confirmed_metadata
                ),
            )
        )
    if previous_round is not None:
        output = await _with_follow_up_round(output, parts, previous_round, token_sink)
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
        )
        output = replace(output, calculation_items=public, calculation_traces=private)
    return output


async def _with_follow_up_round(
    output: GraphOutput,
    parts: list[TaskQuestions],
    previous: PendingRound,
    token_sink: TokenSink,
) -> GraphOutput:
    """Chain another panel, unless MAX_CHAIN_DEPTH panels were already asked for this
    question - then say what is still missing instead of asking again."""

    depth = previous.chain_depth + 1
    if depth > MAX_CHAIN_DEPTH:
        note = unanswered_note(parts)
        if not note:
            return output
        await token_sink(note)
        return replace(output, response_text=output.response_text + note)
    follow_up = build_round(parts, original_query=previous.original_query, chain_depth=depth)
    return replace(output, pending_round=follow_up)


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
    before_generation: Callable[[], Awaitable[list[str]]] | None = None,
) -> GraphOutput:
    """Advisory branch: query transformation → retrieval → rerank → web search
    for the sub-queries rerank left empty → generation (or ticket fallback
    when neither found anything). `question` is what gets answered; `None`
    means the whole user message."""

    question = question or graph_input.user_message

    # QueryTransformationNode: HyDE per SINGLE task, decomposer per MULTI task.
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
    # The standalone rewrite only applies to a single HyDE question.
    resolved_query = (
        extract_standalone_question(sub_queries[0].retrieval_text)
        if len(sub_queries) == 1
        else None
    )

    # RetrievalFilteringNode (permission pre-filter on every query).
    trace.node("08_RetrievalFilteringNode")
    per_query_chunks = await retrieve_chunks(
        [sub_query.retrieval_text for sub_query in sub_queries],
        models.retrieval,
        graph_input.security,
    )

    # PostRetrievalRerankNode (per sub-query, then merged).
    trace.node("09_PostRetrievalRerankNode")
    rerank_result = rerank_chunks(per_query_chunks)
    # The standalone question of each sub-query (HyDE's rewrite, or the decomposed
    # question itself) - what the LLM rerank judges against and what web search looks up.
    questions = [extract_standalone_question(sub_query.retrieval_text) for sub_query in sub_queries]
    admin_warnings: list[AdminWarning] = []

    # LLMRerankNode: keep only the chunks that answer each sub-query (RERANK model).
    if settings.CHAT_LLM_RERANK_ENABLED and rerank_result.has_valid_context:
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
                admin_warnings.append(
                    AdminWarning(
                        code="LLM_RERANK_FAILED",
                        message=f"{_LLM_RERANK_SKIPPED}{llm_rerank_outcome.failure}",
                    )
                )
        elif models.rerank_unavailable is not None:
            admin_warnings.append(
                AdminWarning(
                    code="LLM_RERANK_UNAVAILABLE",
                    message=f"{_LLM_RERANK_SKIPPED}{models.rerank_unavailable}",
                )
            )

    # WebSearchNode: only the sub-queries left with no chunk.
    failed_sub_queries = [questions[index] for index in rerank_result.failed_query_indexes]
    web_results: list[WebSearchResult] = []
    if failed_sub_queries and settings.CHAT_WEB_SEARCH_ENABLED:
        trace.node("09b_WebSearchNode")
        web_outcome = await search_web(failed_sub_queries)
        web_results = web_outcome.results
        if web_outcome.failure is not None:
            admin_warnings.append(
                AdminWarning(
                    code=web_outcome.failure.code,
                    message=f"{_WEB_SEARCH_FAILED}{web_outcome.failure}",
                )
            )
        for result in web_results:
            trace.prompt("09b_WebSearchNode", f"{result.score:.2f} {result.url}\n{result.content}")

    # Barrier: the calculation blocks go out before any advisory token.
    calculation_titles_shown = await before_generation() if before_generation else []

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
            token_sink=token_sink,
            purpose="CHAT",
            credential=models.generation_credential,
            snapshot_version=models.snapshot_version,
            on_failover=_make_failover_applier(models),
            on_attempt=usage_recorder.bind("TicketFallbackNode"),
            budget=budget,
        )
        return GraphOutput(
            response_text=fallback_text,
            confirmed_metadata=confirmed_metadata,
            used_ticket_fallback=True,
            admin_warnings=admin_warnings,
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
        purpose="CHAT",
        credential=models.generation_credential,
        snapshot_version=models.snapshot_version,
        on_failover=_make_failover_applier(models),
        on_attempt=usage_recorder.bind("GenerationSynthesisNode"),
        budget=budget,
    )
    return GraphOutput(
        response_text=generation_result.response_text,
        confirmed_metadata=confirmed_metadata,
        ask_forms=generation_result.ask_forms,
        citations=build_citations(
            generation_result.response_text, rerank_result.chunks, web_results
        ),
        used_web_search=bool(web_results),
        admin_warnings=admin_warnings,
    )
