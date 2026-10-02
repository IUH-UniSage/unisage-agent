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

from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai.models import Model

from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.core.registry.model_registry import CredentialConfig
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.nodes.calculation import CALCULATION_PLACEHOLDER_TEMPLATE
from app.graph.nodes.generation_synthesis import build_generation_agent, run_generation_synthesis
from app.graph.nodes.greeting import GREETING_TEMPLATE, detect_greeting
from app.graph.nodes.intent_routing import SOCIAL_CHAT_TEMPLATE, plan_route
from app.graph.nodes.message_classification import build_classification_agent, classify_intent
from app.graph.nodes.off_topic import OFF_TOPIC_TEMPLATE
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.graph.nodes.query_transformation import (
    build_decomposer_agent,
    build_query_transformation_agent,
    extract_standalone_question,
    transform_tasks,
)
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.graph.nodes.security_context import (
    ClarificationGuardResult,
    resolve_clarification_guard,
)
from app.graph.nodes.ticket_fallback import build_ticket_fallback_agent, run_ticket_fallback
from app.graph.nodes.web_search import search_web
from app.graph.streaming import BudgetContext, FailoverCallback, TokenSink
from app.graph.streaming_state import GraphInput, GraphModels, GraphOutput
from app.rag.prompting.citations import build_citations
from app.schemas.clarification import PendingClarification
from app.schemas.intent import ClassifiedTask, RoutingMode
from app.schemas.web_search import WebSearchResult

_ORIGIN_NODE_QUERY_TRANSFORMATION = "QueryTransformationNode"


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
    # GreetingDetectionNode: fast path, no LLM.
    trace.node("01_GreetingDetectionNode")
    if detect_greeting(graph_input.user_message, first_turn=graph_input.is_first_turn):
        await token_sink(GREETING_TEMPLATE)
        return GraphOutput(
            response_text=GREETING_TEMPLATE,
            confirmed_metadata=graph_input.confirmed_metadata,
        )

    # Clarification Guard (SecurityContextExtractionNode).
    trace.node("02_SecurityContextExtractionNode_ClarificationGuard")
    guard_result = resolve_clarification_guard(
        user_message=graph_input.user_message,
        pending=graph_input.pending_clarification,
        confirmed_metadata=graph_input.confirmed_metadata,
        max_retry=graph_input.clarification_max_retry,
    )
    confirmed_metadata = guard_result.confirmed_metadata
    pending_clarification = guard_result.pending_clarification

    if guard_result.route_to_origin or pending_clarification is not None:
        return await _run_advisory_flow(
            graph_input,
            models,
            token_sink,
            trace,
            usage_recorder,
            confirmed_metadata=confirmed_metadata,
            pending_clarification=pending_clarification,
            advisory_tasks=_resume_advisory_tasks(graph_input, guard_result),
            budget=budget,
        )

    # MessageClassificationNode.
    trace.node("03_MessageClassificationNode", model=models.classification)
    classification_agent = build_classification_agent(models.classification)
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

    # IntentRoutingNode (deterministic).
    trace.node("04_IntentRoutingNode")
    route_plan = plan_route(classification)

    if route_plan.end == "SOCIAL_CHAT":
        trace.node("04_IntentRouting_SocialChat")
        await token_sink(SOCIAL_CHAT_TEMPLATE)
        return GraphOutput(
            response_text=SOCIAL_CHAT_TEMPLATE,
            confirmed_metadata=confirmed_metadata,
            pending_clarification=pending_clarification,
        )

    if route_plan.end == "OFF_TOPIC":
        trace.node("05_OffTopicRejectNode")
        await token_sink(OFF_TOPIC_TEMPLATE)
        return GraphOutput(
            response_text=OFF_TOPIC_TEMPLATE,
            confirmed_metadata=confirmed_metadata,
            pending_clarification=pending_clarification,
        )

    if not route_plan.advisory_tasks:
        # Only calculation tasks: CalculationNode answers the whole turn.
        trace.node("07_CalculationNode")
        await token_sink(CALCULATION_PLACEHOLDER_TEMPLATE)
        return GraphOutput(
            response_text=CALCULATION_PLACEHOLDER_TEMPLATE,
            confirmed_metadata=confirmed_metadata,
            pending_clarification=pending_clarification,
        )

    # A mixed turn answers only the advisory question(s), so generation doesn't
    # try to answer the calculation part from regulations.
    advisory_question = (
        " ".join(task.query for task, _mode in route_plan.advisory_tasks)
        if route_plan.calculation_tasks
        else None
    )
    output = await _run_advisory_flow(
        graph_input,
        models,
        token_sink,
        trace,
        usage_recorder,
        confirmed_metadata=confirmed_metadata,
        pending_clarification=pending_clarification,
        advisory_tasks=route_plan.advisory_tasks,
        question=advisory_question,
        budget=budget,
    )
    if not route_plan.calculation_tasks:
        return output

    # CalculationNode placeholder, appended outside the LLM.
    trace.node("07_CalculationNode")
    calculation_part = f"\n\n{CALCULATION_PLACEHOLDER_TEMPLATE}"
    await token_sink(calculation_part)
    return replace(output, response_text=output.response_text + calculation_part)


def _single_advisory_task(query: str) -> tuple[ClassifiedTask, RoutingMode]:
    """A resume turn re-runs the original question as one SINGLE advisory task."""

    return (
        ClassifiedTask(intent="academic_advisory", query=query, routing_mode="SINGLE"),
        "SINGLE",
    )


def _resume_advisory_tasks(
    graph_input: GraphInput, guard_result: ClarificationGuardResult
) -> list[tuple[ClassifiedTask, RoutingMode]]:
    """Task(s) to re-run for a resume turn: several origin tasks are re-run on
    their own queries unchanged (splitting a reply across tasks is a known
    gap); a single task, or no recorded origin_tasks (legacy rows), folds
    the reply into the resume query first."""

    origin_tasks = guard_result.origin_tasks
    if origin_tasks and len(origin_tasks) > 1:
        return [(task, task.routing_mode or "SINGLE") for task in origin_tasks]

    query = _resume_retrieval_query(graph_input, guard_result) or graph_input.user_message
    if origin_tasks:
        task = origin_tasks[0].model_copy(update={"query": query})
        return [(task, task.routing_mode or "SINGLE")]
    return [_single_advisory_task(query)]


def _resume_retrieval_query(
    graph_input: GraphInput,
    guard_result: ClarificationGuardResult,
) -> str | None:
    """What to retrieve on while a clarification round is open.

    Matched reply: the reply is pure data (already folded in via
    `confirmed_metadata`), so the original question alone is the topic.

    Unmatched reply: ambiguous - it may be an answer the deterministic guard
    couldn't parse, or the student abandoning the form to ask something new.
    Searching on the original question alone would ignore a genuinely new
    question; searching on the reply alone loses the topic (the bug this
    whole mechanism exists to fix). Keeping both covers either case.
    """

    original_query = guard_result.original_query
    if not original_query:
        return None
    if guard_result.route_to_origin:
        return original_query
    return f"{original_query} {graph_input.user_message}"


async def _run_advisory_flow(
    graph_input: GraphInput,
    models: GraphModels,
    token_sink: TokenSink,
    trace: GraphTrace,
    usage_recorder: UsageRecorder,
    *,
    confirmed_metadata: dict[str, str],
    pending_clarification: PendingClarification | None,
    advisory_tasks: Sequence[tuple[ClassifiedTask, RoutingMode]],
    question: str | None = None,
    budget: BudgetContext | None = None,
) -> GraphOutput:
    """Advisory branch: query transformation → retrieval → rerank → web search
    for the sub-queries rerank left empty → generation (or ticket fallback
    when neither found anything). `question` is what gets answered; `None`
    means the whole user message."""

    question = question or graph_input.user_message

    # QueryTransformationNode: HyDE per SINGLE task, decomposer per MULTI task.
    trace.node("06_QueryTransformationNode", model=models.query_transformation)
    sub_queries = await transform_tasks(
        build_query_transformation_agent(models.query_transformation),
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
    per_query_chunks = retrieve_chunks(
        [sub_query.retrieval_text for sub_query in sub_queries],
        models.retrieval,
        graph_input.security,
    )

    # PostRetrievalRerankNode (per sub-query, then merged).
    trace.node("09_PostRetrievalRerankNode")
    rerank_result = rerank_chunks(per_query_chunks)

    # WebSearchNode: only the sub-queries rerank left with no chunk.
    failed_sub_queries = [
        sub_queries[index].retrieval_text for index in rerank_result.failed_query_indexes
    ]
    web_results: list[WebSearchResult] = []
    if failed_sub_queries and settings.CHAT_WEB_SEARCH_ENABLED:
        trace.node("09b_WebSearchNode")
        web_results = await search_web(failed_sub_queries)
        for result in web_results:
            trace.prompt("09b_WebSearchNode", f"{result.score:.2f} {result.url}")

    if not rerank_result.has_valid_context and not web_results:
        # TicketFallbackNode (streaming).
        trace.node("11_TicketFallbackNode", model=models.generation)
        fallback_agent = build_ticket_fallback_agent(models.generation)
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
            pending_clarification=pending_clarification,
            used_ticket_fallback=True,
        )

    # GenerationSynthesisNode (streaming).
    trace.node("10_GenerationSynthesisNode", model=models.generation)
    generation_agent = build_generation_agent(models.generation)
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
        previous_pending=pending_clarification,
        origin_node=_ORIGIN_NODE_QUERY_TRANSFORMATION,
        history=graph_input.history,
        token_sink=token_sink,
        trace=trace,
        advisory_tasks=[task for task, _mode in advisory_tasks],
        sub_queries=sub_query_questions,
        purpose="CHAT",
        credential=models.generation_credential,
        snapshot_version=models.snapshot_version,
        on_failover=_make_failover_applier(models),
        on_attempt=usage_recorder.bind("GenerationSynthesisNode"),
        budget=budget,
    )
    return GraphOutput(
        response_text=generation_result.response_text,
        confirmed_metadata=generation_result.confirmed_metadata,
        pending_clarification=generation_result.pending_clarification,
        citations=build_citations(
            generation_result.response_text, rerank_result.chunks, web_results
        ),
        used_web_search=bool(web_results),
    )
