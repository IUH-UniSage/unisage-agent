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

from app.core.graph_trace import GraphTrace
from app.graph.nodes.generation_synthesis import build_generation_agent, run_generation_synthesis
from app.graph.nodes.greeting import GREETING_TEMPLATE, detect_greeting
from app.graph.nodes.intent_routing import SOCIAL_CHAT_TEMPLATE, route_intent
from app.graph.nodes.message_classification import build_classification_agent, classify_intent
from app.graph.nodes.off_topic import OFF_TOPIC_TEMPLATE
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.graph.nodes.query_transformation import (
    build_query_transformation_agent,
    extract_standalone_question,
    transform_query,
)
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.graph.nodes.security_context import (
    ClarificationGuardResult,
    resolve_clarification_guard,
)
from app.graph.nodes.ticket_fallback import build_ticket_fallback_response
from app.graph.streaming import TokenSink
from app.graph.streaming_state import GraphInput, GraphModels, GraphOutput
from app.rag.prompting.citations import build_citations
from app.schemas.clarification import PendingClarification

_ORIGIN_NODE_QUERY_TRANSFORMATION = "QueryTransformationNode"


async def run_graph(
    graph_input: GraphInput,
    models: GraphModels,
    token_sink: TokenSink,
    trace: GraphTrace,
) -> GraphOutput:
    # Node 01 - GreetingDetectionNode: Fast Path, zero LLM tokens.
    trace.node("01_GreetingDetectionNode")
    if detect_greeting(graph_input.user_message, first_turn=graph_input.is_first_turn):
        await token_sink(GREETING_TEMPLATE)
        return GraphOutput(
            response_text=GREETING_TEMPLATE,
            confirmed_metadata=graph_input.confirmed_metadata,
        )

    # Node 02 - Clarification Guard (part of SecurityContextExtractionNode).
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
            confirmed_metadata=confirmed_metadata,
            pending_clarification=pending_clarification,
            resume_original_query=_resume_retrieval_query(graph_input, guard_result),
        )

    # Node 03 - MessageClassificationNode.
    trace.node("03_MessageClassificationNode")
    classification_agent = build_classification_agent(models.classification)
    intent = await classify_intent(
        classification_agent, graph_input.user_message, history=graph_input.history
    )

    # Node 04 - IntentRoutingNode (deterministic).
    trace.node("04_IntentRoutingNode")
    route = route_intent(intent)

    if route == "END_SOCIAL_CHAT":
        trace.node("05B_SocialChat")
        await token_sink(SOCIAL_CHAT_TEMPLATE)
        return GraphOutput(
            response_text=SOCIAL_CHAT_TEMPLATE,
            confirmed_metadata=confirmed_metadata,
            pending_clarification=pending_clarification,
        )

    if route == "OffTopicRejectNode":
        trace.node("05B_OffTopicRejectNode")
        await token_sink(OFF_TOPIC_TEMPLATE)
        return GraphOutput(
            response_text=OFF_TOPIC_TEMPLATE,
            confirmed_metadata=confirmed_metadata,
            pending_clarification=pending_clarification,
        )

    # route == "QueryTransformationNode": the unified advisory/procedure/document/calendar flow.
    return await _run_advisory_flow(
        graph_input,
        models,
        token_sink,
        trace,
        confirmed_metadata=confirmed_metadata,
        pending_clarification=pending_clarification,
        resume_original_query=guard_result.original_query,
    )


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
    *,
    confirmed_metadata: dict[str, str],
    pending_clarification: PendingClarification | None,
    resume_original_query: str | None,
) -> GraphOutput:
    # Node 06 - QueryTransformationNode (HyDE).
    trace.node("06_QueryTransformationNode")
    query_transformation_agent = build_query_transformation_agent(models.query_transformation)
    retrieval_query = resume_original_query or graph_input.user_message
    hyde_doc = await transform_query(
        query_transformation_agent,
        retrieval_query,
        confirmed_metadata=confirmed_metadata,
        history=graph_input.history,
    )
    trace.prompt("06_QueryTransformationNode_HyDE", hyde_doc)
    resolved_query = extract_standalone_question(hyde_doc)

    # Node 10 - RetrievalFilteringNode (no permission filter yet).
    trace.node("10_RetrievalFilteringNode")
    chunks = retrieve_chunks(hyde_doc, models.retrieval)

    # Node 11 - PostRetrievalRerankNode.
    trace.node("11_PostRetrievalRerankNode")
    rerank_result = rerank_chunks(chunks)

    if not rerank_result.has_valid_context:
        # Node 13 - TicketFallbackNode.
        trace.node("13_TicketFallbackNode")
        fallback = build_ticket_fallback_response(graph_input.user_message)
        await token_sink(fallback.message)
        return GraphOutput(
            response_text=fallback.message,
            confirmed_metadata=confirmed_metadata,
            pending_clarification=pending_clarification,
            used_ticket_fallback=True,
        )

    # Node 12 - GenerationSynthesisNode (streaming fan-in).
    trace.node("12_GenerationSynthesisNode")
    generation_agent = build_generation_agent(models.generation)
    generation_result = await run_generation_synthesis(
        generation_agent,
        user_query=graph_input.user_message,
        resolved_query=resolved_query,
        security=graph_input.security,
        confirmed_metadata=confirmed_metadata,
        chunks=rerank_result.chunks,
        previous_pending=pending_clarification,
        origin_node=_ORIGIN_NODE_QUERY_TRANSFORMATION,
        history=graph_input.history,
        token_sink=token_sink,
        trace=trace,
    )
    return GraphOutput(
        response_text=generation_result.response_text,
        confirmed_metadata=generation_result.confirmed_metadata,
        pending_clarification=generation_result.pending_clarification,
        citations=build_citations(generation_result.response_text, rerank_result.chunks),
    )
