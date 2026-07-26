from typing import cast

from pydantic_graph import GraphBuilder, StepContext

from app.graph.deps import ChatDeps
from app.graph.nodes.generate import generate_response
from app.graph.nodes.intent import detect_intent
from app.graph.nodes.rerank import rerank_context
from app.graph.nodes.retrieve import retrieve_context
from app.graph.state import ChatState

builder = GraphBuilder(
    state_type=ChatState,
    deps_type=ChatDeps,
    input_type=str,
    output_type=str,
)


@builder.step(label="intent")
async def intent_step(ctx: StepContext) -> str:
    state = cast(ChatState, ctx.state)
    query = cast(str, ctx.inputs) or state.query
    state.query = query
    state.intent = detect_intent(query)
    return query


@builder.step(label="retrieve")
async def retrieve_step(ctx: StepContext) -> str:
    state = cast(ChatState, ctx.state)
    deps = cast(ChatDeps, ctx.deps)
    retrieve_context(state, deps)
    return state.query


@builder.step(label="rerank")
async def rerank_step(ctx: StepContext) -> str:
    state = cast(ChatState, ctx.state)
    rerank_context(state)
    return state.query


@builder.step(label="generate")
async def generate_step(ctx: StepContext) -> str:
    state = cast(ChatState, ctx.state)
    return generate_response(state)


builder.add(builder.edge_from(builder.start_node).to(intent_step))
builder.add(builder.edge_from(intent_step).to(retrieve_step))
builder.add(builder.edge_from(retrieve_step).to(rerank_step))
builder.add(builder.edge_from(rerank_step).to(generate_step))
builder.add(builder.edge_from(generate_step).to(builder.end_node))

chat_graph = builder.build()
