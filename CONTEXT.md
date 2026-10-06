# UniSage Agent Context

Short orientation for anyone (or any agent) opening this repo. The rules are in
[`docs/product/PRODUCT.md`](docs/product/PRODUCT.md), the reasons in
[`docs/product/DECISIONS.md`](docs/product/DECISIONS.md), the full flow in
[`docs/architecture/rag-pipeline.md`](docs/architecture/rag-pipeline.md).

## Ownership

`unisage-agent` is the AI service of UniSage: chat over academic documents (RAG),
document ingestion into Qdrant, and the AI-side model plumbing (calling providers with
credentials read from the model registry, usage and cost recording).

Not here: authentication, users, roles, conversations and messages, and LLM
credential management. Those belong to `unisage-backend` (Java), reached through
`BackendJavaClient`.

## Runtime Flow

```text
API Gateway (JWT -> 5 trusted headers + X-Internal-Secret)
    -> POST /api/v1/chat/stream: validate (<= 2000 chars), sanitize, log suspected injection
    -> backend-java: read history, create USER message + ASSISTANT placeholder
    -> run_graph (app/graph/streaming_graph.py), streamed over SSE:
         greeting / clarification guard / classification / routing
         -> advisory: HyDE or sub-queries -> access-filtered dense search in Qdrant
            -> score threshold -> [optional LLM rerank] -> [optional web search]
            -> grounded generation with [n] citations, or ticket fallback
    -> backend-java: patch the final answer and citations
```

`run_graph` is a plain async function that branches with `if`, not a
`pydantic_graph.Graph`; node names exist for trace logs.

## Key Contracts

- `AcademicSecurityContext`: who is asking and which departments/levels they may read,
  rebuilt from gateway headers on every request.
- `GraphInput` / `GraphModels` / `GraphOutput` (`app/graph/streaming_state.py`): the
  orchestrator's input, the per-request models, and the answer + citations.
- `RetrievedChunk` (`app/schemas/retrieval.py`): the boundary between retrieval and
  generation.
- `PendingClarification` / `confirmed_metadata`: an open "ask the student for more" round,
  stored per conversation in this service's Postgres.
- Every error response carries a stable `code` that mirrors `ErrorCode.java` in
  `unisage-backend`.
