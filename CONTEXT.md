# UniSage Agent Context

## Ownership

`unisage-agent` is the AI-specific service for UniSage. It handles document
ingestion, RAG retrieval, reranking, grounded generation, graph orchestration,
evaluation, and model configuration.

Authentication, users, roles, and general administration belong in
`unisage-backend`.

## Runtime Flow

```text
HTTP request
    -> API validation and sanitization
    -> Graph state
    -> intent detection
    -> retrieval with faculty and level metadata
    -> reranking
    -> grounded generation and citations
    -> HTTP response
```

## Key Contracts

- `ChatState` carries query, access metadata, retrieved chunks, response, and citations.
- `ChatDeps` carries request-scoped database and model dependencies.
- `RetrievedChunk` is the boundary between retrieval and generation.
- `ChatResponse` is the public chat response contract.

The current base uses deterministic fallback retrieval and generation so the
service can run without an external provider key. PostgreSQL/pgvector,
Hugging Face embeddings, and Pydantic AI generation can be connected behind
their existing stage boundaries as implementation work is added.
