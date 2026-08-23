# Spec: Ingestion Service (unisage-agent, Python)

## Objective

Add three independent, client-facing REST APIs to `unisage-agent` that implement
the document ingestion pipeline: **preview** (read raw text), **chunking**
(split into reviewable chunks with a chosen strategy), and **embedding**
(enrich + embed the client-approved chunks into Qdrant). The client calls all
three directly through the API Gateway. Java is not an orchestrator for this
flow — it only performs the earlier, out-of-scope step of storing the raw file
in MinIO and creating a `Document` metadata row.

Success = a KLTN defense-ready slice: three working APIs matching the
contracts below, embedding running asynchronously so real-time chat/query
embedding calls are not starved under load, and the untouched parts of
`unisage-agent` (chat/graph/retrieval/reranking/generation) still import and
run exactly as before.

This spec supersedes `Diagrams/architecture/arc.md` for the ingestion flow
(that document's Java-driven Kafka orchestration and PostgreSQL/pgvector vector
store are stale and not implemented here).

## Tech Stack

- Python 3.12, FastAPI (existing `unisage-agent` scaffold)
- `minio` (Python SDK) — internal docker-network MinIO client, own credentials
- `celery[redis]` + Redis — async task queue and result/progress backend for
  the embedding step (chosen over Kafka: single producer/single consumer here,
  no other service needs to subscribe to this event; Celery gives native
  priority queues so real-time query embedding can be prioritized over
  ingestion under load, and free progress state via `task.update_state` —
  Kafka would need both a broker *and* a hand-rolled state store to match this)
- `qdrant-client` — vector store for chunk + multi-representation vectors
- `openai` SDK — `text-embedding-3-small` for embeddings, `gpt-4o-mini` for the
  multi-representation summary/hypothetical-question generation step
- `tiktoken` (already a dependency) — token-based chunking
- `pymupdf4llm` (already a dependency) — PDF/DOCX text and table-as-markdown
  extraction
- `python-docx` or existing `pymupdf4llm` path for `.docx`/`.doc`
- `openpyxl` — `.xlsx` row-based chunking (Java does not yet whitelist XLSX
  uploads — see Open Questions / Known Gaps)
- FastAPI `WebSocket` — realtime embedding progress push (chosen over SSE per
  explicit user decision, to leave room for future cancel/pause commands)

## Commands

```
Dev:    uvicorn app.main:app --reload --port 8402
Worker: celery -A app.worker.celery_app worker --loglevel=info
Test:   pytest
Lint:   ruff check app tests
Format: ruff format --check app tests
Types:  mypy app tests
```

## Project Structure (additions/changes only)

```
app/
├── api/v1/
│   └── ingestion.py          # 3 endpoints: preview, chunking, embedding + WS route
├── core/
│   └── config.py             # + MinIO, Redis/Celery, Qdrant, OpenAI settings
├── rag/
│   ├── ingestion/
│   │   ├── minio_client.py   # object fetch by object_key
│   │   ├── parser.py         # per-filetype raw text extraction (preview)
│   │   └── table_aware_parser.py  # text/table region split (chunking only)
│   ├── chunking/
│   │   ├── recursive.py      # existing, reused
│   │   ├── semantic.py       # upgraded: real semantic chunking (~400 tok, 20% overlap)
│   │   ├── token_based.py    # new: tiktoken-based
│   │   ├── markdown_aware.py # new: heading/table-block aware
│   │   ├── excel_rows.py     # new: row-based .xlsx chunking
│   │   └── strategy.py       # new: strategy registry/dispatch + shared params schema
│   ├── embeddings/
│   │   ├── openai_embedder.py     # replaces huggingface.py as the active provider
│   │   └── provider.py            # new: EmbeddingProvider protocol (swap point)
│   ├── enrichment/
│   │   └── multi_representation.py  # new: LLM summary + hypothetical questions
│   └── vectorstore/
│       └── qdrant_store.py   # new: collection bootstrap + upsert
├── worker/
│   └── celery_app.py         # new: Celery app + embed_chunks task
└── schemas/
    └── ingestion.py          # new: Preview/Chunking/Embedding request+response models

app/api/deps.py                # + get_trusted_context (reads X-User-Department,
                                #   X-User-Access-Level headers injected by gateway)
app/core/security.py           # + verify_internal_secret (reads X-Internal-Secret,
                                #   applied at router level to every ingestion route)
```

## Code Style

Match existing conventions in the repo (see `AGENTS.md`): Python 3.12 syntax,
strict type hints, request/response models in `app/schemas`, provider-specific
code stays inside its own RAG stage module, no DDD layers/ports without a
second real implementation.

```python
class ChunkingStrategy(Protocol):
    """One chunking strategy: pure function from parsed regions to chunks."""

    def split(self, regions: list[ParsedRegion], params: ChunkingParams) -> list[Chunk]: ...
```

## Testing Strategy

- `pytest` + `pytest-asyncio`, tests under `tests/`, mirroring existing
  `tests/test_ingestion.py` naming.
- Unit tests per chunking strategy (pure functions, no I/O) — fixed input text
  → expected chunk boundaries/counts.
- Unit tests for the embedding-provider and multi-representation modules using
  mocked OpenAI client (no live API calls in CI).
- Integration test for preview/chunking endpoints against a MinIO test
  container or a mocked MinIO client.
- Celery task tested in eager mode (`task_always_eager=True`) to assert
  progress states are emitted in order and Qdrant upsert is called with the
  expected payload shape (mocked Qdrant client).
- No live external calls (OpenAI, Qdrant, MinIO) in the default test run.

## Boundaries

- **Always**: keep chat/graph/retrieval/reranking/generation modules
  untouched; keep embedding model swappable behind `EmbeddingProvider`; run
  `ruff`/`mypy`/`pytest` before considering a task done; never silently invent
  a Java-side change (XLSX whitelist, DocStatus callback) — flag instead.
- **Ask first**: removing the existing Postgres/SQLAlchemy/alembic plumbing
  (still used by out-of-scope modules — default is leave it alone, see Open
  Questions); changing the embedding model or multi-representation LLM;
  renaming the Qdrant collection/payload schema once chosen.
- **Never**: call back into the Java backend from Python; implement Kafka;
  cache preview/chunking file state between requests; re-derive chunks from
  `object_key` inside the embedding step (client-sent chunk list is the only
  input); accept a client-facing ingestion request without a valid
  `X-Internal-Secret` header (see Decision 6) — this is what makes the
  Gateway the only caller and the trusted headers in Decision 3 trustworthy.

## Success Criteria

- `POST /api/v1/ingestion/preview` returns raw text for a given `object_key`
  for txt/pdf/docx/doc files fetched from MinIO.
- `POST /api/v1/ingestion/chunking` returns a chunk list for any of the 5
  strategies, re-parsing (incl. table-aware split) from `object_key` on every
  call, no server-side caching.
- `POST /api/v1/ingestion/embedding` accepts a client-supplied chunk list,
  returns a `task_id` immediately (202), and a Celery worker performs
  multi-representation enrichment + OpenAI embedding + Qdrant upsert in the
  background.
- `WS /api/v1/ingestion/embedding/{task_id}/progress` streams percent-complete
  updates in realtime until the task finishes or fails.
- Existing chat/graph/retrieval endpoints and tests remain green,
  unmodified.
- `ruff check`, `ruff format --check`, `mypy`, and `pytest` all pass.

## Resolved Decisions (confirmed by user, superseding the earlier draft)

1. **Postgres/alembic plumbing**: confirmed dead code — `get_db_session` is
   threaded into `ChatDeps` but nothing queries it (`RetrievalService` uses an
   in-memory demo corpus; `DocumentRepository`/`ChunkRepository`/
   `ConversationRepository` are empty stub classes with no methods). **Leave
   untouched** — do not remove, do not add ingestion tables to it.

2. **Qdrant point/vector design**: **one point per original chunk**, using
   Qdrant **named vectors** so content, summary, and hypothetical-questions
   are each independently searchable on the same point (not separate points
   linked by `parent_chunk_id`):
   - Named vectors: `content_vector`, `summary_vector`, `questions_vector`
     (the joined/concatenated hypothetical questions embedded as one vector).
     Three OpenAI embedding calls per chunk.
   - Payload fields: `document_id`, `object_key`, `chunk_id`, `content`,
     `summary`, `questions` (`list[str]`), `department`, `access_level`,
     `region_type` (`"text"` | `"table"` | `"excel_row"`).
   - Collection name: `unisage_chunks`.

3. **`department` / `access_level` source**: read from **trusted headers
   injected by the API Gateway** after JWT verification (e.g.
   `X-User-Department`, `X-User-Access-Level`), the same Access-Context-
   Verifier pattern already implied by the system architecture — **not**
   passed in the request body, and **not** fetched by Python calling Java.
   The ingested document inherits the calling user's department/access level.
   A FastAPI dependency (e.g. `get_trusted_context`) reads and validates the
   presence of these headers for the embedding endpoint.

4. **Multi-representation LLM**: confirmed — `gpt-4o-mini` via OpenAI, one
   summary + 3 hypothetical questions per chunk, count configurable via env.

5. **Known gaps, documented not silently patched**: Java's `AllowedFileType`
   has no `.xlsx` yet (Excel chunking strategy is unreachable end-to-end until
   Java adds it); Java's `DocStatus` lifecycle (`PENDING → COMPLETED`) has no
   caller in this scope since Python never calls back to Java.

6. **Service-to-service auth: shared-secret gate, not just the trusted
   headers**: every route on the ingestion router (`preview`, `chunking`,
   `embedding`, and the WebSocket progress endpoint) requires a matching
   `X-Internal-Secret` header, verified by a router-level FastAPI dependency
   (`verify_internal_secret` in `app/core/security.py`) before any handler
   runs. The value is a shared secret configured on both sides — this
   service's `INTERNAL_SECRET_KEY` setting and the corresponding value the
   API Gateway sends on its `python-ai-agent-route` proxy filters — not a
   per-user credential. A request without it (or with the wrong value) is
   rejected with `403` before it ever reaches the handler, which is what
   makes the Gateway the only path into this service and is precisely what
   makes the `X-User-Department`/`X-User-Access-Level` headers in Decision 3
   trustworthy: only the Gateway holds the shared secret, so only the
   Gateway can be the one injecting those headers. `/api/v1/health` is
   intentionally exempt (not gated) so infra health checks don't need the
   secret. Same pattern already used in `KLTN-Academic-Agent-AI`'s
   `app/security.py`; ported here rather than reinvented.
   - **Settings**: `INTERNAL_SECRET_KEY` (default matches the Gateway
     route's own fallback, so local dev works out of the box — override
     both sides together in real deployments).
   - **Ask first**: rotating or removing the shared secret, or changing
     which routes it gates.
   - **Never**: accept an ingestion request without a valid
     `X-Internal-Secret`, even for local/dev convenience.
