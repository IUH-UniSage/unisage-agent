# Task List: Ingestion Service (unisage-agent)

Spec: `docs/specs/SPEC-ingestion.md` · Plan: `tasks/plan.md`

---

## Phase 0: Foundation

### Task 1: Dependencies + settings

**Description:** Add the new runtime dependencies and typed settings needed
by every later task: MinIO SDK, Celery+Redis, Qdrant client, OpenAI SDK,
openpyxl (Excel), python-docx (if `pymupdf4llm` doesn't cover `.docx`
cleanly). Extend `Settings` with MinIO endpoint/credentials/bucket, Redis
broker/result-backend URL, Qdrant host/port/collection name, OpenAI API
key/embedding model/chat model, and multi-representation question count.

**Acceptance criteria:**
- [x] `pyproject.toml` lists `minio`, `celery[redis]`, `qdrant-client`,
      `openai`, `openpyxl` (plus `python-docx` if needed) as dependencies
- [x] `Settings` exposes `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`,
      `MINIO_SECRET_KEY`, `MINIO_BUCKET`, `REDIS_URL`, `QDRANT_HOST`,
      `QDRANT_PORT`, `QDRANT_COLLECTION` (default `unisage_chunks`),
      `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`),
      `MULTI_REP_LLM_MODEL` (default `gpt-4o-mini`),
      `MULTI_REP_QUESTION_COUNT` (default `3`)
- [x] App still boots with defaults (no crash on import) when these env vars
      are unset in dev

**Verification:**
- [x] `pip install -e .` succeeds
- [x] `python -c "from app.core.config import settings; print(settings.QDRANT_COLLECTION)"` prints `unisage_chunks`
- [x] `mypy app` passes on the changed file

**Dependencies:** None

**Files likely touched:**
- `pyproject.toml`
- `app/core/config.py`

**Estimated scope:** Small (2 files)

---

### Task 2: MinIO client wrapper

**Description:** A thin, testable wrapper around the MinIO SDK exposing
`get_object_bytes(object_key: str) -> bytes`, using the internal
docker-network endpoint and Python's own credentials (Task 1's settings). No
presigned URLs — direct SDK access, matching Java's `FileServiceImpl`
pattern but as an independent client.

**Acceptance criteria:**
- [x] `get_object_bytes` raises a clear, typed exception (not a raw MinIO
      SDK exception) when the object doesn't exist
- [x] Client is constructed once (module-level or dependency-injected), not
      per-request

**Verification:**
- [x] `pytest tests/test_minio_client.py` — covers success path and
      not-found path, using a mocked `Minio` client (no real MinIO needed)
- [x] `mypy app/rag/ingestion/minio_client.py` passes

**Dependencies:** Task 1

**Files likely touched:**
- `app/rag/ingestion/minio_client.py`
- `tests/test_minio_client.py`

**Estimated scope:** Small (2 files)

---

### Task 3: Ingestion schemas skeleton

**Description:** Pydantic models for all 3 APIs: `PreviewRequest`
(`object_key`), `PreviewResponse` (`raw_text`); `ChunkingRequest`
(`object_key`, `strategy`, `params`), `Chunk` (`chunk_index`, `content`,
`region_type`), `ChunkingResponse` (`chunks: list[Chunk]`);
`EmbeddingRequest` (`chunks: list[Chunk]`), `EmbeddingAcceptedResponse`
(`task_id`). `strategy` is a `Literal`/enum of the 5 strategy names;
`params` is a strategy-agnostic dict validated further inside
`chunking/strategy.py` (Task 9).

**Acceptance criteria:**
- [x] All 5 chunking strategy names are valid `strategy` values
- [x] Invalid `region_type`/`strategy` values are rejected by Pydantic at the
      schema level (422, not a 500 downstream)

**Verification:**
- [x] `pytest tests/test_ingestion_schemas.py` — valid/invalid payload cases
      for each model
- [x] `mypy app/schemas/ingestion.py` passes

**Dependencies:** Task 1

**Files likely touched:**
- `app/schemas/ingestion.py`
- `tests/test_ingestion_schemas.py`

**Estimated scope:** Small (2 files)

---

## Checkpoint A: Foundation
- [x] `ruff check app tests` and `ruff format --check app tests` pass
- [x] `mypy app tests` passes
- [x] `pytest tests/test_minio_client.py tests/test_ingestion_schemas.py` green
- [x] Full `pytest` suite still green (no regression in chat/graph/retrieval tests)
- [ ] **Review with human before proceeding**

---

## Phase 1: API #1 — Preview

### Task 4: Per-filetype raw text parser

**Description:** A pure function `extract_raw_text(content: bytes, filename: str) -> str`
dispatching by extension: `.txt` → decode UTF-8; `.pdf`/`.docx`/`.doc` →
`pymupdf4llm` (or a docx-specific fallback if `pymupdf4llm` doesn't handle
`.doc`/`.docx` cleanly) to plain text (no markdown/table structure needed
here — that's Task 6, chunking-only). Unsupported extensions raise a typed
error.

**Acceptance criteria:**
- [x] Correct raw text extracted from one fixture file per supported
      extension (txt/pdf/docx/doc)
- [x] Unsupported extension raises a typed, catchable exception (not a bare
      `Exception`)

**Verification:**
- [x] `pytest tests/test_parser.py` — one test per filetype fixture +
      one unsupported-extension test
- [x] `mypy app/rag/ingestion/parser.py` passes

**Dependencies:** Task 1

**Files likely touched:**
- `app/rag/ingestion/parser.py`
- `tests/test_parser.py`
- `tests/fixtures/` (sample .txt/.pdf/.docx/.doc files)

**Estimated scope:** Medium (3 files + fixtures)

---

### Task 5: Preview endpoint

**Description:** `POST /api/v1/ingestion/preview` — validates
`PreviewRequest`, fetches bytes via `minio_client.get_object_bytes`, extracts
raw text via `parser.extract_raw_text`, returns `PreviewResponse`. Maps the
MinIO not-found exception to a 404, and unsupported-filetype to a 415/422.

**Acceptance criteria:**
- [x] Returns 200 + raw text for a valid `object_key` (mocked MinIO)
- [x] Returns 404 when the object doesn't exist
- [x] Router is registered under `/api/v1` in `app/main.py` if not already

**Verification:**
- [x] `pytest tests/test_ingestion_preview.py` — success + not-found +
      unsupported-filetype cases via FastAPI `TestClient`
- [x] Full `pytest` suite green (existing chat/graph tests unaffected)

**Dependencies:** Tasks 2, 3, 4

**Files likely touched:**
- `app/api/v1/ingestion.py`
- `app/main.py` (only if router prefix/registration needs adjusting)
- `tests/test_ingestion_preview.py`

**Estimated scope:** Small (2-3 files)

---

## Checkpoint B: Preview works end-to-end
- [x] `POST /api/v1/ingestion/preview` returns correct raw text for txt/pdf/docx/doc fixtures (mocked MinIO)
- [x] Full test suite green
- [ ] **Review with human before proceeding**

---

## Phase 2: API #2 — Chunking

### Task 6: Table-aware parser

**Description:** `split_regions(content: bytes, filename: str) -> list[ParsedRegion]`
where `ParsedRegion` has `region_type` (`"text"` | `"table"`) and `content`.
Uses `pymupdf4llm`'s markdown output (which preserves table structure as
markdown tables) to separate table blocks from surrounding text for
PDF/DOCX; `.txt`/`.xlsx` short-circuit (txt → one text region; xlsx handled
entirely by Task 9's row-based strategy, not this parser).

**Acceptance criteria:**
- [x] A fixture PDF/DOCX containing at least one table produces both
      `"table"` and `"text"` regions, in original document order
- [x] A fixture with no tables produces only `"text"` regions

**Verification:**
- [x] `pytest tests/test_table_aware_parser.py`
- [x] `mypy app/rag/ingestion/table_aware_parser.py` passes

**Dependencies:** Task 1 (uses `pymupdf4llm`, already a dependency)

**Files likely touched:**
- `app/rag/ingestion/table_aware_parser.py`
- `tests/test_table_aware_parser.py`
- `tests/fixtures/` (a table-containing PDF/DOCX fixture)

**Estimated scope:** Medium (3 files + fixture)

---

### Task 7: Recursive + token-based chunkers

**Description:** Extend the existing `RecursiveChunker` (character-based,
size+overlap — likely unchanged, just confirm it operates on a
`list[ParsedRegion]` → `list[Chunk]` interface) and add
`TokenBasedChunker` using `tiktoken` to count real tokens instead of
characters, with the same `chunk_size`/`overlap` parameter shape.

**Acceptance criteria:**
- [x] `RecursiveChunker.split` still passes its existing tests (no
      regression) and now returns `Chunk` objects carrying `region_type`
- [x] `TokenBasedChunker.split` produces chunks whose token count (per
      `tiktoken`) is within `chunk_size`, with the configured overlap

**Verification:**
- [x] `pytest tests/test_chunking_recursive.py tests/test_chunking_token_based.py`
- [x] `mypy app/rag/chunking/recursive.py app/rag/chunking/token_based.py` passes

**Dependencies:** Task 6 (consumes `ParsedRegion`)

**Files likely touched:**
- `app/rag/chunking/recursive.py`
- `app/rag/chunking/token_based.py`
- `tests/test_chunking_recursive.py`
- `tests/test_chunking_token_based.py`

**Estimated scope:** Medium (4 files)

---

### Task 8: Semantic chunker upgrade + markdown-aware chunker

**Description:** Replace the `split_by_paragraphs` stub with a real semantic
chunker: group sentences/paragraphs toward a ~400-token target with ~20%
overlap, using cosine similarity between adjacent sentence embeddings via
`EmbeddingProvider`/`OpenAIEmbedder` (Task 12) — **confirmed: this endpoint
makes synchronous OpenAI embedding calls during chunking**, accepted for
quality over the zero-API-call alternative. Add `MarkdownAwareChunker`:
text regions get semantic/recursive chunking, table regions become one
dedicated markdown-table chunk each (no further splitting).

**Acceptance criteria:**
- [x] Semantic chunker's output chunks average close to the configured
      token target on a representative fixture (documented tolerance, not
      exact)
- [x] Markdown-aware chunker emits exactly one chunk per table region,
      verbatim as markdown, and independently-chunked text regions

**Verification:**
- [x] `pytest tests/test_chunking_semantic.py tests/test_chunking_markdown_aware.py`
- [x] `mypy` passes on both files

**Dependencies:** Task 6, Task 7, **and Task 12** (`EmbeddingProvider` /
`OpenAIEmbedder`) — since semantic chunking now makes live OpenAI embedding
calls, **Task 12 must be implemented before this task**, out of numeric
order. Pull Task 12 forward in your working order (it has no other
dependency besides Task 1, so this is safe).

**Files likely touched:**
- `app/rag/chunking/semantic.py`
- `app/rag/chunking/markdown_aware.py`
- `tests/test_chunking_semantic.py`
- `tests/test_chunking_markdown_aware.py`

**Estimated scope:** Medium (4 files)

**⚠ Open question to flag to the user before/at this task:** does semantic
chunking calling OpenAI embeddings synchronously inside the chunking API
(sync, per-request) fit the latency budget, or should semantic chunking use
a local (non-API) similarity heuristic instead? Spec doesn't pin this down.

---

### Task 9: Excel row-based chunker + strategy registry

**Description:** `ExcelRowChunker` using `openpyxl` — each row (or a
configurable group of N rows) becomes one chunk, with the header row
prepended as context to each chunk. Add `chunking/strategy.py`: a registry
mapping the 5 strategy names to their chunker classes plus a
`ChunkingParams` validator per strategy (e.g. `chunk_size`/`overlap` for
recursive/token, `rows_per_chunk` for excel), and a single
`dispatch(strategy, params, regions_or_bytes) -> list[Chunk]` entry point.

**Acceptance criteria:**
- [x] A fixture `.xlsx` with N data rows + 1 header row produces the
      expected chunk count and each chunk includes the header
- [x] `dispatch("excel_row", ...)` on a non-xlsx file raises a clear typed
      error (strategy/filetype mismatch)
- [x] All 5 strategies are reachable through `dispatch`

**Verification:**
- [x] `pytest tests/test_chunking_excel_rows.py tests/test_chunking_strategy.py`
- [x] `mypy` passes on both new files

**Dependencies:** Tasks 6, 7, 8 (registry wires all previously-built
strategies together)

**Files likely touched:**
- `app/rag/chunking/excel_rows.py`
- `app/rag/chunking/strategy.py`
- `tests/test_chunking_excel_rows.py`
- `tests/test_chunking_strategy.py`
- `tests/fixtures/` (sample `.xlsx`)

**Estimated scope:** Medium (5 files incl. fixture)

---

### Task 10: Chunking endpoint

**Description:** `POST /api/v1/ingestion/chunking` — fetches the object from
MinIO fresh (no cache), runs `table_aware_parser.split_regions` (skipped for
`.xlsx`/`.txt` per Task 6), dispatches to the requested strategy via
`chunking/strategy.dispatch`, returns `ChunkingResponse`.

**Acceptance criteria:**
- [x] Each of the 5 strategies is callable end-to-end through this endpoint
      against an appropriate fixture file, returning a non-empty chunk list
- [x] Requesting `excel_row` against a non-xlsx `object_key` returns a 422
      with a clear message (not a 500)

**Verification:**
- [x] `pytest tests/test_ingestion_chunking.py` — one case per strategy +
      the mismatch case
- [x] Full `pytest` suite green

**Dependencies:** Tasks 2, 3, 9

**Files likely touched:**
- `app/api/v1/ingestion.py`
- `tests/test_ingestion_chunking.py`

**Estimated scope:** Small (2 files)

---

## Checkpoint C: All 5 chunking strategies verified
- [x] Each strategy produces correct, size/boundary-verified chunks on fixture inputs (text/table/excel)
- [x] `POST /api/v1/ingestion/chunking` returns the expected response shape for all 5 strategies
- [x] Full test suite green
- [ ] **Review with human before proceeding** (especially Task 8's flagged open question)

---

## Phase 3: API #3 — Embedding

### Task 11: Trusted-context dependency

**Description:** `get_trusted_context` FastAPI dependency reading
`X-User-Department` and `X-User-Access-Level` request headers (injected by
the API Gateway after JWT verification, per spec). Returns a small
`TrustedContext` dataclass/model; raises 400 if either header is missing or
malformed.

**Acceptance criteria:**
- [x] Present, valid headers → `TrustedContext(department=..., access_level=...)`
- [x] Missing either header → 400 with a message naming which header is missing

**Verification:**
- [x] `pytest tests/test_trusted_context.py`
- [x] `mypy app/api/deps.py` passes

**Dependencies:** Task 1

**Files likely touched:**
- `app/api/deps.py`
- `tests/test_trusted_context.py`

**Estimated scope:** Small (2 files)

---

### Task 12: EmbeddingProvider protocol + OpenAI implementation

**Description:** `EmbeddingProvider` `Protocol` with
`embed(texts: list[str]) -> list[list[float]]`. `OpenAIEmbedder`
implementation using `text-embedding-3-small` (batched calls). Remove
`app/rag/embeddings/huggingface.py` (non-functional stub, superseded) —
**flagged as a deletion in the plan; confirm before merging if this task
lands as-is.**

**Acceptance criteria:**
- [x] `OpenAIEmbedder.embed(["a", "b"])` returns two vectors of the
      configured dimension, using a mocked OpenAI client in tests (no live
      API calls)
- [x] Swapping providers requires only a config/DI change, no call-site
      changes (verified by a test double provider used in Task 15's tests)

**Verification:**
- [x] `pytest tests/test_embedding_provider.py`
- [x] `mypy app/rag/embeddings/provider.py app/rag/embeddings/openai_embedder.py` passes

**Dependencies:** Task 1

**Confirmed:** `huggingface.py` is deleted outright (not kept as an unused
alternative). **Confirmed:** semantic chunking (Task 8) depends on this
task's output — implement this one first if working out of numeric order.

**Files likely touched:**
- `app/rag/embeddings/provider.py`
- `app/rag/embeddings/openai_embedder.py`
- `app/rag/embeddings/huggingface.py` (deleted)
- `tests/test_embedding_provider.py`

**Estimated scope:** Medium (3-4 files)

---

### Task 13: Multi-representation enrichment

**Description:** `enrich_chunk(chunk: Chunk) -> EnrichedChunk` (adds
`summary: str` and `questions: list[str]`, length = configured
`MULTI_REP_QUESTION_COUNT`) via one `gpt-4o-mini` call per chunk (mocked in
tests).

**Acceptance criteria:**
- [x] Returns exactly `MULTI_REP_QUESTION_COUNT` questions and a non-empty
      summary for a fixture chunk, using a mocked LLM response
- [x] A malformed/short LLM response is handled without crashing the whole
      embedding task (documented fallback: e.g. empty questions list + a
      logged warning, not a raised exception that kills the batch)

**Verification:**
- [x] `pytest tests/test_multi_representation.py`
- [x] `mypy app/rag/enrichment/multi_representation.py` passes

**Dependencies:** Task 1

**Files likely touched:**
- `app/rag/enrichment/multi_representation.py`
- `tests/test_multi_representation.py`

**Estimated scope:** Small (2 files)

---

### Task 14: Qdrant vectorstore (named vectors)

**Description:** `qdrant_store.py`: `ensure_collection()` bootstraps
`unisage_chunks` with three named vectors (`content_vector`,
`summary_vector`, `questions_vector`) at the configured embedding dimension;
`upsert_chunk(point)` builds one point with the resolved payload
(`document_id`, `object_key`, `chunk_id`, `content`, `summary`,
`questions`, `department`, `access_level`, `region_type`) and the three
named vectors.

**Acceptance criteria:**
- [x] `ensure_collection()` is idempotent (safe to call every startup/task)
- [x] `upsert_chunk` builds the exact payload/vector-name shape from the
      spec, verified against a mocked Qdrant client's call arguments

**Verification:**
- [x] `pytest tests/test_qdrant_store.py`
- [x] `mypy app/rag/vectorstore/qdrant_store.py` passes

**Dependencies:** Task 1, Task 3 (payload shape from schemas)

**Files likely touched:**
- `app/rag/vectorstore/qdrant_store.py`
- `tests/test_qdrant_store.py`

**Estimated scope:** Medium (2 files, higher care needed — see plan's risk table)

---

### Task 15: Celery app + `embed_chunks` task

**Description:** `worker/celery_app.py` — Celery app configured with the
Redis broker/backend (Task 1 settings). `embed_chunks(task_id, chunks,
department, access_level)` task: for each chunk, call `enrich_chunk`
(Task 13) → embed all 3 representations (Task 12) → `upsert_chunk` (Task
14), calling `self.update_state(state="PROGRESS", meta={"percent": ...})`
after each chunk.

**Acceptance criteria:**
- [x] Run with `task_always_eager=True` + mocked enrichment/embedding/Qdrant:
      progress states observed strictly increasing to 100%, final state
      `SUCCESS`
- [x] One chunk's enrichment/embedding failure doesn't silently drop it —
      either retried or recorded per-chunk in the task result, not swallowed

**Verification:**
- [x] `pytest tests/test_embed_chunks_task.py`
- [x] `mypy app/worker/celery_app.py` passes

**Dependencies:** Tasks 12, 13, 14

**Files likely touched:**
- `app/worker/celery_app.py`
- `tests/test_embed_chunks_task.py`

**Estimated scope:** Medium (2 files, higher care — orchestration logic)

---

### Task 16: Embedding endpoint + WebSocket progress endpoint

**Description:** `POST /api/v1/ingestion/embedding` — depends on
`get_trusted_context` (Task 11); validates `EmbeddingRequest`; dispatches
`embed_chunks.delay(...)`; returns `202` + `EmbeddingAcceptedResponse(task_id=...)`.
`WS /api/v1/ingestion/embedding/{task_id}/progress` — accepts the
connection, polls `AsyncResult(task_id)` on a short interval, sends JSON
frames `{"percent": ..., "state": ...}`, closes on `SUCCESS`/`FAILURE`.

**Acceptance criteria:**
- [x] Missing trusted-context headers → 400, task never dispatched
- [x] Valid request → 202 + `task_id`
- [x] WebSocket client receives at least one progress frame and a terminal
      frame, then the connection closes cleanly

**Verification:**
- [x] `pytest tests/test_ingestion_embedding.py` — REST + WebSocket
      (FastAPI `TestClient` WebSocket support), Celery in eager mode
- [x] Full `pytest` suite green

**Dependencies:** Tasks 11, 15

**Files likely touched:**
- `app/api/v1/ingestion.py`
- `tests/test_ingestion_embedding.py`

**Estimated scope:** Medium (2 files)

---

## Checkpoint D: Embedding pipeline verified end-to-end (mocked)
- [x] `embed_chunks` task, run eagerly, calls enrichment → embed (3 named vectors) → Qdrant upsert with the confirmed payload shape, in order
- [x] Progress states observed via the task's `update_state` calls in the expected sequence
- [x] `POST /api/v1/ingestion/embedding` returns 202 + `task_id`; missing trusted headers returns 400
- [x] WebSocket test client receives progress frames and a terminal state
- [x] Full test suite green
- [ ] **Review with human before proceeding**

---

## Phase 4: Integration & Polish

### Task 17: Dev infra + docs note

**Description:** Add `qdrant` and `redis` services to the project's local
`docker-compose`/devcontainer config for manual dev/testing. Add a short
note (in `docs/specs/SPEC-ingestion.md` or a new `docs/specs/known-gaps.md`)
documenting the two external gaps: Java's `AllowedFileType` missing `.xlsx`,
and no caller updates Java's `DocStatus` after ingestion completes in this
scope.

**Acceptance criteria:**
- [x] `docker compose up qdrant redis` (or equivalent) starts both services
      locally
- [x] The two known gaps are written down somewhere discoverable, not just
      in chat history

**Verification:**
- [x] Manual: services start and are reachable on their configured ports
- [x] Doc file renders/reads correctly

**Dependencies:** None (can run anytime after Task 1)

**Files likely touched:**
- `.devcontainer/docker-compose.yml` (or project's compose file)
- `docs/specs/SPEC-ingestion.md` or `docs/specs/known-gaps.md`

**Estimated scope:** Small (2 files)

---

### Task 18: Full verification pass

**Description:** Run the complete quality gate across the whole project,
not just the new modules.

**Acceptance criteria:**
- [x] `ruff check app tests` — no errors
- [x] `ruff format --check app tests` — no diffs
- [x] `mypy app tests` — no errors
- [x] `pytest` — full suite green, including all pre-existing
      chat/graph/retrieval/reranking/generation tests, unmodified

**Verification:**
- [x] All four commands above run and pass in one sitting; paste/attach
      output as evidence

**Dependencies:** All previous tasks

**Files likely touched:** None (verification only)

**Estimated scope:** XS

---

## Checkpoint: Complete
- [x] All acceptance criteria across Tasks 1–17 met
- [x] `ruff check`, `ruff format --check`, `mypy`, `pytest` all pass (Task 18)
- [x] Existing chat/graph/retrieval endpoints/tests unmodified and green
- [x] Ready for human review
