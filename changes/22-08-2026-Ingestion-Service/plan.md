# Implementation Plan: Ingestion Service (unisage-agent)

## Overview

Add three independent REST APIs — preview, chunking, embedding — to the
existing `unisage-agent` FastAPI scaffold, per
`docs/specs/SPEC-ingestion.md`. Built as three vertical slices (one per API)
on top of a shared foundation (config, MinIO client, schemas), finishing with
infra/polish. Chat/graph/retrieval/reranking/generation modules and the
Postgres/alembic plumbing are not touched.

## Architecture Decisions

- **Vertical slicing by API, not by layer**: Preview ships first (smallest,
  validates MinIO connectivity end-to-end), then Chunking (reuses preview's
  parsing primitives, adds the table-aware split + 5 strategies), then
  Embedding (largest — Celery, Qdrant, OpenAI, WebSocket all land together
  since they're only meaningful combined).
- **No caching between steps**: chunking always re-fetches + re-parses from
  MinIO (per spec — chunking logic differs too much per content type to make
  a shared cache safe). Embedding never re-derives chunks — it only accepts
  the client-sent list. This removes an entire class of cache-invalidation
  work from the plan.
- **Celery in eager mode for tests**: no live Redis/Qdrant/OpenAI in CI. A
  local `docker-compose` addition provides Redis + Qdrant for manual/dev
  runs; OpenAI calls are always mocked in the test suite.
- **`EmbeddingProvider` protocol first, OpenAI implementation second**: keeps
  the self-host migration path (`ura-embed-v2`) a config swap, not a rewrite,
  per the spec's explicit requirement.
- **`huggingface.py` embedding stub removed, not kept as a dead alternative**:
  it's non-functional (raises `RuntimeError`) and superseded by
  `EmbeddingProvider` + `openai_embedder.py`; keeping it would just be a
  second, misleading "embeddings" entry point. **Confirmed by user.**
- **Semantic chunking calls OpenAI embeddings synchronously**: confirmed
  over the zero-API-call local-heuristic alternative, for quality. This
  means `EmbeddingProvider`/`OpenAIEmbedder` (originally Task 12, Phase 3)
  is a **Phase 2 dependency** — build it right after Task 1, ahead of its
  numeric position, since Task 8 (semantic chunker) needs it. Tasks are not
  renumbered to avoid a large diff; the task list notes this explicitly at
  both ends of the dependency.

## Dependency Graph

```
config (MinIO/Celery/Qdrant/OpenAI settings)
    │
    ├── minio_client.py ──────────────┐
    │                                 │
    ├── schemas/ingestion.py ─────────┼── Preview endpoint (Phase 1)
    │                                 │
    │   parser.py (per-filetype) ─────┘
    │        │
    │        ▼
    │   table_aware_parser.py
    │        │
    │        ▼
    │   chunking/{recursive,semantic,token_based,markdown_aware,excel_rows}
    │        │
    │        ▼
    │   chunking/strategy.py (registry) ──── Chunking endpoint (Phase 2)
    │
    ├── embeddings/provider.py → openai_embedder.py ─┐
    ├── enrichment/multi_representation.py ──────────┼── worker/celery_app.py
    ├── vectorstore/qdrant_store.py ──────────────────┘        │
    ├── api/deps.py (get_trusted_context)                      │
    │                                                            ▼
    └──────────────────────────────────────────── Embedding + WS endpoint (Phase 3)
```

Implementation order follows this graph top-to-bottom, left-to-right within
each phase.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| OpenAI cost/latency/flakiness leaking into tests | Med | Every test mocks the OpenAI client; no live calls in the default `pytest` run (checked in Task 18). |
| Qdrant named-vector collection config wrong (vector names/dimensions mismatch) | High | `qdrant_store.py` owns collection bootstrap as one function, unit-tested against a mocked client asserting the exact `vectors_config` dict; a one-time manual check against a real local Qdrant is in Task 17. |
| Semantic chunking quality is genuinely hard to get right in v1 | Med | Ship a documented heuristic (OpenAI-embedding cosine-similarity grouping toward ~400 tokens / 20% overlap) — explicitly not claimed to be state-of-the-art; acceptance criteria only check boundary/size correctness, not semantic quality. |
| Chunking endpoint (sync) now makes live OpenAI calls (semantic strategy only) | Med | Scoped to one document per request, not a batch; if latency becomes a problem later, swap to a local heuristic behind the same `ChunkingStrategy` interface — no call-site changes needed. Non-semantic strategies (4 of 5) make zero API calls. |
| Gateway may not yet inject `X-User-Department`/`X-User-Access-Level` in practice | High (blocks embedding endpoint in real use) | `get_trusted_context` fails loudly (400) when headers are missing rather than silently defaulting — makes the gap visible immediately instead of writing bad payloads to Qdrant. Documented in Task 17's notes. |
| Celery/Redis not running in a dev environment | Low | Tests use `task_always_eager=True`; `docker-compose` addition (Task 17) gives a one-command local Redis+Qdrant. |
| Removing `huggingface.py` turns out to be unwanted | Low | Called out explicitly above and in Task 12 — easy to revert (single file) if the user objects. |
| Excel chunking strategy unreachable end-to-end (Java whitelist gap) | Low (documented, not our bug) | Task 17 adds a README/spec note; Task 9's tests exercise `excel_rows.py` directly with a fixture `.xlsx`, independent of the upload path. |

## Task List

Tracked in `tasks/todo.md` (this project has no external tracker configured).

### Phase 0: Foundation
- Task 1: Dependencies + settings
- Task 2: MinIO client wrapper
- Task 3: Ingestion schemas skeleton

### Checkpoint A: Foundation
- [ ] `ruff check`, `ruff format --check`, `mypy` pass
- [ ] `pytest tests/test_minio_client.py tests/test_ingestion_schemas.py` green
- [ ] Existing chat/graph/retrieval tests unaffected (`pytest` full suite still green)

### Phase 1: API #1 — Preview
- Task 4: Per-filetype raw text parser
- Task 5: Preview endpoint

### Checkpoint B: Preview works end-to-end
- [ ] `POST /api/v1/ingestion/preview` returns correct raw text for txt/pdf/docx/doc fixtures (mocked MinIO)
- [ ] Full test suite green

### Phase 2: API #2 — Chunking
- Task 6: Table-aware parser
- Task 7: Recursive + token-based chunkers
- Task 8: Semantic chunker upgrade + markdown-aware chunker
- Task 9: Excel row-based chunker + strategy registry
- Task 10: Chunking endpoint

### Checkpoint C: All 5 chunking strategies verified
- [ ] Each strategy produces correct, size/boundary-verified chunks on fixture inputs (text/table/excel)
- [ ] `POST /api/v1/ingestion/chunking` returns the expected response shape for all 5 strategies
- [ ] Full test suite green

### Phase 3: API #3 — Embedding
- Task 11: Trusted-context dependency
- Task 12: EmbeddingProvider protocol + OpenAI implementation
- Task 13: Multi-representation enrichment
- Task 14: Qdrant vectorstore (named vectors)
- Task 15: Celery app + `embed_chunks` task
- Task 16: Embedding endpoint + WebSocket progress endpoint

### Checkpoint D: Embedding pipeline verified end-to-end (mocked)
- [ ] `embed_chunks` task, run eagerly, calls enrichment → embed (3 named vectors) → Qdrant upsert with the confirmed payload shape, in order
- [ ] Progress states observed via the task's `update_state` calls in the expected sequence
- [ ] `POST /api/v1/ingestion/embedding` returns 202 + `task_id`; missing trusted headers returns 400
- [ ] WebSocket test client receives progress frames and a terminal state
- [ ] Full test suite green

### Phase 4: Integration & Polish
- Task 17: Dev infra (docker-compose) + docs note for known external gaps
- Task 18: Full verification pass

### Checkpoint: Complete
- [ ] All acceptance criteria across Tasks 1–17 met
- [ ] `ruff check`, `ruff format --check`, `mypy`, `pytest` all pass
- [ ] Existing chat/graph/retrieval endpoints/tests unmodified and green
- [ ] Ready for human review

## Open Questions

- Deleting `app/rag/embeddings/huggingface.py` (Task 12) — confirm before that
  task lands, or say "keep it" and it'll be left in place unused.
- Real (non-mocked) manual smoke test against a live Qdrant + Redis + OpenAI
  key is not part of any task's automated acceptance criteria — flag if you
  want an explicit manual end-to-end smoke-test task added before calling
  this done.
