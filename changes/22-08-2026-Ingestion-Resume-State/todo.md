# Task List: Ingestion Resume State (unisage-agent)

Spec: `docs/specs/SPEC-ingestion-resume.md` · Plan: `plan.md` (this folder)

This work is **additive** on top of the already-planned/implemented
`changes/22-08-2026-Ingestion-Service/` ingestion API surface. None of its
tasks are reopened or modified here.

---

## Phase 1: Foundation (data layer)

### Task 1: ORM models + `DocumentProcessStep` enum

**Description:** Add `DocumentProcessStep` (a one-value `str` Enum:
`CHUNKED`), `DocumentProcessLog`, and `DocumentChunk` SQLAlchemy models to
`app/database/models.py`, per the schema in `SPEC-ingestion-resume.md`.
`DocumentChunk.process_log_id` FKs to `DocumentProcessLog.id` with
`ondelete="CASCADE"`. Before writing tests, read `tests/conftest.py` to
reuse whatever test-DB fixture pattern already exists in the project rather
than introducing a new one.

**Acceptance criteria:**
- [x] `DocumentProcessStep` has exactly one member, `CHUNKED`
- [x] `DocumentProcessLog.document_id` has a unique constraint, no FK
- [x] `DocumentChunk.process_log_id` cascades on delete of its parent

**Verification:**
- [x] `pytest tests/test_database_models.py` — model instantiation + cascade
      behavior against a test DB session
- [x] `mypy app/database/models.py` passes

**Dependencies:** None

**Files likely touched:**
- `app/database/models.py`
- `tests/test_database_models.py`

**Estimated scope:** Small (2 files)

---

### Task 2: Alembic migration

**Description:** Generate/write the Alembic migration creating
`document_process_logs`, `document_chunks`, and the `documentprocessstep`
enum type, targeting Python's own schema/database (explicitly not Java's).

**Acceptance criteria:**
- [x] `alembic upgrade head` succeeds on a fresh database and creates both
      tables with the exact columns from the spec
- [x] `alembic downgrade -1` cleanly drops both tables and the enum type
- [x] The migration's generated SQL is manually inspected to confirm it
      targets Python's own schema, not any schema Java owns

**Verification:**
- [x] `alembic upgrade head` then `alembic downgrade -1` then
      `alembic upgrade head` again, all succeed
- [x] Manual: `psql` (or equivalent) inspection of the resulting table DDL

**Dependencies:** Task 1

**Files likely touched:**
- `migrations/versions/xxxx_add_document_process_logs_and_chunks.py`

**Estimated scope:** Small (1 file)

---

### Task 3: Repository (upsert / get / delete)

**Description:** `app/database/repositories/ingestion_job.py` with three
functions: `upsert_chunking_draft(session, *, document_id, object_key,
strategy, params, chunks)` (insert-or-update the log row, delete+insert its
chunks), `get_draft(session, document_id) -> DraftDTO | None`,
`delete_draft(session, document_id) -> None`.

**Acceptance criteria:**
- [x] Calling `upsert_chunking_draft` twice for the same `document_id`
      results in exactly one `document_process_logs` row (updated, not
      duplicated) and its chunks fully replaced (old chunk rows gone)
- [x] `delete_draft` removes the log row and, via cascade, its chunks
- [x] `get_draft` for a `document_id` with no row returns `None` (not an
      exception)

**Verification:**
- [x] `pytest tests/test_ingestion_job_repository.py` — covers all three
      functions and the double-upsert/replace case
- [x] `mypy app/database/repositories/ingestion_job.py` passes

**Dependencies:** Task 1, Task 2

**Files likely touched:**
- `app/database/repositories/ingestion_job.py`
- `tests/test_ingestion_job_repository.py`

**Estimated scope:** Medium (2 files, core logic of this feature)

---

## Checkpoint A: Data layer works in isolation
- [x] `alembic upgrade head` creates both tables + enum cleanly on a fresh DB
- [x] Repository unit tests (Task 3) green
- [x] `ruff check`, `mypy` pass on all new files
- [x] Existing SPEC-ingestion.md test suite untouched and green (nothing wired into any endpoint yet)
- [ ] **Review with human before proceeding**

---

## Phase 2: Wiring

### Task 4: Side-effect in the chunking handler

**Description:** In the existing `POST /api/v1/ingestion/chunking` handler
(from `changes/22-08-2026-Ingestion-Service`'s work), add **one call** to
`upsert_chunking_draft(...)` after the handler has already built its normal
`ChunkingResponse` — do not touch any existing line of that handler's
parsing/chunking logic or its response construction.

**Acceptance criteria:**
- [x] A successful chunking call results in a `document_process_logs` +
      matching `document_chunks` rows for that `document_id`
- [x] The endpoint's HTTP response is byte-for-byte identical to before this
      task (verified by running SPEC-ingestion.md's existing chunking tests
      unmodified and green)
- [x] A failed chunking call (e.g. unsupported strategy/filetype mismatch)
      does not write a draft row

**Verification:**
- [x] `pytest tests/test_ingestion_chunking.py` (existing, from
      SPEC-ingestion.md) — still green, unmodified
- [x] New assertion (in the same or a new test file) that the draft row
      exists after a successful call

**Dependencies:** Task 3

**Files likely touched:**
- `app/api/v1/ingestion.py`
- `tests/test_ingestion_chunking.py` (only additive assertions, no existing
  assertions changed)

**Estimated scope:** Small (1-2 files)

---

### Task 5: Side-effect in the embedding handler

**Description:** In the existing `POST /api/v1/ingestion/embedding` handler,
add **one call** to `delete_draft(...)` immediately after the Celery task
is successfully dispatched (i.e. after the `202` response's `task_id` is
already determined) — do not touch any existing dispatch/validation logic.

**Acceptance criteria:**
- [x] A successful embedding dispatch removes any existing draft row for
      that `document_id`
- [x] The endpoint's HTTP response (202 + `task_id`) is unchanged from
      before this task
- [x] If there was no draft row to begin with (e.g. embedding called
      without a prior chunking call, if that's ever a valid path), the
      delete is a no-op, not an error

**Verification:**
- [x] `pytest tests/test_ingestion_embedding.py` (existing) — still green,
      unmodified
- [x] New assertion that the draft row is gone after a successful dispatch

**Dependencies:** Task 3

**Files likely touched:**
- `app/api/v1/ingestion.py`
- `tests/test_ingestion_embedding.py` (only additive assertions)

**Estimated scope:** Small (1-2 files)

---

### Task 6: New `GET /ingestion/jobs/{document_id}` endpoint

**Description:** New read-only endpoint calling `get_draft`, returning
`IngestionJobResponse` (object_key, current_step, chunking_strategy,
chunking_params, chunks) or `404` if no row exists.

**Acceptance criteria:**
- [x] Returns 200 + the expected shape for a `document_id` with a draft
- [x] Returns 404 for a `document_id` with no draft (never chunked, or
      already embedded/deleted)

**Verification:**
- [x] `pytest tests/test_ingestion_jobs_get.py` — both cases
- [x] Full `pytest` suite green (including all of
      `changes/22-08-2026-Ingestion-Service`'s existing tests)

**Dependencies:** Task 3 (independent of Tasks 4/5 — can be built in
parallel with them)

**Files likely touched:**
- `app/api/v1/ingestion.py`
- `app/schemas/ingestion.py` (+ `IngestionJobResponse`)
- `tests/test_ingestion_jobs_get.py`

**Estimated scope:** Small (2-3 files)

---

## Checkpoint B: Feature complete
- [x] Chunking a document then `GET`-ing its job returns the matching draft
- [x] Re-chunking the same document replaces (not duplicates) the draft
- [x] Dispatching embedding for that document makes the `GET` return 404 afterward
- [x] Full SPEC-ingestion.md test suite (preview/chunking/embedding) still green, unmodified
- [x] `ruff check`, `ruff format --check`, `mypy`, `pytest` all pass
- [x] Ready for human review
