# Spec: Ingestion Resume State (unisage-agent)

> **This spec is additive, with one narrow exception.** It does not modify,
> override, or change the *behavior* of any endpoint in
> `docs/specs/SPEC-ingestion.md` (the 3-API ingestion pipeline: preview,
> chunking, embedding) — response shapes, re-fetch-every-call semantics, and
> the embedding pipeline are all unchanged. The one exception (see
> `SPEC-ingestion.md`'s Decision 7): `ChunkingRequest` gains a new required
> `document_id` field, needed so a chunking draft can be keyed and later
> retrieved by `document_id`. That's the only schema this feature touches;
> everything else — including chunking's request-independent behavior once
> `document_id` is read — is unaffected.

## Objective

An admin who finishes chunking a document (step 2 of the ingestion pipeline)
but leaves before confirming embedding (step 3) currently loses that work —
returning to the document means re-picking a chunking strategy and re-running
chunking from scratch. This feature persists the chunking draft (strategy,
params, and the resulting chunk list) so returning to an in-progress
document restores exactly what was there, with no re-work.

Success = an admin can chunk a document, close the tab, come back an hour
(or a day) later, open the same document, and see the same chunks/strategy/
params they left — until they either re-run chunking (which overwrites the
draft with the fresh result) or confirm embedding (which clears the draft,
since it's no longer a draft at that point).

## Scope Boundaries (read this before anything else)

- **Preview (step 1): not persisted at all.** Re-opening a document at the
  preview stage always re-fetches and re-parses from MinIO. Nothing to
  restore, nothing to go stale.
- **Chunking (step 2): response behavior is unchanged; request gains one
  required field.** `POST /api/v1/ingestion/chunking` still always
  re-fetches the object from MinIO and re-parses from scratch on every
  call — it never reads a cache to build its response, and its response
  shape (`ChunkingResponse`) is untouched. The one change is on the request
  side: `ChunkingRequest` now requires `document_id` (see
  `SPEC-ingestion.md`'s Decision 7), so this feature has a key to persist
  the draft under. Persistence itself is a **side effect only**: after a
  successful call, the result is additionally persisted for later retrieval
  through a *different* endpoint (see below).
- **Embedding (step 3): out of scope.** If an admin leaves while an embedding
  Celery task is in flight, reconnecting to it is not handled here — that's
  what SPEC-ingestion.md's `task_id` + WebSocket mechanism is for, already.
  This feature does not touch that.
- **No cross-service calls added.** Consistent with SPEC-ingestion.md's
  existing principle, Python does not call Java and Java does not call
  Python. A document's full detail view (title/source_url/DocStatus/
  department/category from Java's existing `GET /documents/{id}`, plus the
  ingestion draft from this feature's new endpoint) is composed by the
  **frontend calling both APIs independently** — not by either backend
  aggregating on the other's behalf. No BFF/aggregation endpoint is added by
  this feature.
- **No locking.** Two admins touching the same `document_id` concurrently is
  an accepted, unmitigated edge case for this KLTN scope — last write wins.

## Tech Stack

- Python 3.12, FastAPI, SQLAlchemy (async) + Alembic — reviving the
  currently-dead Postgres/SQLAlchemy/alembic plumbing already present in
  `unisage-agent` (see `SPEC-ingestion.md`'s note that it was confirmed
  unused before this feature), now given a real purpose.
- **Same physical Postgres server as Java, but a separate schema/database.**
  Not the same schema as Java's `documents`/`categories`/etc. tables. No
  real foreign key from this feature's tables to Java's tables — even
  though Postgres supports cross-schema FKs within one database, adding one
  here would couple Python's Alembic migrations to Java's Hibernate
  `ddl-auto=update` schema, and either side changing shape could silently
  break the other. `document_id` is a plain, unconstrained column;
  validity is an application-level concern (the client already holds a real
  `document_id` from Java), not a database-level constraint.

## Data Model

### `DocumentProcessStep` (enum)

> **Amended after initial ship.** The paragraph below described the
> single-value enum as originally shipped; a follow-up change (see
> "Extension: resuming into an in-flight embed" at the end of this doc) added
> a second value, `EMBEDDING`, using exactly the extension point this
> paragraph anticipated.

A marker of which ingestion stage last wrote this draft record — originally
`CHUNKED` only, meaning "a chunking draft exists for this document, not yet
embedded." Preview isn't tracked (no row created for it). The enum type
exists as a **forward-compatible extension point**: if a later feature
needs to track, say, "embedding in progress" or "previewed but not yet
chunked," it adds a new enum value, not a new column or a new table.

```python
class DocumentProcessStep(str, Enum):
    CHUNKED = "chunked"
    EMBEDDING = "embedding"  # added by the resume-into-embed extension
```

### Table `document_process_logs`

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `document_id` | VARCHAR | **Unique.** Java's Document id, read from `ChunkingRequest.document_id` (SPEC-ingestion.md Decision 7). No FK — see above. |
| `object_key` | VARCHAR | MinIO key, mirrored from the chunking request for convenience. |
| `current_step` | VARCHAR (`DocumentProcessStep`) | `CHUNKED` after a chunking call, `EMBEDDING` after an embedding call dispatches a Celery task (see extension section). |
| `chunking_strategy` | VARCHAR | One of SPEC-ingestion.md's 5 strategy names. |
| `chunking_params` | JSONB | Whatever params were used for that strategy. |
| `celery_task_id` | VARCHAR, nullable | Set when `current_step` becomes `EMBEDDING`; the Celery task id to reconnect the WebSocket progress view to. `NULL` while `current_step` is `CHUNKED`. |
| `created_at` | TIMESTAMPTZ | Set on first upsert. |
| `updated_at` | TIMESTAMPTZ | Set on every upsert. |

### Table `document_chunks`

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | PK |
| `process_log_id` | UUID | FK → `document_process_logs.id`, `ON DELETE CASCADE`. |
| `chunk_index` | INTEGER | Position in the chunk list. |
| `content` | TEXT | Chunk text. |
| `region_type` | VARCHAR | `"text"` \| `"table"` \| `"excel_row"` — matches SPEC-ingestion.md's chunk region types. |

## Lifecycle

1. **On a successful `POST /api/v1/ingestion/chunking` call** (existing
   endpoint, response contract unchanged; request now requires
   `document_id` per SPEC-ingestion.md Decision 7): upsert the
   `document_process_logs` row for that `document_id` (insert if absent;
   otherwise update `chunking_strategy`, `chunking_params`, `updated_at`),
   and replace all of that row's `document_chunks` (delete existing, insert
   the freshly computed list). This runs *after* the endpoint has already
   computed its normal response — it never influences what that endpoint
   returns.
2. **On a successful `POST /api/v1/ingestion/embedding` call** (existing
   endpoint, unchanged contract) that successfully dispatches the Celery
   task: **(amended, see extension section)** flip the row's `current_step`
   to `EMBEDDING` and record `celery_task_id`, rather than deleting it.
3. **`GET /api/v1/ingestion/jobs/{document_id}`** (new): returns
   `{object_key, current_step, chunking_strategy, chunking_params, chunks: [...], task_id}`
   if a row exists, else `404`. `task_id` is `null` while `current_step` is
   `CHUNKED`.
4. **`DELETE /api/v1/ingestion/jobs/{document_id}`** (new, see extension
   section): deletes the row, once a client has observed the embed task
   reach a terminal state.

## Commands

```
Dev:    uvicorn app.main:app --reload --port 8402
Test:   pytest
Lint:   ruff check app tests
Format: ruff format --check app tests
Types:  mypy app tests
Migrate: alembic upgrade head
```

## Project Structure (additions only)

```
app/
├── database/
│   ├── models.py                 # + DocumentProcessLog, DocumentChunk ORM models
│   └── repositories/
│       └── ingestion_job.py      # new: upsert/replace/get/delete for the two tables
├── schemas/
│   └── ingestion.py               # + document_id field on existing ChunkingRequest
├── api/v1/
│   └── ingestion.py               # + GET /ingestion/jobs/{document_id};
│                                   #   existing preview/chunking/embedding handlers
│                                   #   gain a call to the repository as a side effect
migrations/versions/
└── xxxx_add_document_process_logs_and_chunks.py   # new Alembic migration
```

## Code Style

Match `AGENTS.md` conventions already established in `unisage-agent`: strict
type hints, request/response models in `app/schemas`, repositories own SQL.

```python
class DocumentProcessStep(str, Enum):
    CHUNKED = "chunked"

async def upsert_chunking_draft(
    session: AsyncSession,
    *,
    document_id: str,
    object_key: str,
    strategy: str,
    params: dict[str, Any],
    chunks: list[Chunk],
) -> None:
    """Persist (or replace) the chunking draft for one document."""
```

The chunking handler passes `document_id`, `object_key`, `strategy`, and
`params` straight from the already-validated `request: ChunkingRequest` it
already has — no second DTO or re-parsing needed; only `chunks` comes from
the response it just computed.

## Testing Strategy

- `pytest` + `pytest-asyncio`, using a test Postgres schema (or SQLite for
  unit-level repository tests if the project's existing test setup already
  has a pattern for this — check `tests/conftest.py` before introducing a
  new one).
- Repository unit tests: upsert creates a row; a second upsert for the same
  `document_id` updates in place (not a duplicate) and replaces chunks;
  delete cascades to `document_chunks`.
- Endpoint integration tests: a `POST /ingestion/chunking` call is followed
  by a `GET /ingestion/jobs/{document_id}` returning the same
  strategy/params/chunks; a `POST /ingestion/embedding` call is followed by
  the same `GET` returning 404.
- `POST /ingestion/chunking`'s own *response* shape does not change —
  SPEC-ingestion.md's existing tests for that endpoint's response must stay
  green. Its *request* does change (new required `document_id`): every
  existing chunking-endpoint test (and any other caller) must be updated to
  send `document_id` as part of this feature's Task 4, not left broken.

## Boundaries

- **Always**: keep `SPEC-ingestion.md`'s endpoints' *response* contracts and
  chunking's re-fetch-every-call *behavior* unchanged; keep this feature's
  tables in Python's own schema/database, no FK into Java's tables.
- **Ask first**: changing `DocumentProcessStep` to have more than one value
  (a real state machine) — that's a bigger design than what's shipped here;
  reusing this feature's tables for anything beyond the resume use case;
  any further change to `SPEC-ingestion.md`'s request/response schemas
  beyond the one `ChunkingRequest.document_id` addition already decided.
- **Never**: add a cross-schema FK to Java's `documents` table; have Python
  call Java or vice versa to compose a document-detail view; modify Java's
  `DocStatus` enum or any Java code; change any *response* shape or
  chunking/embedding *behavior* defined in `SPEC-ingestion.md`.

## Success Criteria

- Chunking a document, then calling `GET /ingestion/jobs/{document_id}`,
  returns the same strategy/params/chunks just computed.
- Re-chunking the same document with different params replaces (not
  duplicates) the draft.
- Successfully dispatching embedding for that document makes the `GET`
  endpoint return 404 afterward.
- `SPEC-ingestion.md`'s existing preview/embedding tests remain green,
  unmodified; its chunking-endpoint tests are updated to send
  `document_id` and remain green.
- `ruff check`, `ruff format --check`, `mypy`, `pytest` all pass.
- A migration (`alembic upgrade head`) creates both tables cleanly against a
  fresh database.

## Open Questions

None outstanding — all resolved during the interview (DB placement, no
cross-service FK, `DocumentProcessStep`'s single-value nature, frontend-side
composition, no locking, step-3 out of scope) plus the later `document_id`
source decision (see `SPEC-ingestion.md` Decision 7).

## Extension: resuming into an in-flight embed

Originally, step 3 (embedding) was explicitly out of scope (see "Scope
Boundaries" above) and a successful embed dispatch deleted the draft row.
In practice this meant: a client that dispatched embedding and then
navigated away (or reloaded) before the WebSocket reported a terminal state
had no way back in — `GET /ingestion/jobs/{document_id}` 404'd, and the
frontend fell back to restarting the wizard from preview even though the
document was mid-embed server-side.

This extension closes that gap using exactly the mechanism the original
`DocumentProcessStep` docstring reserved for it:

- `DocumentProcessStep` gains `EMBEDDING`. `document_process_logs` gains a
  nullable `celery_task_id` column.
- `POST /ingestion/embedding`, on a successful dispatch, now calls
  `mark_embedding(document_id, task.id)` instead of `delete_draft` — the row
  survives, flipped to `current_step="embedding"` with `celery_task_id` set.
- `GET /ingestion/jobs/{document_id}`'s response gains `task_id` (from
  `celery_task_id`, `null` while `current_step` is `chunked`). A frontend
  resuming a document whose job has `current_step="embedding"` reconnects
  the `/ingestion/embedding/{task_id}/progress` WebSocket directly, instead
  of hydrating into the chunk-review step.
- New `DELETE /ingestion/jobs/{document_id}` (204, no-op if no row) deletes
  the row. The frontend calls this once it observes the WebSocket reach a
  terminal state (`SUCCESS`/`FAILURE`) — the same moment it already calls
  Java's `PATCH /documents/{id}/status`, so both cleanups happen together
  from the one place that's actually watching the task finish.
- Re-chunking a document whose row is still `EMBEDDING` (e.g. the user
  starts over) resets `current_step` back to `CHUNKED` and clears
  `celery_task_id`, same as any other chunking upsert.

**Known gap, accepted rather than solved here**: the Celery worker
(`app/worker/celery_app.py`) has no DB session and does not write to
`document_process_logs` itself — it only reports progress via the Celery
result backend, which is what the WebSocket endpoint already polls. So the
row transitions to a terminal state only when *some* client observes the
WebSocket's terminal frame and calls the `DELETE`. If the browser is closed
for the entire duration of an embed and never reopened, the row lingers
forever with `current_step="embedding"` and a task id whose Celery result
has long since expired from the backend — reopening the document then shows
a progress view stuck on `state: "error"` (the WebSocket's own "can't
determine state" fallback) rather than resolving one way or the other.
Fixing this for real would mean the Celery task itself writing terminal
state to the DB (a sync DB session inside the worker, a separate design
decision this extension does not make) or a scheduled sweep for stale
`EMBEDDING` rows. Neither is implemented; this is the same category of
accepted, documented gap as this project's existing `known-gaps.md` entries.
