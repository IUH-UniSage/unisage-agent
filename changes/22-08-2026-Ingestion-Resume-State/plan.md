# Implementation Plan: Ingestion Resume State (unisage-agent)

## Overview

Persist a chunking draft (strategy, params, chunk list) per `document_id` so
an admin who chunked a document but hasn't yet confirmed embedding can
resume exactly where they left off. Per
`docs/specs/SPEC-ingestion-resume.md`. This plan is fully separate from, and
does not reopen, `changes/22-08-2026-Ingestion-Service/todo.md` — it adds a
persistence side-effect on top of that feature's endpoints without touching
their logic.

## Architecture Decisions

- **Side effect, not a rewrite**: the existing `chunking`/`embedding`
  handlers gain exactly one new function call each (at the end, after their
  normal response is already built). No existing line in those handlers'
  computation logic is edited. This is the main correctness risk in this
  plan (see Risks) and the reason Task ordering puts the repository fully
  built and unit-tested *before* it's wired into any endpoint.
- **Revive, don't reinvent, the Postgres/Alembic plumbing**: `unisage-agent`
  already has SQLAlchemy async session plumbing and an Alembic setup,
  confirmed dead code by the earlier SPEC-ingestion.md investigation. This
  feature is the first real consumer — no new ORM/migration tooling
  introduced.
- **Separate schema/DB from Java, no cross-service FK**: `document_id` is a
  plain column, not a foreign key into Java's `documents` table, even
  though same-database cross-schema FKs are technically possible. Keeps
  Alembic and Java's Hibernate `ddl-auto=update` from ever being able to
  break each other.
- **`DocumentProcessStep` ships as a one-value enum on purpose**: no
  transitions to test or design around in this feature — verified by
  Task 1's tests asserting only `CHUNKED` is ever written.

## Dependency Graph

```
ORM models (DocumentProcessLog, DocumentChunk, DocumentProcessStep)
    │
    ▼
Alembic migration (creates both tables)
    │
    ▼
Repository (upsert_chunking_draft / get_draft / delete_draft)
    │
    ├──────────────┬───────────────────────┐
    ▼              ▼                       ▼
Wire into      Wire into            New GET /ingestion/jobs/{document_id}
chunking       embedding            endpoint (reads via get_draft)
handler        handler
(side effect)  (side effect:
               delete_draft)
```

Wiring into the chunking/embedding handlers and building the new GET
endpoint are independent of each other once the repository exists — they
can be done in any order or in parallel.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Wiring the side-effect accidentally changes chunking/embedding's existing response or control flow | High (would break `changes/22-08-2026-Ingestion-Service` acceptance criteria) | Task 4/5 diffs are reviewed to confirm they are pure *additions* (one new line each, after the existing `return`-building code) — no existing line touched. Full SPEC-ingestion test suite re-run as part of both tasks' verification. |
| Test DB setup for these new tests doesn't match the project's existing pattern (if any) | Med | Task 1 starts by reading `tests/conftest.py` before writing any test, and reuses whatever fixture pattern already exists rather than inventing a second one. |
| `document_id` uniqueness race (two chunking calls for the same document at nearly the same time) | Low | Accepted per spec ("last write wins", no locking) — upsert uses a single transaction per call; worst case is one of the two writes losing, not corruption. |
| Alembic migration conflicts with Java's Hibernate-managed schema if ever pointed at the same database/schema by mistake | Med | Migration explicitly targets a distinct schema (or database) name, verified manually in Task 2's verification step by inspecting the generated SQL before applying it. |

## Task List

### Phase 1: Foundation (data layer)
- Task 1: ORM models + `DocumentProcessStep` enum
- Task 2: Alembic migration
- Task 3: Repository (upsert / get / delete)

### Checkpoint A: Data layer works in isolation
- [ ] `alembic upgrade head` creates both tables + enum cleanly on a fresh DB
- [ ] Repository unit tests (Task 3) green
- [ ] `ruff`, `mypy` pass on all new files
- [ ] Existing SPEC-ingestion test suite untouched and green (nothing wired in yet)

### Phase 2: Wiring
- Task 4: Side-effect in the chunking handler
- Task 5: Side-effect in the embedding handler
- Task 6: New `GET /ingestion/jobs/{document_id}` endpoint

### Checkpoint B: Feature complete
- [ ] Chunking a document then `GET`-ing its job returns the matching draft
- [ ] Re-chunking the same document replaces (not duplicates) the draft
- [ ] Dispatching embedding for that document makes the `GET` return 404 afterward
- [ ] Full SPEC-ingestion.md test suite (preview/chunking/embedding) still green, unmodified
- [ ] `ruff check`, `ruff format --check`, `mypy`, `pytest` all pass
- [ ] Ready for human review

## Open Questions

None — resolved during the interview (see SPEC-ingestion-resume.md's Open
Questions section, which is empty for the same reason).
