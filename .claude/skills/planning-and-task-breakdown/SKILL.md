---
name: planning-and-task-breakdown
description: Breaks work into ordered tasks. Use when you have a spec or clear requirements and need to break work into implementable tasks. Use when a task feels too large to start, when you need to estimate scope, or when parallel work is possible.
---

# Planning and Task Breakdown

## Overview

Decompose work into small, verifiable tasks with explicit acceptance criteria. Good task breakdown is the difference between an agent that completes work reliably and one that produces a tangled mess. Every task should be small enough to implement, test, and verify in a single focused session.

## When to Use

- You have a spec and need to break it into implementable units
- A task feels too large or vague to start
- Work needs to be parallelized across multiple agents or sessions
- You need to communicate scope to a human
- The implementation order isn't obvious

**When NOT to use:** Single-file changes with obvious scope, or when the spec already contains well-defined tasks.

## The Planning Process

### Step 1: Enter Plan Mode

Before writing any code, operate in read-only mode:

- Read the spec and relevant codebase sections
- Identify existing patterns and conventions
- Map dependencies between components
- Note risks and unknowns

**Do NOT write code during planning.** The output is a plan document and a task list saved into a dated `changes/dd-MM-yyyy-Title/` folder (see Output Files; default files inside it are `plan.md` and `todo.md`), not implementation.

### Step 2: Identify the Dependency Graph

Map what depends on what:

```
Database schema
    │
    ├── API models/types
    │       │
    │       ├── API endpoints
    │       │       │
    │       │       └── Frontend API client
    │       │               │
    │       │               └── UI components
    │       │
    │       └── Validation logic
    │
    └── Seed data / migrations
```

Implementation order follows the dependency graph bottom-up: build foundations first.

### Step 3: Slice Vertically

Instead of building all the database, then all the API, then all the UI — build one complete feature path at a time:

**Bad (horizontal slicing):**

```
Task 1: Build entire database schema
Task 2: Build all API endpoints
Task 3: Build all UI components
Task 4: Connect everything
```

**Good (vertical slicing):**

```
Task 1: User can create an account (schema + API + UI for registration)
Task 2: User can log in (auth schema + API + UI for login)
Task 3: User can create a task (task schema + API + UI for creation)
Task 4: User can view task list (query + API + UI for list view)
```

Each vertical slice delivers working, testable functionality.

### Step 4: Write Tasks

Each task follows this structure, whether it lands in the markdown task list or as an item in an external tracker (see Output Files):

```markdown
## Task [N]: [Short descriptive title]

**Description:** One paragraph explaining what this task accomplishes.

**Acceptance criteria:**

- [ ] [Specific, testable condition]
- [ ] [Specific, testable condition]

**Verification:**

- [ ] Tests pass: [the repository's focused-test command]
- [ ] Build succeeds: [the repository's build command]
- [ ] Manual check: [description of what to verify]

**Dependencies:** [Task numbers this depends on, or "None"]

**Files likely touched:**

- `src/path/to/file.ts`
- `tests/path/to/test.ts`

**Estimated scope:** [Small: 1-2 files | Medium: 3-5 files | Large: 5+ files]
```

### Step 5: Order and Checkpoint

Arrange tasks so that:

1. Dependencies are satisfied (build foundation first)
2. Each task leaves the system in a working state
3. Verification checkpoints occur after every 2-3 tasks
4. High-risk tasks are early (fail fast)

Add explicit checkpoints to the task list target:

```markdown
## Checkpoint: After Tasks 1-3

- [ ] All tests pass
- [ ] Application builds without errors
- [ ] Core user flow works end-to-end
- [ ] Review with human before proceeding
```

## Task Sizing Guidelines

| Size   | Files | Scope                                 | Example                              |
| ------ | ----- | ------------------------------------- | ------------------------------------ |
| **XS** | 1     | Single function or config change      | Add a validation rule                |
| **S**  | 1-2   | One component or endpoint             | Add a new API endpoint               |
| **M**  | 3-5   | One feature slice                     | User registration flow               |
| **L**  | 5-8   | Multi-component feature               | Search with filtering and pagination |
| **XL** | 8+    | **Too large — break it down further** | —                                    |

If a task is L or larger, it should be broken into smaller tasks. An agent performs best on S and M tasks.

**When to break a task down further:**

- It would take more than one focused session (roughly 2+ hours of agent work)
- You cannot describe the acceptance criteria in 3 or fewer bullet points
- It touches two or more independent subsystems (e.g., auth and billing)
- You find yourself writing "and" in the task title (a sign it is two tasks)

## Output Files

Plans are written directly into a dated folder under `changes/` — there is no intermediate `tasks/` staging directory to archive later. This keeps every plan's history in place from the start instead of needing a separate archive step once work finishes.

### Compute the destination folder

As soon as you know the feature/project name (usually by the time you'd write the plan's `# Implementation Plan: [Feature/Project Name]` heading — see the template below), derive the destination directly — no script needed, this is a one-line decision:

`changes/<dd-MM-yyyy>-<Title>/`, where `<dd-MM-yyyy>` is today's date and `<Title>` is the feature/project name collapsed to a single CamelCase token (drop diacritics, drop a leading label like "Plan:"/"Implementation Plan:", drop a trailing parenthetical, keep only letters/digits, capitalize the first letter of each word). Match this exactly to how `archive-task-plan` derives titles elsewhere, so a plan folder created here looks the same as one that skill would have produced.

Example: feature name "Access Level CRUD" on 22-08-2026 → `changes/22-08-2026-AccessLevelCRUD/`.

Before creating it, check whether that exact folder already exists (`ls changes/` or similar) — if it does, **reuse it** rather than making a new one; the common case is continuing to refine the same plan across multiple turns or sessions on the same day. Only pick a distinct, numerically-suffixed name (`-2`, `-3`, ...) if the user explicitly wants a second, separate plan under the same title today.

- **Plan document:** Save the implementation plan to `<destination>/plan.md`. This is always a markdown file — design decisions, risks, and open questions don't map cleanly onto individual tracker issues.
- **Task list:** Record each task in the **task list target** (defined below).

### Task List Target

The task list target is where tasks and checkpoints are recorded. It is defined once, here; every other reference in this skill defers to it.

- **Default: a checklist-style markdown file at `<destination>/todo.md`.** This is the convention the `/build` command and other downstream tooling expect. Use it unless the project says otherwise.
- **External tracker:** if the project's agent rules (`CLAUDE.md`, `AGENTS.md`, etc.) or the user designate an issue tracker (e.g. GitHub Issues, Jira, Linear, `bd`/beads), create one tracker item per task instead of writing `<destination>/todo.md`. Map the Step 4 structure onto the tracker's fields: acceptance criteria and verification steps in the item body, dependencies via the tracker's linking mechanism (`bd dep add`, "blocked by", etc.). Record Step 5 checkpoints as tracker items too, or as a checklist in the plan document if the tracker has no natural equivalent.

When using an external tracker, note it in `<destination>/plan.md` (e.g. "Tasks tracked in Linear project FOO") so downstream steps and future sessions know where to look, and keep the plan document's Task List section as an ordered index of tracker item IDs or links rather than a duplicate checklist.

Since plans no longer pass through a `tasks/` staging directory, the `archive-task-plan` skill has nothing left to do for plans created this way — it only matters for older `tasks/plan.md` + `tasks/todo.md` pairs left over from before this change.

## Plan Document Template

```markdown
# Implementation Plan: [Feature/Project Name]

## Overview

[One paragraph summary of what we're building]

## Architecture Decisions

- [Key decision 1 and rationale]
- [Key decision 2 and rationale]

## Task List

### Phase 1: Foundation

- [ ] Task 1: ...
- [ ] Task 2: ...

### Checkpoint: Foundation

- [ ] Tests pass, builds clean

### Phase 2: Core Features

- [ ] Task 3: ...
- [ ] Task 4: ...

### Checkpoint: Core Features

- [ ] End-to-end flow works

### Phase 3: Polish

- [ ] Task 5: ...
- [ ] Task 6: ...

### Checkpoint: Complete

- [ ] All acceptance criteria met
- [ ] Ready for review

## Risks and Mitigations

| Risk   | Impact         | Mitigation |
| ------ | -------------- | ---------- |
| [Risk] | [High/Med/Low] | [Strategy] |

## Open Questions

- [Question needing human input]
```

When tasks live in an external tracker, keep the Task List section above as an ordered index of tracker item IDs or links instead of a duplicate checklist.

## Parallelization Opportunities

When multiple agents or sessions are available:

- **Safe to parallelize:** Independent feature slices, tests for already-implemented features, documentation
- **Must be sequential:** Database migrations, shared state changes, dependency chains
- **Needs coordination:** Features that share an API contract (define the contract first, then parallelize)

## Common Rationalizations

| Rationalization                | Reality                                                                                      |
| ------------------------------ | -------------------------------------------------------------------------------------------- |
| "I'll figure it out as I go"   | That's how you end up with a tangled mess and rework. 10 minutes of planning saves hours.    |
| "The tasks are obvious"        | Write them down anyway. Explicit tasks surface hidden dependencies and forgotten edge cases. |
| "Planning is overhead"         | Planning is the task. Implementation without a plan is just typing.                          |
| "I can hold it all in my head" | Context windows are finite. Written plans survive session boundaries and compaction.         |

## Red Flags

- Starting implementation without a written task list
- Writing `<destination>/todo.md` when the project has designated an external tracker (or scattering tasks across both)
- Writing to a `tasks/` staging directory out of habit instead of the `changes/dd-MM-yyyy-Title/` destination
- Tasks that say "implement the feature" without acceptance criteria
- No verification steps in the plan
- All tasks are XL-sized
- No checkpoints between tasks
- Dependency order isn't considered

## Verification

Before starting implementation, confirm:

- [ ] Every task has acceptance criteria
- [ ] Every task has a verification step
- [ ] Task dependencies are identified and ordered correctly
- [ ] Tasks are recorded in the task list target (default `<destination>/todo.md` inside the `changes/dd-MM-yyyy-Title/` folder)
- [ ] No task touches more than ~5 files
- [ ] Checkpoints exist between major phases
- [ ] The human has reviewed and approved the plan

## Cleanup after writing to a connected device folder

`todo.md` in particular gets rewritten repeatedly during planning — checking off tasks, adding new ones, revising checkpoints — and `plan.md` often gets a few rounds of edits too. When these files live in a connected local folder reached through the remote-devices bridge (not the cloud workspace), repeatedly overwriting the same file is exactly the pattern that leaves orphaned `.fuse_hiddenXXXXXXXXXXXXXXXX` artifacts behind in `changes/dd-MM-yyyy-Title/` — noise that shows up as untracked files in `git status`.

After writing or updating `plan.md`/`todo.md` on a connected folder — and again before telling the user the plan is done — run the `device-fuse-cleanup` skill's steps: scan `changes/dd-MM-yyyy-Title/` (and the `changes/` root) for `.fuse_hidden*` files and move any found out of the way. Do this proactively; don't wait for the user to notice stray files.

## See Also

Acceptance criteria are per-task and answer "did we build the right thing?". They sit on top of the project-wide Definition of Done, the standing bar every task clears before it counts as done. See `../../references/definition-of-done.md`.
