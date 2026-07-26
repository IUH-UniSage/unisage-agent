---
name: git-commit-instructions
description: >-
  Git workflow and commit guidelines for UniSage AI Agent. Use when creating or naming branches,
  preparing commits, reviewing staged changes, pushing branches, or opening pull requests.
  Enforces owner and task-based branch names plus English conventional commits without emojis.
---

# UniSage Git Workflow

> [!IMPORTANT]
> **Safety and quality rules**
>
> 1. Do not commit or push without explicit user instruction.
> 2. Never commit `.env`, credentials, API keys, tokens, caches, or local-only configuration.
> 3. Never force-push a shared branch.
> 4. Do not commit directly to `main`.
> 5. Commit only after the relevant tests, Ruff, and Mypy checks pass.

---

## Branch Naming

Use exactly one of these prefixes:

| Prefix | Use for |
| --- | --- |
| `feature` | New user-facing or system capability |
| `fix` | Defect correction |
| `enhance` | Improvement to an existing capability without being a bug fix |

Use this format:

```text
<prefix>/<owner>-<task-id>-<short-name>
```

Validate the final branch name with:

```regex
^(feature|fix|enhance)/(huy|huyen)-unisage-[0-9]+-[a-z0-9]+(?:-[a-z0-9]+)*$
```

Rules:

- Allow only `huy` or `huyen` as `<owner>`.
- Require a task ID in the form `unisage-<number>`.
- Use lowercase ASCII kebab-case for the entire branch name.
- Keep `<short-name>` concise and descriptive.
- Ask for the owner or task ID when either is missing; do not invent them.
- Create the branch from a clean and current `main` unless the user names another base.
- Use `feature` for branches and `feat` for feature commit messages.

Examples:

```text
feature/huy-unisage-303-hybrid-retrieval
fix/huyen-unisage-317-citation-mapping
enhance/huy-unisage-325-reranker-config
```

Reject inconsistent forms such as:

```text
feat/unisage-303
feature/UNISAGE-303-hybrid-retrieval
huy/feature-unisage-303
feature/minh-unisage-303-hybrid-retrieval
```

Before creating a branch:

```text
git status --short --branch
git fetch origin
git switch main
git pull --ff-only
git switch -c <branch-name>
```

## Commit Message Format (English - NO Emojis)

Use Conventional Commits in English without emojis and include the UniSage task ID:

```text
<type>(<scope>): [<TASK-ID>] <short title in English>

<Context/Root cause>:
- <Explain reason for change or root cause of bug>

<Changes/Fix>:
- <Detail technical modifications made in code>

Verification:
- <List automated tests or manual verifications passed>
```

Use uppercase task IDs in commit messages even though branch names are lowercase:

```text
Branch: feature/huy-unisage-303-hybrid-retrieval
Commit: feat(rag): [UNISAGE-303] implement hybrid retrieval
```

### Commit Types and Body Labels

| Type             | Description                           | Part 1 Label  | Part 2 Label |
| ---------------- | ------------------------------------- | ------------- | ------------ |
| `feat`           | New feature                           | `Context:`    | `Changes:`   |
| `fix`            | Bug fix                               | `Root cause:` | `Fix:`       |
| `refactor`       | Code improvement (no behavior change) | `Motivation:` | `Refactor:`  |
| `chore` / `deps` | Config or dependency update           | `Reason:`     | `Changes:`   |
| `docs`           | Documentation changes                 | `Context:`    | `Changes:`   |
| `test`           | Adding or updating tests              | `Context:`    | `Changes:`   |

---

## Atomic Commit Workflow

1. Inspect `git status --short --branch`.
2. Review the complete diff and identify unrelated changes.
3. Stage only one coherent concern.
4. Run the checks relevant to that concern.
5. Review `git diff --cached`.
6. Build the commit message from the staged diff, not from intent alone.
7. Commit only after user authorization.
8. Push only after separate or explicit combined authorization.

Do not:

- Mix skills, Taskfile, database, application, and documentation changes in one commit.
- Commit generated caches, virtual environments, uploaded files, or secrets.
- Amend or rewrite a pushed commit.
- Use `git reset --hard`, force push, or broad destructive cleanup.

## Examples

### Feature Commit

```text
feat(rag): [UNISAGE-303] implement hybrid retrieval

Context:
- Need to filter academic regulations by student faculty and security access level.

Changes:
- Implement hybrid retrieval combining BM25 and pgvector cosine similarity.
- Add SQL pre-filtering for `faculty` and `security_level` claims.

Verification:
- Pytest suite: 18 passed.
- Mypy strict and Ruff check passed 100%.
```

### Bug Fix Commit

```text
fix(citation): [UNISAGE-317] preserve source mapping

Root cause:
- Reranking changed chunk order without preserving source metadata.

Fix:
- Preserve citation metadata when rebuilding the reranked result list.
- Add a regression test for reordered chunks.

Verification:
- Pytest suite passed.
- Ruff and Mypy checks passed.
```
