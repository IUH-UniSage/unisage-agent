---
name: pr
description: >-
  Draft polished UniSage pull request titles and descriptions from the current Git branch,
  commits, diff, Jira task, verification evidence, and repository PR template. Use when the user
  invokes $pr, selects this skill through /skills, or asks to write, prepare, improve, or summarize
  a UniSage pull request. Do not create, publish, merge, or modify a remote PR unless explicitly
  requested.
---

# Write UniSage PR

Produce a concise, evidence-backed PR title and body that can be pasted directly into GitHub.

## Workflow

1. Read `.github/pull_request_template.md`.
2. Inspect:

   ```text
   git status --short --branch
   git log --oneline --decorate origin/main..HEAD
   git diff --stat origin/main...HEAD
   git diff --name-status origin/main...HEAD
   ```

3. Read the relevant changed files when filenames and commit messages do not establish behavior.
4. Derive the Jira key from the PR title request, branch, or commit messages in that order.
   Normalize zero-padded identifiers to Jira's canonical form:
   `UNISAGE-02` becomes `UNISAGE-2`.
5. Include only verification that has actually run. Never invent test counts or claim manual
   verification without evidence.
6. Draft the title and body using the output contract below.

## Title

Use English and this format:

```text
<type>: [UNISAGE-N] <concise outcome>
```

Choose `feat`, `fix`, `enhance`, `refactor`, `chore`, or `docs` from the dominant PR intent.
Keep the title under 72 characters when practical.

## Body

Follow the repository template and these rules:

- Link Jira as `[UNISAGE-N](https://tranngochuyen.atlassian.net/browse/UNISAGE-N)`.
- Explain motivation and behavior, not a line-by-line file inventory.
- Use 2-5 bullets in `Change` and keep related changes together.
- State affected services, endpoints, storage, or developer workflows in `Impact`.
- Use exact commands and results in `Test`.
- Write `None` in `Note` when there are no limitations or migration steps.
- Preserve the Jira marker comments so the GitHub workflow remains idempotent.

## Mermaid

Include a compact Mermaid diagram when the PR changes architecture, orchestration, data flow,
request flow, or three or more connected components. Omit the section for simple docs, config,
or isolated fixes.

Use `flowchart LR` for request/data flow. Keep labels short and quote labels containing
punctuation. Ensure every node and edge represents code present in the diff.

## Output Contract

Return exactly:

```markdown
## PR Title

<title>

## PR Description

<completed repository template>
```

Do not add analysis, review findings, commit commands, or publishing instructions unless the
user asks for them.
