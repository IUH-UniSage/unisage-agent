---
name: taskfile-management
description: >-
  Manage Go Task (Taskfile) in UniSage AI Agent project with modular structure,
  conventions, and patterns. Activate when adding, modifying, or creating taskfiles or tasks.
---

# Taskfile Management (UniSage AI Agent)

Manage Go Task (`Taskfile.yml`) in the UniSage AI Agent project following modular structure and task conventions.

## Core Principles

1. **Modular Structure** - Each category has its own dedicated taskfile in `taskfiles/`.
2. **Consistency** - Follow existing task naming patterns (`namespace:action`).
3. **Help Sync** - **ALWAYS** update `taskfiles/help.yml` when adding new tasks.
4. **YAML Safety** - Quote string commands properly to prevent YAML syntax errors.

## Active Project Structure

```text
Taskfile.yml              # Main: global variables + includes + default help task
taskfiles/
├── help.yml              # Terminal help menu (colored, boxed)
├── backend.yml           # Backend dev/prod server (be:dev, be:prod)
├── code.yml              # Formatting, linting, type checking (code:fix, code:check)
├── test.yml              # Pytest testing & coverage (test, test:file, test:cov)
└── db.yml                # Database migrations (db:up, db:migrate)
```

## Task Naming Convention

- Use `:` for namespace separator (e.g., `be:dev`, `code:check-strict`, `test:cov:html`).
- Use `-` for action variants (e.g., `lint-fix`, `check-strict`).

## Mandatory Help Synchronization

Whenever adding or modifying a task in any `taskfiles/*.yml` file, you MUST update `taskfiles/help.yml` so that `task` displays accurate help entries.
