# AGENTS.md

Project guidance for coding agents working on `unisage-agent`.

## Architecture

`unisage-agent` owns the UniSage RAG engine and graph orchestration. Use the
pipeline-oriented modular monolith below:

```text
app/
├── api/          # FastAPI routes and dependency wiring
├── core/         # Settings and cross-cutting concerns
├── rag/          # Ingestion, chunking, embeddings, retrieval, reranking, generation
├── graph/        # Pydantic Graph state and orchestration nodes
├── database/     # SQLAlchemy session, models, and repositories
└── schemas/      # API and pipeline contracts
```

Dependency direction is `API -> Graph -> RAG services -> repositories`.
Handlers validate and delegate. Graph nodes orchestrate. Repositories own SQL.

Do not add DDD layers, ports, or provider factories without a concrete second
implementation or a test boundary that needs the abstraction.

Keep authentication, users, roles, and general administration in
`unisage-backend`. This service owns AI-specific administration such as
document ingestion, retrieval evaluation, and model configuration.

## Coding Rules

- Use Python 3.12 syntax and strict type hints.
- Keep request and response models in `app/schemas`.
- Apply `user_faculty` and `user_level` metadata filtering before returning RAG chunks.
- Keep provider-specific code inside its RAG stage.
- Keep runtime text files UTF-8 encoded.
- Do not commit `.env`, credentials, tokens, caches, or virtual environments.

## Checks

Before committing:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check app tests
.venv\Scripts\python.exe -m ruff format --check app tests
.venv\Scripts\python.exe -m mypy app tests
```

## Git Workflow

Never commit directly to `main` or force-push. Branches use:

```text
feature/<owner>-<task-id>-<short-name>
fix/<owner>-<task-id>-<short-name>
enhance/<owner>-<task-id>-<short-name>
```

For Huyền's work on task `UNISAGE-02`, use lowercase branch IDs such as:

```text
feature/huyen-unisage-02-rag-agent-base
```

Commits use English Conventional Commits without emoji:

```text
feat(rag): [UNISAGE-02] implement retrieval flow
```

Keep commits atomic. Do not mix skills, database, application, and
documentation changes in one commit.
