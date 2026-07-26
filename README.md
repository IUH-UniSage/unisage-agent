# UniSage Agent

`unisage-agent` is the RAG and orchestration service for the UniSage academic
assistant. It is a pipeline-oriented modular monolith built with FastAPI,
Pydantic Graph, SQLAlchemy, PostgreSQL, and pgvector.

## Quick Start

Requirements:

- Python 3.12+
- PostgreSQL 16 with the `vector` extension
- `uv` or `pip`
- `go-task` is optional

Create a local environment:

```powershell
Copy-Item .env.example .env
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run the API:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Useful endpoints:

- Swagger UI: `http://localhost:8000/docs`
- Health: `GET /api/v1/health`
- Chat: `POST /api/v1/chat`
- Text ingestion: `POST /api/v1/ingestion`

The ingestion endpoint currently parses and chunks text. Persistence,
embeddings, and external LLM calls are intentionally separate next steps.

## Architecture

The project uses a pipeline-oriented modular monolith. Each package maps to a
recognizable RAG stage without introducing full DDD or Hexagonal ceremony before
the project needs it.

```text
app/
├── api/              # FastAPI routes and request dependencies
├── core/             # Settings, errors, middleware, sanitization, tracing
├── rag/
│   ├── ingestion/    # Loading, parsing, and ingestion orchestration
│   ├── chunking/     # Recursive and paragraph chunking
│   ├── embeddings/   # Embedding provider boundaries
│   ├── retrieval/    # Vector, keyword, hybrid, and context building
│   ├── reranking/    # Reranking implementations
│   └── generation/   # Grounded answers, citations, and suggestions
├── graph/            # Intent -> retrieve -> rerank -> generate orchestration
├── database/         # Async session, models, and repositories
└── schemas/          # API and pipeline data contracts
```

Dependency direction:

```text
API -> Graph -> RAG services -> Database repositories
```

Graph nodes orchestrate services. They do not contain SQL, embedding
algorithms, prompt construction, or provider-specific SDK calls.

## Development Checks

Run the checks directly with the project virtual environment:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check app tests
.venv\Scripts\python.exe -m ruff format --check app tests
.venv\Scripts\python.exe -m mypy app tests
```

Equivalent Taskfile commands are available:

```powershell
task test
task code:check-strict
```

## Database

The local Dev Container provides PostgreSQL 16 and pgvector. Run migrations
after the database is available:

```powershell
.venv\Scripts\python.exe -m alembic upgrade head
```

Keep authentication, user, role, and general administration in
`unisage-backend`. This service owns AI-specific administration such as
ingestion, evaluation, and model configuration.
