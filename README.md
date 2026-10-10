# UniSage AI Agent (`unisage-agent`)

`unisage-agent` is the AI service of the UniSage academic assistant. It answers
students' academic questions with retrieval-augmented generation over the
documents they are allowed to read, and runs the document ingestion pipeline
(preview → chunk → embed into Qdrant).

Built with FastAPI, pydantic-ai, SQLAlchemy (PostgreSQL), Qdrant, Redis and
Celery. Product rules live in [`docs/product/PRODUCT.md`](docs/product/PRODUCT.md)
and the reasons behind them in [`docs/product/DECISIONS.md`](docs/product/DECISIONS.md);
where those are silent, the code is the source of truth.

## Quick Start

### Requirements

- Python 3.12+
- [go-task](https://taskfile.dev/) for Taskfile commands
- PostgreSQL 16 (this service's own database: ingestion drafts, clarification state)
- Qdrant, Redis and MinIO (`.devcontainer/docker-compose.yml` starts Redis and Qdrant;
  MinIO is shared with `unisage-backend`)
- `unisage-backend` running: it owns conversations and messages, and, with the model
  registry enabled, the LLM credentials this service uses

### Windows PowerShell

```powershell
cd unisage-agent

py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env

task be:dev
```

Celery worker + beat (usage outbox drain, budget reconciliation, model
registry verification, ...), each in its own terminal:

```powershell
task be:worker
task be:beat
```

Replay usage-log payloads that landed in the dead-letter queue (after fixing
whatever made backend-java reject them):

```powershell
task usage:replay-dead
```

Run everything directly when `go-task` is unavailable:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8402
.venv\Scripts\python.exe -m celery -A app.worker.celery_app worker --loglevel=info
.venv\Scripts\python.exe -m celery -A app.worker.celery_app beat --loglevel=info
```

### Linux / macOS

```bash
cd unisage-agent

python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env

task be:dev
```

Celery worker + beat, each in its own terminal:

```bash
task be:worker
task be:beat
```

Run everything directly when `go-task` is unavailable:

```bash
.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8402
.venv/bin/python -m celery -A app.worker.celery_app worker --loglevel=info
.venv/bin/python -m celery -A app.worker.celery_app beat --loglevel=info
```

## API Endpoints

Every route sits under `/api/v1` and, except health, requires the `X-Internal-Secret`
header the API Gateway adds; clients never call this service directly.

- Swagger UI: `http://127.0.0.1:8402/docs`
- Health: `GET /api/v1/health` (Postgres, Redis, Qdrant, usage outbox, model registry)
- Chat: `POST /api/v1/chat/stream` (Server-Sent Events)
- Ingestion: `POST /api/v1/ingestion/preview`, `POST /api/v1/ingestion/chunking`,
  `POST /api/v1/ingestion/embedding`, `GET /api/v1/ingestion/jobs/{document_id}`,
  `WS /api/v1/ingestion/events`
- Indexed chunks: `GET|DELETE /api/v1/documents/{document_id}/chunks/...`

## Development Commands

```powershell
# Show the command menu
task

# Application
task be:dev

# Tests and coverage
task test
task test:cov
task test:cov:html

# Code quality
task code:check
task code:check-strict
task code:fix

# Database / infrastructure (add env=eval|prod for another environment, default dev)
task db:up                 # Postgres + MinIO (backend-java) + Redis + Qdrant; creates this env's DBs if missing
task db:upgrade            # alembic upgrade head
task env:show env=eval     # which DB / bucket / collection / Redis an env points at
task db:current
task db:history
task db:migrate -- "migration message"
```

Run `task --list-all` to see every available command.

## Project Structure

```text
unisage-agent/
|-- app/
|   |-- main.py              # FastAPI application entry point
|   |-- api/                 # Routes, dependency wiring, error handlers
|   |-- core/                # Settings, security, errors, observability, model registry,
|   |                        # LLM providers, budget, pricing, usage tracking
|   |-- graph/               # Chat orchestrator (`streaming_graph.py`) and its nodes
|   |-- rag/
|   |   |-- ingestion/       # Parsing (PDF/DOCX/TXT/XLSX), table-aware regions
|   |   |-- chunking/        # Recursive, token, semantic, markdown, table-row, Excel-row
|   |   |-- enrichment/      # Summary + sample questions per chunk (multi-representation)
|   |   |-- embeddings/      # Embedding provider
|   |   |-- vectorstore/     # Qdrant collection, permission filter
|   |   |-- retrieval/       # Dense search over the 3 named vectors
|   |   |-- reranking/       # Score threshold
|   |   `-- prompting/       # YAML prompt templates, prompt builder, citations
|   |-- integrations/        # backend-java, Tavily, Slack clients
|   |-- worker/              # Celery app, periodic and ingestion tasks
|   |-- database/            # Async sessions, models, repositories
|   |-- schemas/             # API and pipeline contracts
|   `-- tools/               # Operator CLIs (embedding identity, dead-letter replay)
|-- docs/                    # Product rules, specs, configuration guide
|-- migrations/              # Alembic environment and versions
|-- changes/                 # Dated plans and task lists for each piece of work
|-- tests/                   # Pytest suite (`tests/e2e` needs live services)
`-- .devcontainer/           # Python, Redis and Qdrant development environment
```

Dependency direction:

```text
API -> Graph -> RAG services -> Database repositories / Qdrant
```

API handlers validate and delegate. Graph nodes orchestrate the RAG stages.
Provider logic stays in its RAG package or `core/llm`, SQL stays in repositories.

## Chat Flow

`app/graph/streaming_graph.py` is a plain async function, `run_graph`, that
branches with ordinary `if` statements. It is not a `pydantic_graph.Graph`,
because `Graph.run()` returns one final output and can't stream tokens. Node
names (`01_…` to `11_…`) exist for `GraphTrace` logs and to match the original
design. See [`docs/architecture/rag-pipeline.md`](docs/architecture/rag-pipeline.md)
for the full flow.

## Configuration

Every setting is an environment variable read by `app/core/config.py`; see
[`docs/guide/cau-hinh.md`](docs/guide/cau-hinh.md). With `APP_ENV=production`
the service refuses to start on an unsafe internal secret or backend URL.

## Verification

```bash
task test
task code:check-strict
```

## Commit and Branch Conventions

- Branch: `<prefix>/<owner>-<task-id>-<short-name>`; an `enhance/` branch without a
  Jira ticket uses `enhance/<owner>-<short-name>`
- Commit: `<type>(<scope>): [UNISAGE-xxx] <short English title>`, without the ticket
  when there is none
- Use English Conventional Commits without emoji.
- Never commit directly to `main` or force-push a shared branch.

## Documentation

See [`docs/README.md`](docs/README.md) for the documentation index.
