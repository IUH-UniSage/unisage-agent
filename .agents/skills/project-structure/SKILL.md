---
name: project-structure
description: >-
  UniSage Agent package structure and dependency-boundary rules. Use when scaffolding the
  application, adding or moving Python modules, deciding where RAG, graph, API, schema,
  database, tests, or evaluation code belongs, or reviewing changes for architectural drift.
---

# UniSage Agent Project Structure

## Overview

Organize `unisage-agent` as a pipeline-oriented modular monolith. Prefer direct, readable
RAG pipeline packages over full DDD or Hexagonal ceremony while preserving clear boundaries.

## Canonical Layout

```text
unisage-agent/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── deps.py
│   │   └── v1/
│   ├── core/
│   ├── rag/
│   │   ├── ingestion/
│   │   ├── chunking/
│   │   ├── embeddings/
│   │   ├── retrieval/
│   │   ├── reranking/
│   │   └── generation/
│   ├── graph/
│   │   └── nodes/
│   ├── database/
│   │   └── repositories/
│   └── schemas/
├── evals/
├── migrations/
├── scripts/
├── taskfiles/
├── tests/
├── pyproject.toml
└── Taskfile.yml
```

Add `__init__.py` to every Python package even when omitted from diagrams.

## Package Responsibilities

| Package | Owns | Must not own |
| --- | --- | --- |
| `app/api` | FastAPI routes, request dependencies, HTTP mapping | RAG logic, SQL, provider calls |
| `app/core` | Settings, errors, logging, cross-cutting constants | Database clients, repositories, business logic |
| `app/rag/ingestion` | Document loading, parsing coordination, ingestion service | HTTP routes |
| `app/rag/chunking` | Recursive, semantic, and metadata-aware chunking | Persistence |
| `app/rag/embeddings` | Active embedding implementation and configuration | Graph decisions |
| `app/rag/retrieval` | Vector, keyword, hybrid search, context assembly | API response mapping |
| `app/rag/reranking` | Reranking implementations | Generation prompts |
| `app/rag/generation` | Agent, prompts, grounded answer and citations | SQL |
| `app/graph` | Pydantic Graph state, dependencies, nodes, routing | SQL, parsing, embedding algorithms |
| `app/database` | Async session, persistence models, repositories | Prompt or graph logic |
| `app/schemas` | API and pipeline data contracts | Service implementation |
| `evals` | Golden datasets and retrieval/answer evaluation | Runtime API code |
| `tests` | Unit, integration, and end-to-end verification | Production code |

## Dependency Direction

Follow this direction:

```text
API -> Graph -> RAG services -> Database repositories
```

Apply these rules:

1. Make API handlers validate and delegate; keep them thin.
2. Make graph nodes orchestrate services; never write SQL or call SDKs directly.
3. Put database access only in repositories.
4. Keep provider-specific code in its pipeline package, such as
   `rag/embeddings/huggingface.py`.
5. Pass `user_faculty` and `user_level` into retrieval and enforce metadata filtering before
   returning chunks.
6. Keep `auth`, `user`, `role`, and general admin business logic in `unisage-backend`.
7. Let `unisage-agent` own only AI-specific administration such as ingestion, evaluation, and
   model configuration.

## Placement Guide

| Change | Place it in |
| --- | --- |
| Add a FastAPI chat endpoint | `app/api/v1/chat.py` |
| Add PDF or DOCX parsing | `app/rag/ingestion/` |
| Add a chunking strategy | `app/rag/chunking/` |
| Add an embedding model | `app/rag/embeddings/` |
| Add BM25, vector, or hybrid search | `app/rag/retrieval/` |
| Add a cross-encoder reranker | `app/rag/reranking/` |
| Add answer prompts or citation logic | `app/rag/generation/` |
| Add intent routing or graph state | `app/graph/` |
| Add a PostgreSQL query | `app/database/repositories/` |
| Add request or response models | `app/schemas/` |
| Add quality datasets or metrics | `evals/` |

## Avoid Premature Abstraction

- Do not introduce full DDD layers or `ports.py` everywhere.
- Add a protocol or base class only when at least one condition is true:
  - two implementations exist,
  - a provider is expected to be swapped soon,
  - tests need a stable fake boundary.
- Do not create empty future modules solely to match the complete tree.
- Prefer one clear implementation first, then extract an interface when evidence justifies it.
- Split a file when it owns multiple responsibilities, not merely because it becomes long.

## Validation

Before finishing structural work:

1. Search for stale imports from moved packages.
2. Run the full pytest suite.
3. Run Ruff formatting and lint checks.
4. Run Mypy strict checks.
5. Verify FastAPI startup and `/api/v1/health`.
6. Update `README.md` and `AGENTS.md` when the canonical layout changes.
