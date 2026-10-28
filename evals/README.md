# evals - RAG evaluation of UniSage

Offline harness, not part of the service image. Design, criteria and budget:
`backend-java/changes/30-09-2026-RAG-Evaluation/` (`spec.md`, `scoring-rules.md`, `todo.md`).
Data lives outside this repo, in the gateway repo's `dataset/official/` (`manifest.csv`,
`files/`, `questions.jsonl`); run outputs go to `runs/<name>/` (git-ignored).

```
evals/
  dataset/     1. build the knowledge base
    crawl/        discover -> download -> triage the IUH PDFs, keep manifest.csv
    label.py      simulated access labels (label_map.yaml)
    titles.py     readable titles for citations
    cost_estimate.py  ingest + evaluation cost before spending anything
    ingest.py     upload through the real flow (backend -> MinIO -> agent -> Qdrant)
    qdrant_titles.py  rename already-ingested chunks' citation without re-ingesting
  questions/   2. the question set: plan.py (which document backs which slot), build.py
  run/         3. run the graph in-process and score what needs no model
    runner.py     python -m evals.run
    recording.py  captures nodes, retrieval, context and TTFT of a turn
    metrics.py    deterministic checks: outcome, recall@5, leakage, citations, latency
  judge/       4. RAGAS scores, in its own environment
    cli.py        python -m evals.judge (flags, picks the providers, prints the summary)
    tasks.py      which metrics a row gets and their inputs
    cost.py       list-price estimate and the --max-cost guard
    runner.py     the run: concurrency, quota waits, fallback re-scoring
    checkpoint.py judge.jsonl, merge into results.jsonl, judge_summary.json
    ledger.py     calls, tokens, USD, models and masked keys of each score
    ragas_metrics.py  the RAGAS metrics and the JudgeLLM/JudgeEmbeddings bridge to an adapter
    providers/    one adapter per judge provider, behind one interface
      base.py         JudgeAdapter: generate_json(model, prompt, schema), embed(model, texts)
      pool.py         key pool: per-key RPM, daily-quota reset, 429/5xx/timeout retries, 401/403 drop
      gemini.py       google-genai (GEMINI_API_KEYS)
      openai_compat.py  any Chat Completions endpoint, e.g. Qwen (JUDGE_OPENAI_*)
      __init__.py     registry + choice from JUDGE_PROVIDER / JUDGE_EMBEDDING_PROVIDER
    prompts/      RAGAS prompt examples translated to Vietnamese (cached, reused)
    requirements.in / requirements.txt  pinned judge environment
```

## Commands (from `unisage-agent/`)

| Stage | Command |
|---|---|
| Crawl | `python -m evals.dataset.crawl.discover --dataset <dataset>`, then `.download`, `.triage` |
| Label, titles | `python -m evals.dataset.label --dataset <dataset>`; `python -m evals.dataset.titles extract/apply ...` |
| Cost estimate | `python -m evals.dataset.cost_estimate ...` |
| Ingest | `task eval:ingest env=eval [set=official] -- --limit 1` |
| Questions | `python -m evals.questions.plan ...`, `python -m evals.questions.build ...` |
| Run | `task eval:run env=eval -- --limit 20 --variant user` (writes `runs/<name>/results.jsonl`) |
| Judge | `task eval:judge -- --out runs/<name> --limit 20` (`--dry-run` estimates the cost only) |

The judge cannot share the service's dependencies (RAGAS 0.4.3 pins `instructor`, which would
downgrade the service's `openai` and `pydantic-ai`), so `task eval:judge` runs it with
`uv run --isolated --no-project --with-requirements evals/judge/requirements.txt`. Its tests:
`pytest tests/evals/test_judge.py tests/evals/test_judge_openai_compat.py` in the service
environment, and `tests/evals/test_judge_ragas.py` with `--noconftest` in the judge environment.

## Judge provider

Picked from `.env.eval` (never committed) or the environment; `--judge-provider` and
`--embedding-provider` override, `--judge-model` / `--judge-fallback` / `--embedding-model`
override the models.

| Variable | Meaning |
|---|---|
| `JUDGE_PROVIDER` | `gemini` (default) or `openai_compat` |
| `JUDGE_EMBEDDING_PROVIDER` | who embeds for answer_relevancy; default the judge provider if it can embed, else `gemini` |
| `GEMINI_API_KEYS` | Gemini keys, comma separated |
| `JUDGE_OPENAI_BASE_URL`, `JUDGE_OPENAI_API_KEYS` | endpoint and keys of `openai_compat` |
| `JUDGE_OPENAI_MODEL`, `JUDGE_OPENAI_FALLBACK_MODEL` | judge model (required) and optional fallback |
| `JUDGE_OPENAI_EMBEDDING_MODEL` | set only if the endpoint has `/embeddings` |
| `JUDGE_OPENAI_RESPONSE_FORMAT` | `json_object` (default), `json_schema` or `none` |
| `JUDGE_OPENAI_RPM` | requests per minute per key (default `--rpm`) |

A new provider is one module in `providers/` implementing `JudgeAdapter` plus one line in
`providers.ADAPTERS`. Scores made by different judges are not comparable: judge a run with a
new provider in its own folder (copy `results.jsonl` and `config.json`, not `judge.jsonl`);
`judge_summary.json` records the providers and models used.

## What the judge scores

Per row of `results.jsonl` (`spec.md` §4); every score lands in `runs/<name>/judge.jsonl` as it is
computed (a rerun only scores what is missing or NaN), then in `results.jsonl` (`row["ragas"]`)
and `judge_summary.json`. Pass mark 0.7.

| Metric | Rows | Question it answers |
|---|---|---|
| `id_context_recall` | with an expected answer | Did retrieval bring the expected documents? (no LLM) |
| `id_context_precision` | with an expected answer | Of the documents retrieved, how many were expected? (no LLM) |
| `context_recall` | with an expected answer | Does the retrieved text hold the facts of the expected answer? |
| `faithfulness` | with an expected answer | Is every claim of the answer supported by the retrieved text? |
| `answer_coverage` | with an expected answer | Share of the expected answer's claims the answer states (criterion #3) |
| `answer_precision`, `factual_correctness_f1` | same calls as above | Share of the answer's claims the expected answer backs; RAGAS' F1. Informative only: expected answers are terse |
| `answer_relevancy` | with an expected answer | Does the answer address the question asked? |
| `refusal` | every answered row | Did the answer refuse or hand over to a ticket? (1 = refused) |
| `web_faithfulness` | answered from the web | Is every claim supported by the fetched pages? |
