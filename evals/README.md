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
  judge/       4. RAGAS scores on Gemini, in its own environment
    cli.py        python -m evals.judge
    ragas_metrics.py  the RAGAS metrics and the Gemini LLM/embeddings they call
    gemini_pool.py    key pool: per-key RPM, daily-quota reset (or --rpd-recheck), 503/504 move on to another key
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
`uv run --isolated --no-project --with-requirements evals/judge/requirements.txt`. Its keys are
`GEMINI_API_KEYS` in `.env.eval` (comma-separated, never committed). Its tests:
`pytest tests/evals/test_judge.py` in the service environment, and
`tests/evals/test_judge_ragas.py` with `--noconftest` in the judge environment.

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
