"""Stage 4 - score the recorded turns with RAGAS (`python -m evals.judge`), in its own
environment: `uv run --isolated --no-project --with-requirements evals/judge/requirements.txt
python -m evals.judge --out <run>`. The judge model comes from a provider adapter
(`evals.judge.providers`), picked by `JUDGE_PROVIDER` or `--judge-provider`."""
