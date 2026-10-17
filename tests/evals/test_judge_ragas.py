"""Task 11: the real RAGAS metrics driven through `GeminiJudgeLLM` with a fake Gemini client
that answers every structured-output request with JSON built from the requested schema.

Runs in the judge environment only (RAGAS cannot share the service's dependencies):
    uv run --isolated --no-project --with-requirements evals/judge/requirements.txt \\
        python -m pytest tests/evals/test_judge_ragas.py
"""

import asyncio
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("ragas.metrics.collections")
pytest.importorskip("google.genai")

from evals.judge import cli as judge, ragas_metrics as judge_ragas  # noqa: E402
from evals.judge.cli import Task  # noqa: E402


def fake_instance(schema: dict[str, Any], node: dict[str, Any] | None = None) -> Any:
    """The smallest JSON value that satisfies `node` (a part of `schema`): 2 items per list,
    1 for integers (a `verdict`/`attributed` of 1), the first allowed value of an enum."""

    node = schema if node is None else node
    if "$ref" in node:
        return fake_instance(schema, schema["$defs"][node["$ref"].rsplit("/", 1)[-1]])
    if "enum" in node:
        return node["enum"][0]
    if "const" in node:
        return node["const"]
    if "anyOf" in node:
        return fake_instance(schema, node["anyOf"][0])
    kind = node.get("type")
    if kind == "object":
        return {name: fake_instance(schema, sub) for name, sub in node.get("properties", {}).items()}
    if kind == "array":
        return [fake_instance(schema, node.get("items", {"type": "string"})) for _ in range(2)]
    return {"integer": 1, "number": 1.0, "boolean": True}.get(kind, "Học phí 15 triệu đồng.")


class FakeModels:
    def __init__(self, unavailable: set[str]) -> None:
        self.unavailable = unavailable
        self.models: list[str] = []

    async def generate_content(self, model: str, contents: str, config: Any) -> Any:
        self.models.append(model)
        if model in self.unavailable:
            error = Exception("model overloaded")
            error.code = 503  # type: ignore[attr-defined]
            raise error
        schema = config.response_json_schema
        if schema.get("title") == "_TranslatedStrings":  # prompt translation keeps the count
            found = json.loads(re.search(r"\[.*\]", contents, re.DOTALL).group(0))
            value: Any = {"statements": [f"(vi) {text}" for text in found]}
        else:
            value = fake_instance(schema)
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5)
        return SimpleNamespace(text=json.dumps(value, ensure_ascii=False), usage_metadata=usage)

    async def embed_content(self, model: str, contents: list[str]) -> Any:
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1.0, 0.0, 0.0]) for _ in contents],
            usage_metadata=None,
        )


@pytest.fixture
def fake_gemini(monkeypatch) -> FakeModels:
    models = FakeModels(unavailable=set())
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    monkeypatch.setattr(judge_ragas, "_client_cache", lambda: (lambda key: client))
    return models


def _factory(tmp_path: Path, adapt: bool = False) -> judge_ragas.RagasScorerFactory:
    return judge_ragas.RagasScorerFactory(
        keys=["key-one-123456", "key-two-123456"],
        model="gemini-3.1-flash-lite-preview",
        fallback="gemini-3.5-flash-lite",
        embedding_model="gemini-embedding-001",
        thinking="minimal",
        rpm=100,
        prompts_dir=tmp_path / "prompts",
        adapt_prompts=adapt,
        metrics=list(judge.METRICS),
    )


def _row() -> dict[str, Any]:
    return {
        "id": "q1",
        "variant": "original",
        "asked": "Học phí ngành Kế toán bao nhiêu?",
        "category": "normal",
        "expected_answer": "15 triệu đồng mỗi học kỳ.",
        "expected_document_ids": ["doc-a", "doc-b"],
        "response": "Học phí là 15 triệu đồng mỗi học kỳ.",
        "nodes": ["08_RetrievalFilteringNode"],
        "context": [{"document_id": "doc-a", "content": "Học phí 15 triệu đồng."}],
        "web_results": [],
        "citations": [],
    }


def test_every_metric_scores_a_row_through_the_fake_gemini(tmp_path: Path, fake_gemini) -> None:
    scorer = _factory(tmp_path)(False)

    async def go() -> dict[str, judge.Score]:
        return {task.metric: await scorer.score(task) for task in judge.tasks_for(_row())}

    scores = asyncio.run(go())

    assert set(scores) == {*judge.ANSWER_METRICS, "refusal"}
    for metric, score in scores.items():
        assert not math.isnan(score.value), (metric, score.reason)
        assert 0.0 <= score.value <= 1.0
    assert scores["id_context_recall"].value == 0.5  # doc-a of doc-a, doc-b
    assert scores["refusal"].value == 1.0  # the fake picks the first allowed value, "refused"
    usage = scores["faithfulness"].usage
    assert usage["calls"] == 2 and usage["input_tokens"] == 200 and usage["output_tokens"] == 50
    assert usage["model"] == "gemini-3.1-flash-lite-preview" and usage["fallback"] is False
    assert scores["id_context_precision"].usage["calls"] == 0


def test_an_overloaded_main_model_falls_back_and_the_score_says_so(
    tmp_path: Path, fake_gemini, monkeypatch
) -> None:
    async def no_wait(seconds: float) -> None:
        return None

    fake_gemini.unavailable.add("gemini-3.1-flash-lite-preview")
    factory = _factory(tmp_path)
    scorer = factory(False)
    scorer.caller.sleep = no_wait

    score = asyncio.run(scorer.score(Task("q1:original", "refusal", {
        "user_input": "Học phí?", "response": "Không tìm thấy thông tin."})))

    assert score.usage["fallback"] is True
    assert score.usage["model"] == "gemini-3.5-flash-lite"

    primary_only = factory(True)
    primary_only.caller.sleep = no_wait
    failed = asyncio.run(primary_only.score(Task("q1:original", "refusal", {
        "user_input": "Học phí?", "response": "Không tìm thấy thông tin."})))
    assert math.isnan(failed.value) and "overloaded" in failed.reason


def test_missing_context_is_a_nan_with_a_reason_not_a_crash(tmp_path: Path, fake_gemini) -> None:
    scorer = _factory(tmp_path)(False)
    row = {**_row(), "context": []}

    scores = asyncio.run(_score_all(scorer, judge.tasks_for(row)))

    assert math.isnan(scores["faithfulness"].value)
    assert scores["faithfulness"].reason == "no retrieved context"
    assert scores["id_context_recall"].value == 0.0


async def _score_all(scorer: Any, tasks: list[Task]) -> dict[str, judge.Score]:
    return {task.metric: await scorer.score(task) for task in tasks}


def test_prompts_are_translated_once_and_loaded_from_the_cache(tmp_path: Path, fake_gemini) -> None:
    task = Task("q1:original", "context_recall", {
        "user_input": "Học phí?", "retrieved_contexts": ["Học phí 15 triệu."], "reference": "15 triệu."})

    asyncio.run(_factory(tmp_path, adapt=True)(False).score(task))
    cached = sorted(p.name for p in (tmp_path / "prompts").iterdir())
    assert "context_recall.prompt.json" in cached and "faithfulness.nli_statement_prompt.json" in cached
    data = json.loads((tmp_path / "prompts" / "context_recall.prompt.json").read_text(encoding="utf-8"))
    assert data["language"] == "vietnamese"

    calls_before = len(fake_gemini.models)
    scorer = _factory(tmp_path, adapt=False)(False)
    asyncio.run(scorer.score(task))
    assert scorer.metrics["context_recall"].prompt.language == "vietnamese"
    assert len(fake_gemini.models) - calls_before == 1  # the metric call only, no translation


def test_a_correct_answer_with_extra_detail_covers_the_expected_answer(monkeypatch) -> None:
    # d0008: the answer's one claim carries the topic and term the terse expected answer lacks
    # (not backed by it), while the expected answer's one claim is in the answer
    import numpy as np

    metric = judge_ragas.AnswerCorrectness(llm=judge_ragas.GeminiJudgeLLM(None, "m", None, ""))
    checks = {"answer": np.array([False]), "expected": np.array([True])}

    async def verify(text: str, against: str) -> np.ndarray:
        return checks["answer"] if text == "A" else checks["expected"]

    monkeypatch.setattr(metric, "_decompose_and_verify_claims", verify)

    coverage, precision, f1 = asyncio.run(metric.parts(response="A", reference="R"))

    assert (coverage, precision, f1) == (1.0, 0.0, 0.0)
