"""End-to-end run of `evals.run` against the real graph with fake models and retrieval."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.function import FunctionModel

from app.core.config import settings
from app.graph.streaming_state import GraphModels
from app.schemas.retrieval import RetrievedChunk
from evals.recording import RecordingRetrieval
from evals.run import Job, Runner, access_of, select_jobs, spread
from tests.llm_mocks import FakeRetrievalService, make_classification_llm_model

CHUNK = RetrievedChunk(
    chunk_id="doc-1:0",
    content="Sinh viên được phúc khảo trong 7 ngày.",
    source="Quy chế.pdf",
    score=0.9,
    metadata={
        "document_id": "doc-1",
        "department": "uuid-pdt",
        "access_level": 0,
        "is_public": True,
    },
)

QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "d0001",
        "source_set": "demo",
        "category": "normal",
        "question": "Thời hạn phúc khảo là bao lâu?",
        "question_user": "phuc khao trong bao lau v ad",
        "expected_answer": "7 ngày",
        "expected_doc_ids": ["f1"],
        "expected_intent": "academic_advisory",
        "persona": "guest",
        "asker": {"department_access": []},
        "expect_visible": True,
    },
    {
        "id": "q0001",
        "source_set": "official",
        "category": "normal",
        "question": "Câu chỉ có trong bộ official",
        "expected_doc_ids": ["f2"],
    },
    {
        "id": "q0002",
        "source_set": "official",
        "category": "off_topic",
        "question": "Mai trời mưa không?",
        "expected_doc_ids": [],
    },
]


def test_select_jobs_demo_set_and_variants() -> None:
    jobs = select_jobs(QUESTIONS, subset="demo", variant="both")
    assert [job.key for job in jobs] == ["d0001:exact", "d0001:user", "q0002:exact"]
    assert jobs[1].text == "phuc khao trong bao lau v ad"
    assert len(select_jobs(QUESTIONS, subset="official", variant="exact")) == 3


def test_access_of_maps_department_codes_to_uuids() -> None:
    question = {
        "asker": {
            "department_access": [
                {"department_id": "PHONG_DAO_TAO", "access_level": 3},
                {"department_id": "*", "access_level": 5},
            ]
        }
    }
    assert access_of(question, {"PHONG_DAO_TAO": "uuid-pdt"}) == [
        {"department_id": "uuid-pdt", "access_level": 3},
        {"department_id": "*", "access_level": 5},
    ]


@pytest.mark.asyncio
async def test_runner_records_a_rag_turn_and_resumes(
    tmp_path: Path,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_ENABLED", False)
    models = GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        query_transformation=mock_sync_llm_model("HyDE"),
        generation=mock_streaming_llm_model(["Phúc khảo trong 7 ngày [1]."]),
        retrieval=RecordingRetrieval(FakeRetrievalService([CHUNK])),
    )
    runner = Runner(models, {}, {"f1": {"document_id": "doc-1"}}, tmp_path, web="off")
    jobs = [Job(QUESTIONS[0], "exact")]

    await runner.run(jobs)
    await runner.run(jobs)  # already done: nothing runs again

    lines = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["checks"]["outcome"] == ["rag"]
    assert row["checks"]["recall_at_5"] is True
    assert row["checks"]["leaked"] == []
    assert row["context"][0]["document_id"] == "doc-1"
    assert row["retrievals"][0]["chunks"][0]["document_id"] == "doc-1"
    assert row["citations"][0]["documentId"] == "doc-1"
    assert "08_RetrievalFilteringNode" in row["nodes"]
    assert row["ttft_ms"] is not None and row["error"] is None


def test_spread_takes_every_category_first() -> None:
    jobs = select_jobs(QUESTIONS, subset="official", variant="exact")
    assert [job.question["category"] for job in spread(jobs, 2)] == ["normal", "off_topic"]
