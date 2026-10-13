"""Built-in formula params read from the documents: unit prices and a programme's
credits are looked up before the student is asked, and a number is kept only when
the chunk it cites really contains it."""

import json
from dataclasses import dataclass, field

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.config import settings
from app.graph.nodes.calculation import CalculationDeps, fill_from_documents
from app.graph.streaming_state import GraphModels
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import RetrieveManyMixin

_FEES = RetrievedChunk(
    chunk_id="hp:1",
    content=(
        "| 3.1 | Khóa tuyển sinh năm học 2025-2026 | | |\n"
        "| | - Khối Kinh tế (các ngành khác): | 980.000 / 1.050.000 | 36.260.000 |\n"
        "| | - Khối Công nghệ: Môn lý thuyết: 980.000 / Môn thực hành: 1.600.000 | | |"
    ),
    source="Quyet Dinh 1035 - Hoc phi 2025-2026",
    score=0.9,
)


@dataclass
class _Retrieval(RetrieveManyMixin):
    chunks: list[RetrievedChunk]
    queries: list[str] = field(default_factory=list)

    def retrieve(
        self, query: str, *, security: AcademicSecurityContext, limit: int | None = None
    ) -> list[RetrievedChunk]:
        del security, limit
        self.queries.append(query)
        return self.chunks


def _deps(answer: object, chunks: list[RetrievedChunk]) -> tuple[CalculationDeps, list[str]]:
    prompts: list[str] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        prompts.append(str(getattr(messages[-1].parts[-1], "content", "")))
        return ModelResponse(parts=[TextPart(content=json.dumps(answer, ensure_ascii=False))])

    model = FunctionModel(function=respond)
    models = GraphModels(
        classification=model,
        query_transformation=model,
        generation=model,
        retrieval=_Retrieval(chunks),
    )
    return CalculationDeps(models=models), prompts


@pytest.fixture(autouse=True)
def _score_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)


@pytest.mark.asyncio
async def test_unit_prices_are_read_from_the_fee_schedule() -> None:
    answer = {
        "values": {
            "don_gia_lt": {"value": 980000, "chunk": 1},
            "don_gia_th": {"value": "1.600.000", "chunk": 1},
        },
        "ask": [],
    }
    deps, prompts = _deps(answer, [_FEES])
    known = {"tc_lt": 15, "tc_th": 4, "khoa_nhap_hoc": "2025-2026"}

    found = await fill_from_documents("tuition_semester", "học phí kỳ này", known, deps)

    assert found.values == {"don_gia_lt": 980000, "don_gia_th": 1600000}
    assert found.sources["don_gia_th"] is _FEES
    assert found.questions == []
    assert "khoa_nhap_hoc: 2025-2026" in prompts[0]
    assert "don_gia_lt: Đơn giá 1 tín chỉ lý thuyết" in prompts[0]
    assert "tc_lt" not in prompts[0].split("<wanted>")[1].split("</wanted>")[0]


@pytest.mark.asyncio
async def test_a_number_the_cited_chunk_does_not_contain_is_dropped() -> None:
    answer = {
        "values": {
            "don_gia_lt": {"value": 950000, "chunk": 1},
            "don_gia_th": {"value": 1600000, "chunk": 2},
        }
    }
    deps, _ = _deps(answer, [_FEES])

    found = await fill_from_documents("tuition_year", "học phí một năm", {}, deps)

    assert found.values == {}


@pytest.mark.asyncio
async def test_several_intakes_ask_which_one_first() -> None:
    answer = {
        "values": {},
        "ask": [
            {"field": "khoa_nhap_hoc", "options": ["2025-2026", "2024-2025", "2025-2026"]},
            {"field": "khoi_nganh", "options": ["Khối Công nghệ"]},
            {"field": "don_gia_lt", "options": ["1", "2"]},
        ],
    }
    deps, _ = _deps(answer, [_FEES])

    found = await fill_from_documents("tuition_program", "học phí cả khoá", {}, deps)

    choice, text = found.questions
    assert choice["field"] == "khoa_nhap_hoc" and choice["kind"] == "choice"
    assert [option["label"] for option in choice["options"]] == ["2025-2026", "2024-2025"]
    assert text["field"] == "khoi_nganh" and text["kind"] == "text"


@pytest.mark.asyncio
async def test_nothing_retrieved_means_no_model_call() -> None:
    deps, prompts = _deps({"values": {}}, [])

    found = await fill_from_documents("tuition_semester", "học phí", {}, deps)

    assert found.values == {} and found.questions == [] and prompts == []


@pytest.mark.asyncio
async def test_nothing_to_look_up_means_no_search() -> None:
    deps, _ = _deps({"values": {}}, [_FEES])
    known = {"tc_lt": 1, "tc_th": 1, "don_gia_lt": 1, "don_gia_th": 1}

    await fill_from_documents("tuition_semester", "học phí", known, deps)

    assert deps.models.retrieval.queries == []  # type: ignore[attr-defined]
