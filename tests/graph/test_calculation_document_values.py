"""Built-in formula params read from the documents: unit prices and a programme's
credits are looked up before the student is asked, and a number is kept only when
the chunk it cites really contains it."""

import json
from dataclasses import dataclass, field

import pytest
from pydantic import JsonValue
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


_ROW = {
    "khoa_nhap_hoc": "2025-2026",
    "khoi_nganh": "Khối Công nghệ",
    "he_dao_tao": "Đại học chính quy (chương trình chuẩn)",
}


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
    known: dict[str, JsonValue] = {"tc_lt": 15, "tc_th": 4, **_ROW}

    found = await fill_from_documents("tuition_semester", "học phí kỳ này", known, deps)

    assert found.values == {"don_gia_lt": 980000, "don_gia_th": 1600000}
    assert found.sources["don_gia_th"] is _FEES
    assert found.questions == []
    assert dict(found.row)["Khối ngành"] == "Khối Công nghệ"
    assert "khoa_nhap_hoc: 2025-2026" in prompts[0]
    assert "don_gia_lt: Đơn giá 1 tín chỉ lý thuyết" in prompts[0]
    assert "tc_lt" not in prompts[0].split("<wanted>")[1].split("</wanted>")[0]


@pytest.mark.asyncio
async def test_a_price_is_never_read_for_an_assumed_row() -> None:
    """Only the intake was known, and the model priced the student as "Khối Công nghệ"
    on its own (seen 28/10/2026). The prices are dropped and the faculty group and the
    training type (with its programme) are asked before anything is computed."""

    answer = {
        "values": {
            "don_gia_lt": {"value": 980000, "chunk": 1},
            "don_gia_th": {"value": 1600000, "chunk": 1},
        },
        "ask": [{"field": "khoi_nganh", "options": ["Khối Kinh tế", "Khối Công nghệ"]}],
    }
    deps, _ = _deps(answer, [_FEES])
    known: dict[str, JsonValue] = {"tc_lt": 12, "tc_th": 4, "khoa_nhap_hoc": "2022"}

    found = await fill_from_documents("tuition_semester", "tính học phí giùm tôi", known, deps)

    assert found.values == {} and found.row == ()
    assert [question["field"] for question in found.questions] == ["khoi_nganh", "he_dao_tao"]
    faculty, training = found.questions
    assert [option["label"] for option in faculty["options"]] == [
        "Khối Kinh tế",
        "Khối Công nghệ",
        "Ngành Dược",
    ]
    # One question for the training type and its programme: the fee schedule's sections.
    assert training["kind"] == "choice"
    assert [option["label"] for option in training["options"]] == [
        "Đại học chính quy (chương trình chuẩn)",
        "Đại học chính quy tăng cường tiếng Anh",
        "Đại học liên thông, văn bằng 2",
        "Cao học",
        "Nghiên cứu sinh",
    ]


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

    choice, text, training = found.questions
    assert choice["field"] == "khoa_nhap_hoc" and choice["kind"] == "choice"
    assert [option["label"] for option in choice["options"]] == ["2025-2026", "2024-2025"]
    # The faculty group is a choice of the fee schedule's groups, never free text ("cntt").
    assert text["field"] == "khoi_nganh" and text["kind"] == "choice"
    assert training["field"] == "he_dao_tao"


@pytest.mark.asyncio
async def test_nothing_retrieved_means_no_model_call() -> None:
    deps, prompts = _deps({"values": {}}, [])

    found = await fill_from_documents("tuition_semester", "học phí", {}, deps)

    assert found.values == {} and found.questions == [] and prompts == []


@pytest.mark.asyncio
async def test_nothing_to_look_up_means_no_search() -> None:
    deps, _ = _deps({"values": {}}, [_FEES])
    known: dict[str, JsonValue] = {"tc_lt": 1, "tc_th": 1, "don_gia_lt": 1, "don_gia_th": 1}

    await fill_from_documents("tuition_semester", "học phí", known, deps)

    assert deps.models.retrieval.queries == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_document_value_outside_the_params_range_is_dropped() -> None:
    """The 2026 formula sheet says "thang điểm 30"; a model that reads that 30 as the
    aptitude test's top score (ĐTK, a 1200-point scale) is overruled by the param's range -
    otherwise the student is asked their own result again and again (seen 27/10/2026)."""

    sheet = RetrievedChunk(
        chunk_id="xt:1",
        content="Thang điểm: 30 điểm. ĐĐGNL = (Kết quả thi đánh giá năng lực) x 30 / ĐTK",
        source="Công thức tính điểm xét tuyển 2026",
        score=0.9,
    )
    deps, _ = _deps({"values": {"dtk": {"value": 30, "chunk": 1}}, "ask": []}, [sheet])
    known: dict[str, JsonValue] = {"dtn": 26, "dhb": 30, "dgnl": 1100, "uu_tien": 1, "diem_cong": 0}

    found = await fill_from_documents("admission_score_2026", "tính điểm xét tuyển", known, deps)

    assert found.values == {} and found.questions == []


_ENGLISH_ENHANCED = RetrievedChunk(
    chunk_id="hp:4",
    content=(
        "| 4 | Đại học chính quy tăng cường tiếng Anh | | |\n"
        "| 4.1 | Khóa 2025-2026: Khối KT (1.300k/TC) | Khối CN (1.400k/TC) | QPAN (400k)"
        " | 1.300.000 / 1.400.000 | 48.770.000 / 48.000.000 |"
    ),
    source="Quyet Dinh 1035 - Hoc phi 2025-2026",
    score=0.9,
)


@pytest.mark.asyncio
async def test_a_row_with_one_price_per_credit_prices_theory_and_practice_alike() -> None:
    """The English-enhanced programme lists one price per credit (1.400.000 for Khối CN,
    intake 2025): it is read for both prices instead of nothing (seen 28/10/2026, the
    student was asked to type the prices)."""

    answer = {
        "values": {
            "don_gia_lt": {"value": 1400000, "chunk": 1},
            "don_gia_th": {"value": 1400000, "chunk": 1},
        },
        "ask": [],
    }
    deps, _ = _deps(answer, [_ENGLISH_ENHANCED])
    known: dict[str, JsonValue] = {
        "tc_lt": 12,
        "tc_th": 5,
        "khoa_nhap_hoc": "2025",
        "khoi_nganh": "Khối Công nghệ",
        "he_dao_tao": "Đại học chính quy tăng cường tiếng Anh",
    }

    found = await fill_from_documents("tuition_semester", "tính học phí", known, deps)

    assert found.values == {"don_gia_lt": 1400000, "don_gia_th": 1400000}
    assert found.questions == []


@pytest.mark.asyncio
async def test_training_type_choices_keep_the_fee_schedule_sections() -> None:
    """28/10/2026: the model listed the documents' training types (with graduate rows)
    and "tăng cường tiếng Anh" was gone. The undergraduate sections come first, a
    document option that repeats one is dropped, the others follow."""

    answer = {
        "values": {},
        "ask": [
            {
                "field": "he_dao_tao",
                "options": ["Đại học chính quy", "Đại học liên thông, văn bằng 2", "Cao học"],
            }
        ],
    }
    deps, _ = _deps(answer, [_FEES])
    known: dict[str, JsonValue] = {
        "tc_lt": 12,
        "tc_th": 4,
        "khoa_nhap_hoc": "2025",
        "khoi_nganh": "Khối Công nghệ",
    }

    found = await fill_from_documents("tuition_semester", "tính học phí", known, deps)

    [training] = found.questions
    assert [option["label"] for option in training["options"]] == [
        "Đại học chính quy (chương trình chuẩn)",
        "Đại học chính quy tăng cường tiếng Anh",
        "Đại học liên thông, văn bằng 2",
        "Cao học",  # the document's "Cao học" repeats this one: listed once
        "Nghiên cứu sinh",
    ]


_CAMPUSES = RetrievedChunk(
    chunk_id="hp:10",
    content=(
        "| 3.4 | - Khối Kinh tế [LT: 750.000, TH: 790.000] | 750.000 / 790.000 | 23.294.000 |\n"
        "| B | PHÂN HIỆU QUẢNG NGÃI, CƠ SỞ THANH HÓA |  |  |\n"
        "| 1.1 | Khóa tuyển sinh năm học 2025-2026 - Khối Kinh tế | 416.000 | 15.900.000 |"
    ),
    source="Quyet Dinh 1035 - Hoc phi 2025-2026",
    score=0.9,
)
_BRANCH_ROW: dict[str, JsonValue] = {
    "tc_lt": 12,
    "tc_th": 4,
    "khoa_nhap_hoc": "2025",
    "khoi_nganh": "Khối Kinh tế",
    "he_dao_tao": "Đại học chính quy",
    "co_so_hoc": "Phân hiệu Quảng Ngãi",
}


@pytest.mark.asyncio
async def test_a_branch_campus_price_comes_from_the_branch_section_only() -> None:
    """28/10/2026: a Quảng Ngãi student was priced with the main campus's 2022 row."""

    main_row = {
        "values": {
            "don_gia_lt": {"value": 750000, "chunk": 1},
            "don_gia_th": {"value": 790000, "chunk": 1},
        }
    }
    deps, _ = _deps(main_row, [_CAMPUSES])
    found = await fill_from_documents("tuition_semester", "học phí", _BRANCH_ROW, deps)
    assert found.values == {}

    branch_row = {
        "values": {
            "don_gia_lt": {"value": 416000, "chunk": 1},
            "don_gia_th": {"value": 416000, "chunk": 1},
        }
    }
    deps, _ = _deps(branch_row, [_CAMPUSES])
    found = await fill_from_documents("tuition_semester", "học phí", _BRANCH_ROW, deps)
    assert found.values == {"don_gia_lt": 416000, "don_gia_th": 416000}


_FEE_ROWS = """QUYẾT ĐỊNH: > MỨC THU HỌC PHÍ CÁC HỆ ĐÀO TẠO NĂM HỌC 2025 - 2026

| TT | HỆ ĐÀO TẠO / KHÓA TUYỂN SINH | Mức thu 01 tín chỉ | Mức thu theo năm học |
| --- | --- | --- | --- |
[A ĐỐI VỚI TRỤ SỞ CHÍNH > 3 Đại học chính quy > 3.2 Khóa tuyển sinh năm học 2024-2025]
| 3.2 | - Khối Công nghệ [LT: 930.000, TH: 1.530.000] | 930.000 / 1.530.000 | 35.340.000 |
[A ĐỐI VỚI TRỤ SỞ CHÍNH > 3 Đại học chính quy > 3.3 Khóa tuyển sinh năm học 2023-2024]
| 3.3 | - Khối Công nghệ [LT: 840.000, TH: 1.560.000] | 840.000 / 1.560.000 | 33.428.000 |
[A ĐỐI VỚI TRỤ SỞ CHÍNH > 3 Đại học chính quy]
| 3.5 | Khóa tuyển sinh năm học 2021-2022 trở về trước |  |  |
[A ĐỐI VỚI TRỤ SỞ CHÍNH > 3 Đại học chính quy > 3.5 Khóa tuyển sinh năm học 2021-2022 trở về trước]
| 3.5 | - Khối Công nghệ [LT: 700.000, TH: 1.000.000] | 700.000 / 1.000.000 | 25.000.000 |
"""


def test_a_price_must_be_written_under_the_students_intake() -> None:
    from app.graph.nodes.calculation import _in_intake_section

    # Live 29/10/2026: intake 2024 read the 2023-2024 row (840.000 / 1.560.000).
    assert _in_intake_section("2024", _FEE_ROWS, 930000)
    assert _in_intake_section("2024", _FEE_ROWS, 1530000)
    assert not _in_intake_section("2024", _FEE_ROWS, 840000)
    assert not _in_intake_section("2024", _FEE_ROWS, 1560000)
    assert _in_intake_section("2023", _FEE_ROWS, 1560000)
    # "trở về trước" covers older intakes; a chunk naming no intake cannot be checked.
    assert _in_intake_section("2019", _FEE_ROWS, 700000)
    assert not _in_intake_section("2022", _FEE_ROWS, 700000)
    assert _in_intake_section("2024", "| Khối Công nghệ | 840.000 |", 840000)
    assert _in_intake_section(None, _FEE_ROWS, 840000)


@pytest.mark.asyncio
async def test_a_price_read_from_another_intakes_row_is_read_again() -> None:
    # Live 29/10/2026: intake 2024 read the 2023 row; the student was then asked to type
    # the prices. One more read naming the row finds them.
    answers = [
        {"values": {"don_gia_lt": {"value": 840000, "chunk": 1}}, "ask": []},
        {"values": {"don_gia_lt": {"value": 930000, "chunk": 1}}, "ask": []},
    ]
    prompts: list[str] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        prompts.append(str(getattr(messages[-1].parts[-1], "content", "")))
        answer = answers[len(prompts) - 1]
        return ModelResponse(parts=[TextPart(content=json.dumps(answer))])

    model = FunctionModel(function=respond)
    deps = CalculationDeps(
        models=GraphModels(
            classification=model,
            query_transformation=model,
            generation=model,
            retrieval=_Retrieval(
                [RetrievedChunk(chunk_id="hp:rows", content=_FEE_ROWS, source="QĐ 1035", score=0.9)]
            ),
        )
    )
    known: dict[str, JsonValue] = {
        "tc_lt": 12,
        "tc_th": 0,
        "khoa_nhap_hoc": "2024",
        "khoi_nganh": "Khối Công nghệ",
        "he_dao_tao": "Đại học chính quy (chương trình chuẩn)",
    }

    found = await fill_from_documents("tuition_semester", "học phí", known, deps)

    assert found.values.get("don_gia_lt") == 930000
    assert len(prompts) == 2 and "<recheck>" in prompts[1] and "2024-2025" in prompts[1]
    # A right first read costs no second call.
    del answers[0]
    prompts.clear()
    found = await fill_from_documents("tuition_semester", "học phí", known, deps)
    assert found.values.get("don_gia_lt") == 930000 and len(prompts) == 1
