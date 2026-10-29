"""A student correcting what they declared ("tôi nhầm, khoá 2023 mới đúng"): node 03
returns `metadata_updates`, applied and saved at once, before anything else runs."""

import json
from dataclasses import replace

import pytest
from pydantic import JsonValue
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.database.repositories.user_profile import ProfileOwner, UserProfileRepository
from app.graph.address import LECTURER, STAFF
from app.graph.metadata_fields import update_notice
from app.graph.nodes.message_classification import (
    build_classification_agent,
    classify_intent,
    parse_classification,
)
from app.graph.streaming_graph import run_graph
from app.graph.streaming_session import _profile_saver
from app.graph.streaming_state import GraphModels
from app.schemas.clarification import CalculationPlan, LastCalculation
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from tests.graph.test_calculation_node import (
    _FEE_TABLE,
    _PRICES_FOUND,
    _TRACE,
    _input,
    _sink,
    _tuition_models,
    _usage_recorder,
)
from tests.llm_mocks import FakeRetrievalService

_PROFILE = {"khoa_nhap_hoc": "2022", "nganh_hoc": "CNTT"}


def _classifier(updates: object, intent: str = "social_chat") -> FunctionModel:
    payload = {
        "tasks": [{"intent": intent, "query": "q", "routing_mode": None}],
        "confidence": 0.9,
        "metadata_updates": updates,
    }

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=json.dumps(payload, ensure_ascii=False))])

    return FunctionModel(function=respond)


def _models(classification: FunctionModel) -> GraphModels:
    return GraphModels(
        classification=classification,
        query_transformation=classification,
        generation=classification,
        retrieval=FakeRetrievalService(),
    )


def test_metadata_updates_are_parsed_leniently() -> None:
    raw = json.dumps(
        {
            "tasks": [{"intent": "social_chat", "query": "q"}],
            "metadata_updates": {"khoa_nhap_hoc": 2023, "nganh_hoc": None, "x": [1], "": "y"},
        }
    )
    assert parse_classification(raw, "q").metadata_updates == {
        "khoa_nhap_hoc": "2023",
        "nganh_hoc": None,
    }
    without = json.dumps({"tasks": [{"intent": "social_chat", "query": "q"}]})
    assert parse_classification(without, "q").metadata_updates == {}


@pytest.mark.asyncio
async def test_the_classifier_sees_the_profile_and_the_profile_field_table() -> None:
    prompts: list[str] = []
    classifier = _classifier({})

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts.append(str(getattr(messages[-1].parts[-1], "content", "")))
        assert classifier.function is not None
        return classifier.function(messages, info)  # type: ignore[return-value]

    agent = build_classification_agent(FunctionModel(function=respond))
    await classify_intent(agent, "tôi nhầm, khoá 2023 mới đúng", profile=_PROFILE)

    assert (
        "<declared_attributes>\n- khoa_nhap_hoc: 2022 (khóa K18)\n- nganh_hoc: CNTT\n"
        "</declared_attributes>" in (prompts[0])
    )
    system = "\n".join(str(prompt) for prompt in agent._system_prompts)
    assert "{profile_fields}" not in system
    assert "| `khoa_nhap_hoc` |" in system and "`tc_lt`" not in system


def test_the_notice_names_each_change() -> None:
    assert update_notice({"khoa_nhap_hoc": "2023", "nganh_hoc": None}) == (
        "Mình đã cập nhật thông tin của bạn - khóa nhập học: 2023; bỏ ngành học."
    )


def test_the_notice_addresses_staff_and_lecturers_by_role() -> None:
    assert update_notice({"don_vi_cong_tac": "Phòng Đào tạo"}, STAFF).startswith(
        "Em đã cập nhật thông tin của anh/chị - "
    )
    assert update_notice({"hoc_vi": "Tiến sĩ"}, LECTURER).startswith(
        "Em đã cập nhật thông tin của Thầy/Cô - "
    )


@pytest.mark.asyncio
async def test_a_correction_is_saved_first_and_announced_at_the_top_of_the_reply() -> None:
    saved: list[dict[str, str]] = []
    tokens: list[str] = []

    async def save(profile: dict[str, str]) -> None:
        assert tokens == []  # saved before anything is shown
        saved.append(profile)

    async def sink(token: str) -> None:
        tokens.append(token)

    # An alias, a calculation field and an unknown field: only the profile field applies.
    updates = {"khoa_tuyen_sinh": "khoá 2023", "tc_lt": "15", "truong_la": "x"}
    output = await run_graph(
        _input("à tôi nhầm, khoá 2023 mới đúng", confirmed_metadata=dict(_PROFILE)),
        _models(_classifier(updates)),
        sink,
        _TRACE,
        _usage_recorder(),
        on_profile_change=save,
    )

    notice = "Mình đã cập nhật thông tin của bạn - khóa nhập học: 2023."
    assert saved == [{"khoa_nhap_hoc": "2023", "nganh_hoc": "CNTT"}]
    assert output.confirmed_metadata == saved[0]
    assert tokens[0] == f"{notice}\n\n"
    assert output.response_text.startswith(f"{notice}\n\n") and len(tokens) == 2


@pytest.mark.asyncio
async def test_the_correction_is_kept_even_when_the_rest_of_the_turn_fails() -> None:
    saved: list[dict[str, str]] = []

    async def save(profile: dict[str, str]) -> None:
        saved.append(profile)

    async def sink(token: str) -> None:
        if not token.startswith("Mình đã cập nhật"):
            raise RuntimeError("stream broke")

    with pytest.raises(RuntimeError):
        await run_graph(
            _input("tôi học Kế toán chứ không phải CNTT", confirmed_metadata=dict(_PROFILE)),
            _models(_classifier({"nganh_hoc": "Kế toán"})),
            sink,
            _TRACE,
            _usage_recorder(),
            on_profile_change=save,
        )
    assert saved == [{"khoa_nhap_hoc": "2022", "nganh_hoc": "Kế toán"}]


@pytest.mark.asyncio
async def test_nothing_to_update_saves_nothing_and_adds_no_notice() -> None:
    saved: list[dict[str, str]] = []
    tokens: list[str] = []

    async def save(profile: dict[str, str]) -> None:
        saved.append(profile)

    async def sink(token: str) -> None:
        tokens.append(token)

    # The same value as the profile is not a change.
    output = await run_graph(
        _input("khoá 2022 nhé", confirmed_metadata=dict(_PROFILE)),
        _models(_classifier({"khoa_nhap_hoc": "2022"})),
        sink,
        _TRACE,
        _usage_recorder(),
        on_profile_change=save,
    )
    assert saved == [] and len(tokens) == 1
    assert not output.response_text.startswith("Mình đã cập nhật")


@pytest.mark.asyncio
async def test_the_session_saver_writes_the_profile_at_once(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    owner = ProfileOwner(user_id="u1")
    await _profile_saver(db_session_factory, owner)({"khoa_nhap_hoc": "2023"})
    async with db_session_factory() as session:
        assert await UserProfileRepository(session).get(owner) == {"khoa_nhap_hoc": "2023"}


@pytest.mark.asyncio
async def test_a_failed_immediate_save_does_not_fail_the_turn() -> None:
    def broken() -> AsyncSession:
        raise RuntimeError("db down")

    await _profile_saver(broken, ProfileOwner(user_id="u1"))({"khoa_nhap_hoc": "2023"})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_follow_up_after_a_correction_recomputes_with_the_new_intake() -> None:
    previous = LastCalculation(
        plan=CalculationPlan(formula_id="tuition_semester"),
        params={
            "tc_lt": 15,
            "tc_th": 4,
            "khoa_nhap_hoc": "2022",
            "khoi_nganh": "Khối Công nghệ",
            "he_dao_tao": "Đại học chính quy",
        },
        title="Học phí một học kỳ",
    )
    query = "tính lại học phí học kỳ này giúp tôi"
    result = await run_graph(
        _input(
            "à tôi nhầm, khoá 2023 mới đúng, tính lại giúp tôi",
            confirmed_metadata=dict(_PROFILE),
            recent_calculations=[previous],
        ),
        _tuition_models(
            {
                "tasks": [{"intent": "academic_calculation", "query": query}],
                "metadata_updates": {"khoa_nhap_hoc": "2023"},
            },
            {"formula_id": "previous", "params": {}, "retrieval_query": None},
            _PRICES_FOUND,
            chunks=[_FEE_TABLE],
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert result.confirmed_metadata["khoa_nhap_hoc"] == "2023"
    assert result.last_calculation is not None
    assert result.last_calculation.params["khoa_nhap_hoc"] == "2023"
    assert result.response_text.startswith("Mình đã cập nhật thông tin của bạn")


@pytest.mark.asyncio
async def test_a_corrected_training_type_reprices_the_tuition_instead_of_reusing_old_prices() -> (
    None
):
    """28/10/2026: "tôi nhầm, tôi học cao học" after a tuition result must save the new
    training type and look the prices up again for that row - the previous turn's prices
    belonged to the undergraduate row."""

    previous = LastCalculation(
        plan=CalculationPlan(formula_id="tuition_semester"),
        params={
            "tc_lt": 12,
            "tc_th": 4,
            "don_gia_lt": 930000,
            "don_gia_th": 980000,
            "khoa_nhap_hoc": "2025-2026",
            "khoi_nganh": "Khối Công nghệ",
            "he_dao_tao": "Đại học chính quy",
        },
        title="Học phí một học kỳ",
    )
    result = await run_graph(
        _input(
            "tôi nhầm, tôi học cao học",
            confirmed_metadata={"khoa_nhap_hoc": "2025", "khoi_nganh": "Khối Công nghệ"},
            recent_calculations=[previous],
        ),
        _tuition_models(
            {
                "tasks": [
                    {"intent": "academic_calculation", "query": "Tính lại học phí, hệ Cao học"}
                ],
                "metadata_updates": {"he_dao_tao": "Cao học"},
            },
            {"formula_id": "previous", "params": {"he_dao_tao": "Cao học"}},
            _PRICES_FOUND,
            chunks=[_FEE_TABLE],
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert result.confirmed_metadata["he_dao_tao"] == "Cao học"
    assert result.last_calculation is not None
    params = result.last_calculation.params
    assert params["he_dao_tao"] == "Cao học"
    # Read again for the new row (the lookup answered 980.000 / 1.600.000), not reused.
    assert (params["don_gia_lt"], params["don_gia_th"]) == (980000, 1600000)


@pytest.mark.asyncio
async def test_an_echoed_old_price_does_not_survive_a_corrected_intake() -> None:
    """28/10/2026: "tôi nhầm, tôi khoá 20" after a K18 tuition result. The extractor
    echoed the whole previous calculation (2022 prices included) and the 2022 prices were
    reused; an echoed value is not one the student gave, so the prices are read again."""

    earlier: dict[str, JsonValue] = {
        "tc_lt": 12,
        "tc_th": 4,
        "don_gia_lt": 750000,
        "don_gia_th": 1130000,
        "khoa_nhap_hoc": "Nhập học 2022 (đang học kỳ 9)",
        "khoi_nganh": "cntt",
        "he_dao_tao": "Đại học chính quy (chương trình chuẩn)",
    }
    previous = LastCalculation(
        plan=CalculationPlan(formula_id="tuition_semester"),
        params=earlier,
        title="Học phí một học kỳ",
    )
    result = await run_graph(
        _input(
            "tôi nhầm , tôi khoá 20",
            confirmed_metadata={"khoa_nhap_hoc": "2022", "khoi_nganh": "cntt"},
            recent_calculations=[previous],
        ),
        _tuition_models(
            {
                "tasks": [{"intent": "academic_calculation", "query": "tôi nhầm , tôi khoá 20"}],
                "metadata_updates": {"khoa_nhap_hoc": "2024"},
            },
            {"formula_id": "previous", "params": {**earlier, "khoa_nhap_hoc": "2024"}},
            _PRICES_FOUND,
            chunks=[_FEE_TABLE],
        ),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )

    assert result.last_calculation is not None
    params = result.last_calculation.params
    assert params["khoa_nhap_hoc"] == "2024"
    assert (params["don_gia_lt"], params["don_gia_th"]) == (980000, 1600000)


def test_a_row_selector_written_differently_is_the_same_row() -> None:
    from pydantic import JsonValue

    from app.graph.nodes.calculation import drop_stale_lookups

    earlier: dict[str, JsonValue] = {
        "khoa_nhap_hoc": "2025-2026",
        "khoi_nganh": "Khối Công nghệ",
        "don_gia_lt": 1,
    }
    same = drop_stale_lookups(
        "tuition_semester", earlier, {**earlier, "khoa_nhap_hoc": "2025"}, keep=set()
    )
    assert same["don_gia_lt"] == 1
    other = drop_stale_lookups(
        "tuition_semester", earlier, {**earlier, "khoi_nganh": "Khối Kinh tế"}, keep=set()
    )
    assert "don_gia_lt" not in other
    typed = drop_stale_lookups(
        "tuition_semester",
        earlier,
        {**earlier, "khoi_nganh": "Khối Kinh tế"},
        keep={"don_gia_lt"},
    )
    assert typed["don_gia_lt"] == 1  # the student gave it this turn


@pytest.mark.asyncio
async def test_a_lecturer_declaring_their_own_degree_is_saved_like_a_student() -> None:
    """The profile is the caller's own, whatever the role: "tôi là tiến sĩ" picks the
    branch of a lecturer's own regulation next time."""

    saved: list[dict[str, str]] = []

    async def save(profile: dict[str, str]) -> None:
        saved.append(profile)

    lecturer = AcademicSecurityContext(user_id="u1", role="GIANG_VIEN")
    output = await run_graph(
        replace(_input("tôi là tiến sĩ, định mức giờ chuẩn của tôi?"), security=lecturer),
        _models(_classifier({"hoc_vi": "Tiến sĩ"})),
        _sink([]),
        _TRACE,
        _usage_recorder(),
        on_profile_change=save,
    )

    assert saved == [{"hoc_vi": "Tiến sĩ"}]
    assert output.confirmed_metadata == {"hoc_vi": "Tiến sĩ"}


_ROW_2022: dict[str, JsonValue] = {
    "tc_lt": 12,
    "tc_th": 4,
    "don_gia_lt": 750000,
    "don_gia_th": 1130000,
    "khoa_nhap_hoc": "2022",
    "khoi_nganh": "Khối Công nghệ",
    "he_dao_tao": "Đại học chính quy (chương trình chuẩn)",
}


@pytest.mark.parametrize(
    ("change", "reprice"),
    [
        ({"khoa_nhap_hoc": "2024"}, True),
        ({"khoi_nganh": "Khối Kinh tế"}, True),
        ({"he_dao_tao": "Đại học chính quy tăng cường tiếng Anh"}, True),
        ({"co_so_hoc": "Phân hiệu Quảng Ngãi"}, True),  # the campus was assumed before
        ({"khoa_nhap_hoc": "Nhập học 2022 (đang học kỳ 9)"}, False),  # same intake
        ({"khoi_nganh": "khối công nghệ"}, False),  # same group, other case
        ({"tc_lt": 15}, False),  # credits are not part of the fee row
        ({"gom_qpan": "có", "tc_qp": 8}, False),
    ],
)
def test_which_follow_up_changes_look_the_unit_prices_up_again(
    change: dict[str, JsonValue], reprice: bool
) -> None:
    from app.graph.nodes.calculation import drop_stale_lookups

    after = drop_stale_lookups("tuition_semester", _ROW_2022, {**_ROW_2022, **change}, keep=set())

    assert ("don_gia_lt" not in after and "don_gia_th" not in after) is reprice
    assert after["tc_lt"] == {**_ROW_2022, **change}["tc_lt"]  # never dropped


def test_a_changed_campus_reprices_after_a_known_campus_too() -> None:
    from app.graph.nodes.calculation import drop_stale_lookups

    main = {**_ROW_2022, "co_so_hoc": "Trụ sở chính TP.HCM"}
    moved = drop_stale_lookups(
        "tuition_semester", main, {**main, "co_so_hoc": "Phân hiệu Quảng Ngãi"}, keep=set()
    )
    same = drop_stale_lookups("tuition_semester", main, dict(main), keep=set())
    assert "don_gia_lt" not in moved and same["don_gia_lt"] == 750000


_ALL_PRICES = RetrievedChunk(
    chunk_id="hp:all",
    content=(
        "| 3.2 | 2024-2025 Khối Kinh tế: 930.000 / 980.000 |\n"
        "| 3.2 | 2024-2025 Khối Công nghệ: 930.000 / 1.530.000 |\n"
        "| 3.3 | 2023-2024 Khối Công nghệ: 840.000 / 1.560.000 |\n"
        "| 3.4 | 2022-2023 Khối Công nghệ: 750.000 / 1.130.000 |"
    ),
    source="Quyet Dinh 1035 - Hoc phi 2025-2026",
    score=0.9,
)


def _prices(lt: int, th: int) -> dict[str, object]:
    return {
        "values": {
            "don_gia_lt": {"value": lt, "chunk": 1},
            "don_gia_th": {"value": th, "chunk": 1},
        },
        "ask": [],
    }


async def _turn(
    message: str,
    profile: dict[str, str],
    previous: LastCalculation,
    *,
    updates: dict[str, str],
    extracted: dict[str, JsonValue],
    prices: dict[str, object] | None,
) -> tuple[dict[str, str], LastCalculation]:
    """One follow-up turn; the extractor echoes the whole previous calculation plus the
    change (the worst case seen live). `prices` is what a lookup would read; None = the
    turn must not look anything up (the scripted model has no answer for it)."""

    scripted: list[object] = [
        {
            "tasks": [{"intent": "academic_calculation", "query": message}],
            "metadata_updates": updates,
        },
        {"formula_id": "previous", "params": {**previous.params, **extracted}},
    ]
    if prices is not None:
        scripted.append(prices)
    result = await run_graph(
        _input(message, confirmed_metadata=dict(profile), recent_calculations=[previous]),
        _tuition_models(*scripted, chunks=[_ALL_PRICES]),
        _sink([]),
        _TRACE,
        _usage_recorder(),
    )
    assert result.last_calculation is not None, result.response_text
    return result.confirmed_metadata, result.last_calculation


@pytest.mark.asyncio
async def test_a_chain_of_corrections_always_computes_with_the_latest_values() -> None:
    profile = {
        "khoa_nhap_hoc": "2022",
        "khoi_nganh": "Khối Công nghệ",
        "he_dao_tao": "Đại học chính quy (chương trình chuẩn)",
    }
    start = LastCalculation(
        plan=CalculationPlan(formula_id="tuition_semester"),
        params=dict(_ROW_2022),
        title="Học phí một học kỳ",
    )

    # 1. The intake: a profile correction, prices of 2024 Khối Công nghệ.
    profile, calc = await _turn(
        "tôi nhầm, tôi khoá 20",
        profile,
        start,
        updates={"khoa_nhap_hoc": "2024"},
        extracted={"khoa_nhap_hoc": "2024"},
        prices=_prices(930000, 1530000),
    )
    assert profile["khoa_nhap_hoc"] == "2024"
    assert (calc.params["don_gia_lt"], calc.params["don_gia_th"]) == (930000, 1530000)

    # 2. The faculty group, on top of turn 1 (2024 stays).
    profile, calc = await _turn(
        "à ngành tôi thuộc khối kinh tế",
        profile,
        calc,
        updates={"khoi_nganh": "Khối Kinh tế"},
        extracted={"khoi_nganh": "Khối Kinh tế"},
        prices=_prices(930000, 980000),
    )
    assert (profile["khoa_nhap_hoc"], profile["khoi_nganh"]) == ("2024", "Khối Kinh tế")
    assert calc.params["khoa_nhap_hoc"] == "2024"
    assert (calc.params["don_gia_lt"], calc.params["don_gia_th"]) == (930000, 980000)

    # 3. Only the credits: same row, the prices are kept (no lookup at all).
    profile, calc = await _turn(
        "kỳ này tôi học 15 tín lý thuyết",
        profile,
        calc,
        updates={},
        extracted={"tc_lt": 15},
        prices=None,
    )
    assert calc.params["tc_lt"] == 15 and calc.params["don_gia_th"] == 980000

    # 4. A hypothetical: computed for 2023 Khối Công nghệ, the profile is not touched.
    profile, calc = await _turn(
        "nếu là khoá 2023 khối công nghệ thì sao",
        profile,
        calc,
        updates={},
        extracted={"khoa_nhap_hoc": "2023", "khoi_nganh": "Khối Công nghệ"},
        prices=_prices(840000, 1560000),
    )
    assert (profile["khoa_nhap_hoc"], profile["khoi_nganh"]) == ("2024", "Khối Kinh tế")
    assert (calc.params["don_gia_lt"], calc.params["don_gia_th"]) == (840000, 1560000)

    # 5. Back to the student's own row: the profile wins over the hypothetical.
    profile, calc = await _turn(
        "tính lại cho khoá và khối của tôi",
        profile,
        calc,
        updates={},
        extracted={"khoa_nhap_hoc": "2024", "khoi_nganh": "Khối Kinh tế"},
        prices=_prices(930000, 980000),
    )
    assert (calc.params["don_gia_lt"], calc.params["don_gia_th"]) == (930000, 980000)
    assert calc.params["tc_lt"] == 15  # the credits of turn 3 carried along


@pytest.mark.asyncio
async def test_the_corrected_profile_wins_over_an_intake_the_extractor_misread() -> None:
    """ "tôi nhầm, tôi khoá 20": the classifier saves 2024 while the extractor reads
    "khoá 20" as 2020 (seen live 28/10/2026). The turn computes with 2024."""

    start = LastCalculation(
        plan=CalculationPlan(formula_id="tuition_semester"), params=dict(_ROW_2022)
    )
    profile, calc = await _turn(
        "tôi nhầm, tôi khoá 20",
        {"khoa_nhap_hoc": "2022", "khoi_nganh": "Khối Công nghệ"},
        start,
        updates={"khoa_nhap_hoc": "2024"},
        extracted={"khoa_nhap_hoc": "2020"},
        prices=_prices(930000, 1530000),
    )
    assert profile["khoa_nhap_hoc"] == "2024" and calc.params["khoa_nhap_hoc"] == "2024"
    assert (calc.params["don_gia_lt"], calc.params["don_gia_th"]) == (930000, 1530000)


@pytest.mark.asyncio
async def test_a_price_the_student_writes_is_kept_and_an_echoed_one_is_not_a_new_value() -> None:
    start = LastCalculation(
        plan=CalculationPlan(formula_id="tuition_semester"), params=dict(_ROW_2022)
    )
    _profile, calc = await _turn(
        "nếu đơn giá lý thuyết là 1.000.000 thì sao",
        {},
        start,
        updates={},
        extracted={"don_gia_lt": 1000000},
        prices=None,  # same row: nothing is looked up
    )
    assert calc.params["don_gia_lt"] == 1000000
    assert calc.params["don_gia_th"] == _ROW_2022["don_gia_th"]


def test_what_the_student_states_about_themselves_in_a_calculation_is_their_profile() -> None:
    from app.graph.nodes.calculation import PlannedTask, declared_profile

    task = PlannedTask(
        "T1",
        "q",
        "tuition_semester",
        {
            "co_so_hoc": "Phân hiệu Quảng Ngãi",
            "khoi_nganh": "Khối Kinh tế",
            "he_dao_tao": "Đại học chính quy",
            "khoa_nhap_hoc": "2025",
            "tc_lt": 12,
        },
    )
    own = "Tôi học ở phân hiệu Quảng Ngãi, khối Kinh tế, đại học chính quy, nhập học 2025"
    assert declared_profile(task, own + ". Tính học phí kỳ này") == {
        "co_so_hoc": "Phân hiệu Quảng Ngãi",
        "khoi_nganh": "Khối Kinh tế",
        "he_dao_tao": "Đại học chính quy",
        "khoa_nhap_hoc": "2025",
    }
    # A hypothetical or a question about nobody in particular changes nothing.
    assert declared_profile(task, "nếu là phân hiệu Quảng Ngãi khối Kinh tế thì sao") == {}
    assert declared_profile(task, "học phí phân hiệu Quảng Ngãi khối Kinh tế 2025") == {}
    # A value the message does not contain (from the profile or a panel) is not re-declared.
    assert declared_profile(task, "tôi học khối Kinh tế, tính học phí") == {
        "khoi_nganh": "Khối Kinh tế"
    }


@pytest.mark.asyncio
async def test_back_to_my_own_row_after_a_hypothetical_even_when_the_extractor_echoes_it() -> None:
    hypothetical = LastCalculation(
        plan=CalculationPlan(formula_id="tuition_semester"),
        params={
            **_ROW_2022,
            "khoa_nhap_hoc": "2023",
            "don_gia_lt": 840000,
            "don_gia_th": 1560000,
        },
    )
    _profile, calc = await _turn(
        "vậy tính lại học phí cho đúng khoá và khối của tôi",
        {"khoa_nhap_hoc": "2024", "khoi_nganh": "Khối Công nghệ"},
        hypothetical,
        updates={},
        extracted={"khoa_nhap_hoc": "2023"},  # the echo of the hypothetical
        prices=_prices(930000, 1530000),
    )
    assert calc.params["khoa_nhap_hoc"] == "2024"
    assert (calc.params["don_gia_lt"], calc.params["don_gia_th"]) == (930000, 1530000)


@pytest.mark.asyncio
async def test_a_correction_written_as_an_intake_number_is_not_an_echo() -> None:
    # Live 28/10/2026: the classifier left `metadata_updates` empty and "khoá 20" has no
    # "2024" in it, so the extractor's 2024 was dropped as an echo.
    start = LastCalculation(
        plan=CalculationPlan(formula_id="tuition_semester"), params=dict(_ROW_2022)
    )
    _profile, calc = await _turn(
        "tôi nhầm, tôi khoá 20",
        {"khoa_nhap_hoc": "2022"},
        start,
        updates={},
        extracted={"khoa_nhap_hoc": "2024"},
        prices=_prices(930000, 1530000),
    )
    assert calc.params["khoa_nhap_hoc"] == "2024"
    assert (calc.params["don_gia_lt"], calc.params["don_gia_th"]) == (930000, 1530000)


@pytest.mark.asyncio
async def test_a_campus_correction_worded_unlike_the_option_label_is_kept() -> None:
    branch = LastCalculation(
        plan=CalculationPlan(formula_id="tuition_semester"),
        params={**_ROW_2022, "co_so_hoc": "Phân hiệu Quảng Ngãi"},
    )
    _profile, calc = await _turn(
        "tôi nhầm, tôi học ở trụ sở chính TP.HCM",
        {"co_so_hoc": "Phân hiệu Quảng Ngãi"},
        branch,
        updates={},
        extracted={"co_so_hoc": "Trụ sở chính TP. Hồ Chí Minh"},
        prices=_prices(750000, 1130000),
    )
    assert calc.params["co_so_hoc"] == "Trụ sở chính TP. Hồ Chí Minh"
    assert (calc.params["don_gia_lt"], calc.params["don_gia_th"]) == (750000, 1130000)


def test_an_intake_number_declares_the_intake_year() -> None:
    from app.graph.nodes.calculation import PlannedTask, declared_profile

    task = PlannedTask("T1", "q", "tuition_semester", {"khoa_nhap_hoc": "2024", "tc_lt": 12})
    assert declared_profile(task, "tôi K20, tính học phí 12 tín chỉ") == {"khoa_nhap_hoc": "2024"}
    assert declared_profile(task, "em khoá 20 tính học phí") == {"khoa_nhap_hoc": "2024"}
    assert declared_profile(task, "tôi khoá 19 tính học phí") == {}
