"""A student correcting what they declared ("tôi nhầm, khoá 2023 mới đúng"): node 03
returns `metadata_updates`, applied and saved at once, before anything else runs."""

import json
from dataclasses import replace

import pytest
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
