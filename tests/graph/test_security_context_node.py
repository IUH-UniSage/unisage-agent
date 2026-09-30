import pytest

from app.core.errors.exceptions import InvalidTrustedContextException
from app.graph.nodes.security_context import parse_security_headers, resolve_clarification_guard
from app.schemas.clarification import PendingClarification


@pytest.mark.asyncio
async def test_missing_headers_yields_guest_context() -> None:
    context = await parse_security_headers(None, None, None, None, None)

    assert context.role == "KHACH"
    assert context.is_guest is True
    assert context.user_id is None
    assert context.department_access == []
    assert context.permissions == []


@pytest.mark.asyncio
async def test_valid_headers_yield_full_context() -> None:
    context = await parse_security_headers(
        "user-1",
        "SINH_VIEN",
        "SV001",
        '[{"department_id": "KHOA_CNTT", "access_level": 2}]',
        '["DOCUMENT_READ"]',
    )

    assert context.role == "SINH_VIEN"
    assert context.is_guest is False
    assert context.user_id == "user-1"
    assert context.user_code == "SV001"
    assert context.department_access[0].department_id == "KHOA_CNTT"
    assert context.department_access[0].access_level == 2
    assert context.permissions == ["DOCUMENT_READ"]


@pytest.mark.asyncio
async def test_present_but_malformed_header_is_400_not_401() -> None:
    with pytest.raises(InvalidTrustedContextException):
        await parse_security_headers("user-1", "SINH_VIEN", "SV001", "not-json", '["X"]')


def test_guard_no_pending_passes_through() -> None:
    result = resolve_clarification_guard(
        user_message="hi", pending=None, confirmed_metadata={}, max_retry=2
    )

    assert result.matched is False
    assert result.route_to_origin is False
    assert result.pending_clarification is None


def test_guard_matches_option_by_typed_reply_writes_confirmed_metadata() -> None:
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong", "vlvh"]],
        retry_count=0,
    )

    result = resolve_clarification_guard(
        user_message="Chính quy ạ", pending=pending, confirmed_metadata={}, max_retry=2
    )

    assert result.matched is True
    assert result.route_to_origin is True
    assert result.origin_node == "QueryTransformationNode"
    assert result.confirmed_metadata == {"training_type": "chinh_quy"}
    assert result.pending_clarification is None
    assert result.skip_classification is True


def test_guard_no_match_below_retry_limit_bumps_retry_count() -> None:
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong", "vlvh"]],
        retry_count=0,
    )

    result = resolve_clarification_guard(
        user_message="mình không hiểu câu hỏi",
        pending=pending,
        confirmed_metadata={},
        max_retry=2,
    )

    assert result.matched is False
    assert result.route_to_origin is False
    assert result.pending_clarification is not None
    assert result.pending_clarification.retry_count == 1


def test_guard_matches_option_containing_the_dj_stroke_letter() -> None:
    """Regression: "đ" (U+0111) is a standalone Unicode code point, not a
    base letter + combining mark - NFD normalization alone leaves it
    untouched ("đại" -> "đai", not "dai"), so it silently failed to match
    "chinh_quy_dai_tra" until handled explicitly."""

    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["he_dao_tao"],
        options=[["chinh_quy_dai_tra", "chinh_quy_clc", "lien_thong", "vlvh"]],
        retry_count=0,
    )

    result = resolve_clarification_guard(
        user_message="Chính quy đại trà, khoá K21",
        pending=pending,
        confirmed_metadata={},
        max_retry=2,
    )

    assert result.matched is True
    assert result.confirmed_metadata == {"he_dao_tao": "chinh_quy_dai_tra"}
    assert result.pending_clarification is None


def test_guard_matches_all_fields_at_once_from_one_comma_separated_reply() -> None:
    """Regression: a reply answering every field of a multi-field form in one
    go ("Chính quy, K21, Cử nhân") used to only ever save the LAST field -
    both because the matcher returned on its first hit, and because commas
    broke the surrounding-space substring check for the middle field."""

    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["he_dao_tao", "khoa_nhap_hoc", "chuong_trinh_dao_tao"],
        options=[
            ["chinh_quy", "lien_thong", "vlvh"],
            ["k21", "k22", "k23"],
            ["cu_nhan", "ky_su"],
        ],
        retry_count=0,
    )

    result = resolve_clarification_guard(
        user_message="Chính quy, K21, Cử nhân",
        pending=pending,
        confirmed_metadata={},
        max_retry=2,
    )

    assert result.matched is True
    assert result.route_to_origin is True
    assert result.confirmed_metadata == {
        "he_dao_tao": "chinh_quy",
        "khoa_nhap_hoc": "k21",
        "chuong_trinh_dao_tao": "cu_nhan",
    }
    assert result.pending_clarification is None
    assert result.skip_classification is True


def test_guard_partial_match_keeps_only_unresolved_fields_pending() -> None:
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["he_dao_tao", "khoa_nhap_hoc"],
        options=[["chinh_quy", "lien_thong"], ["k21", "k22"]],
        retry_count=1,
    )

    result = resolve_clarification_guard(
        user_message="Chính quy ạ",
        pending=pending,
        confirmed_metadata={"existing": "value"},
        max_retry=3,
    )

    assert result.matched is True
    assert result.route_to_origin is True
    assert result.confirmed_metadata == {"existing": "value", "he_dao_tao": "chinh_quy"}
    assert result.pending_clarification is not None
    assert result.pending_clarification.missing_fields == ["khoa_nhap_hoc"]
    assert result.pending_clarification.options == [["k21", "k22"]]
    # Partial progress resets retry_count - it isn't a failed attempt.
    assert result.pending_clarification.retry_count == 0
    assert result.skip_classification is True


def test_guard_no_match_at_retry_limit_discards_pending() -> None:
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong", "vlvh"]],
        retry_count=1,
    )

    result = resolve_clarification_guard(
        user_message="mình không hiểu câu hỏi",
        pending=pending,
        confirmed_metadata={},
        max_retry=2,
    )

    assert result.matched is False
    assert result.route_to_origin is False
    assert result.pending_clarification is None
