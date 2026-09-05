import pytest

from app.core.exceptions import InvalidTrustedContextException
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
