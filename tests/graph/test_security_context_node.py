import pytest

from app.core.errors.exceptions import InvalidTrustedContextException
from app.graph.nodes.security_context import parse_security_headers


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
