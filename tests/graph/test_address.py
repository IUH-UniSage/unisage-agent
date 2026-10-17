"""How code-written replies address the caller: by role, like XƯNG HÔ in the prompts."""

import pytest

from app.graph.address import LECTURER, STAFF, STUDENT, address_for
from app.graph.nodes.greeting import GREETING_TEMPLATE, greeting_for
from app.graph.nodes.social_chat import SOCIAL_CHAT_TEMPLATES, social_chat_reply
from app.schemas.security import AcademicSecurityContext


@pytest.mark.parametrize(
    ("role", "address"),
    [
        ("KHACH", STUDENT),
        ("SINH_VIEN", STUDENT),
        ("GIANG_VIEN", LECTURER),
        ("CAN_BO", STAFF),
        ("SUPER_ADMIN", STAFF),
    ],
)
def test_address_follows_the_role(role: str, address: object) -> None:
    caller = AcademicSecurityContext(user_id=None if role == "KHACH" else "u1", role=role)
    assert address_for(caller) == address


def test_a_student_keeps_the_greeting_and_a_lecturer_is_greeted_as_thay_co() -> None:
    student = AcademicSecurityContext(user_id="u1", role="SINH_VIEN")
    lecturer = AcademicSecurityContext(user_id="u1", role="GIANG_VIEN")
    staff = AcademicSecurityContext(user_id="u1", role="CAN_BO")

    assert greeting_for(student) == GREETING_TEMPLATE
    assert greeting_for(lecturer).startswith("Em chào Thầy/Cô!")
    assert "Anh/chị cần em hỗ trợ" in greeting_for(staff)
    for text in (greeting_for(lecturer), greeting_for(staff)):
        assert " bạn" not in text and "mình" not in text.lower()


@pytest.mark.parametrize("message", ["cảm ơn", "tạm biệt", "chào", "bạn là ai", "ok"])
def test_the_social_fallback_never_calls_a_lecturer_ban(message: str) -> None:
    reply = social_chat_reply(message, LECTURER)

    assert reply not in SOCIAL_CHAT_TEMPLATES
    assert "Thầy/Cô" in reply and " bạn" not in reply and "học tốt" not in reply
