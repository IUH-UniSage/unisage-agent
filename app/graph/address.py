"""How the assistant addresses the caller, by role: the `XƯNG HÔ` rule of
`common/response_style.yaml`, for the replies code writes without the model (the
first-turn greeting, the social-chat fallback)."""

from dataclasses import dataclass

from app.schemas.security import AcademicSecurityContext


@dataclass(frozen=True)
class Address:
    me: str
    you: str

    @property
    def you_capitalized(self) -> str:
        """The form that starts a sentence: "Anh/chị", "Thầy/Cô"."""

        return self.you[:1].upper() + self.you[1:]


STUDENT = Address(me="mình", you="bạn")
LECTURER = Address(me="em", you="Thầy/Cô")
STAFF = Address(me="em", you="anh/chị")


def address_for(security: AcademicSecurityContext) -> Address:
    """A student or guest is "bạn"; a lecturer "Thầy/Cô"; any other staff "anh/chị"."""

    if security.is_guest or security.is_student:
        return STUDENT
    if "GIANG_VIEN" in security.role.upper():
        return LECTURER
    return STAFF
