"""`AcademicSecurityContext` — caller identity for the graph.

Shape matches the real JWT/gateway contract:
`department_access: list[{department_id, access_level}]`.
"""

from pydantic import BaseModel


class DepartmentAccessEntry(BaseModel):
    department_id: str
    access_level: int


class AcademicSecurityContext(BaseModel):
    """Trusted caller identity, built entirely from gateway-injected headers.

    A guest (`role == "KHACH"`) has `user_id=None`, `department_access=[]`,
    `permissions=[]` — this is a normal, expected shape (unauthenticated
    lookup is allowed), not an error state.
    """

    user_id: str | None = None
    role: str = "KHACH"
    user_code: str | None = None
    department_access: list[DepartmentAccessEntry] = []
    permissions: list[str] = []

    @property
    def is_guest(self) -> bool:
        return self.role == "KHACH"
