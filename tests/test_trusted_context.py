import json

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.api.deps import (
    TrustedContext,
    get_trusted_context,
    require_department_membership,
    require_document_permission,
)
from app.core.exceptions import UniSageException

app = FastAPI()


@app.exception_handler(UniSageException)
async def _unisage_exception_handler(request: Request, exc: UniSageException) -> JSONResponse:
    del request
    return JSONResponse(status_code=exc.error_code.http_status, content={"message": exc.message})


@app.get("/whoami")
async def whoami(context: TrustedContext = Depends(get_trusted_context)) -> dict[str, object]:
    return {
        "department_access": [
            {"department_id": e.department_id, "access_level": e.access_level}
            for e in context.department_access
        ],
        "permissions": context.permissions,
    }


@app.get("/document-gate")
async def document_gate(
    context: TrustedContext = Depends(require_document_permission),
) -> dict[str, str]:
    del context
    return {"ok": "true"}


@app.get("/department-gate/{department_id}")
async def department_gate(
    department_id: str, context: TrustedContext = Depends(get_trusted_context)
) -> dict[str, str]:
    require_department_membership(department_id, context)
    return {"ok": "true"}


client = TestClient(app)

_VALID_HEADERS = {
    "X-User-Department-Access": json.dumps(
        [
            {"department_id": "CNTT", "access_level": 3},
            {"department_id": "KHOA_KINH_TE", "access_level": 1},
        ]
    ),
    "X-User-Permissions": json.dumps(["DOCUMENT_ALL", "DOCUMENT_CREATE"]),
}


def test_present_valid_headers_build_trusted_context() -> None:
    response = client.get("/whoami", headers=_VALID_HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "department_access": [
            {"department_id": "CNTT", "access_level": 3},
            {"department_id": "KHOA_KINH_TE", "access_level": 1},
        ],
        "permissions": ["DOCUMENT_ALL", "DOCUMENT_CREATE"],
    }


def test_missing_department_access_header_returns_400() -> None:
    response = client.get(
        "/whoami", headers={"X-User-Permissions": _VALID_HEADERS["X-User-Permissions"]}
    )

    assert response.status_code == 400
    assert "X-User-Department-Access" in response.json()["message"]


def test_missing_permissions_header_returns_400() -> None:
    response = client.get(
        "/whoami",
        headers={"X-User-Department-Access": _VALID_HEADERS["X-User-Department-Access"]},
    )

    assert response.status_code == 400
    assert "X-User-Permissions" in response.json()["message"]


def test_malformed_department_access_json_returns_400() -> None:
    response = client.get(
        "/whoami",
        headers={
            "X-User-Department-Access": "not json",
            "X-User-Permissions": _VALID_HEADERS["X-User-Permissions"],
        },
    )

    assert response.status_code == 400
    assert "X-User-Department-Access" in response.json()["message"]


def test_department_access_entry_missing_field_returns_400() -> None:
    response = client.get(
        "/whoami",
        headers={
            "X-User-Department-Access": json.dumps([{"department_id": "CNTT"}]),
            "X-User-Permissions": _VALID_HEADERS["X-User-Permissions"],
        },
    )

    assert response.status_code == 400
    assert "X-User-Department-Access" in response.json()["message"]


def test_malformed_permissions_json_returns_400() -> None:
    response = client.get(
        "/whoami",
        headers={
            "X-User-Department-Access": _VALID_HEADERS["X-User-Department-Access"],
            "X-User-Permissions": "not json",
        },
    )

    assert response.status_code == 400
    assert "X-User-Permissions" in response.json()["message"]


def test_valid_header_with_single_entry_parses_correctly() -> None:
    response = client.get(
        "/whoami",
        headers={
            "X-User-Department-Access": json.dumps([{"department_id": "CNTT", "access_level": 5}]),
            "X-User-Permissions": _VALID_HEADERS["X-User-Permissions"],
        },
    )

    assert response.status_code == 200
    assert response.json()["department_access"] == [{"department_id": "CNTT", "access_level": 5}]


def test_require_document_permission_allows_document_all() -> None:
    headers = {**_VALID_HEADERS, "X-User-Permissions": json.dumps(["DOCUMENT_ALL"])}
    response = client.get("/document-gate", headers=headers)

    assert response.status_code == 200


def test_require_document_permission_allows_document_create() -> None:
    headers = {**_VALID_HEADERS, "X-User-Permissions": json.dumps(["DOCUMENT_CREATE"])}
    response = client.get("/document-gate", headers=headers)

    assert response.status_code == 200


def test_require_document_permission_rejects_unrelated_permission() -> None:
    headers = {**_VALID_HEADERS, "X-User-Permissions": json.dumps(["CHAT_MODEL_READ"])}
    response = client.get("/document-gate", headers=headers)

    assert response.status_code == 403


def test_require_department_membership_allows_granted_department() -> None:
    response = client.get("/department-gate/CNTT", headers=_VALID_HEADERS)

    assert response.status_code == 200


def test_require_department_membership_rejects_ungranted_department() -> None:
    response = client.get("/department-gate/KHOA_LY_LUAN_CHINH_TRI", headers=_VALID_HEADERS)

    assert response.status_code == 403
