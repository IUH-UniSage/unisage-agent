from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.api.deps import TrustedContext, get_trusted_context
from app.core.exceptions import UniSageException

app = FastAPI()


@app.exception_handler(UniSageException)
async def _unisage_exception_handler(request: Request, exc: UniSageException) -> JSONResponse:
    del request
    return JSONResponse(status_code=exc.status_code, content={"message": exc.message})


@app.get("/whoami")
async def whoami(context: TrustedContext = Depends(get_trusted_context)) -> dict[str, str]:
    return {"department": context.department, "access_level": context.access_level}


client = TestClient(app)


def test_present_valid_headers_build_trusted_context() -> None:
    response = client.get(
        "/whoami",
        headers={"X-User-Department": "CNTT", "X-User-Access-Level": "STUDENT"},
    )

    assert response.status_code == 200
    assert response.json() == {"department": "CNTT", "access_level": "STUDENT"}


def test_missing_department_header_returns_400() -> None:
    response = client.get("/whoami", headers={"X-User-Access-Level": "STUDENT"})

    assert response.status_code == 400
    assert "X-User-Department" in response.json()["message"]


def test_missing_access_level_header_returns_400() -> None:
    response = client.get("/whoami", headers={"X-User-Department": "CNTT"})

    assert response.status_code == 400
    assert "X-User-Access-Level" in response.json()["message"]
