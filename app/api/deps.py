import json
from dataclasses import dataclass

from fastapi import Depends, Header
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.exceptions import (
    DepartmentAccessDeniedException,
    InsufficientDocumentPermissionException,
    InvalidTrustedContextException,
    MissingTrustedContextException,
)
from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client
from app.database.session import async_session_factory
from app.graph.streaming_state import GraphModels
from app.integrations.backend_java_client import BackendJavaClient
from app.rag.retrieval.service import RetrievalService

_DOCUMENT_WRITE_PERMISSIONS = {"DOCUMENT_ALL", "DOCUMENT_CREATE"}

# Static .env-sourced credential for now — registry-driven provider selection
# (llmProvider -> Model/Provider class mapping from the model registry snapshot)
# is Task 5/9's job.
_OPENAI_API_BASE_URL = "https://api.openai.com/v1"


def get_backend_java_client() -> BackendJavaClient:
    """FastAPI dependency: one `BackendJavaClient` per request.

    Overridden in tests (see tests/api/test_chat_stream_endpoint.py) with a
    client built on `httpx.MockTransport` - never hits a live Java instance
    in the default test run.
    """

    return BackendJavaClient()


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """FastAPI dependency: the session factory `run_and_persist` uses for its
    OWN database session (independent of the request's `db_session`, since
    that background task must keep running after the request scope ends).

    Overridden in tests so `run_and_persist`'s clarification-state write
    lands in the same in-memory SQLite engine the rest of the test uses,
    instead of the real (unreachable in CI) Postgres `DB_URL`.
    """

    return async_session_factory


def get_graph_models() -> GraphModels:
    """FastAPI dependency: the 3 LLM-backed nodes' models for the streaming graph.

    Overridden in tests with `pydantic_ai.models.function.FunctionModel`
    doubles (see tests/llm_mocks.py). Production builds a real `OpenAIChatModel`
    with `settings.OPENAI_API_KEY` passed explicitly - a bare `"openai:<name>"`
    string instead relies on pydantic_ai reading `OPENAI_API_KEY` from the OS
    environment, which `.env` alone does not set.
    """

    model = OpenAIChatModel(
        settings.OPENAI_MODEL,
        provider=OpenAIProvider(
            api_key=settings.OPENAI_API_KEY,
            http_client=build_provider_http_client(
                ProviderConnectionInfo(api_base_url=_OPENAI_API_BASE_URL)
            ),
        ),
    )
    return GraphModels(
        classification=model,
        query_transformation=model,
        generation=model,
        retrieval=RetrievalService(),
    )


@dataclass(frozen=True)
class DepartmentAccessEntry:
    """One department the caller is granted access to, and at what level."""

    department_id: str
    access_level: int


WILDCARD_DEPARTMENT_ID = "*"


@dataclass(frozen=True)
class TrustedContext:
    """Caller identity trusted because the API Gateway injected it after JWT verification."""

    department_access: list[DepartmentAccessEntry]
    permissions: list[str]

    @property
    def has_wildcard_department_access(self) -> bool:
        return any(e.department_id == WILDCARD_DEPARTMENT_ID for e in self.department_access)

    def granted_access_level(self, department_id: str) -> int | None:
        """Highest access_level the caller holds for `department_id`, or None if not granted.

        A wildcard ("*") grant covers every department at its own level.
        """

        levels = [
            e.access_level
            for e in self.department_access
            if e.department_id in (department_id, WILDCARD_DEPARTMENT_ID)
        ]
        return max(levels) if levels else None


async def get_trusted_context(
    x_user_department_access: str | None = Header(default=None),
    x_user_permissions: str | None = Header(default=None),
) -> TrustedContext:
    """Read the gateway-injected trusted headers, failing loudly if either is absent."""

    if not x_user_department_access:
        raise MissingTrustedContextException("X-User-Department-Access")
    if not x_user_permissions:
        raise MissingTrustedContextException("X-User-Permissions")

    return TrustedContext(
        department_access=_parse_department_access(x_user_department_access),
        permissions=_parse_permissions(x_user_permissions),
    )


def _parse_department_access(raw: str) -> list[DepartmentAccessEntry]:
    header_name = "X-User-Department-Access"
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidTrustedContextException(header_name, str(exc)) from exc
    if not isinstance(entries, list):
        raise InvalidTrustedContextException(header_name, "expected a JSON array")
    try:
        return [
            DepartmentAccessEntry(
                department_id=str(entry["department_id"]),
                access_level=int(entry["access_level"]),
            )
            for entry in entries
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidTrustedContextException(header_name, str(exc)) from exc


def _parse_permissions(raw: str) -> list[str]:
    header_name = "X-User-Permissions"
    try:
        permissions = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidTrustedContextException(header_name, str(exc)) from exc
    if not isinstance(permissions, list) or not all(isinstance(p, str) for p in permissions):
        raise InvalidTrustedContextException(header_name, "expected a JSON array of strings")
    return permissions


async def require_document_permission(
    context: TrustedContext = Depends(get_trusted_context),
) -> TrustedContext:
    """FastAPI dependency: 403 unless the caller has DOCUMENT_ALL or DOCUMENT_CREATE.

    Returns the resolved context so callers that also need department
    membership checks (preview/chunking) don't have to depend on
    `get_trusted_context` a second time.
    """

    if _DOCUMENT_WRITE_PERMISSIONS.isdisjoint(context.permissions):
        raise InsufficientDocumentPermissionException()
    return context


def require_department_membership(department_id: str, context: TrustedContext) -> None:
    """Raise 403 unless `department_id` is present in the caller's department_access.

    Only checks membership, not `access_level` — that ceiling only has
    meaning at the embed step, which uses the fuller check in
    `app/api/v1/ingestion.py` instead of this helper.
    """

    if context.granted_access_level(department_id) is None:
        raise DepartmentAccessDeniedException(department_id)
