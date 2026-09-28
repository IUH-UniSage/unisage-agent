"""Input/output/deps shapes for the streaming graph orchestrator."""

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.models import Model

from app.core.registry.model_registry import CredentialConfig
from app.rag.retrieval.service import RetrievalServiceProtocol
from app.schemas.chat_history import HistoryMessage
from app.schemas.clarification import PendingClarification
from app.schemas.security import AcademicSecurityContext


@dataclass
class GraphModels:
    """The graph's injected dependencies: the 3 LLM-backed nodes' models
    (so tests can pass `pydantic_ai.models.function.FunctionModel` doubles,
    see tests/llm_mocks.py, instead of hitting a real provider) plus the
    retrieval service (so tests can inject a fake Qdrant client/embedder
    instead of hitting the network)."""

    classification: Model | str
    query_transformation: Model | str
    generation: Model | str
    retrieval: RetrievalServiceProtocol
    generation_credential: CredentialConfig | None = None
    snapshot_version: int | None = None


@dataclass
class GraphInput:
    conversation_id: str
    user_message: str
    is_first_turn: bool
    security: AcademicSecurityContext
    confirmed_metadata: dict[str, str] = field(default_factory=dict)
    pending_clarification: PendingClarification | None = None
    clarification_max_retry: int = 2
    history: list[HistoryMessage] = field(default_factory=list)


@dataclass
class GraphOutput:
    response_text: str
    confirmed_metadata: dict[str, str] = field(default_factory=dict)
    pending_clarification: PendingClarification | None = None
    used_ticket_fallback: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)
