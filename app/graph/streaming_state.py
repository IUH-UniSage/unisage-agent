"""Input/output/deps shapes for the streaming graph orchestrator."""

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.models import Model

from app.core.registry.model_registry import CredentialConfig
from app.rag.retrieval.service import RetrievalServiceProtocol
from app.schemas.chat_history import HistoryMessage
from app.schemas.clarification import PendingClarification, PendingRound
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
    # LLMRerankNode's model - the top-priority RERANK credential (EXTRACTION while no
    # RERANK row exists), not CHAT's. None skips the node; `rerank_unavailable` then says
    # why (for the AI-admin warning). `rerank_purpose` is the pool failover draws from.
    rerank: Model | str | None = None
    rerank_credential: CredentialConfig | None = None
    rerank_purpose: str = "RERANK"
    rerank_unavailable: str | None = None


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


@dataclass(frozen=True)
class AdminWarning:
    code: str
    message: str


@dataclass
class GraphOutput:
    response_text: str
    confirmed_metadata: dict[str, str] = field(default_factory=dict)
    pending_clarification: PendingClarification | None = None
    used_ticket_fallback: bool = False
    used_web_search: bool = False
    # Things an AI admin should fix that did not stop the turn (web search or the
    # LLM rerank failing, ...) - sent to AI admins only, as `event: warning`.
    admin_warnings: list[AdminWarning] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    # The clarification panel this turn raises (stored, projected, then sent as
    # `event: clarification` by run_and_persist), or None.
    pending_round: PendingRound | None = None
    # The advisory answer's captured ask_user_form blocks - input to the panel.
    ask_forms: tuple[dict[str, Any], ...] = ()
