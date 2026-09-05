"""Input/output/deps shapes for the streaming graph orchestrator (T1.13a)."""

from dataclasses import dataclass, field

from pydantic_ai.models import Model

from app.schemas.clarification import PendingClarification
from app.schemas.security import AcademicSecurityContext


@dataclass(frozen=True)
class GraphModels:
    """The 4 LLM-backed nodes' models, injected so tests can pass
    `pydantic_ai.models.function.FunctionModel` doubles (see tests/llm_mocks.py)
    instead of hitting a real provider."""

    classification: Model | str
    direct_llm: Model | str
    query_transformation: Model | str
    generation: Model | str


@dataclass
class GraphInput:
    conversation_id: str
    user_message: str
    is_first_turn: bool
    security: AcademicSecurityContext
    confirmed_metadata: dict[str, str] = field(default_factory=dict)
    pending_clarification: PendingClarification | None = None
    clarification_max_retry: int = 2


@dataclass
class GraphOutput:
    response_text: str
    confirmed_metadata: dict[str, str] = field(default_factory=dict)
    pending_clarification: PendingClarification | None = None
    used_ticket_fallback: bool = False
