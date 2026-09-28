"""Ticket fallback node - streams a zero-hallucination fallback via LLM.

Activates when retrieval reports `has_valid_context = False`. The model is
given no chunks and no clarification-form machinery, so it structurally
cannot cite a regulation it has no source for - it can only write the
fallback message defined by `common/ticket_fallback.yaml`.
"""

from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.registry.model_registry import CredentialConfig
from app.graph.streaming import AttemptRecorder, FailoverCallback, TokenSink, stream_agent_text
from app.rag.prompting import build_ticket_fallback_prompt
from app.schemas.chat_history import HistoryMessage
from app.schemas.security import AcademicSecurityContext


def build_ticket_fallback_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model)


async def run_ticket_fallback(
    agent: Agent[None, str],
    user_query: str,
    *,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    history: Sequence[HistoryMessage] = (),
    token_sink: TokenSink,
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
    on_failover: FailoverCallback | None = None,
    on_attempt: AttemptRecorder | None = None,
) -> str:
    prompt = build_ticket_fallback_prompt(
        user_query=user_query,
        security=security,
        confirmed_metadata=confirmed_metadata,
        history=history,
    )
    return await stream_agent_text(
        agent,
        prompt,
        token_sink,
        purpose=purpose,
        credential=credential,
        snapshot_version=snapshot_version,
        agent_factory=build_ticket_fallback_agent,
        on_failover=on_failover,
        on_attempt=on_attempt,
    )
