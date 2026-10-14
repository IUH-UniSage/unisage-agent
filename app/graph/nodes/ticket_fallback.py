"""Ticket fallback node - streams a zero-hallucination fallback via LLM.

Activates when retrieval reports `has_valid_context = False`. The model is
given no chunks and no clarification-form machinery, so it structurally
cannot cite a regulation it has no source for - it can only write the
fallback message defined by `common/ticket_fallback.yaml`, plus links to the
official pages the web judge kept without an answer in their snippet
(`common/web_candidate_sources.yaml`). A reply that names none of those pages
gets them listed after it, so a page the judge kept is never lost.
"""

from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.registry.model_registry import CredentialConfig
from app.graph.streaming import (
    AttemptRecorder,
    BudgetContext,
    FailoverCallback,
    TokenSink,
    generation_model_settings,
    stream_agent_text,
)
from app.rag.prompting import build_ticket_fallback_prompt
from app.schemas.chat_history import HistoryMessage
from app.schemas.security import AcademicSecurityContext
from app.schemas.web_search import WebSearchResult


def build_ticket_fallback_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, model_settings=generation_model_settings(model))


async def run_ticket_fallback(
    agent: Agent[None, str],
    user_query: str,
    *,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    history: Sequence[HistoryMessage] = (),
    web_sources: Sequence[WebSearchResult] = (),
    token_sink: TokenSink,
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
    on_failover: FailoverCallback | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
) -> str:
    prompt = build_ticket_fallback_prompt(
        user_query=user_query,
        security=security,
        confirmed_metadata=confirmed_metadata,
        history=history,
        web_sources=web_sources,
    )
    text = await stream_agent_text(
        agent,
        prompt,
        token_sink,
        purpose=purpose,
        credential=credential,
        snapshot_version=snapshot_version,
        agent_factory=build_ticket_fallback_agent,
        on_failover=on_failover,
        on_attempt=on_attempt,
        budget=budget,
    )
    links = missing_source_links(text, web_sources)
    if links:
        await token_sink(links)
    return text + links


def missing_source_links(text: str, web_sources: Sequence[WebSearchResult]) -> str:
    """The pages to list after a reply that links none of them, or nothing."""

    if not web_sources or any(source.url in text for source in web_sources):
        return ""
    lines = []
    for source in web_sources:
        year = f" (năm học {source.academic_year})" if source.academic_year else ""
        title = source.title.replace("[", "(").replace("]", ")").strip() or source.url
        lines.append(f"- [{title}]({source.url}){year}")
    return "\n\nVăn bản liên quan trên website của Trường:\n" + "\n".join(lines)
