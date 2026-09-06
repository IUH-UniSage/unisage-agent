"""Streaming direct-LLM node - answers general-knowledge questions without RAG.

Activates for `general_knowledge` (simple, non-academic-specific questions).
Uses `chat_direct_llm.yaml` - identity-aware (still renders the two
`{academic_metadata}` tags) but has no `<academic_context>`/`task_1`/`task_2`,
since there's no retrieved context to reason about for this intent.
"""

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.graph.streaming import TokenSink, stream_agent_text
from app.rag.prompting import build_direct_llm_prompt
from app.schemas.security import AcademicSecurityContext


def build_direct_llm_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model)


async def run_direct_llm(
    agent: Agent[None, str],
    *,
    user_query: str,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    token_sink: TokenSink,
) -> str:
    prompt = build_direct_llm_prompt(
        user_query=user_query,
        security=security,
        confirmed_metadata=confirmed_metadata,
    )
    return await stream_agent_text(agent, prompt, token_sink)
