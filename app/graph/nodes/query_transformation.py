"""Query transformation node - entry point of the unified
Advisory/Procedure/Document/Calendar flow.

Implements HyDE-only query rewriting (single query, no sub-query fan-out);
other modes can be added later without changing `transform_query`'s
signature.

Follow-up questions: a turn like "còn nghiên cứu sinh khóa 2024-2025?" carries
no topic of its own (the "học phí" lives in the previous turn). Without the
conversation, the model guesses a topic from whatever words are left
("Khóa tuyển sinh" -> admissions procedure), and retrieval lands on the wrong
document. So the last few history messages are passed along, and the model
first rewrites the turn into a self-contained question, then writes the HyDE
document for it. The whole output (question line + document) is used as the
retrieval text, so the question's keywords always reach the embedding even
if the document part drifts.

Resume behavior: when the clarification flow routes back here with a newly
confirmed field, that value is folded into the query passed to the LLM so
retrieval benefits from it immediately (e.g. student confirms "chinh_quy" ->
the query mentions it before retrieval runs again).
"""

from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.rag.prompting import append_recent_history, get_templates
from app.schemas.chat_history import HistoryMessage


def build_query_transformation_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, system_prompt=get_templates().agent_hyde_generator)


def _fold_confirmed_metadata_into_query(user_query: str, confirmed_metadata: dict[str, str]) -> str:
    if not confirmed_metadata:
        return user_query
    declared = ", ".join(f"{field}={value}" for field, value in confirmed_metadata.items())
    return f"{user_query} (thông tin sinh viên đã xác nhận: {declared})"


async def transform_query(
    agent: Agent[None, str],
    user_query: str,
    *,
    confirmed_metadata: dict[str, str] | None = None,
    history: Sequence[HistoryMessage] = (),
) -> str:
    """Returns the retrieval text: the self-contained question followed by
    the HyDE document written for it."""

    enriched_query = _fold_confirmed_metadata_into_query(user_query, confirmed_metadata or {})
    result = await agent.run(append_recent_history(enriched_query, history))
    return result.output


def extract_standalone_question(hyde_output: str) -> str:
    """The self-contained question `transform_query`'s output leads with
    (see `agent_hyde_generator.yaml`'s output format: question line, blank
    line, then the HyDE document) - handed to `GenerationSynthesisNode` so it
    doesn't have to re-derive a follow-up's topic from raw history on its
    own (see `GenerationResult`/`build_system_prompt`'s `resolved_query`).
    Falls back to the first line as-is if the blank-line separator is
    missing (a malformed model output shouldn't crash the turn)."""

    return hyde_output.split("\n\n", 1)[0].strip()
