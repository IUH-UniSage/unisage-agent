"""Node 05A: `DirectLLMNode` (T1.7) — streaming, no RAG.

Activates for `general_knowledge` (simple, non-academic-specific questions).
No `<academic_context>` block, per the reference design's
`main/chat_direct_llm.yaml` (not ported - see app/rag/prompting/loader.py's
scope-deviation note; this uses a plain instruction string instead).
"""

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.graph.streaming import TokenSink, stream_agent_text

_SYSTEM_PROMPT = (
    "Bạn là Trợ Lý AI Học Vụ. Trả lời ngắn gọn, chính xác cho câu hỏi kiến thức "
    "phổ thông này - không cần tra cứu quy chế, không cần trích dẫn nguồn."
)


def build_direct_llm_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, system_prompt=_SYSTEM_PROMPT)


async def run_direct_llm(agent: Agent[None, str], message: str, token_sink: TokenSink) -> str:
    return await stream_agent_text(agent, message, token_sink)
