"""Node 06: `QueryTransformationNode` (T1.8) — entry point of the unified
Advisory/Procedure/Document/Calendar flow.

Simplified relative to the reference design's 4 modes (HyDE/Multi-query/
Procedure/Document) - documented deviation: this implements HyDE-only
(single query, no sub-query fan-out). Multi-query/Procedure/Document mode
selection can be added later without changing `transform_query`'s signature
(a `mode` parameter is threaded through, currently only affecting the
instruction text).

Resume behavior (the part plan.md calls out specifically): when the
Clarification Guard routes back here with a newly confirmed field, that
value is folded into the query passed to the LLM so retrieval benefits from
it immediately - per missing_metadata_clarification_design.md section 8's
worked example (student confirms "chinh_quy" -> query mentions it before
node 10 retrieval runs again).
"""

from pydantic_ai import Agent
from pydantic_ai.models import Model

_SYSTEM_PROMPT = (
    "Bạn sinh một đoạn văn bản hành chính giả định (HyDE) trả lời câu hỏi của sinh viên, "
    "văn phong giống một điều khoản quy chế thật, để dùng làm truy vấn tìm kiếm ngữ nghĩa. "
    "Chỉ trả về đoạn văn bản đó, không giải thích thêm."
)


def build_query_transformation_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, system_prompt=_SYSTEM_PROMPT)


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
) -> str:
    """Returns the HyDE document text to use for retrieval (node 10)."""

    enriched_query = _fold_confirmed_metadata_into_query(user_query, confirmed_metadata or {})
    result = await agent.run(enriched_query)
    return result.output
