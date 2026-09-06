"""Intent routing node - deterministic, no LLM.

`academic_comparison` and `academic_calculation` have no dedicated node yet,
so both fall back to the unified advisory flow (`QueryTransformationNode`)
instead of a dedicated node - a degraded but functional answer instead of a
dead end. Swap these two entries to real nodes once those are implemented.
"""

from typing import Literal

NextNode = Literal[
    "END_SOCIAL_CHAT",
    "DirectLLMNode",
    "OffTopicRejectNode",
    "QueryTransformationNode",
]

_ROUTING_MAP: dict[str, NextNode] = {
    "social_chat": "END_SOCIAL_CHAT",
    "general_knowledge": "DirectLLMNode",
    "off_topic": "OffTopicRejectNode",
    "academic_advisory": "QueryTransformationNode",
    "academic_procedure": "QueryTransformationNode",
    "academic_calendar": "QueryTransformationNode",
    "academic_document": "QueryTransformationNode",
    # Not implemented yet - degrade to the advisory flow.
    "academic_comparison": "QueryTransformationNode",
    "academic_calculation": "QueryTransformationNode",
    # Should never route here post-greeting-node, but a safe fallback beats a KeyError.
    "greeting": "QueryTransformationNode",
}

SOCIAL_CHAT_TEMPLATE = "Không có gì đâu, bạn cần hỏi thêm gì cứ nhắn cho mình nhé!"


def route_intent(intent: str) -> NextNode:
    return _ROUTING_MAP.get(intent, "QueryTransformationNode")
