"""Node 04: `IntentRoutingNode` (T1.6) — deterministic, no LLM.

Deviation from the reference routing table (documented, matches todo.md's
Phase 1 scope): `academic_comparison` (node 07, Phase 2) and
`academic_calculation` (node 08, Phase 3) are not implemented yet, so both
fall back to the unified advisory flow (`QueryTransformationNode`) instead
of a dedicated node - a degraded but functional answer instead of a dead
end. Swap these two entries to the real nodes once Phase 2/3 land.
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
    # Phase 2/3 not implemented yet - degrade to the advisory flow.
    "academic_comparison": "QueryTransformationNode",
    "academic_calculation": "QueryTransformationNode",
    # Should never route here post-greeting-node, but a safe fallback beats a KeyError.
    "greeting": "QueryTransformationNode",
}

SOCIAL_CHAT_TEMPLATE = "Không có gì đâu, bạn cần hỏi thêm gì cứ nhắn cho mình nhé!"


def route_intent(intent: str) -> NextNode:
    return _ROUTING_MAP.get(intent, "QueryTransformationNode")
