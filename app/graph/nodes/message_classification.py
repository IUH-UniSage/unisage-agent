"""Node 03: `MessageClassificationNode` (T1.5).

Simplified relative to the reference design's full 9-intent classifier
prompt (`agents/message_classification.yaml`, with confidence score and
`routing_mode` SINGLE/MULTI) - documented deviation: the LLM is asked to
output exactly one intent label as plain text, not a structured
confidence-scored object. This is enough for IntentRoutingNode (T1.6) to
route correctly; a follow-up task can port the richer prompt + structured
output + MULTI routing_mode without changing this function's signature.
"""

from pydantic_ai import Agent
from pydantic_ai.models import Model

INTENT_TYPES = (
    "greeting",
    "social_chat",
    "general_knowledge",
    "academic_advisory",
    "academic_comparison",
    "academic_calculation",
    "academic_procedure",
    "academic_calendar",
    "academic_document",
    "off_topic",
)

_DEFAULT_INTENT = "academic_advisory"

_SYSTEM_PROMPT = (
    "Bạn phân loại tin nhắn của sinh viên vào ĐÚNG MỘT nhãn trong danh sách sau, "
    "trả về CHÍNH XÁC nhãn đó, không thêm chữ nào khác:\n" + ", ".join(INTENT_TYPES)
)


def build_classification_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, system_prompt=_SYSTEM_PROMPT)


async def classify_intent(agent: Agent[None, str], message: str) -> str:
    """Run the classification agent and normalize its output to a known intent.

    Falls back to `academic_advisory` for any unrecognized output — the
    default flow (retrieve + rerank + generate) is the safest guess for an
    academic question the classifier didn't cleanly label, and never worse
    than crashing the turn.
    """

    result = await agent.run(message)
    label = (result.output or "").strip().lower()
    return label if label in INTENT_TYPES else _DEFAULT_INTENT
