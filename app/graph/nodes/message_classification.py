"""Message classification node - labels an incoming message with one intent.

The LLM is asked to output exactly one intent label as plain text, not a
structured confidence-scored object; this is enough for the routing node to
pick the right next step.
"""

from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.rag.prompting import append_recent_history, get_templates
from app.schemas.chat_history import HistoryMessage

INTENT_TYPES = (
    "greeting",
    "social_chat",
    "academic_advisory",
    "academic_calculation",
    "academic_procedure",
    "academic_calendar",
    "academic_document",
    "off_topic",
)

_DEFAULT_INTENT = "academic_advisory"


def build_classification_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, system_prompt=get_templates().agent_message_classification)


async def classify_intent(
    agent: Agent[None, str],
    message: str,
    *,
    history: Sequence[HistoryMessage] = (),
) -> str:
    """Run the classification agent and normalize its output to a known intent.

    Falls back to `academic_advisory` for any unrecognized output — the
    default flow (retrieve + rerank + generate) is the safest guess for an
    academic question the classifier didn't cleanly label, and never worse
    than crashing the turn.

    `history` lets a short follow-up ("sao không có ngành X?") be read in
    the context of the previous turn instead of being judged off_topic on
    its own.
    """

    result = await agent.run(append_recent_history(message, history))
    label = (result.output or "").strip().strip("`").strip().lower()
    return label if label in INTENT_TYPES else _DEFAULT_INTENT
