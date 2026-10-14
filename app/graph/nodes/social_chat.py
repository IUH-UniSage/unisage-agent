"""Social chat reply - streamed by the LLM, static templates as the fallback.

`main/chat_social.yaml` gives the model the profile and the security rules but no
chunks, so it can answer "what do you know about me" from the declared attributes
and must not state any regulation; it always ends with one leading question.

When the call fails before the first token, the reply falls back to
`social_chat_reply`: one fixed sentence for every social message answered "hi"
with "Không có gì đâu", so the reply is picked by the message's kind (thanks /
goodbye / "who are you" / greeting / anything else), then at random within that
kind so repeated small talk does not read as a canned bot.
"""

import logging
import random
import re

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.registry.model_registry import CredentialConfig
from app.graph.streaming import (
    AttemptRecorder,
    BudgetContext,
    FailoverCallback,
    TokenSink,
    auxiliary_model_settings,
    stream_agent_text,
)
from app.rag.prompting import build_social_chat_prompt
from app.schemas.security import AcademicSecurityContext

logger = logging.getLogger(__name__)

THANKS_TEMPLATES: list[str] = [
    "Không có gì đâu, bạn cần hỏi thêm gì cứ nhắn cho mình nhé!",
    "Rất vui vì giúp được bạn! Có thắc mắc gì khác về học vụ thì cứ hỏi mình.",
    "Không có chi! Chúc bạn học tốt, cần gì cứ quay lại hỏi mình nhé.",
    "Mình luôn sẵn sàng hỗ trợ. Bạn còn câu hỏi nào nữa không?",
]
GOODBYE_TEMPLATES: list[str] = [
    "Tạm biệt bạn, chúc bạn học tập thật tốt!",
    "Hẹn gặp lại bạn! Khi cần tra cứu học vụ cứ quay lại tìm mình nhé.",
    "Chào bạn nhé, chúc bạn một ngày thật suôn sẻ!",
]
GREETING_TEMPLATES: list[str] = [
    "Chào bạn! Bạn cần mình hỗ trợ gì về học vụ nào?",
    "Hi bạn, mình vẫn ở đây. Bạn muốn hỏi về quy chế, thủ tục hay học phí cứ nhắn nhé!",
    "Xin chào! Bạn cứ gửi câu hỏi, mình sẽ tra cứu trong văn bản của Nhà trường giúp bạn.",
    "Chào bạn, hôm nay mình có thể giúp gì cho việc học của bạn?",
]
# "Bạn là ai / làm được gì": the assistant's own role, worded like the system prompt
# (common/header.yaml) so the bot describes itself the same way everywhere.
IDENTITY_TEMPLATES: list[str] = [
    "Mình là Trợ lý AI Học vụ của Nhà trường, giúp sinh viên, giảng viên và cán bộ tra cứu "
    "chính xác quy chế đào tạo, học phí, học bổng và thủ tục học vụ từ các văn bản chính thức. "
    "Bạn cần mình hỗ trợ gì?",
    "Mình là Trợ lý AI Học vụ của Nhà trường. Mình giúp sinh viên, giảng viên và cán bộ tra cứu "
    "quy chế đào tạo, học phí, học bổng, thủ tục học vụ - mọi câu trả lời đều dựa trên văn bản "
    "chính thức của Trường. Bạn muốn hỏi gì nào?",
]
OTHER_TEMPLATES: list[str] = [
    "Mình là Trợ lý AI Học vụ của trường, chuyên giải đáp quy chế, thủ tục, học phí "
    "và các vấn đề học vụ. Bạn cần mình hỗ trợ gì?",
    "Mình ở đây để hỗ trợ bạn các vấn đề học vụ. Bạn cứ đặt câu hỏi nhé!",
    "Cảm ơn bạn đã trò chuyện cùng mình! Nếu có câu hỏi về học vụ, mình sẵn sàng giúp.",
]
SOCIAL_CHAT_TEMPLATES = (
    THANKS_TEMPLATES + GOODBYE_TEMPLATES + GREETING_TEMPLATES + IDENTITY_TEMPLATES + OTHER_TEMPLATES
)

# Thanks before goodbye before greeting: "cảm ơn, tạm biệt" is answered as thanks,
# "chào tạm biệt" as goodbye.
_THANKS = re.compile(r"\b(c[ảáa]m\s+[ơo]n|thanks?|thank\s+you|tks|thx)\b", re.IGNORECASE)
_GOODBYE = re.compile(r"\b(t[ạa]m\s+bi[ệe]t|bye|goodbye|h[ẹe]n\s+g[ặa]p\s+l[ạa]i)\b", re.IGNORECASE)
# Questions about the assistant itself, with or without diacritics: "bạn là ai", "em là gì",
# "bạn tên gì", "bạn làm/giúp được gì", "ai tạo ra bạn", "who are you".
_IDENTITY = re.compile(
    r"\b(b[ạa]n|em|m[àa]y|c[ậa]u|bot)\s+(l[àa]\s+(ai|g[ìi])|t[êe]n\s+(l[àa]\s+)?g[ìi]"
    r"|(c[óo]\s+th[ểe]\s+)?(l[àa]m|gi[úu]p)\s+(đ|d)[ưuượ]+c\s+g[ìi])"
    r"|\bai\s+t[ạa]o\s+ra\s+(b[ạa]n|em)|\bwho\s+are\s+you\b",
    re.IGNORECASE,
)
_GREETING = re.compile(r"\b(xin\s+ch[àa]o|ch[àa]o|hello|hi|hey|alo)\b", re.IGNORECASE)


def social_chat_reply(message: str) -> str:
    templates = _templates_for(message)
    return templates[random.randrange(len(templates))]


def _templates_for(message: str) -> list[str]:
    if _THANKS.search(message):
        return THANKS_TEMPLATES
    if _GOODBYE.search(message):
        return GOODBYE_TEMPLATES
    if _IDENTITY.search(message):
        return IDENTITY_TEMPLATES
    if _GREETING.search(message):
        return GREETING_TEMPLATES
    return OTHER_TEMPLATES


def build_social_chat_agent(model: Model | str) -> Agent[None, str]:
    # A short reply with no sources to weigh - no thinking, like the auxiliary agents.
    return Agent(model=model, model_settings=auxiliary_model_settings(model))


async def run_social_chat(
    agent: Agent[None, str],
    user_query: str,
    *,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    token_sink: TokenSink,
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
    on_failover: FailoverCallback | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
) -> str:
    prompt = build_social_chat_prompt(
        user_query=user_query,
        security=security,
        confirmed_metadata=confirmed_metadata,
    )
    streamed = False

    async def sink(token: str) -> None:
        nonlocal streamed
        streamed = True
        await token_sink(token)

    try:
        return await stream_agent_text(
            agent,
            prompt,
            sink,
            purpose=purpose,
            credential=credential,
            snapshot_version=snapshot_version,
            agent_factory=build_social_chat_agent,
            on_failover=on_failover,
            on_attempt=on_attempt,
            budget=budget,
        )
    except Exception:
        # Half a reply cannot be finished by a template.
        if streamed:
            raise
        logger.warning("social_chat.llm_failed - falling back to a template", exc_info=True)
    reply = social_chat_reply(user_query)
    await token_sink(reply)
    return reply
