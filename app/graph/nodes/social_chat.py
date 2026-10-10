"""Social chat reply - deterministic, no LLM.

One fixed sentence for every social message answered "hi" with "Không có gì
đâu", so the reply is picked by the message's kind (thanks / goodbye /
greeting / anything else), then at random within that kind so repeated
small talk does not read as a canned bot.
"""

import random
import re

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
OTHER_TEMPLATES: list[str] = [
    "Mình là Trợ lý AI Học vụ của trường, chuyên giải đáp quy chế, thủ tục, học phí "
    "và các vấn đề học vụ. Bạn cần mình hỗ trợ gì?",
    "Mình ở đây để hỗ trợ bạn các vấn đề học vụ. Bạn cứ đặt câu hỏi nhé!",
    "Cảm ơn bạn đã trò chuyện cùng mình! Nếu có câu hỏi về học vụ, mình sẵn sàng giúp.",
]
SOCIAL_CHAT_TEMPLATES = THANKS_TEMPLATES + GOODBYE_TEMPLATES + GREETING_TEMPLATES + OTHER_TEMPLATES

# Thanks before goodbye before greeting: "cảm ơn, tạm biệt" is answered as thanks,
# "chào tạm biệt" as goodbye.
_THANKS = re.compile(r"\b(c[ảáa]m\s+[ơo]n|thanks?|thank\s+you|tks|thx)\b", re.IGNORECASE)
_GOODBYE = re.compile(r"\b(t[ạa]m\s+bi[ệe]t|bye|goodbye|h[ẹe]n\s+g[ặa]p\s+l[ạa]i)\b", re.IGNORECASE)
_GREETING = re.compile(r"\b(xin\s+ch[àa]o|ch[àa]o|hello|hi|hey|alo)\b", re.IGNORECASE)


def social_chat_reply(message: str) -> str:
    templates = _templates_for(message)
    return templates[random.randrange(len(templates))]


def _templates_for(message: str) -> list[str]:
    if _THANKS.search(message):
        return THANKS_TEMPLATES
    if _GOODBYE.search(message):
        return GOODBYE_TEMPLATES
    if _GREETING.search(message):
        return GREETING_TEMPLATES
    return OTHER_TEMPLATES
