import re
import unicodedata

# Each pattern runs on `_fold(text)`: lower-case, diacritics stripped, "đ" -> "d",
# whitespace collapsed - so "Bỏ qua hướng dẫn" and "bo qua huong dan" match alike.
# Patterns aim at instructions to the assistant itself, not at ordinary academic
# wording: "bỏ qua môn này", "hướng dẫn đăng ký học phần" or "bạn là ai" must not
# match. Matches are only logged (see `chat_stream_endpoint`), never blocked.
_INSTRUCTION_NOUNS = (
    r"(?:huong dan|chi dan|chi thi|quy tac|luat le|lenh|instructions?|rules|prompts?)"
)
PROMPT_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions_en": re.compile(
        r"\b(?:ignore|disregard|forget)\b(?:\s+\w+){0,3}\s+"
        r"(?:previous|prior|above|earlier|all|your)\b(?:\s+\w+){0,2}\s+"
        r"(?:instructions?|rules|prompts?|directions)\b"
    ),
    "ignore_instructions_vi": re.compile(
        r"\b(?:bo qua|phot lo|lo di|quen di|dung tuan theo|khong can tuan theo)\s+"
        r"(?:(?:moi|tat ca|toan bo|het)\s+(?:(?:cac|nhung)\s+)?" + _INSTRUCTION_NOUNS + r"\b"
        r"|(?:\w+\s+){0,2}" + _INSTRUCTION_NOUNS + r"\s+(?:\w+\s+)?"
        r"(?:truoc|tren|cu|cua ban|he thong|ban dau|da cho|da duoc giao)\b)"
    ),
    "system_prompt_probe": re.compile(
        r"\b(?:system prompt|prompt he thong|chi dan he thong|chi thi he thong|"
        r"huong dan he thong cua ban)\b"
    ),
    "role_override_en": re.compile(
        r"\byou are now\b|\bdeveloper mode\b|\bjailbreak\b|\bdan mode\b|\byou are dan\b"
        r"|\boverride (?:the |your |all )?(?:rules|instructions|safety)\b"
    ),
    "role_override_vi": re.compile(
        r"\b(?:tu gio|bay gio|ke tu bay gio|tu bay gio)\b(?:\s+\w+){0,2}\s+"
        r"ban\s+(?:la|se la|dong vai)\b"
        r"|\bban (?:gio|bay gio) (?:la|se la|dong vai)\b"
        r"|\b(?:dong vai|gia lam|gia vo la)\b(?:\s+\w+){0,4}\s+"
        r"(?:khong (?:bi )?gioi han|khong co quy tac|quan tri vien|admin|developer"
        r"|nha phat trien)\b"
    ),
}


def _fold(text: str) -> str:
    """Case/diacritic/whitespace-insensitive form for pattern matching.

    "đ"/"Đ" are standalone code points that NFD does not decompose, so they
    are mapped to "d" before stripping combining marks."""

    without_dj = re.sub(r"[đĐ]", "d", text)
    decomposed = unicodedata.normalize("NFD", without_dj)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(stripped.lower().split())


def sanitize_input_text(text: str) -> str:
    """Strip HTML tags and collapse whitespace in a student query.

    Never truncates: the only length limit is `ChatStreamRequest.message`'s
    `max_length`, enforced before this runs (400 / code 4009). Both steps
    here only shorten the text, so the result stays within that limit.
    """
    if not text:
        return ""

    clean_text = re.sub(r"<[^>]*>", "", text)
    return re.sub(r"\s+", " ", clean_text).strip()


def detect_prompt_injection(text: str) -> str | None:
    """Name of the first prompt-injection pattern the text matches, or `None`."""

    folded = _fold(text)
    for name, pattern in PROMPT_INJECTION_PATTERNS.items():
        if pattern.search(folded):
            return name
    return None
