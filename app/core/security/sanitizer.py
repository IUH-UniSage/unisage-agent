import re

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"you are now DAN",
    r"override rules",
]


def sanitize_input_text(text: str, max_length: int = 1000) -> str:
    """Sanitize student query text from HTML tags, excessive whitespace, and injection patterns."""
    if not text:
        return ""

    # Strip HTML tags
    clean_text = re.sub(r"<[^>]*>", "", text)

    # Normalize whitespace
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # Truncate if exceeds max length
    if len(clean_text) > max_length:
        clean_text = clean_text[:max_length]

    return clean_text


def detect_prompt_injection(text: str) -> bool:
    """Detect common prompt injection patterns in input text."""
    lower_text = text.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lower_text):
            return True
    return False
