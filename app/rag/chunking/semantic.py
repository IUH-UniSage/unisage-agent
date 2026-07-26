def split_by_paragraphs(text: str) -> list[str]:
    """Use paragraph boundaries as a lightweight semantic chunking fallback."""

    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
