def detect_intent(query: str) -> str:
    """Classify whether the query contains one or multiple academic intents."""

    query_lower = query.lower()
    if any(marker in query_lower for marker in (" và ", " hoặc ", "so sánh")):
        return "MULTI_INTENT"
    return "SINGLE_INTENT"
