def vector_similarity_score(query: str, content: str) -> float:
    """Return a deterministic lexical proxy until pgvector search is wired in."""

    query_terms = {term for term in query.lower().split() if term}
    content_terms = set(content.lower().split())
    if not query_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms)
