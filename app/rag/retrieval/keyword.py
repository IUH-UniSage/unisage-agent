def keyword_overlap_score(query: str, content: str) -> float:
    """Score exact query terms for the keyword side of hybrid retrieval."""

    query_terms = {term.strip(".,?!") for term in query.lower().split() if term}
    content_terms = {term.strip(".,?!") for term in content.lower().split()}
    if not query_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms)
