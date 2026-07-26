from app.rag.retrieval.keyword import keyword_overlap_score
from app.rag.retrieval.vector import vector_similarity_score


def hybrid_score(query: str, content: str) -> float:
    """Combine vector and keyword signals with an explicit baseline weight."""

    return 0.7 * vector_similarity_score(query, content) + 0.3 * keyword_overlap_score(
        query, content
    )
