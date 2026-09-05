from app.graph.state import ChatState
from app.rag.reranking.cross_encoder import rerank
from app.schemas.retrieval import RetrievedChunk


def rerank_context(state: ChatState) -> None:
    """Apply the reranking boundary before generation.

    This is the pre-Phase-1 prototype's `/chat` endpoint, being superseded by
    the streaming graph (T1.13a-e) - kept working as-is in the meantime with
    `score_threshold=0.0` (no filtering) so it doesn't regress against the
    demo corpus's scores, which sit below the new T1.10 default threshold.
    The real graph's `PostRetrievalRerankNode` (app/graph/nodes/post_retrieval_rerank.py)
    uses the real `settings.RERANK_SCORE_THRESHOLD` default instead.
    """

    chunks = [RetrievedChunk.model_validate(chunk) for chunk in state.retrieved_chunks]
    state.retrieved_chunks = [
        chunk.model_dump() for chunk in rerank(chunks, score_threshold=0.0).chunks
    ]
