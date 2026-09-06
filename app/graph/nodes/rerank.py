from app.graph.state import ChatState
from app.rag.reranking.cross_encoder import rerank
from app.schemas.retrieval import RetrievedChunk


def rerank_context(state: ChatState) -> None:
    """Apply the reranking boundary before generation.

    Used by the older non-streaming `/chat` prototype endpoint, which is
    being superseded by the streaming graph. Kept working as-is with
    `score_threshold=0.0` (no filtering) so it doesn't regress against the
    demo corpus's scores, which sit below the streaming graph's
    `PostRetrievalRerankNode` default threshold
    (app/graph/nodes/post_retrieval_rerank.py, `settings.RERANK_SCORE_THRESHOLD`).
    """

    chunks = [RetrievedChunk.model_validate(chunk) for chunk in state.retrieved_chunks]
    state.retrieved_chunks = [
        chunk.model_dump() for chunk in rerank(chunks, score_threshold=0.0).chunks
    ]
