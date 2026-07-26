from app.graph.state import ChatState
from app.rag.reranking.cross_encoder import rerank
from app.schemas.retrieval import RetrievedChunk


def rerank_context(state: ChatState) -> None:
    """Apply the reranking boundary before generation."""

    chunks = [RetrievedChunk.model_validate(chunk) for chunk in state.retrieved_chunks]
    state.retrieved_chunks = [chunk.model_dump() for chunk in rerank(chunks)]
