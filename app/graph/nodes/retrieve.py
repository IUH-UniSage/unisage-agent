from app.graph.deps import ChatDeps
from app.graph.state import ChatState
from app.rag.retrieval.service import RetrievalService

retrieval_service = RetrievalService()


def retrieve_context(state: ChatState, deps: ChatDeps) -> None:
    """Populate graph state through the RAG retrieval service."""

    del deps
    chunks = retrieval_service.retrieve(state.query)
    state.retrieved_chunks = [chunk.model_dump() for chunk in chunks]
