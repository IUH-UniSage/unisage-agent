from app.graph.state import ChatState
from app.rag.generation.agent import generate_answer
from app.schemas.retrieval import RetrievedChunk


def generate_response(state: ChatState) -> str:
    """Generate the final grounded response and citations."""

    chunks = [RetrievedChunk.model_validate(chunk) for chunk in state.retrieved_chunks]
    response, citations = generate_answer(state.query, chunks)
    state.final_response = response
    state.citations = [citation.model_dump() for citation in citations]
    return response
