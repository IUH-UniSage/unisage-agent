from collections.abc import Sequence

from app.rag.generation.citation import build_citations
from app.rag.retrieval.context_builder import build_context
from app.schemas.chat import Citation
from app.schemas.retrieval import RetrievedChunk


def generate_answer(
    query: str,
    chunks: Sequence[RetrievedChunk],
) -> tuple[str, list[Citation]]:
    """Generate a grounded base response without requiring an external LLM key."""

    if not chunks:
        return (
            "Mình chưa tìm thấy tài liệu phù hợp để trả lời câu hỏi này. "
            "Bạn hãy thử bổ sung khoa hoặc khóa học.",
            [],
        )

    context = build_context(chunks)
    response = (
        f"UniSage đã tìm thông tin cho câu hỏi: “{query}”.\n\n"
        f"Thông tin liên quan:\n{context}\n\n"
        "Bạn nên đối chiếu thêm thông báo chính thức của nhà trường "
        "nếu cần đưa ra quyết định cụ thể."
    )
    return response, build_citations(chunks)
