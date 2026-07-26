from app.rag.retrieval.service import RetrievalService


def test_retrieval_returns_only_visible_faculties() -> None:
    service = RetrievalService()

    chunks = service.retrieve(
        "quy chế đào tạo",
        user_faculty="FIT",
        user_level=1,
    )

    assert chunks
    assert all(chunk.faculty in {"GLOBAL", "FIT"} for chunk in chunks)
    assert all(int(chunk.metadata.get("min_user_level", 1)) <= 1 for chunk in chunks)
    assert all(chunk.chunk_id != "demo-003" for chunk in chunks)
    assert chunks == sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
