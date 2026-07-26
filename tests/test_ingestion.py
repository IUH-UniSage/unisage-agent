from fastapi.testclient import TestClient

from app.rag.chunking.recursive import RecursiveChunker


def test_ingestion_endpoint_chunks_text(client: TestClient) -> None:
    response = client.post(
        "/api/v1/ingestion",
        json={"source": "handbook.txt", "content": "Quy chế học vụ\n\nNội dung mẫu."},
    )

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 1


def test_recursive_chunker_preserves_trailing_content() -> None:
    chunker = RecursiveChunker(chunk_size=5, overlap=1)

    chunks = chunker.split("abcdefghij")

    assert chunks[-1] == "ij"
    reconstructed = chunks[0] + "".join(chunk[1:] for chunk in chunks[1:])
    assert reconstructed == "abcdefghij"
